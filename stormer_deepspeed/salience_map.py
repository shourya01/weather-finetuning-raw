# load mpi first to avoid GTL errors
from mpi4py import MPI

# now do standard imports
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from deepspeed import initialize, comm, init_distributed
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
import netCDF4
import h5py
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

# enable matplotlib to use latex
plt.rcParams.update({
    "text.usetex": True,           # switch to LaTeX
    "font.family": "serif",        # optional: match LaTeX default
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}"  # extra pkgs
})

# import custom model functions
from TimesFM import TimesFM
from stormer_specific_data_utils import WeatherEnergySet
from utils import generate_and_save_ds_config, generate_sharded_dataloader, train_one_epoch, evaluate

def parse_args():

    # Minimal arguments needed to run DeepSpeed.
    parser = argparse.ArgumentParser(description="MWE for running a DeepSpeed training.")
    # Deepspeed configs
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    # Batch size and optimizer arguments
    parser.add_argument("--per_device_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=lambda x: float(x), default=5e-5) # will be overridden if tuning is enabled
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # File paths
    parser.add_argument("--timesfm_ckpt_path", type=str, default="/home/shourya01/timesfm/timesfm-1.0-200m-pytorch/torch_model.ckpt")
    parser.add_argument("--energy_nc_base", type=str, default="/eagle/ParaLLMs/weather_load_forecasting/comstock_datasets")
    parser.add_argument("--dataset_name", type=str, default="california")
    parser.add_argument("--weather_tensor_h5_path", type=str, default="/eagle/ParaLLMs/weather_load_forecasting/stormer/saved_tensors/tensors.h5")
    # Model configuration arguments
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--lookahead", type=int, default=96)
    parser.add_argument("--use_weather", action="store_true") # note that if this behavior is necessary, this flag is MANDATORY
    parser.add_argument("--weather_tokens_to_use", type=int, default=16)
    parser.add_argument("--weather_decoder_layers", type=int, default=16)
    # Results directory and experiment name
    parser.add_argument("--results_base_dir", type=str, default="/eagle/ParaLLMs/weather_load_forecasting/results")
    parser.add_argument("--experiment_name", type=str, default="california-timesfm-stormer")
    # Tuning results/override
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--tuning_file", type=str, default="tune.csv")
    parser.add_argument("--val_epochs", type=int, default=2)
    parser.add_argument("--lr_candidates", type=lambda s: [float(x.strip()) for x in s.split(',') if x.strip()], default=[5e-5])
    # Train configurations
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--train_epochs", type=int, default=20)
    # Test set frequency and file name
    parser.add_argument("--test_file", type=str, default="eval.csv")
    parser.add_argument("--test_every", type=int, default=2)
    # Salience
    parser.add_argument("--load_epoch", type=int, default=19)
    parser.add_argument("--salience_samples", type=int, default=50)
    # Top what percentage
    parser.add_argument("--top_percs", type=lambda s: [float(x.strip()) for x in s.split(',') if x.strip()], default=[1.,5.,10.,20.,50.])
    # generate args and modify the config file name
    args = parser.parse_args()
    args.deepspeed_config = f"ds_config_{args.experiment_name}.json"
    # return
    return args

def set_seed(seed: int = 42):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args, print_missing_unexpected: bool = False, seed: int = 42):

    # set the random seed
    set_seed(seed)

    # define model
    model = TimesFM(
        lookback = args.lookback,
        lookahead = args.lookahead,
        lora_rank = args.lora_rank if args.lora_rank > 0 else None,
        weather_model = args.use_weather,
        weather_tokens_to_use = args.weather_tokens_to_use,
        weather_decoder_layers = args.weather_decoder_layers,
        ckpt = args.timesfm_ckpt_path,
    )

    # load weights 
    model.load_weights(print_missing_unexpected = print_missing_unexpected)

    # return
    return model

def get_dataset(args = None, splits = [0.8,0.1,0.1]):

    # assert that inputs are valid
    assert args is not None, "Configuration args not passed into get_dataloader()."
    assert all([i>0 for i in splits]) and np.isclose(sum(splits),1.), "Ensure train-val-test split is a 3-list that is non-negative and adds upto 1."
    cum_splits = [0] + list(np.cumsum(splits))

    # load raw data into memory
    energy_ds = netCDF4.Dataset(str(Path(args.energy_nc_base) / Path(args.dataset_name) / Path(args.dataset_name+".nc")), mode="r")
    energy_np = energy_ds.variables["energy_consumption"][:]           # (B, T)
    with h5py.File(str(args.weather_tensor_h5_path), mode="r") as h5f:
        root = h5f["/data"]
        if isinstance(root, h5py.Dataset):               # single-dataset layout
            weather_np = root[...]
        else:                                            # per-hour group layout
            hour_keys = sorted(root.keys(), key=int)
            sample = root[hour_keys[0]]
            num_hours = len(hour_keys)
            num_tiers, num_vars, H, W = sample.shape
            weather_np = np.empty(
                (num_hours, num_tiers, num_vars, H, W), dtype=sample.dtype
            )
            for idx, k in enumerate(hour_keys):
                weather_np[idx] = root[k][...]

    # generate train dataset object
    train_dset = WeatherEnergySet(
        lookback = args.lookback,
        lookahead = args.lookahead,
        energy_np = energy_np,
        weather_np = weather_np,
        start_boundary_ratio = cum_splits[0],
        end_boundary_ratio = cum_splits[1],
    )

    # generate validation dataset object
    val_dset = WeatherEnergySet(
        lookback = args.lookback,
        lookahead = args.lookahead,
        energy_np = energy_np,
        weather_np = weather_np,
        start_boundary_ratio = cum_splits[1],
        end_boundary_ratio = cum_splits[2],
        energy_mean = train_dset.energy_mean,
        energy_std = train_dset.energy_std,
        weather_mean = train_dset.weather_mean,
        weather_std = train_dset.weather_std
    )

    # generate test dataset object
    test_dset = WeatherEnergySet(
        lookback = args.lookback,
        lookahead = args.lookahead,
        energy_np = energy_np,
        weather_np = weather_np,
        start_boundary_ratio = cum_splits[2],
        end_boundary_ratio = cum_splits[3],
        energy_mean = train_dset.energy_mean,
        energy_std = train_dset.energy_std,
        weather_mean = train_dset.weather_mean,
        weather_std = train_dset.weather_std
    )

    return train_dset, val_dset, test_dset

def get_loss_dict(mean, std):

    # This function provides the loss function dictionary assu,ing that y and yhat are already normalized
    # and mean/std are the normalization factors
    unnormalize = lambda z, mean = mean, std = std: std.item() * z + mean.item()

    loss_dict = {
        'nmse_loss': lambda yhat, y: F.mse_loss(yhat, y, reduction="mean"),
        'nmae_loss': lambda yhat, y: F.l1_loss(yhat, y, reduction="mean"),
        'mse_loss': lambda yhat, y: F.mse_loss(unnormalize(yhat), unnormalize(y), reduction="mean"),
        'mae_loss': lambda yhat, y: F.l1_loss(unnormalize(yhat), unnormalize(y), reduction="mean"),
        'mape_loss': lambda yhat, y: 100 * torch.mean( torch.abs(unnormalize(yhat) - unnormalize(y)) / (torch.abs(unnormalize(y)) + 1e-6) ),
        'smape_loss': lambda yhat, y: 100 * torch.mean( torch.abs(unnormalize(yhat) - unnormalize(y)) / (torch.abs(unnormalize(yhat)) + torch.abs(unnormalize(y)) + 1e-6) )
    }

    # return
    return loss_dict

"""
Length of train dataset is 329100, validation dataset is 34764, and test dataset is 34764.
Each dataloader entry contains 3 items.
The shape of item number 1 (x) is torch.Size([512]). # lookback
The shape of item number 2 (y) is torch.Size([96]). # lookahead
The shape of item number 3 (weather) is torch.Size([152, 5, 32, 64]). # (lookback / 4, channels, height, width)
"""

if __name__ == "__main__":

    # Get arguments
    args = parse_args()

    # # Get model and datasets
    model = get_model(args)
    train_ds, val_ds, test_ds = get_dataset(args)

    # # Generate dataloader
    set_seed(0)
    dl = DataLoader(train_ds, batch_size = 128, shuffle=True)

    # provide experiment_name, pass --use_weather, and provide load_epoch

    # run an example 
    epoch_path = Path(args.results_base_dir) / Path(args.experiment_name) / Path('checkpoint') / Path(f"epoch_{args.load_epoch}") / Path("mp_rank_00_model_states.pt")
    wt = torch.load(str(epoch_path))
    results = model.load_state_dict(wt['module'],strict=False)
    model = nn.DataParallel(model)  # <-- replicate on all visible GPUs
    model.cuda() 
    
    # calculate salience
    s = None
    count = 0
    var = 0
    for x, y, w in tqdm(dl):
        x,y,w = x.cuda(), y.cuda(), w.cuda()
        x,w = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
        out = model(x,w)
        target = out.pow(2).sum()**(1/2)
        model.zero_grad()
        target.backward()
        temp = w.grad[:,:,var,:,:].abs().detach().cpu().numpy() 
        count += 1
        for tx in temp:
            for idx,ti in enumerate(tx):
                if s is None:
                    s = ti
                else:
                    s += ti
        if count == args.salience_samples:
            break
    # s = (s - s.min()) / (s.max() - s.min())

    # print frame shape
    print(f"\n---Salience map shape is {s.shape}.---\n\n")

    # plot
    lat_lon = lambda M,N: (np.linspace(-90 + (180/M + 320/N)/2 / 2, 90 + (180/M + 320/N)/2 / 2 , M), 
                       np.linspace(0, 360, N, endpoint=True))
    lat, lon = lat_lon(32, 64)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cf = ax.contourf(lon_grid, lat_grid, s, transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, orientation='vertical', label=r'$\mathbb{E}_{t\in[L+T],s\in\text{unif}([100])}\left(\delta\{\text{weather}_{t,s}\}\right)$')
    plt.title(f"Dataset: {args.dataset_name}, (TimesFM,Stormer)")
    plt.savefig(f"fram_{args.experiment_name}_{var}.png", format = "png", bbox_inches="tight", dpi=300)
    plt.close()

    for perc in args.top_percs:

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()             

        # Calculate the 90th percentile of s
        threshold = np.percentile(s, int(100-perc))

        # Define two-color colormap
        colors = ['gray', 'white']
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm([s.min(), threshold, s.max()], cmap.N)

        # Plot with custom colormap
        cf = ax.contourf(lon_grid, lat_grid, s, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        plt.colorbar(cf, ax=ax, orientation='vertical', label=r'$\mathbb{E}_{t\in[L+T],s\in\text{unif}([100])}\left(\delta\{\text{weather}_{t,s}\}\right)$')
        plt.title(f"Dataset: {args.dataset_name}, (TimesFM,Stormer), Thres={perc}")
        plt.savefig(f"fram_{args.experiment_name}_{var}_top{int(perc)}.png", format="png", bbox_inches="tight", dpi=300)
        plt.close()