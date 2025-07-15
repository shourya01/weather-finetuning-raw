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
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mtick
from itertools import islice
from datetime import datetime, timedelta

# enable matplotlib to use latex
plt.rcParams.update({
    "text.usetex": True,           # switch to LaTeX
    "font.family": "serif",        # optional: match LaTeX default
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",  # extra pkgs
    "font.size": 14
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
    # for plotting
    parser.add_argument("--batch_number", type=int, default=0)
    parser.add_argument("--in_batch_sample", type=int, default=0)
    # generate args and modify the config file name
    args = parser.parse_args()
    args.deepspeed_config = f"ds_config_{args.experiment_name}.json"
    # clamp in-batch sample
    clamp = lambda value, min_value, max_value: max(min_value, min(max_value - 1, value))
    args.in_batch_sample = clamp(args.in_batch_sample, 0, 4*args.per_device_batch_size)
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

def get_model(args, print_missing_unexpected: bool = False, seed: int = 42, load_weights: bool = True):

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
    if load_weights is True:
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

    # here we modify the args
    args.dataset_name = "california"
    args.experiment_name = "exp4_cali_timesfm_only"
    args.use_weather = False

    # get no-weather model and variables to configure it
    model_noweather = get_model(args, load_weights=False)
    epoch_to_load = 1
    noweather_epoch_path = Path(args.results_base_dir) / Path(args.experiment_name) / Path('checkpoint') / Path(f"epoch_{epoch_to_load-1}") / Path("mp_rank_00_model_states.pt")

    # modify args for weather model
    args.experiment_name = "exp7_cali_timesfm_stormer"
    args.use_weather = True

    # get weather model
    model_weather = get_model(args)
    epoch_to_load = 40
    weather_epoch_path = Path(args.results_base_dir) / Path(args.experiment_name) / Path('checkpoint') / Path(f"epoch_{epoch_to_load-1}") / Path("mp_rank_00_model_states.pt")

    # load weights into no-weather model
    wt = torch.load(str(noweather_epoch_path))
    model_noweather.load_state_dict(wt['module'])
    model_noweather = nn.DataParallel(model_noweather)
    model_noweather.cuda()

    # load weights into weather model
    wt = torch.load(str(weather_epoch_path))
    model_weather.load_state_dict(wt['module'])
    model_weather = nn.DataParallel(model_weather)
    model_weather.cuda()

    # get dataset
    set_seed(0)
    train_ds, val_ds, test_ds = get_dataset(args)
    dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    # run the dataloader
    def nth(it, i, default=None):
        return next(islice(it, i, None), default)
    x, y, w = nth(dl, args.batch_number)
    x, y, w = x[args.in_batch_sample,...].unsqueeze(0).cuda(), y[args.in_batch_sample,...].unsqueeze(0).cuda(), w[args.in_batch_sample,...].unsqueeze(0).cuda()
    lookback = x.detach().clone().cpu().squeeze().numpy()
    output_noweather = model_noweather(x).detach().cpu().squeeze().numpy()
    output_weather = model_weather(x,w).detach().cpu().squeeze().numpy()
    ground_truth = y.detach().clone().cpu().squeeze().numpy()
    done = True

    # define axes
    x1 = range(0,lookback.size)
    x2 = range(lookback.size,lookback.size+output_weather.size)

    # function
    BASE = datetime(2018, 1, 1, 0, 15)
    def quarter_hours(idx_range):
        """range(1,3) ➜ [2018-01-01 00:30, 00:45] (list of datetime objects)."""
        return [BASE + timedelta(minutes=15*i) for i in idx_range]
    x1, x2 = quarter_hours(x1), quarter_hours(x2)

    # plot
    fig = plt.figure()
    plt.title("Residential Building in Alameda County, CA")
    plt.plot(x1,lookback,'k',label="Past energy consumption")
    plt.plot(x2,output_noweather,'r',label="Forecast, TimesFM only")
    plt.plot(x2,output_weather,'b',label="Forecast, TimesFM+Stormer", linewidth=2)
    plt.plot(x2,ground_truth,'k--',label="Ground truth")
    # plt.xlim(400,lookback.size+output_weather.size)
    plt.legend()

    # do funky stuff
    ax = plt.gca()                        # current axes
    end   = x2[-1]
    start = end - timedelta(minutes=15*299)   # 300 points incl. end
    ax.set_xlim(start, end)

    # 2. exactly three boxed x‑tick labels
    ax.xaxis.set_major_locator(LinearLocator(3))
    for lab in ax.get_xticklabels():
        lab.set_bbox(dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=.5))

    # 3. show y‑ticks shifted by +0.6 (data unchanged)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v+0.63:g}"))
    plt.ylabel("Load (kWh)")

    # Save
    plt.savefig("COSMOS.png", format="png", bbox_inches="tight", dpi=300)
    plt.close()
    