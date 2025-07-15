import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import os, sys
import netCDF4
import numpy as np
from typing import Optional, Union, Dict, List
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from TimesFM import TimesFM
from stormer_specific_data_utils import WeatherEnergySet

def save_movie(T, channel=1, title='', fps=15):
    os.makedirs('movie', exist_ok=True)
    T_cpu = T.detach().cpu()
    frames = T_cpu[0, :, channel]  # (seq, height, width)
    fig, ax = plt.subplots()
    ax.axis('off')
    if title:
        ax.set_title(title)
    im = ax.imshow(frames[0], animated=True)
    def update(i):
        im.set_array(frames[i])
        return im,
    ani = FuncAnimation(fig, update, frames=frames.shape[0], blit=True)
    ani.save(os.path.join('movie', f'channel{channel}.gif'), writer=PillowWriter(fps=fps))
    plt.close(fig)

def get_model(
        lookback: int = 512,
        lookahead: int = 96,
        lora_rank: Optional[int] = 4,
        weather_model: bool = True,
        load_weights_into_model: bool = True, # load weights via TimesFM load_weights() method
        model_load_kwargs: Dict = {}, # to be used for general TimesFM weights loading
        model_weight_kwargs: Dict = {}, # to be used for TimesFM load_weights
):
    
    model = TimesFM(
        lookback = lookback,
        lookahead = lookahead,
        lora_rank = lora_rank,
        weather_model = weather_model,
        **model_load_kwargs
    )

    if load_weights_into_model is True:
        model.load_weights(**model_weight_kwargs)

    return model

def get_dataloader(
        lookback: int = 512,
        lookahead: int = 96,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        batch_size: int = 128, # in case of multi-node training, this becomes the per-GPU batch size
        is_val: bool = False, # in this case, return the val instead of the test set
        splits: List = [0.8,0.1,0.1],
        energy_nc_path: Union[str, Path] = Path("/eagle/ParaLLMs/weather_load_forecasting/comstock_datasets/small_ca/dset.nc"),
        weather_tensor_h5_path: Union[str, Path] = Path("/eagle/ParaLLMs/weather_load_forecasting/stormer/saved_tensors/tensors.h5"),
        device: Optional[torch.device] = None
):
    
    # check splits are legal and generate cumulatives
    assert len(splits) == 3 and np.isclose(sum(splits),1), "Splits must contain 3 ratios that add upto 1."
    cum_splits = [0] + list(np.cumsum(splits))

    # load the raw data into RAM
    energy_ds = netCDF4.Dataset(str(energy_nc_path), mode="r")
    energy_np = energy_ds.variables["energy_consumption"][:]           # (B, T)
    with h5py.File(str(weather_tensor_h5_path), mode="r") as h5f:
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

    # load train dataset
    train_dset = WeatherEnergySet(
        lookback = lookback,
        lookahead = lookahead,
        energy_np = energy_np,
        weather_np = weather_np,
        start_boundary_ratio = cum_splits[0],
        end_boundary_ratio = cum_splits[1],
        device=device
    )

    # if we have ranks we should create our own sampler
    if (global_rank is not None) and (world_size is not None):
        sampler = DistributedSampler(
            train_dset,
            num_replicas = world_size,
            rank = global_rank,
            shuffle = False,
            seed = 42
        )
    else:
        sampler = None

    # define train loader
    train_loader = DataLoader(
        train_dset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = False,
        sampler = sampler,
        pin_memory = True
    )

    # in case of non-distributed settings or when global rank is 0, generate train or validation set
    if (global_rank is None or world_size is None) or (global_rank == 0):
        val_dset = WeatherEnergySet(
            lookback = lookback,
            lookahead = lookahead,
            energy_np = energy_np,
            weather_np = weather_np,
            start_boundary_ratio = cum_splits[1] if is_val else cum_splits[2],
            end_boundary_ratio = cum_splits[2] if is_val else cum_splits[3]
        )
        val_loader = DataLoader(
            val_dset,
            batch_size = batch_size,
            shuffle = False,
            drop_last = False,
            pin_memory = True
        )
    else:
        val_loader = None

    # return the results
    return train_loader, val_loader

if __name__ == "__main__":

    dl_train, dl_val = get_dataloader(global_rank = 0, world_size = 1, device=torch.device("cpu"),batch_size=1)
    model = get_model()

    loaded_states = torch.load("/home/shourya01/states/model_state.pt")
    new_states = {}
    for k,v in loaded_states.items():
        new_states[k.replace("_orig_mod.module.","")] = v
    model.load_state_dict(new_states, strict=False)
    model = model.cuda()

    for x,y,w in dl_train:
        x,y,w = torch.tensor(x,requires_grad=True).cuda(), y.cuda(), torch.tensor(w,requires_grad=True).cuda()
        loss = F.mse_loss(model(x,w),w,reduction="mean")
        grad_w = torch.autograd.grad(loss, w, retain_graph=True)[0]
        print(grad_w)
        save_movie(w.grad,channel=1,title="Temperature")
        print("Movie saved!")
        break

