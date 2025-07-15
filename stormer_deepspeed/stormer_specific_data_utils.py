import os
import gc
import h5py
import math
import time
import torch
import netCDF4
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path
from functools import cache
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union, List


def get_data_given_path(path, variables):
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items() if sub_key in variables + ['time']
        } for main_key, group in f.items() if main_key in ['input']}

    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0)

class ERA5ModuloInpOnly(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        data_freq = 6,
        modulo=0
    ):
        super().__init__()

        # ensure that modulo values are well defined
        assert any(modulo==i for i in range(data_freq)), "Initial modulo must be an integer between 0 and data_freq-1."

        self.root_dir = root_dir
        self.variables = variables
        
        # do modulo selection
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = [file_paths[i] for i in range(modulo, len(file_paths), 6)]
        # print(f"For modulo {modulo}, detected {len(file_paths)} files, start: {file_paths[0]}, end: {file_paths[-1]}")
        
        self.inp_file_paths = file_paths
        
    def __len__(self):
        return len(self.inp_file_paths)
    
    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)
        inp_data = torch.from_numpy(inp_data)
        
        return inp_data

class InterleavedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = np.cumsum([len(ds) for ds in datasets])

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.lengths, idx, side='right')
        if dataset_idx > 0:
            idx -= self.lengths[dataset_idx - 1]
        return self.datasets[dataset_idx][idx]
    
class WeatherEnergySet(Dataset):
    """
    Returns (past_energy, future_energy, weather_tensor).

    Implements:
      • Item 5  – down-sample weather from 128×256 to 32×64
                  with AdaptiveAvgPool2d, executed once in __init__.
    """

    def __init__(
        self,
        lookback: int = 512,
        lookahead: int = 96,
        energy_np: Optional[np.ndarray] = None,
        weather_np: Optional[Union[List,np.ndarray]] = None,
        start_boundary_ratio: float = 0.0,
        end_boundary_ratio: float = 0.8,
        weather_mean: Optional[Sequence[float]] = None,
        weather_std:  Optional[Sequence[float]] = None,
        energy_mean: Optional[float] = None,
        energy_std:  Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
    ):
        # ───────── Assert weather and energy exist ────────────────────────────
        assert (energy_np is not None) and (weather_np is not None), "Energy and weather must be supplied."

        # ───────── Load energy ────────────────────────────────────────────────
        self.energy_tensor = torch.tensor(energy_np, dtype=dtype)          # CPU
        num_buildings, num_steps = self.energy_tensor.shape

        # ── Compute or assign global energy stats ─────────
        if energy_mean is None or energy_std is None:
            start_idx = int(start_boundary_ratio * num_steps)
            end_idx   = int(end_boundary_ratio   * num_steps)
            flat_e = self.energy_tensor[:, start_idx:end_idx].reshape(-1)
            mean_e = flat_e.mean()
            std_e  = flat_e.std() + 1e-6
        else:
            mean_e = torch.as_tensor(energy_mean, dtype=dtype)
            std_e  = torch.as_tensor(energy_std,  dtype=dtype) + 1e-6
        self.energy_mean, self.energy_std = mean_e, std_e

        # ───────── Down-sample to 32×64 (Item 5) ─────────────────────────────
        weather_torch = torch.as_tensor(weather_np, dtype=dtype)           # CPU
        del weather_np                                                     # free NumPy
        gc.collect()

        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        weather_torch = weather_torch.to(device, non_blocking=True)        # GPU accel
        h, t, v, H, W = weather_torch.shape                                # (hours, tiers, vars, 128,256)

        weather_flat = weather_torch.view(-1, v, H, W)                     # flatten (hours*tier)
        pooled = F.adaptive_avg_pool2d(weather_flat, output_size=(32, 64))
        weather_small = pooled.view(h, t, v, 32, 64).cpu().contiguous()    # back to CPU

        # clean GPU & large tensor
        del weather_torch, pooled
        torch.cuda.empty_cache()

        self.weather_flat = weather_small.view(h * t, v, 32, 64)           # flattened for fast gather
        self.num_hours, self.num_tiers, self.num_vars, self.H, self.W = (
            h, t, v, 32, 64,
        )

        # ── Compute or assign per-channel stats on flattened data ────────
        if weather_mean is None or weather_std is None:
            flat = self.weather_flat               # (h*t, v,32,64)
            h0 = int(start_boundary_ratio * flat.size(0))
            h1 = int(end_boundary_ratio   * flat.size(0))
            subset = flat[h0:h1]                  # (h'*t, v,32,64)
            mean = subset.mean(dim=(0,2,3))       # (v,)
            std  = subset.std(dim=(0,2,3)) + 1e-6  # (v,)
        else:
            mean = torch.as_tensor(weather_mean, dtype=dtype)
            std  = torch.as_tensor(weather_std,  dtype=dtype) + 1e-6
        self.weather_mean, self.weather_std = mean, std

        # ───────── Time bookkeeping ─────────────────────────────────────────
        assert 0.0 <= start_boundary_ratio < end_boundary_ratio <= 1.0
        self.dtype = dtype

        self.lookback = lookback
        self.lookahead = lookahead
        self.past_hours = lookback // 4
        self.future_hours = lookahead // 4
        window = lookback + lookahead

        start_step = int(start_boundary_ratio * num_steps)
        end_step = int(end_boundary_ratio * num_steps)
        assert end_step - start_step >= window, "Boundary window too small."

        # index_map holds (building, start_step) pairs
        self.index_map: List[Tuple[int, int]] = [
            (b, s) for b in range(num_buildings)
            for s in range(start_step, end_step - window + 1)
        ]

        self.tiers_per_hour = self.num_tiers                               # =5

    # ───────── Dataset API ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        building, start_step = self.index_map[idx]

        # Energy slices
        e_series = self.energy_tensor[building]
        past_e = e_series[start_step : start_step + self.lookback]
        future_e = e_series[
            start_step + self.lookback : start_step + self.lookback + self.lookahead
        ]

        # normalize energy
        past_e   = (past_e   - self.energy_mean) / self.energy_std
        future_e = (future_e - self.energy_mean) / self.energy_std

        # Weather slices
        base_hour = start_step // 4

        # past observations (tier 0)
        past_hours = torch.arange(base_hour,
                                  base_hour + self.past_hours,
                                  dtype=torch.long)
        past_lin = past_hours * self.tiers_per_hour
        w_past = self.weather_flat.index_select(0, past_lin)

        # future forecasts (tiers 1-4)
        f1 = torch.arange(1, self.future_hours + 1, dtype=torch.long)
        tier_idx = ((f1 - 1) // 6) + 1
        offset = (6 * tier_idx) - f1
        future_hours = base_hour + self.past_hours - offset
        fut_lin = future_hours * self.tiers_per_hour + tier_idx
        w_future = self.weather_flat.index_select(0, fut_lin)

        w = torch.cat([w_past, w_future], dim=0)

        # normalize per-channel
        w = (w - self.weather_mean.view(1, -1, 1, 1)) \
            / self.weather_std .view(1, -1, 1, 1)

        return past_e, future_e, w
    
class LocalWeatherEnergySet(Dataset):
    """
    Returns (past_energy, future_energy, weather_tensor).

    Implements:
      • Item 5  – down-sample weather from 128×256 to 32×64
                  with AdaptiveAvgPool2d, executed once in __init__.
    """

    def __init__(
        self,
        lookback: int = 512,
        lookahead: int = 96,
        energy_np: Optional[np.ndarray] = None,
        weather_np: Optional[np.ndarray] = None,
        start_boundary_ratio: float = 0.0,
        end_boundary_ratio: float = 0.8,
        weather_mean: Optional[np.ndarray] = None,
        weather_std:  Optional[np.ndarray] = None,
        energy_mean: Optional[float] = None,
        energy_std:  Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # ───────── Load energy ────────────────────────────────────────────────
        # energy_ds = netCDF4.Dataset(str(energy_nc_path), mode="r")
        # energy_np = energy_ds.variables["energy_consumption"][:]           # (B, T)
        self.energy_tensor = torch.as_tensor(energy_np, dtype=dtype)          # CPU
        # weather_np = energy_ds.variables["weather"][:]                      # (B, T, V)
        self.weather_tensor = torch.as_tensor(weather_np, dtype=dtype)
        num_buildings, num_steps = self.energy_tensor.shape

        # ── Compute or assign global energy stats ─────────
        if energy_mean is None or energy_std is None:
            start_idx = int(start_boundary_ratio * num_steps)
            end_idx   = int(end_boundary_ratio   * num_steps)
            flat_e = self.energy_tensor[:, start_idx:end_idx].reshape(-1)
            mean_e = flat_e.mean()
            std_e  = flat_e.std() + 1e-6
        else:
            mean_e = torch.as_tensor(energy_mean, dtype=dtype)
            std_e  = torch.as_tensor(energy_std,  dtype=dtype) + 1e-6
        self.energy_mean, self.energy_std = mean_e, std_e

        # ── Compute or assign local weather stats ─────────
        if weather_mean is None or weather_std is None:
            start_idx = int(start_boundary_ratio * num_steps)
            end_idx   = int(end_boundary_ratio   * num_steps)
            w = self.weather_tensor[:, start_idx:end_idx, :]
            mean_w = torch.tensor(w.mean(axis=(0,1), keepdims=True), dtype=dtype)
            std_w = torch.tensor(w.std(axis=(0,1), keepdims=True), dtype=dtype) + 1e-6
        else:
            mean_w = torch.tensor(weather_mean, dtype=dtype)
            std_w  = torch.tensor(weather_std,  dtype=dtype) + 1e-6
        self.mean_w, self.std_w = mean_w, std_w 
        self.weather_mean, self.weather_std = mean_w.cpu().numpy(), std_w.cpu().numpy()

        # ───────── Time bookkeeping ─────────────────────────────────────────
        assert 0.0 <= start_boundary_ratio < end_boundary_ratio <= 1.0
        self.dtype = dtype

        self.lookback = lookback
        self.lookahead = lookahead
        self.past_hours = lookback // 4
        self.future_hours = lookahead // 4
        window = lookback + lookahead

        start_step = int(start_boundary_ratio * num_steps)
        end_step = int(end_boundary_ratio * num_steps)
        assert end_step - start_step >= window, "Boundary window too small."

        # index_map holds (building, start_step) pairs
        self.index_map: List[Tuple[int, int]] = [
            (b, s) for b in range(num_buildings)
            for s in range(start_step, end_step - window + 1)
        ]

    # ───────── Dataset API ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        building, start_step = self.index_map[idx]

        # Energy slices
        e_series = self.energy_tensor[building]
        past_e = e_series[start_step : start_step + self.lookback]
        future_e = e_series[
            start_step + self.lookback : start_step + self.lookback + self.lookahead
        ]

        # normalize energy
        past_e   = (past_e   - self.energy_mean) / self.energy_std
        future_e = (future_e - self.energy_mean) / self.energy_std

        # Weather slices
        w_series = self.weather_tensor[building]
        past_w = w_series[start_step : start_step + self.lookback]
        future_w = w_series[
            start_step + self.lookback : start_step + self.lookback + self.lookahead
        ]

        # normalize weather
        past_w = (past_w - self.mean_w[0,:,:]) / self.std_w[0,:,:]
        future_w = (future_w - self.mean_w[0,:,:]) / self.std_w[0,:,:]

        return past_e, future_e, past_w, future_w
    
class WeatherEnergySetWithCoords(Dataset):
    """
    Returns (past_energy, future_energy, weather_tensor).

    Implements:
      • Item 5  – down-sample weather from 128×256 to 32×64
                  with AdaptiveAvgPool2d, executed once in __init__.
    """

    def __init__(
        self,
        lookback: int = 512,
        lookahead: int = 96,
        energy_np: Optional[np.ndarray] = None,
        weather_np: Optional[Union[List,np.ndarray]] = None,
        building_coords_np: Optional[Union[List, np.ndarray]] = None,
        weather_resolution: tuple = (32, 64),
        start_boundary_ratio: float = 0.0,
        end_boundary_ratio: float = 0.8,
        weather_mean: Optional[Sequence[float]] = None,
        weather_std:  Optional[Sequence[float]] = None,
        energy_mean: Optional[float] = None,
        energy_std:  Optional[float] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
    ):
        # ───────── Assert weather and energy exist ────────────────────────────
        assert (energy_np is not None) and (weather_np is not None), "Energy and weather must be supplied."

        # ───────── Load energy ────────────────────────────────────────────────
        self.energy_tensor = torch.tensor(energy_np, dtype=dtype)          # CPU
        num_buildings, num_steps = self.energy_tensor.shape

        # ── Compute or assign global energy stats ─────────
        if energy_mean is None or energy_std is None:
            start_idx = int(start_boundary_ratio * num_steps)
            end_idx   = int(end_boundary_ratio   * num_steps)
            flat_e = self.energy_tensor[:, start_idx:end_idx].reshape(-1)
            mean_e = flat_e.mean()
            std_e  = flat_e.std() + 1e-6
        else:
            mean_e = torch.as_tensor(energy_mean, dtype=dtype)
            std_e  = torch.as_tensor(energy_std,  dtype=dtype) + 1e-6
        self.energy_mean, self.energy_std = mean_e, std_e

        # save building coordinates and resolution
        self.building_coords = building_coords_np
        M,N = weather_resolution
        self.lat = np.linspace(-90 + (180/M + 320/N)/2 / 2, 90 - (180/M + 320/N)/2 / 2 , M) # vector of latitude grid points
        self.lon = np.linspace(0, 360, N, endpoint=False) # vector of longitude grid points
        self.s2us = lambda x: x + 180 # lambda function for changing longitudes in range [0, 360]

        # ───────── Down-sample to 32×64 (Item 5) ─────────────────────────────
        weather_torch = torch.as_tensor(weather_np, dtype=dtype)           # CPU
        del weather_np                                                     # free NumPy
        gc.collect()

        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        weather_torch = weather_torch.to(device, non_blocking=True)        # GPU accel
        h, t, v, H, W = weather_torch.shape                                # (hours, tiers, vars, 128,256)

        weather_flat = weather_torch.view(-1, v, H, W)                     # flatten (hours*tier)
        pooled = F.adaptive_avg_pool2d(weather_flat, output_size=(32, 64))
        weather_small = pooled.view(h, t, v, 32, 64).cpu().contiguous()    # back to CPU

        # clean GPU & large tensor
        del weather_torch, pooled
        torch.cuda.empty_cache()

        self.weather_flat = weather_small.view(h * t, v, 32, 64)           # flattened for fast gather
        self.num_hours, self.num_tiers, self.num_vars, self.H, self.W = (
            h, t, v, 32, 64,
        )

        # ── Compute or assign per-channel stats on flattened data ────────
        if weather_mean is None or weather_std is None:
            flat = self.weather_flat               # (h*t, v,32,64)
            h0 = int(start_boundary_ratio * flat.size(0))
            h1 = int(end_boundary_ratio   * flat.size(0))
            subset = flat[h0:h1]                  # (h'*t, v,32,64)
            mean = subset.mean(dim=(0,2,3))       # (v,)
            std  = subset.std(dim=(0,2,3)) + 1e-6  # (v,)
        else:
            mean = torch.as_tensor(weather_mean, dtype=dtype)
            std  = torch.as_tensor(weather_std,  dtype=dtype) + 1e-6
        self.weather_mean, self.weather_std = mean, std

        # ───────── Time bookkeeping ─────────────────────────────────────────
        assert 0.0 <= start_boundary_ratio < end_boundary_ratio <= 1.0
        self.dtype = dtype

        self.lookback = lookback
        self.lookahead = lookahead
        self.past_hours = lookback // 4
        self.future_hours = lookahead // 4
        window = lookback + lookahead

        start_step = int(start_boundary_ratio * num_steps)
        end_step = int(end_boundary_ratio * num_steps)
        assert end_step - start_step >= window, "Boundary window too small."

        # index_map holds (building, start_step) pairs
        self.index_map: List[Tuple[int, int]] = [
            (b, s) for b in range(num_buildings)
            for s in range(start_step, end_step - window + 1)
        ]

        self.tiers_per_hour = self.num_tiers                               # =5

    # ───────── Functions to generate coordinate indices ─────────────────────
    @cache
    def get_coord_idx(self, lat, lon):
        return int(np.argmin(np.abs(self.lat - lat))), int(np.argmin(np.abs(self.lon - self.s2us(lon))))

    # ───────── Dataset API ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        building, start_step = self.index_map[idx]
        building_coords = self.building_coords[building]
        i_bldg, j_bldg = self.get_coord_idx(building_coords[0].item(), building_coords[1].item())

        # Energy slices
        e_series = self.energy_tensor[building]
        past_e = e_series[start_step : start_step + self.lookback]
        future_e = e_series[
            start_step + self.lookback : start_step + self.lookback + self.lookahead
        ]

        # normalize energy
        past_e   = (past_e   - self.energy_mean) / self.energy_std
        future_e = (future_e - self.energy_mean) / self.energy_std

        # Weather slices
        base_hour = start_step // 4

        # past observations (tier 0)
        past_hours = torch.arange(base_hour,
                                  base_hour + self.past_hours,
                                  dtype=torch.long)
        past_lin = past_hours * self.tiers_per_hour
        w_past = self.weather_flat.index_select(0, past_lin)

        # future forecasts (tiers 1-4)
        f1 = torch.arange(1, self.future_hours + 1, dtype=torch.long)
        tier_idx = ((f1 - 1) // 6) + 1
        offset = (6 * tier_idx) - f1
        future_hours = base_hour + self.past_hours - offset
        fut_lin = future_hours * self.tiers_per_hour + tier_idx
        w_future = self.weather_flat.index_select(0, fut_lin)

        w = torch.cat([w_past, w_future], dim=0)

        # normalize per-channel
        w = (w - self.weather_mean.view(1, -1, 1, 1)) \
            / self.weather_std .view(1, -1, 1, 1)
        
        # select the appropriate coordinate
        w = w[:,:,i_bldg,j_bldg]

        return past_e, future_e, w
    
class LocalWeatherEnergySet2(Dataset):
    """
    Returns (past_energy, future_energy, weather_tensor).

    Implements:
      • Item 5  – down-sample weather from 128×256 to 32×64
                  with AdaptiveAvgPool2d, executed once in __init__.
    Does not output future weather.
    """

    def __init__(
        self,
        lookback: int = 512,
        lookahead: int = 96,
        energy_np: Optional[np.ndarray] = None,
        weather_np: Optional[np.ndarray] = None,
        start_boundary_ratio: float = 0.0,
        end_boundary_ratio: float = 0.8,
        weather_mean: Optional[np.ndarray] = None,
        weather_std:  Optional[np.ndarray] = None,
        energy_mean: Optional[float] = None,
        energy_std:  Optional[float] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # ───────── Load energy ────────────────────────────────────────────────
        # energy_ds = netCDF4.Dataset(str(energy_nc_path), mode="r")
        # energy_np = energy_ds.variables["energy_consumption"][:]           # (B, T)
        self.energy_tensor = torch.as_tensor(energy_np, dtype=dtype)          # CPU
        # weather_np = energy_ds.variables["weather"][:]                      # (B, T, V)
        self.weather_tensor = torch.as_tensor(weather_np, dtype=dtype)
        num_buildings, num_steps = self.energy_tensor.shape

        # ── Compute or assign global energy stats ─────────
        if energy_mean is None or energy_std is None:
            start_idx = int(start_boundary_ratio * num_steps)
            end_idx   = int(end_boundary_ratio   * num_steps)
            flat_e = self.energy_tensor[:, start_idx:end_idx].reshape(-1)
            mean_e = flat_e.mean()
            std_e  = flat_e.std() + 1e-6
        else:
            mean_e = torch.as_tensor(energy_mean, dtype=dtype)
            std_e  = torch.as_tensor(energy_std,  dtype=dtype) + 1e-6
        self.energy_mean, self.energy_std = mean_e, std_e

        # ── Compute or assign local weather stats ─────────
        if weather_mean is None or weather_std is None:
            start_idx = int(start_boundary_ratio * num_steps)
            end_idx   = int(end_boundary_ratio   * num_steps)
            w = self.weather_tensor[:, start_idx:end_idx, :]
            mean_w = torch.tensor(w.mean(axis=(0,1), keepdims=True), dtype=dtype)
            std_w = torch.tensor(w.std(axis=(0,1), keepdims=True), dtype=dtype) + 1e-6
        else:
            mean_w = torch.tensor(weather_mean, dtype=dtype)
            std_w  = torch.tensor(weather_std,  dtype=dtype) + 1e-6
        self.mean_w, self.std_w = mean_w, std_w 
        self.weather_mean, self.weather_std = mean_w.cpu().numpy(), std_w.cpu().numpy()

        # ───────── Time bookkeeping ─────────────────────────────────────────
        assert 0.0 <= start_boundary_ratio < end_boundary_ratio <= 1.0
        self.dtype = dtype

        self.lookback = lookback
        self.lookahead = lookahead
        self.past_hours = lookback // 4
        self.future_hours = lookahead // 4
        window = lookback + lookahead

        start_step = int(start_boundary_ratio * num_steps)
        end_step = int(end_boundary_ratio * num_steps)
        assert end_step - start_step >= window, "Boundary window too small."

        # index_map holds (building, start_step) pairs
        self.index_map: List[Tuple[int, int]] = [
            (b, s) for b in range(num_buildings)
            for s in range(start_step, end_step - window + 1)
        ]

    # ───────── Dataset API ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        building, start_step = self.index_map[idx]

        # Energy slices
        e_series = self.energy_tensor[building]
        past_e = e_series[start_step : start_step + self.lookback]
        future_e = e_series[
            start_step + self.lookback : start_step + self.lookback + self.lookahead
        ]

        # normalize energy
        past_e   = (past_e   - self.energy_mean) / self.energy_std
        future_e = (future_e - self.energy_mean) / self.energy_std

        # Weather slices
        w_series = self.weather_tensor[building]
        past_w = w_series[start_step : start_step + self.lookback]
        # future_w = w_series[
        #     start_step + self.lookback : start_step + self.lookback + self.lookahead
        # ]

        # normalize weather
        past_w = (past_w - self.mean_w[0,:,:]) / self.std_w[0,:,:]
        # future_w = (future_w - self.mean_w[0,:,:]) / self.std_w[0,:,:]

        # downsample the weather
        k, target_dim = 4, 0 # subsample factor 15 min - > 1 hr
        idx = torch.arange(0, past_w.size(target_dim), k, device=past_w.device)
        past_w = past_w.index_select(target_dim, idx)

        return past_e, future_e, past_w