import h5py
import sys
from pathlib import Path
from netCDF4 import Dataset
import numpy as np
from time import perf_counter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
from math import fmod

# enable matplotlib to use latex
plt.rcParams.update({
    "text.usetex": True,           # switch to LaTeX
    "font.family": "serif",        # optional: match LaTeX default
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}"  # extra pkgs
})

# hack to import from higher level dir
sys.path.append(str(Path(__file__).resolve().parent.parent))

# import the energy datasets
from stormer_specific_data_utils import WeatherEnergySetWithCoords, LocalWeatherEnergySet2
from TimesFM import TimesFM3

# latitude/longitude function
# this function gives a basic representation of te ground truth of the 
# "image" which represents one frame of Stormer data.
# This returns lat, lon, delta
lat_lon = lambda M,N: (np.linspace(-90 + (180/M + 320/N)/2 / 2, 90 - (180/M + 320/N)/2 / 2 , M), 
                       np.linspace(0, 360, N, endpoint=False), (180/M + 320/N)/2)
us2s = lambda x: x - 180
s2us = lambda x: x + 180

if __name__ == "__main__":

    # file paths
    energy_path = "/eagle/ParaLLMs/weather_load_forecasting/comstock_datasets/california/california.nc"
    tensors_path = "/eagle/ParaLLMs/weather_load_forecasting/stormer/saved_tensors/tensors.h5"

    # lookback and lookahead
    lookback, lookahead = 512, 96

    # first we extract the energy dataset
    energy_ds = Dataset(energy_path, mode="r")
    print("\n---Printing out energy .nc file variables and shapes:---\n\n", flush=True)
    for var in energy_ds.variables:
        print(var,energy_ds.variables[var].dimensions,energy_ds.variables[var].shape, flush=True)
    print("\n---Generating numpy energy array:---\n\n", flush=True)
    energy_np = energy_ds["energy_consumption"][:]
    building_coords_np = energy_ds['building_coord'][:]

    # load the h5 file
    print("\n---Generating numpy weather array:---\n\n", flush=True)
    t0 = perf_counter()
    with h5py.File(tensors_path, mode="r") as h5f:
        root = h5f["/data"]
        print(f"Seeking to root took {((t1:=perf_counter())-t0)*1e3:.2f} ms.", flush=True)
        hour_keys = sorted(root.keys(), key=int)
        sample = root[hour_keys[0]] # sample to infer the size of the variables
        num_hours = len(hour_keys)
        num_tiers, num_vars, H, W = sample.shape # tiers is basically (1 + future forecasts)
        print(f"Calculating shapes took {((t2:=perf_counter())-t1)*1e3:.2f} ms.", flush=True)
        weather_np = np.empty(
            (num_hours, num_tiers, num_vars, H, W), dtype=sample.dtype
        )
        print(f"Allocating empty array took {((t3:=perf_counter())-t2)*1e3:.2f} ms.", flush=True)
        for idx, k in enumerate(hour_keys):
            weather_np[idx] = root[k][...]
        print(f"Populating empty array took {((t4:=perf_counter())-t3)*1e3:.2f} ms.", flush=True)
        
    # generate train dataset object
    print("\n---Generating dataset:---\n\n", flush=True)
    t0 = perf_counter()
    train_dset = WeatherEnergySetWithCoords(
        lookback = lookback,
        lookahead = lookahead,
        energy_np = energy_np,
        building_coords_np = building_coords_np,
        weather_resolution = (32, 64),
        weather_np = weather_np,
        start_boundary_ratio = 0.,
        end_boundary_ratio = 0.8
    )
    weather_np = energy_ds['weather'][:]
    train_dset_2 = LocalWeatherEnergySet2(
        lookback = lookback,
        lookahead = lookahead,
        energy_np = energy_np,
        weather_np = weather_np,
        start_boundary_ratio = 0.,
        end_boundary_ratio = 0.8
    )
    print(f"Generating dataset took {((t1:=perf_counter())-t0)*1e3:.2f} ms.", flush=True)

    # generate model
    model = TimesFM3(
        lookback = 512,
        lookahead = 96,
        lora_rank = 4,
        weather_model = True
    )
    model.load_weights()
    model.cuda()

        # check sizes
    loader = DataLoader(train_dset, batch_size=128, shuffle=False)
    for x, y, w in loader:
        x, y, w = x.cuda(), y.cuda(), w.cuda()
        print(f"(coords) x shape is {x.shape}.")
        print(f"(coords) y shape is {y.shape}.")
        print(f"(coords) w shape is {w.shape}")
        out = model(x,w)
        print(f"Model output shape is {out.shape}")
        break

    loader = DataLoader(train_dset_2, batch_size=128, shuffle=False)
    for x, y, w in loader:
        x, y, w = x.cuda(), y.cuda(), w.cuda()
        print(f"(weather) x shape is {x.shape}.")
        print(f"(weather) y shape is {y.shape}.")
        print(f"(weather) w shape is {w.shape}")
        # out = model(x,w)
        # print(f"Model output shape is {out.shape}")
        break
