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
from stormer_specific_data_utils import WeatherEnergySet, LocalWeatherEnergySet

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
    train_dset = WeatherEnergySet(
        lookback = lookback,
        lookahead = lookahead,
        energy_np = energy_np,
        weather_np = weather_np,
        start_boundary_ratio = 0.,
        end_boundary_ratio = 0.8
    )
    print(f"Generating dataset took {((t1:=perf_counter())-t0)*1e3:.2f} ms.", flush=True)

    # We now generate a block to verify tat indeed the basic logic for selecting locations work
    loader = DataLoader(train_dset, shuffle=False, batch_size=128)
    for x,y,w in loader:
        weather_frame = w[0,0,0,:,:] # batch 0, sample 0, variable 0 (temperature), H, W: tensor of shape 128 x 256
        print(f"\n---Weather frame shape is {weather_frame.shape}.---\n\n")
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.))
        ax.coastlines()

        # get required coords
        coord_data = np.load("/eagle/ParaLLMs/weather_load_forecasting/coords/coords.npz")
        lat, lon, ddeg = lat_lon(32, 64)
        print(f"\n---For plotting, Delta(degrees) = {ddeg}. min(lat): {np.min(lat)}, max(lat): {np.max(lat)}, min(lon): {np.min(lon)}, max(lon): {np.max(lon)}.---\n\n")

        # indices
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

        # plot the contour
        cf = ax.contourf(lon_grid, lat_grid, weather_frame, transform=ccrs.PlateCarree(central_longitude=0.))

        # plot the grid points
        for pt_lat in lat:
            for pt_lon in lon:
                ax.scatter(us2s(pt_lon), pt_lat,  s=0.1, c='k',
                    transform=ccrs.PlateCarree(central_longitude=0.), zorder=3)

        # san francisco
        sf_lat, sf_lon = 37.7749, -122.4194
        i_raw, j_raw = int(np.argmin(np.abs(lat - sf_lat))), int(np.argmin(np.abs(lon - s2us(sf_lon))))
        ax.scatter(sf_lon, sf_lat,  s=40, c='white', edgecolors='k',
                transform=ccrs.PlateCarree(central_longitude=0.), zorder=3, label='SF (input)')
        ax.scatter(us2s(lon[j_raw]), lat[i_raw], marker='x', s=60, c='red',
                transform=ccrs.PlateCarree(central_longitude=0.), zorder=4, label='Snapped node')
        ax.gridlines(draw_labels=["bottom", "left"])
        ax.legend(loc="lower left")

        # colorbar
        plt.colorbar(cf, ax=ax, orientation='vertical')
        plt.savefig("test_coord_geolocation.png", format="png", bbox_inches="tight", dpi=300)

        # end loop 
        break


