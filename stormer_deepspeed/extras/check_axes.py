import argparse
import xarray as xr
import os
from tqdm import tqdm

if __name__ == "__main__":

    # describe dataset
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-1440x721.zarr',
                      consolidated=False, decode_cf=False)

    # print axes and varaibles
    print("Axes and sizes:")
    for dim, size in ds.dims.items():
        print(dim, size)
    print("\nVariables, sizes, and axes:")
    for var in ds.data_vars:
        print(var, ds[var].shape, ds[var].dims)