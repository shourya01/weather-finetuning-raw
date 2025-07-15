# from pathlib import Path
# import xarray as xr
# from dask.distributed import LocalCluster, Client, progress
# from dask.utils import SerializableLock
# import sys, time

# def main():

#     # ---------- parallel runtime ----------
#     cluster = LocalCluster(n_workers=8, threads_per_worker=1)
#     client  = Client(cluster)          # keep client alive for both writes

#     base_dir = Path("/data/pangu")

#     url = (
#         "gs://gcp-public-data-arco-era5/ar/"
#         "1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
#     )
#     ds = xr.open_zarr(url, consolidated=True, storage_options={"token": "anon"})

#     abbr_sfc = {
#         'MSLP': 'mean_sea_level_pressure',
#         'U10':  '10m_u_component_of_wind',
#         'V10':  '10m_v_component_of_wind',
#         'T2M':  '2m_temperature'
#     }
#     abbr_3d = {
#         'Z': 'geopotential',
#         'Q': 'specific_humidity',
#         'T': 'temperature',
#         'U': 'u_component_of_wind',
#         'V': 'v_component_of_wind'
#     }
#     levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

#     # ---------- slice lazily ----------
#     sel_sfc = ds[list(abbr_sfc.values())].sel(time=slice("2018", "2018"))
#     sel_3d  = ds[list(abbr_3d.values())].sel(time=slice("2018", "2018"))

#     # ---------- surface file ------------------------------------------------
#     # enc_sfc = {v: {"compressor": None} for v in sel_sfc.data_vars}
#     # fut_sfc = sel_sfc.chunk({"time": 24}).to_zarr(
#     #             base_dir / "vars_ground_level.zarr", mode="w", compute=False,
#     #             encoding=enc_sfc, consolidated=True)

#     # future_sfc = client.compute(fut_sfc)          # launch graph
#     # prog_sfc   = progress(future_sfc)             # keep this object!
#     # future_sfc.result()                           # block until done
#     # print("✓ surface file complete\n")

#     # ---------- 3‑D file ----------------------------------------------------
#     enc_3d = {v: {"compressor": None} for v in sel_3d.data_vars}
#     fut_3d   = sel_3d.chunk({"time": 4}).to_zarr(
#                 base_dir / "vars_3d.zarr", mode="w", compute=False,
#                 encoding=enc_3d, consolidated=True)

#     future_3d = client.compute(fut_3d)
#     prog_3d   = progress(future_3d)               # keep reference
#     future_3d.result()
#     print("✓ 3‑D file complete")

# if __name__ == "__main__":
#     main()
# []

from pathlib import Path
import shutil, xarray as xr
from dask.distributed import LocalCluster, Client, progress

def main():

    level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] 
    level.reverse() # docs says maintain reversed order

    store = Path("/eagle/ParaLLMs/weather_load_forecasting/pangu/vars_3d.zarr")
    if store.exists():                                                   # start clean
        shutil.rmtree(store)

    # --- cluster: 8 procs × 1 thread; cap 6 GB each → ≤48 GB total -----
    cluster = LocalCluster(n_workers=8, threads_per_worker=1,
                        memory_limit="3GB")
    client  = Client(cluster)

    url = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
    ds  = xr.open_zarr(url, consolidated=True,
                    storage_options={"token": "anon"})

    vars3d = ["geopotential", "specific_humidity", "temperature",
            "u_component_of_wind", "v_component_of_wind"]

    # 1‑hour chunks → ~0.25 GB each, 8 simultaneous ≈2 GB download + 6 GB decode
    sel = (ds[vars3d]
        .sel(time=slice("2018", "2018"), level=level)
        .chunk({"time": 1}))                    # <<< key change

    enc = {v: {"compressor": None} for v in sel.data_vars}
    fut = sel.to_zarr(store, mode="w", compute=False,
                    encoding=enc, consolidated=True)

    progress(client.compute(fut)).result()
    print("✓ 2018 3‑D file complete")

if __name__ == "__main__":
    main()