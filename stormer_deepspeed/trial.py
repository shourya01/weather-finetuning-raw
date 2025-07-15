import torch
import numpy as np
import os
import argparse
import random
import netCDF4
from pathlib import Path

# specific imports 
from TimesFM import TimesFM2
from stormer_specific_data_utils import LocalWeatherEnergySet

def parse_args():

    parser = argparse.ArgumentParser(description="Trial for detecting the sizes needed for TimesFM-parallel.")
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--lookahead", type=int, default=96)
    parser.add_argument("--per_device_batch_size", type=int, default=128)
    parser.add_argument("--energy_nc_base", type=str,  default="/eagle/ParaLLMs/weather_load_forecasting/comstock_datasets")
    parser.add_argument("--dataset_name", type=str, default="california")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--timesfm_ckpt_path", type=str, default="/home/shourya01/timesfm/timesfm-1.0-200m-pytorch/torch_model.ckpt")
    return parser.parse_args()

def set_seed(seed: int = 42):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args, print_missing_unexpected: bool = False, seed: int = 42):

    # set seed
    set_seed(seed)

    # define model
    model = TimesFM2(
        lookback = args.lookback,
        lookahead = args.lookahead,
        lora_rank = args.lora_rank,
        weather_features = 7,
        ckpt = args.timesfm_ckpt_path
    )  

    # load model weights
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
    energy_np = energy_ds.variables["energy_consumption"][:]
    weather_np = energy_ds.variables["weather"][:]

    # generate train dataset object
    train_dset = LocalWeatherEnergySet(
        lookback = args.lookback,
        lookahead = args.lookahead,
        energy_np = energy_np,
        weather_np = weather_np,
        start_boundary_ratio = cum_splits[0],
        end_boundary_ratio = cum_splits[1]
    )

    # generate validation dataset object
    val_dset = LocalWeatherEnergySet(
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
    test_dset = LocalWeatherEnergySet(
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


if __name__ == "__main__":

    args = parse_args()
    # model = get_model(args)
    # model = model.cuda()

    # # print the weights that have grad as compared to ones which don't
    # param_count = 0
    # for name, val in model.named_parameters():
    #     if val.requires_grad == True:
    #         print(f"Weight {name} has GRAD and has shape {val.shape}.")
    #         param_count += torch.numel(val)
    # print(f"\n\nTotal number of parameters which have GRAD is {param_count}.")
    # param_count = 0
    # for name, val in model.named_parameters():
    #     if val.requires_grad == False:
    #         print(f"Weight {name} has NO GRAD and has shape {val.shape}.")
    #         param_count += torch.numel(val)
    # print(f"\n\nTotal number of parameters which DO NOT have GRAD is {param_count}.")

    # get the datasets
    train_ds, val_ds, test_ds = get_dataset(args)

    print(f"\n\n\nLength of train ds is {len(train_ds)}.")
    print(f"Its weather mean is {train_ds.weather_mean} and std {train_ds.weather_std}.")
    print(f"Its energy mean is {train_ds.energy_mean} and std {train_ds.energy_std}.")
    for itms in train_ds:
        print(f"There are {len(itms)} items per dataset item.")
        for idx,itm in enumerate(itms):
            print(f"Shape of item {idx+1} is {itm.shape}.")
        break
