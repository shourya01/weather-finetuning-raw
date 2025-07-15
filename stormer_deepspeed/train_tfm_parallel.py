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

# import custom model functions
from TimesFM import TimesFM2
from stormer_specific_data_utils import LocalWeatherEnergySet
from utils import generate_and_save_ds_config, generate_sharded_dataloader, train_one_epoch_tfm_parallel, evaluate_tfm_parallel

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

    # Make directories
    (Path(args.results_base_dir) / Path(args.experiment_name)).mkdir(parents=True, exist_ok=True)

    # Initalize deepspeed
    init_distributed(dist_backend="nccl")
    world_size = comm.get_world_size()
    local_rank = comm.get_local_rank()
    global_rank = comm.get_rank()
    args.local_rank = local_rank # "correct" the local rank in the arguments
    print(f"Initialized deepspeed on global rank {global_rank}, local rank {local_rank} with world size {world_size}.")
    torch.cuda.set_device(local_rank)

    # Get model and datasets
    model = get_model(args)
    train_ds, val_ds, test_ds = get_dataset(args)

    # Generate requisite dataloaders
    train_dl = generate_sharded_dataloader(
        ds = train_ds,
        global_rank = global_rank,
        world_size = world_size,
        batch_size = args.per_device_batch_size
    )
    val_dl = generate_sharded_dataloader(
        ds = val_ds,
        global_rank = global_rank,
        world_size = world_size,
        batch_size = args.per_device_batch_size
    )
    test_dl = generate_sharded_dataloader(
        ds = test_ds,
        global_rank = global_rank,
        world_size = world_size,
        batch_size = args.per_device_batch_size
    )

    # Verify if tuning results exist
    if args.tuning == False:
        # create a valid deepspeed config using rank-0 process
        if global_rank == 0:
            generate_and_save_ds_config(
                lr = args.lr,
                per_device_batch_size = args.per_device_batch_size,
                num_devices_per_node = torch.cuda.device_count(),
                num_nodes = world_size // torch.cuda.device_count(),
                path = os.getcwd(),
                config_filename = args.deepspeed_config
            )
        dist.barrier()
    else:
        # define best lr and start it off as None, definte best vlidation loss and start it off as infinity
        best_lr, best_val_loss  = None, np.inf
        # read the tuning file else create an empty dataframe with column headers but no rows
        if (Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.tuning_file)).exists():
            results_tuning = pd.read_csv(str(Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.tuning_file)), dtype={'lr': 'float64', 'val_loss_mse': 'float64'}, index_col="lr")
        else:
            results_tuning = pd.DataFrame(columns = ["lr", "val_loss_mse"]).astype({'lr': 'float64', 'val_loss_mse': 'float64'}).set_index("lr")
        # run the loop 
        for candidate_lr in args.lr_candidates:
            # if the present learning rate already exists, then skip over it
            if not results_tuning.empty:
                mask = pd.Series(np.isclose(results_tuning.index.to_numpy(), candidate_lr, atol=1e-8, rtol=0), index=results_tuning.index,)
                if mask.any():                       # already tuned
                    val_loss = results_tuning.loc[mask, "val_loss_mse"].min()
                    if not np.isnan(val_loss) and val_loss < best_val_loss:
                        best_val_loss, best_lr = val_loss, candidate_lr
                    continue  
            # generate deepspeed config for this lr's training on rank-0
            if global_rank == 0:
                generate_and_save_ds_config(
                    lr = candidate_lr,
                    per_device_batch_size = args.per_device_batch_size,
                    num_devices_per_node = torch.cuda.device_count(),
                    num_nodes = world_size // torch.cuda.device_count(),
                    path = os.getcwd(),
                    config_filename = args.deepspeed_config
                )
            dist.barrier()
            # generate deepspeed engine
            cur_model = deepcopy(model)
            model_engine, optimizer, _, _ = initialize(
                args = args,
                model = cur_model,
                model_parameters = cur_model.parameters()
            )
            # train for one epoch
            for epoch in range(args.val_epochs):
                _ = train_one_epoch_tfm_parallel(
                    model_engine = model_engine,
                    dataloader = train_dl,
                    local_rank = local_rank,
                    global_rank = global_rank,
                    world_size = world_size,
                    desc = f"Validating lr={candidate_lr}, train epoch {epoch}.",
                    save_dir = str(Path(args.results_base_dir) / Path(args.experiment_name)),
                    epoch = epoch
                )
            # calculate loss on the validation set (all-reduced across each GPU)
            loss_dict = evaluate_tfm_parallel(
                model_engine = model_engine,
                dataloader = val_dl,
                loss_fn_dict = {'val_loss_mse': lambda yhat, y: F.mse_loss(yhat,y,reduction="mean")},
                local_rank = local_rank,
                global_rank = global_rank,
                world_size = world_size,
                desc = f"Evaluating for lr={candidate_lr}"
            )
            # flush the model engine
            del model_engine
            # append the loss to the dataframe
            results_tuning.loc[candidate_lr] = {
                "val_loss_mse": loss_dict["val_loss_mse"]
            }
            # if we found a better loss, update the best loss and lr
            if loss_dict["val_loss_mse"] < best_val_loss:
                best_val_loss = loss_dict["val_loss_mse"]
                best_lr = candidate_lr
            # on rank-0, save the file to csv
            if global_rank == 0:
                results_tuning.to_csv(str(Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.tuning_file)))
            dist.barrier()
        # create a valid deepspeed config using rank-0 process
        if best_lr is None:
            raise ValueError(f"best_lr could not be found on global rank {global_rank}!")
        if global_rank == 0:
            print(f"Hyperparameter tuning process completed, using lr {best_lr}!")
            generate_and_save_ds_config(
                lr = best_lr,
                per_device_batch_size = args.per_device_batch_size,
                num_devices_per_node = torch.cuda.device_count(),
                num_nodes = world_size // torch.cuda.device_count(),
                path = os.getcwd(),
                config_filename = args.deepspeed_config
            )
        dist.barrier()

    # Start the main training loop
    if (Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.train_file)).exists():
        results_train = pd.read_csv(str((Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.train_file))), index_col="epoch")
        start_epoch = len(results_train)
    else:
        results_train = pd.DataFrame(columns = ["epoch","test_mse_loss"]).set_index("epoch")
        start_epoch = 0
    # training loop
    model_engine, optimizer, _, _ = initialize(
        args = args,
        model = model,
        model_parameters = model.parameters()
    )
    # load checkpoint if available
    if start_epoch > 0:
        try:
            ckpt_path = str(Path(args.results_base_dir) / Path(args.experiment_name) / Path('checkpoint'))
            model_engine.load_checkpoint(
                load_dir = ckpt_path,
                tag = f"epoch_{start_epoch-1}",
                load_module_strict = False,
                load_optimizer_states = True,
                load_lr_scheduler_states = False
            )
        except:
            raise FileNotFoundError("Checkpoint not found to resume training! Please delete the results and re-start training.")
    # start training
    for epoch in range(start_epoch,args.train_epochs):
        train_loss = train_one_epoch_tfm_parallel(
                    model_engine = model_engine,
                    dataloader = train_dl,
                    local_rank = local_rank,
                    global_rank = global_rank,
                    world_size = world_size,
                    desc = f"Main training loop, train epoch {epoch}.",
                    checkpoint = True,
                    save_dir = str(Path(args.results_base_dir) / Path(args.experiment_name)),
                    epoch = epoch,
                )
        # append the loss to the dataframe
        results_train.loc[epoch] = {
            "val_loss_mse": train_loss
        }
        # on rank-0, save the file to csv
        if global_rank == 0:
            results_train.to_csv(str(Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.train_file)))
        dist.barrier()
        # check if the current epoch requires a test eval
        if (epoch+1) % args.test_every == 0:
            # generate a dataframe or read if data is already available 
            loss_fn_dict = get_loss_dict(train_ds.energy_mean,train_ds.energy_std)
            if (Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.test_file)).exists():
                results_test = pd.read_csv(str(Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.test_file)), index_col = "epoch")
            else:
                results_test = pd.DataFrame(columns = ["epoch"] + list(loss_fn_dict.keys())).set_index("epoch")
            # if the epoch is already present then skip it else carry out the test
            if (results_test.index == epoch).any():
                pass
            else:
                loss_dict = evaluate_tfm_parallel(
                    model_engine = model_engine,
                    dataloader = test_dl,
                    loss_fn_dict = loss_fn_dict,
                    local_rank = local_rank,
                    global_rank = global_rank,
                    world_size = world_size,
                    desc = f"Testing on epoch {epoch}."
                )
                # on rank-0, update the csv file
                results_test.loc[epoch] = loss_dict
                if global_rank == 0:
                    results_test.to_csv(str(Path(args.results_base_dir) / Path(args.experiment_name) / Path(args.test_file)))
                dist.barrier()

    # Dist process termination
    if dist.is_initialized():
        dist.destroy_process_group()