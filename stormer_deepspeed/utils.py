import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Type, Optional, Dict
from tqdm import tqdm

def generate_and_save_ds_config(
        lr: float = 5e-5,
        per_device_batch_size: int = 32,
        num_devices_per_node: int = 4,
        num_nodes: int = 2,
        weight_decay: float = 0.01,
        steps_per_print: int = 100000,
        path: str = os.getcwd(),
        config_filename: str = 'ds_config.json'
):
    
    # Define the relevant dict which is going to be dumped as a json
    ds_config = {
        'train_micro_batch_size_per_gpu': per_device_batch_size,
        'train_batch_size': num_nodes * num_devices_per_node * per_device_batch_size,
        'steps_per_print': steps_per_print,
        'gradient_accumulation_steps': 1,
        'fp16': {
            'enabled': False
        },
        'optimizer': {
            'type': "AdamW",
            'params': {
                'lr': lr,
                'weight_decay': weight_decay,
                'torch_adam': True, # restricts optimizers other than AdamW
                "adam_w_mode": True # restricts optimizers other than AdamW
            },
        },
        'comms_logger': {
            'enabled': True,
            'verbose': False
        },
        'zero_optimization': {
            'stage': 0
        }
    }

    # We place faith that the provided path is a directory and if it does not exist, it can be created
    path = Path(path)
    path.mkdir(parents = True, exist_ok = True)
    config_path = path / Path(config_filename)

    # Dump the dict into the json
    with open(str(config_path), 'w') as f:
        json.dump(ds_config, f)

    # Done
    return

def generate_sharded_dataloader(
        ds: Optional[Dataset] = None,
        global_rank: int = 0,
        world_size: int = 1,
        batch_size: int = 32
):
    
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=True)
    dataloader = DataLoader(ds, batch_size = batch_size, shuffle=False, sampler=sampler, drop_last=True)

    return dataloader

def train_one_epoch(
        model_engine,
        dataloader,
        local_rank = 0,
        global_rank = 0,
        world_size = 1, # to infer the coefficient for converting sum to average
        loss_fn = lambda yhat,y: F.mse_loss(yhat,y,reduction="mean"),
        checkpoint = False,
        save_dir = os.getcwd(),
        epoch = 0,
        two_args_for_model = True,
        desc = ""
):
    data_iter = tqdm(dataloader, desc=desc, disable=global_rank!=0)
    epoch_loss = 0
    with torch.cuda.device(local_rank):
        for items in data_iter:
            if two_args_for_model:
                x, y, weather = items
                x, y, weather = x.cuda(), y.cuda(), weather.cuda()
                loss = loss_fn(model_engine(x,weather),y)
            else:
                x, y, _ = items
                x, y = x.cuda(), y.cuda()
                loss = loss_fn(model_engine(x),y)
            model_engine.backward(loss)
            model_engine.step()
            local_loss = torch.as_tensor(loss.detach().clone().cuda())
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            epoch_loss += local_loss.item() / world_size
    if checkpoint:
        if not (Path(save_dir) / Path('checkpoint')).exists():
            (Path(save_dir) / Path('checkpoint')).mkdir(parents=True, exist_ok=True)
        model_engine.save_checkpoint(
            save_dir = str(Path(save_dir) / Path('checkpoint')),
            tag = f"epoch_{epoch}"
        )
    # return the loss this epoch
    return epoch_loss

def evaluate(
        model_engine,
        dataloader,
        loss_fn_dict  = {"mse": lambda yhat,y: F.mse_loss(yhat,y,reduction="mean")},
        local_rank = 0,
        global_rank = 0,
        world_size = 1, # to infer the coefficient for converting sum to average
        two_args_for_model = True,
        desc = ""
):
    
    data_iter = tqdm(dataloader, desc=desc, disable=global_rank!=0)
    loss_dict = {k:0 for k in loss_fn_dict.keys()}
    with torch.no_grad():
        with torch.cuda.device(local_rank):
            for items in data_iter:
                if two_args_for_model:
                    x, y, weather = items
                    x, y, weather = x.cuda(), y.cuda(), weather.cuda()
                    out = model_engine(x, weather)
                else:
                    x, y, _ = items
                    x, y = x.cuda(), y.cuda()
                    out = model_engine(x)
                for loss_name in loss_fn_dict.keys():
                    loss = loss_fn_dict[loss_name](out, y)
                    local_loss = torch.as_tensor(loss.detach().clone().cuda())
                    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                    loss_dict[loss_name] += local_loss.item() / world_size
    # divide by length of dataloader to get true average
    for loss_name in loss_fn_dict.keys():
        loss_dict[loss_name] /= len(dataloader)
    # return the dict
    return loss_dict

def train_one_epoch_tfm_parallel(
        model_engine,
        dataloader,
        local_rank = 0,
        global_rank = 0,
        world_size = 1, # to infer the coefficient for converting sum to average
        loss_fn = lambda yhat,y: F.mse_loss(yhat,y,reduction="mean"),
        checkpoint = False,
        save_dir = os.getcwd(),
        epoch = 0,
        weather_bias = 0.1,
        desc = ""
):
    data_iter = tqdm(dataloader, desc=desc, disable=global_rank!=0)
    epoch_loss = 0
    with torch.cuda.device(local_rank):
        for items in data_iter:
            x, y, weather_past, _ = items
            x, y, weather_past = x.cuda(), y.cuda(), weather_past.cuda()
            forecast_out, weather_out = model_engine(x, weather_past)
            loss = loss_fn(forecast_out, y)
            model_engine.backward(loss)
            model_engine.step()
            local_loss = torch.as_tensor(loss.detach().clone().cuda())
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            epoch_loss += local_loss.item() / world_size
    if checkpoint:
        if not (Path(save_dir) / Path('checkpoint')).exists():
            (Path(save_dir) / Path('checkpoint')).mkdir(parents=True, exist_ok=True)
        model_engine.save_checkpoint(
            save_dir = str(Path(save_dir) / Path('checkpoint')),
            tag = f"epoch_{epoch}"
        )
    # return the loss this epoch
    return epoch_loss

def evaluate_tfm_parallel(
        model_engine,
        dataloader,
        loss_fn_dict  = {"mse": lambda yhat,y: F.mse_loss(yhat,y,reduction="mean")},
        local_rank = 0,
        global_rank = 0,
        world_size = 1, # to infer the coefficient for converting sum to average
        desc = ""
):
    
    data_iter = tqdm(dataloader, desc=desc, disable=global_rank!=0)
    loss_dict = {k:0 for k in loss_fn_dict.keys()}
    with torch.no_grad():
        with torch.cuda.device(local_rank):
            for items in data_iter:
                x, y, weather_past, _ = items
                x, y, weather_past = x.cuda(), y.cuda(), weather_past.cuda()
                out, _ = model_engine(x, weather_past)
                for loss_name in loss_fn_dict.keys():
                    loss = loss_fn_dict[loss_name](out, y)
                    local_loss = torch.as_tensor(loss.detach().clone().cuda())
                    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                    loss_dict[loss_name] += local_loss.item() / world_size
    # divide by length of dataloader to get true average
    for loss_name in loss_fn_dict.keys():
        loss_dict[loss_name] /= len(dataloader)
    # return the dict
    return loss_dict