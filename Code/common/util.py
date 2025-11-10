import yaml
import json
from pathlib import Path
import pickle
import torch
from datetime import datetime
import os
import numpy as np
import random

def load_config(file_path: str):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file {file_path} does not exist")
    
    if file_path.suffix in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Unsupported config format. Use YAML or JSON.")
    return config


# Creates a directory structure for each experiment
def setup_experiment_dir(base_dir: str, experiment_name: str, task_name = None):
    # Create the experiment directory
    if task_name:
        experiment_dir = os.path.join(base_dir, task_name, experiment_name)
    else:
        experiment_dir = os.path.join(base_dir, experiment_name)
    
    plots_dir = os.path.join(experiment_dir, "plots")

    # Make directories if they don’t exist
    os.makedirs(plots_dir, exist_ok=True)

    return experiment_dir, plots_dir


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value) 
    torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)




def aggregate_metric(metric_list):
    # Pad shorter runs to same length with NaNs
    max_len = max(len(x) for x in metric_list)
    arr = np.full((len(metric_list), max_len), np.nan)
    for i, data in enumerate(metric_list):
        arr[i, :len(data)] = data
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return mean, std



# Saving models and results
def save_checkpoint(model, optimizer, epoch, save_dir, filename=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_dir / filename)


# loading model checkpoints
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch




def save_metadata_json(metadata: dict, save_dir: str, filename=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metadata_{timestamp}.json"
    
    with open(save_dir / filename, 'w') as f:
        json.dump(metadata, f, indent=4)


def save_metadata_pkl(metadata: dict, save_dir: str, filename=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metadata_{timestamp}.pkl"

    with open(save_dir / filename, 'wb') as f:
        pickle.dump(metadata, f)