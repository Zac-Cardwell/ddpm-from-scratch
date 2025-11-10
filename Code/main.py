# systemd-inhibit --what=idle --why="ML training" python main.py

import torch
import numpy as np
import os
import gymnasium as gym
import sys
sys.path.append(os.path.abspath("common"))
import util, graphing

from data import load_data, generate_video
from UNet import UNet
from train import train_ddpm
from noise import get_beta_schedule, generate_and_save_image



def save_exp(config, train_loss, val_loss, save_path):

    summary = {
        "train_loss": {
            "mean": float(np.mean(train_loss)),
            "min": float(np.min(train_loss)),
            "max": float(np.max(train_loss)),
            "final": float(train_loss[-1])
        },
        "val_loss": {
            "mean": float(np.mean(val_loss)),
            "min": float(np.min(val_loss)),
            "max": float(np.max(val_loss)),
            "final": float(val_loss[-1])
        }
    }

    metadata = {
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    # Save config, summary, and metadata
    util.save_metadata_json(config, save_path, "config.json")
    util.save_metadata_json(summary, save_path, "summary.json")
    util.save_metadata_pkl(metadata, save_path, "aggregated_metadata.pkl")


if __name__ == "__main__":
    config = util.load_config("config.json")
    util.set_seed(config["seeds"][0])

    save_path = None
    if config["save_model"]:
        save_path, plots_dir = util.setup_experiment_dir("ddpm", config["name"], config["env"])
        video_dir = os.path.join(save_path, "videos")
        os.makedirs(video_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, channels = load_data(batch_size=config['batch_size'], dataset=config['env'], img_size=config['img_size'])

    print("Training on device:", device)
    print(f"Training on dataset: {config['env']}")

    model = UNet(in_ch=channels, out_ch=channels, base_ch=config['base_ch'], time_emb_dim=config['time_emb_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    beta_t, alpha_cumprod = get_beta_schedule(T=config['T'], beta_start=config['beta_start'], beta_end=config['beta_end'], s=config['s'], schedule_type=config['schedule_type'], device=device)

    train_losses, val_losses = train_ddpm(model, train_loader, val_loader, optimizer, epochs=config['epochs'], T=config['T'], alpha_cumprod=alpha_cumprod, beta_t=beta_t, sample_save_freq=config['sample_save_freq'], sample_grid_size=4,  accum_steps =config['accum_steps'], patience=config['patience'], device=device, save_path=save_path)

    if config["save_model"]:
        save_exp(config, train_losses, val_losses, save_path)

        save_file = os.path.join(plots_dir, f"loss.png")
        graphing.plot_loss(train_losses, loss2=val_losses, title='Train vs Val losses',
                   label1='Train Loss', label2='Val Loss', file_path=save_file)

        # shape = (1, channels, config['img_size'], config['img_size'])
        # save_file = os.path.join(video_dir, f"final_denoise.mp4")
        # generate_video(model, shape, T=config['T'], beta_t=beta_t, alpha_cumprod=alpha_cumprod, filename=save_file, fps=10, device="cpu")

        shape = (4, channels, config['img_size'], config['img_size'])
        save_file = os.path.join(video_dir, f"final_images.png")
        generate_and_save_image(model, shape, config['T'], beta_t, alpha_cumprod, filename=save_file, device="cpu", nrow=2)
