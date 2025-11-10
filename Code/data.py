import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
from torchvision.utils import make_grid
import os

from noise import p_sample


def load_data(batch_size=128, dataset='MNIST', img_size=64):
    # Dataset-specific transforms
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1,1] for grayscale
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        channels=1

    elif dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # [-1,1] for RGB
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        channels=3

    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.CenterCrop(178),           # crop to square
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # [-1,1] for RGB
        ])
        train_dataset = datasets.CelebA(
            root="./data",
            split="train",
            target_type="attr",                  # ignored for unconditional DDPM
            download=True,
            transform=transform
        )
        channels=3
    else:
        raise ValueError(f"No dataset named '{dataset}'")

    # Split into train and validation (10% for validation)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, channels




@torch.no_grad()
def generate_video(model, shape, T, beta_t, alpha_cumprod,
                   filename="denoise.mp4", fps=10, device="cpu",
                   sample_freq=50, nrow=None):
    model.to(device)
    model.eval()
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    x = torch.randn(shape, device=device)
    alpha_cumprod = alpha_cumprod.to(device)
    beta_t = beta_t.to(device)
    batch_size, C, H, W = shape

    # Setup video writer
    temp_frame = np.zeros((H, W, 3), dtype=np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (temp_frame.shape[1], temp_frame.shape[0]))

    print(f"[INFO] Generating video: {filename}")

    for t in reversed(range(T)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, beta_t, alpha_cumprod)

        if t % sample_freq == 0 or t == 0:
            # move to CPU immediately
            x_cpu = x.detach().cpu()

            # make a grid if multiple samples
            if batch_size > 1:
                grid = make_grid(x_cpu, nrow=nrow or batch_size, normalize=True, value_range=(-1, 1))
                img = grid.permute(1, 2, 0).numpy()
            else:
                img = x_cpu[0].permute(1, 2, 0).numpy()

            # scale and convert to uint8
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

            # ensure 3 channels
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)

            out.write(img)
            if batch_size > 1: 
                del x_cpu, img, grid
            else: 
                None
            torch.cuda.empty_cache()

    out.release()