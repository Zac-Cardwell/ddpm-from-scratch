import torch
from torchvision.utils import make_grid, save_image
import math
import numpy as np


def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02, s=0.008, schedule_type="linear", device="cpu"):
    if schedule_type == "linear":
        beta_t = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - beta_t
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return beta_t.to(device), alpha_cumprod.to(device)

    elif schedule_type == "cosine":
        t = torch.linspace(0, T, T+1, dtype=torch.float32) / T
        f_t = torch.cos((t + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_cumprod = f_t / f_t[0]  # Normalize so α̅₀ = 1
        
        # Compute stepwise alphas and betas
        alphas = alpha_cumprod[1:] / alpha_cumprod[:-1]
        beta_t = 1 - alphas
        alpha_cumprod = alpha_cumprod[1:]

    beta_t = torch.clamp(beta_t, max=0.999)
    return beta_t.to(device), alpha_cumprod.to(device)




def add_noise(x0, t, alpha_cumprod):
    epsilon = torch.randn_like(x0)

    # Gather α̅_t for each example in the batch
    alpha_cumprod_t = alpha_cumprod[t].view(-1, 1, 1, 1)
    
    # Compute noisy sample
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod_t)
    xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * epsilon
    
    return xt, epsilon




def p_mean_variance(model, x_t, t, beta_t, alpha_cumprod):
    eps_theta = model(x_t, t)  # predict the noise ε_θ(x_t, t)

    beta_t = beta_t[t].view(-1, 1, 1, 1)
    alpha_t = 1 - beta_t
    alpha_cumprod_t = alpha_cumprod[t].view(-1, 1, 1, 1)

    mu_theta = (
        1 / torch.sqrt(alpha_t)
    ) * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * eps_theta)

    return mu_theta, beta_t, eps_theta


def p_sample(model, x_t, t, beta_t, alpha_cumprod):
    mu_theta, beta_t, eps_theta = p_mean_variance(model, x_t, t, beta_t, alpha_cumprod)
    noise = torch.randn_like(x_t) if t[0] > 0 else 0
    x_prev = mu_theta + torch.sqrt(beta_t) * noise
    return x_prev


def generate_image(model, shape, T, beta_t, alpha_cumprod, device="cpu"):
    x = torch.randn(shape, device=device)  # start from pure noise
    for t in reversed(range(T)):
        t_batch = torch.tensor([t] * shape[0], device=device)
        x = p_sample(model, x, t_batch, beta_t, alpha_cumprod)
    return x


def generate_and_save_image(model, shape, T, beta_t, alpha_cumprod, filename="generated.png", device="cpu", nrow=None):
    model.to(device)
    model.eval()

    # Pre-move parameters to device only once
    alpha_cumprod = alpha_cumprod.to(device)
    beta_t = beta_t.to(device)

    # Initialize with pure noise
    x = torch.randn(shape, device=device)

    # Reverse diffusion process
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, beta_t, alpha_cumprod)

        # optional lightweight memory management for long T
        if t % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Move result to CPU for saving
    x_cpu = x.detach().cpu()

    # Save batch as grid or single image
    if x_cpu.size(0) > 1:
        nrow = nrow or int(x_cpu.size(0) ** 0.5)
        grid = make_grid(
            x_cpu,
            nrow=nrow,
            padding=2,
            normalize=True,
            value_range=(-1, 1),
        )
        save_image(grid, filename)
    else:
        save_image(x_cpu, filename, normalize=True, value_range=(-1, 1))

    # Free memory
    del x, x_cpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()