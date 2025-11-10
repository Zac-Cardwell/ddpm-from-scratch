import torch
from torchvision.utils import make_grid, save_image
import os
import numpy as np
import time

from noise import add_noise, p_sample



@torch.no_grad()
def sample_images_grid(model, shape, T, beta_t, alpha_cumprod, sample_freq=100, device="cpu", filename="denoise_grid.png", nrow=None):
    batch_size, C, H, W = shape
    model.eval()
    x = torch.randn(shape, device=device)
    snapshots = []
    for t in reversed(range(0, T, 1)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, beta_t, alpha_cumprod)
        
        if t % sample_freq == 0 or t == 0:
            snapshots.append(x.cpu().clamp(-1, 1))
        
        del t_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_samples = torch.stack(snapshots, dim=0).view(-1, C, H, W)
    nrow = nrow or batch_size
    grid = make_grid(all_samples, nrow=nrow, padding=2, normalize=True, value_range=(-1, 1))
    save_image(grid, filename)

    


def train_ddpm(model, train_loader, val_loader, optimizer, epochs, T, alpha_cumprod, beta_t, sample_save_freq=500, sample_grid_size=4, accum_steps = 4, patience=5, device="cpu", save_path=None):

    if save_path:
        img_dir = os.path.join(save_path, "imgs")
        os.makedirs(img_dir, exist_ok=True)

    train_losses, val_losses = [], []
    val_accuracy = []

    best_val_loss = float("inf")
    best_val_epoch = 0

    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()

        # Gradient Accumulation
        for i, (x0, _) in enumerate(train_loader):
            x0 = x0.to(device, non_blocking=True)
            alpha_cumprod = alpha_cumprod.to(x0.device)
            beta_t = beta_t.to(x0.device)

            t = torch.randint(0, T, (x0.size(0),), device=device)
            xt, noise = add_noise(x0, t, alpha_cumprod)

            predicted_noise = model(xt, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            # Scale loss so total gradient magnitude stays correct
            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accum_steps  # track true (unscaled) loss

            # Free memory for current batch
            del x0, xt, noise, predicted_noise, loss
            torch.cuda.empty_cache()

        train_losses.append(running_loss / len(train_loader))


        model.eval()
        running_val_loss = 0.0
        noise_diff = 0.0
        with torch.no_grad():
            for x0, _ in val_loader:
                x0 = x0.to(device, non_blocking=True)
                t = torch.randint(0, T, (x0.size(0),), device=device)

                xt, noise = add_noise(x0, t, alpha_cumprod)
                predicted_noise = model(xt, t)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                running_val_loss += loss.item()
                noise_diff += (predicted_noise - noise).abs().mean().item()

            # Free memory
            del x0, xt, noise, predicted_noise, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy.append(noise_diff / len(val_loader))
 

        if avg_val_loss < best_val_loss and save_path:
            best_val_loss = avg_val_loss
            best_val_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_path, "best_optimizer.pth"))

            filename = os.path.join(img_dir, f"denoise_grid_best_model.png")
            sample_batch = next(iter(val_loader))[0][:sample_grid_size]  # small batch
            batch_size, C, H, W = sample_batch.shape
            sample_images_grid(model, shape=(sample_grid_size, C, H, W), T=T,
                               beta_t=beta_t, alpha_cumprod=alpha_cumprod,
                               sample_freq=max(1, T//10),  # 10 snapshots per grid
                               device=device, filename=filename)


        if epoch % sample_save_freq == 0 and save_path:
            filename = os.path.join(img_dir, f"denoise_grid_epoch_{epoch}.png")

            sample_batch = next(iter(val_loader))[0][:sample_grid_size]  # small batch
            batch_size, C, H, W = sample_batch.shape
            sample_images_grid(model, shape=(sample_grid_size, C, H, W), T=T,
                               beta_t=beta_t, alpha_cumprod=alpha_cumprod,
                               sample_freq=max(1, T//10),  # 10 snapshots per grid
                               device=device, filename=filename)
            
            
        no_improve_count = epoch - best_val_epoch
        if no_improve_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs! No improvement for {patience} consecutive epochs.")
            break  
            

        window = 20
        if epoch % window == 0 or epoch == epochs-1:  
            avg_val_loss = np.mean(val_losses[-window:]) if len(val_losses) >= window else val_losses[-1]   
            avg_train_loss = np.mean(train_losses[-window:]) if len(train_losses) >= window else train_losses[-1]   
            avg_val_acc = np.mean(val_accuracy[-window:]) if len(val_accuracy) >= window else val_accuracy[-1]   

            print(f"Epoch {epoch}/{epochs} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f} | Avg Val Noise Diff: {avg_val_acc:.4f}\n")

    end = time.time()

    print(f"Best val loss: {best_val_loss} achieved at epoch: {best_val_epoch}")
    print(f"Total training time: {(end - start) / 60}m")
    return train_losses, val_losses