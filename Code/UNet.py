import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (B, dim)
    



class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)  # For scale and shift
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        # Modulate with time embedding
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        return h + self.res_conv(x)



class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(32, out_ch)
        self.activation = nn.SiLU()

    def forward(self, x):
        h = self.conv(x)
        h = self.norm(h)
        h = self.activation(h)
        return h

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(32, out_ch)
        self.activation = nn.SiLU()

    def forward(self, x):
        h = F.interpolate(x, scale_factor=2, mode='nearest')
        h = self.conv(h)
        h = self.norm(h)
        h = self.activation(h)
        return h



class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = Downsample(base_ch, base_ch)
        self.enc2 = ResidualBlock(base_ch, base_ch*2, time_emb_dim)
        self.down2 = Downsample(base_ch*2, base_ch*2)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_ch*2, base_ch*4, time_emb_dim)

        # Decoder
        self.up1 = Upsample(base_ch*4, base_ch*2)
        self.dec1 = ResidualBlock(base_ch*4, base_ch*2, time_emb_dim)  # Concat skip
        self.up2 = Upsample(base_ch*2, base_ch)
        self.dec2 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)  # Concat skip

        # Output
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        skips = []

        # Encoder
        h = self.enc1(x, t_emb)
        skips.append(h)
        h = self.down1(h)
        h = self.enc2(h, t_emb)
        skips.append(h)
        h = self.down2(h)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder with skip connections
        h = self.up1(h)
        h = torch.cat([h, skips.pop()], dim=1)
        h = self.dec1(h, t_emb)
        h = self.up2(h)
        h = torch.cat([h, skips.pop()], dim=1)
        h = self.dec2(h, t_emb)

        return self.out_conv(h)
    
    
