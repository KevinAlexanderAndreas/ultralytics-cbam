import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Channel Attention (Original CBAM)
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(1, channels // reduction)

        # Shared MLP (FC layers implemented as Conv1x1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        max_ = F.adaptive_max_pool2d(x, 1)

        # Shared MLP applied separately, then summed
        attn = self.mlp(avg) + self.mlp(max_)
        attn = torch.sigmoid(attn)

        return x * attn


# -----------------------------
# Spatial Attention (Original CBAM)
# -----------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]

        attn = torch.cat([avg, max_], dim=1)
        attn = torch.sigmoid(self.conv(attn))

        return x * attn


# -----------------------------
# CBAM Block (Original)
# -----------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        self.cam = ChannelAttention(channels, reduction=reduction)
        self.sam = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.cam(x)
        out = self.sam(out)
        return out