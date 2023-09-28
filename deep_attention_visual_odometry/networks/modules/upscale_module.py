import torch
import torch.nn as nn
import torch.nn.functional as fn


class UpscaleModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.smooth_conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = fn.interpolate(x, size, mode="nearest")
        x = self.smooth_conv(x)
        return x
