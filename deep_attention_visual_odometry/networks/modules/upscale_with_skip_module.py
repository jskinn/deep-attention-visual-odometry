import torch
import torch.nn as nn
from .upscale_module import UpscaleModule


class UpscaleWithSkipModule(nn.Module):
    def __init__(self, in_channels, skip_channels: int):
        super().__init__()
        self.upscale = UpscaleModule(in_channels, skip_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upscale(x, skip.shape)
        return x + skip
