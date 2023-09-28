import torch
import torch.nn as nn
from deep_attention_visual_odometry.types import FeaturePoints
from .upscale_with_skip_module import UpscaleWithSkipModule


class FeatureDetectionModule(nn.Module):
    """
    Feature detection module, finds features in images, and returns descriptors and pixel coordinates for each.
    It's basically a U-Net, although there is plenty of ways for it to get more complex.
    - Smarter generation of pixel coordinates
    - Image pyramids
    """

    def __init__(self, input_channels: int = 3):
        super().__init__()
        # A big initial encoder to pull a large image down to a lower resolution quickly.
        self.encoder1 = nn.Sequential(
            nn.Conv3d(input_channels + 2, 8, 7, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv3d(8, 16, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv3d(16, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv3d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv3d(64, 66, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(66),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=2), nn.ReLU(inplace=True), nn.BatchNorm2d(64)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=2), nn.ReLU(inplace=True), nn.BatchNorm2d(64)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=2), nn.ReLU(inplace=True), nn.BatchNorm2d(64)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 64, 3), nn.ReLU(inplace=True), nn.BatchNorm2d(64)
        )
        self.upsample1 = nn.Sequential(
            UpscaleWithSkipModule(64, 64),
            nn.Conv3d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.upsample2 = nn.Sequential(
            UpscaleWithSkipModule(64, 64),
            nn.Conv3d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.upsample3 = nn.Sequential(
            UpscaleWithSkipModule(64, 64),
            nn.Conv3d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )

    def forward(self, image: torch.Tensor) -> FeaturePoints:
        batch_size = image.size(0)
        height = image.size(2)
        width = image.size(3)
        u, v = torch.meshgrid(
            [torch.arange(0, height), torch.arange(0, width)], indexing="ij"
        )
        image = torch.cat(
            [image, u.tile(batch_size, 1, 1, 1), v.tile(batch_size, 1, 1, 1)], dim=1
        )
        x = self.encoder1(image)
        points = x[:, 0:2, :, :]
        points = points.reshape(batch_size, 2, -1).permute(0, 2, 1)
        skip1 = x[:, 2:, :, :]
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        x = self.encoder4(skip3)
        x = self.bottleneck
        x = self.upsample1(x, skip3)
        x = self.upsample2(x, skip2)
        x = self.upsample3(x, skip1)
        x = x.reshape(batch_size, x.size(1), -1).permute(0, 2, 1)
        return FeaturePoints(points=points, descriptors=x)
