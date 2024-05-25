import torch
import torch.nn as nn
from deep_attention_visual_odometry.networks.weights import (
    get_kaiming_normal_init_function,
)
from .simple_camera_model import SimpleCameraModel
from .lie_rotation import LieRotation


class SimpleCameraModelMLPGuess(nn.Module):
    def __init__(
        self,
        num_views: int,
        num_points: int,
        constrain: bool,
        num_hidden: int = -1,
        init_weights: bool = False,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_points = num_points
        self.constrain = bool(constrain)
        if num_hidden < 0:
            num_hidden = 8 * num_views * num_points
        self.estimator = nn.Sequential(
            nn.Linear(2 * num_views * num_points, num_hidden, bias=True),
            nn.GELU(),
            nn.BatchNorm1d(num_hidden, affine=False),
            nn.Linear(
                num_hidden,
                3 + 6 * num_views + 2 * (num_points - 2) + (num_points - 3),
                bias=True,
            ),
        )
        if init_weights:
            self.init_weights()

    def init_weights(self):
        self.apply(
            get_kaiming_normal_init_function(
                {nn.Linear}, mode="fan_in", nonlinearity="relu"
            )
        )

    def forward(
        self, projected_points: torch.Tensor, visibility_mask: torch.Tensor
    ) -> SimpleCameraModel:
        batch_size = projected_points.size(0)
        x = projected_points.reshape(batch_size, -1)
        x = self.estimator(x)
        orientation_start = 3
        orientation_end = orientation_start + 3 * self.num_views
        translation_end = orientation_end + 3 * self.num_views
        z_end = translation_end + self.num_points - 3
        xy_end = z_end + 2 * (self.num_points - 2)
        focal_length = x[:, 0].view(batch_size, 1)
        cx = x[:, 1].view(batch_size, 1)
        cy = x[:, 2].view(batch_size, 1)
        orientation = x[:, orientation_start:orientation_end].view(
            batch_size, 1, self.num_views, 1, 3
        )
        translation = x[:, orientation_end:translation_end].view(
            batch_size, 1, self.num_views, 3
        )
        z_points = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    dtype=projected_points.dtype,
                    device=projected_points.device,
                ),
                x[:, translation_end:z_end],
            ],
            dim=1,
        ).view(batch_size, 1, self.num_points - 2, 1)
        xy_points = x[:, z_end:xy_end].view(batch_size, 1, self.num_points - 2, 2)
        world_points = torch.cat([xy_points, z_points], dim=3)
        return SimpleCameraModel(
            focal_length=focal_length,
            cx=cx,
            cy=cy,
            orientation=LieRotation(orientation),
            translation=translation,
            world_points=world_points,
            true_projected_points=projected_points,
            visibility_mask=visibility_mask,
            constrain=self.constrain,
        )
