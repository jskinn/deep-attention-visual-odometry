from typing import Literal
import torch
import torch.nn as nn
from deep_attention_visual_odometry.networks.weights import (
    get_kaiming_normal_init_function,
)
from deep_attention_visual_odometry.camera_model.pinhole_camera_model_least_squares import (
    PinholeCameraModelLeastSquares,
)
from deep_attention_visual_odometry.geometry.lie_rotation import LieRotation
from .base_pinhole_camera_model_guess import BasePinholeCameraModelGuess


class PinholeCameraModelMLPGuess(BasePinholeCameraModelGuess):
    def __init__(
        self,
        num_views: int,
        num_points: int,
        num_hidden: int = -1,
        init_weights: bool = False,
        float_precision: Literal["32", "64"] = "32",
        constrain: bool = True,
        max_gradient: float = -1.0,
        minimum_z_distance: float = 1e-3,
        maximum_pixel_ratio: float = 5.0,
        enable_error_gradients: bool = True,
        enable_grad_gradients: bool = False,
    ):
        super().__init__(
            constrain=constrain,
            max_gradient=max_gradient,
            minimum_z_distance=minimum_z_distance,
            maximum_pixel_ratio=maximum_pixel_ratio,
            enable_error_gradients=enable_error_gradients,
            enable_grad_gradients=enable_grad_gradients,
        )
        dtype = torch.float64 if float_precision == "64" else torch.float32
        self.num_views = num_views
        self.num_points = num_points
        if num_hidden < 0:
            num_hidden = 8 * num_views * num_points
        self.estimator = nn.Sequential(
            nn.Linear(2 * num_views * num_points, num_hidden, bias=True, dtype=dtype),
            nn.GELU(),
            nn.BatchNorm1d(num_hidden, affine=False, dtype=dtype),
            nn.Linear(
                num_hidden,
                3 + 6 * num_views + 2 * (num_points - 2) + (num_points - 3),
                bias=True,
                dtype=dtype,
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
    ) -> PinholeCameraModelLeastSquares:
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

        return self._build_model(
            focal_length=focal_length,
            cx=cx,
            cy=cy,
            orientation=LieRotation(orientation),
            translation=translation,
            world_points=world_points,
            projected_points=projected_points,
            visibility_mask=visibility_mask,
        )
