from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from deep_attention_visual_odometry.camera_model.pinhole_camera_model_least_squares import (
    PinholeCameraModelLeastSquares,
)
from deep_attention_visual_odometry.geometry.lie_rotation import LieRotation


class BasePinholeCameraModelGuess(nn.Module, ABC):
    def __init__(
        self,
        num_views: int,
        num_points: int,
        constrain: bool = True,
        max_gradient: float = -1.0,
        minimum_z_distance: float = 1e-3,
        maximum_pixel_ratio: float = 5.0,
        enable_error_gradients: bool = True,
        enable_grad_gradients: bool = False,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_points = num_points
        self.constrain = bool(constrain)
        self.max_gradient = float(max_gradient)
        self.minimum_z_distance = float(minimum_z_distance)
        self.maximum_pixel_ratio = float(maximum_pixel_ratio)
        self.enable_error_gradients = bool(enable_error_gradients)
        self.enable_grad_gradients = bool(enable_grad_gradients)

    @abstractmethod
    def forward(
        self, projected_points: torch.Tensor, visibility_mask: torch.Tensor
    ) -> PinholeCameraModelLeastSquares:
        pass

    def _get_num_model_parameters(self) -> int:
        return (
            3 + 6 * self.num_views + 2 * (self.num_points - 2) + (self.num_points - 3)
        )

    def _build_model(
        self,
        focal_length: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        orientation: LieRotation,
        translation: torch.Tensor,
        world_points: torch.Tensor,
        projected_points: torch.Tensor,
        visibility_mask: torch.Tensor,
    ) -> PinholeCameraModelLeastSquares:
        return PinholeCameraModelLeastSquares(
            focal_length=focal_length,
            cx=cx,
            cy=cy,
            orientation=orientation,
            translation=translation,
            world_points=world_points,
            true_projected_points=projected_points,
            visibility_mask=visibility_mask,
            constrain=self.constrain,
            max_gradient=self.max_gradient,
            minimum_z_distance=self.minimum_z_distance,
            maximum_pixel_ratio=self.maximum_pixel_ratio,
            enable_error_gradients=self.enable_error_gradients,
            enable_grad_gradients=self.enable_grad_gradients,
        )

    def _build_model_from_vector(
        self,
        x: torch.Tensor,
        batch_size: int,
        projected_points: torch.Tensor,
        visibility_mask: torch.Tensor,
    ):
        """
        Unpack a vector
        """
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
