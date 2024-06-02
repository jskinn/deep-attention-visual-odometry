from typing import Literal
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from deep_attention_visual_odometry.camera_model.pinhole_camera_model_least_squares import PinholeCameraModelLeastSquares
from deep_attention_visual_odometry.geometry.lie_rotation import LieRotation


class BasePinholeCameraModelGuess(nn.Module, ABC):
    def __init__(
        self,
        constrain: bool = True,
        max_gradient: float = -1.0,
        minimum_z_distance: float = 1e-3,
        maximum_pixel_ratio: float = 5.0,
        enable_error_gradients: bool = True,
        enable_grad_gradients: bool = False,
    ):
        super().__init__()
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
