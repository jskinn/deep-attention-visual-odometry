import torch
import torch.nn as nn
from deep_attention_visual_odometry.camera_model import (
    PinholeCameraModelLeastSquares,
)


class InitialGuessModel(nn.Module):
    def __init__(
        self,
        initial_guess: nn.Module,
    ):
        super().__init__()
        self.initial_guess = initial_guess

    def forward(
        self, projected_points: torch.Tensor, visiblity_mask: torch.Tensor
    ) -> PinholeCameraModelLeastSquares:
        initial_guess = self.initial_guess(projected_points, visiblity_mask)
        return initial_guess
