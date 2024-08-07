import torch
import torch.nn as nn
from deep_attention_visual_odometry.camera_model import (
    PinholeCameraModelL1,
)


class GuessAndSolverModel(nn.Module):
    def __init__(
        self,
        initial_guess: nn.Module,
        solver: nn.Module,
    ):
        super().__init__()
        self.initial_guess = initial_guess
        self.solver = solver

    def forward(self, projected_points: torch.Tensor, visiblity_mask: torch.Tensor) -> PinholeCameraModelL1:
        # projected_points = projected_points.unsqueeze(1)
        initial_guess = self.initial_guess(projected_points, visiblity_mask)
        solution = self.solver(initial_guess)
        return solution
