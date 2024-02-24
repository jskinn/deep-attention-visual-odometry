import torch
import torch.nn as nn
from deep_attention_visual_odometry.solvers.camera_model import (
    SimpleCameraModel,
    SimpleCameraModelFixedGuess,
)
from deep_attention_visual_odometry.solvers.line_search_strong_wolfe_conditions import (
    LineSearchStrongWolfeConditions,
)
from deep_attention_visual_odometry.solvers import BFGSCameraSolver


class SimpleCameraBGFSModel(nn.Module):
    def __init__(
        self,
        num_views: int,
        num_points: int,
        max_iterations: int,
        epsilon: float,
        max_step_size: float,
        zoom_iterations: int,
        sufficient_decrease: float,
        curvature: float,
    ):
        super().__init__()
        self.initial_guess = SimpleCameraModelFixedGuess(num_views, num_points)
        self.line_search = LineSearchStrongWolfeConditions(
            max_step_size=max_step_size,
            zoom_iterations=zoom_iterations,
            sufficient_decrease=sufficient_decrease,
            curvature=curvature,
        )
        self.solver = BFGSCameraSolver(
            max_iterations=max_iterations,
            epsilon=epsilon,
            max_step_distance=1e3,
            line_search=self.line_search,
        )

    def forward(self, projected_points: torch.Tensor) -> SimpleCameraModel:
        # projected_points = projected_points.unsqueeze(1)
        initial_guess = self.initial_guess(projected_points)
        solution = self.solver(initial_guess)
        return solution
