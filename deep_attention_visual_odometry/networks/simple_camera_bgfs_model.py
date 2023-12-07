import torch
import torch.nn as nn
from deep_attention_visual_odometry.solvers.camera_model import SimpleCameraModel, SimpleCameraModelInitialGuess
from deep_attention_visual_odometry.solvers import BFGSCameraSolver


class SimpleCameraBGFSModel(nn.Module):

    def __init__(self, num_views: int, num_points: int, max_iterations: int,):
        super().__init__()
        num_parameters = 3 + 6 * num_views + 3 * num_points
        self.initial_guess = SimpleCameraModelInitialGuess(num_views)
        self.solver = BFGSCameraSolver(max_iterations=max_iterations, num_parameters=num_parameters)

    def forward(self, projected_points: torch.Tensor, world_points: torch.Tensor) -> SimpleCameraModel:
        initial_guess = self.initial_guess(projected_points, world_points)
        solution = self.solver(initial_guess)
        return solution
