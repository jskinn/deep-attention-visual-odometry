import torch
import torch.nn as nn
from .simple_camera_model import SimpleCameraModel


class SimpleCameraModelInitialGuess(nn.Module):

    def __init__(self, num_views: int):
        super().__init__()
        self.focal_length = nn.Parameter(torch.tensor([[60.0]]))
        self.cx = nn.Parameter(torch.tensor([[320.]]))
        self.cy = nn.Parameter(torch.tensor([[240.]]))
        self.rot_forward = nn.Parameter(torch.tensor([1., 0., 0.]).reshape(1, 1, 1, 3).tile(1, 1, num_views, 1))
        self.rot_up = nn.Parameter(torch.tensor([0., 1., 0.]).reshape(1, 1, 1, 3).tile(1, 1, num_views, 1))
        self.translation = nn.Parameter(torch.zeros(1, 1, num_views, 3))

    def forward(self, projected_points: torch.Tensor, world_points: torch.Tensor) -> SimpleCameraModel:
        batch_size = projected_points.size(0)
        focal_length = self.focal_length.tile(batch_size, 1)
        cx = self.cx.tile(batch_size, 1)
        cy = self.cy.tile(batch_size, 1)
        a = self.rot_forward.tile(batch_size, 1, 1, 1)
        b = self.rot_up.tile(batch_size, 1, 1, 1)
        translation = self.translation.tile(batch_size, 1, 1, 1)
        return SimpleCameraModel(
            focal_length=focal_length,
            cx=cx,
            cy=cy,
            a=a,
            b=b,
            translation=translation,
            world_points=world_points,
            true_projected_points=projected_points
        )
