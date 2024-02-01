import torch
import torch.nn as nn
from .simple_camera_model import SimpleCameraModel
from .lie_rotation import LieRotation


class SimpleCameraModelInitialGuess(nn.Module):

    def __init__(self, num_views: int, num_points: int):
        super().__init__()
        self.focal_length = nn.Parameter(torch.tensor([[0.5]]))
        self.cx = nn.Parameter(torch.tensor([[0.]]))
        self.cy = nn.Parameter(torch.tensor([[0.]]))
        self.orientation = nn.Parameter(torch.tensor([0, 0., 0.]).reshape(1, 1, 1, 1, 3).tile(1, 1, num_views, 1, 1))
        translation = torch.zeros(1, 1, num_views, 3)
        translation[:, :, :, 2] = 10.0
        self.translation = nn.Parameter(translation)
        world_points = torch.randn(1, 1, num_points, 3)
        world_points[0, 0, 0, :] = 0.0
        self.world_points = nn.Parameter(world_points)

    def forward(self, projected_points: torch.Tensor) -> SimpleCameraModel:
        batch_size = projected_points.size(0)
        focal_length = self.focal_length.tile(batch_size, 1)
        cx = self.cx.tile(batch_size, 1)
        cy = self.cy.tile(batch_size, 1)
        orientation = self.orientation.tile(batch_size, 1, 1, 1, 1)
        translation = self.translation.tile(batch_size, 1, 1, 1)
        world_points = self.world_points.tile(batch_size, 1, 1, 1)
        return SimpleCameraModel(
            focal_length=focal_length,
            cx=cx,
            cy=cy,
            orientation=LieRotation(orientation),
            translation=translation,
            world_points=world_points,
            true_projected_points=projected_points
        )
