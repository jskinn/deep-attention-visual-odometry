# Copyright (C) 2024  John Skinner
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
import torch
import torch.nn as nn

from deep_attention_visual_odometry.geometry import pixel_coordinates_to_homogeneous, projective_plane_angle_distance
from deep_attention_visual_odometry.camera_model import get_camera_relative_points, unpack_calibration_parameters
from deep_attention_visual_odometry.autograd_solvers import BFGSSolver


class CalibrationNetwork(nn.Module):

    def __init__(self, num_views: int, num_points: int, hidden_size: int = -1):
        super().__init__()
        self.__num_views = int(num_views)
        self.__num_points = int(num_points)
        num_inputs = num_views * num_points * 2
        num_parameters = 3 + 3 * num_points + 6 * (num_views - 1)
        if hidden_size <= 0:
            hidden_size = 4 * num_inputs
        self.initial_estimator = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_parameters)
        )
        self.solver = BFGSSolver(error_threshold=1e-7, training_error_threshold=1e-3)

    @property
    def num_views(self) -> int:
        return self.__num_views

    @property
    def num_points(self) -> int:
        return self.__num_points

    def forward(self, true_projected_points: torch.Tensor, visibility_mask: torch.Tensor, return_error: bool = False) -> torch.Tensor:
        inputs = true_projected_points.reshape(-1, 2 * self.num_views * self.num_points)
        initial_guess = self.initial_estimator(inputs)

        def error_function(parameters: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
            targets = true_projected_points[batch_mask]
            camera_parameters = unpack_calibration_parameters(parameters, self.num_views, self.num_points)
            homogeneous_points = pixel_coordinates_to_homogeneous(targets, camera_parameters.intrinsics)
            world_points = get_camera_relative_points(world_points=camera_parameters.world_points,
                                                      camera_translations=camera_parameters.camera_translations,
                                                      camera_rotations=camera_parameters.camera_rotations)
            distance = projective_plane_angle_distance(homogeneous_points, world_points)
            error = (distance * visibility_mask[batch_mask]).sum(dim=(-1, -2))
            return error

        result = self.solver(initial_guess, error_function)
        if return_error:
            final_error = error_function(result, torch.ones(result.shape[0], device=result.device, dtype=torch.bool))
            return result, final_error
        return result
