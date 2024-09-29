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
from typing import NamedTuple

import torch

from deep_attention_visual_odometry.geometry import (
    rotate_vector_axis_angle,
    project_points_pinhole_homogeneous,
)


class CalibrationParameters(NamedTuple):
    intrinsics: torch.Tensor
    world_points: torch.Tensor
    camera_translations: torch.Tensor
    camera_rotations: torch.Tensor


def unpack_calibration_parameters(
    parameters: torch.Tensor,
    num_views: int,
    num_points: int,
) -> CalibrationParameters:
    """
    Unpack and reshape a vector of parameters into separate parameters for the camera intrinsics
    """
    if parameters.size(-1) != (3 + 3 * num_points + 6 * (num_views - 1)):
        raise ValueError(
            f"The final dimension of the input tensor must be "
            f"3 + 3 * num_points + 6 * (num_views - 1) = {3 + 3 * num_points + 6 * (num_views - 1)}, "
            f"got {parameters.size(-1)}"
        )
    batch_dimensions = parameters.shape[:-1]
    points_end = 3 + 3 * num_points
    translations_end = points_end + 3 * (num_views - 1)
    intrinsics = parameters[..., 0:3].reshape(batch_dimensions + (1, 1, 3))
    world_points = parameters[..., 3:points_end].reshape(
        batch_dimensions + (1, num_points, 3)
    )
    translations = parameters[..., points_end:translations_end].reshape(
        batch_dimensions + (num_views - 1, 1, 3)
    )
    rotations = parameters[..., translations_end:].reshape(
        batch_dimensions + (num_views - 1, 1, 3)
    )
    return CalibrationParameters(
        intrinsics=intrinsics,
        world_points=world_points,
        camera_translations=translations,
        camera_rotations=rotations,
    )


def calibration_pinhole_camera_model(
    parameters: torch.Tensor,
    num_views: int,
    num_points: int,
) -> torch.Tensor:
    """
    Project world points to camera views as if for camera calibration.
    That is, given n 3D points, project each one to m different cameras.
    Each camera has its own position and orientation in 3D space. All cameras have the same intrinsics.
    This problem is determined up to scale and choice of origin.
    We therefore make the following assumptions to fully constrain the problem:
    - The position and orientation of the first view define the origin
    - The standard deviation of the world points is 1. This sets the scale.

    :param parameters: A tensor of parameters to the model. (B...)x(3 + 3*num_points + 6 * (num_views - 1)).
                       Raises a ValueError if the tensor shape doesn't match the num_views and num_points parameters.
    :param num_views: The number of views in the model, M.
    :param num_points: The number of world points in the model, N.
    :returns: A (B...)xMxNx3 tensor of homogeneous image coordinates.
    """
    calibration_parameters = unpack_calibration_parameters(parameters, num_views, num_points)

    # Transform each point to be relative to each viewpoint
    # The first viewpoint has translation and rotation 0, so we just concatenate untransformed points.
    camera_relative_points = rotate_vector_axis_angle(
        calibration_parameters.world_points, calibration_parameters.camera_rotations
    )
    camera_relative_points = (
        camera_relative_points + calibration_parameters.camera_translations
    )
    camera_relative_points = torch.concatenate(
        [calibration_parameters.world_points, camera_relative_points], dim=-3
    )

    # Project the camera points to homogeneous coordinates
    return project_points_pinhole_homogeneous(
        camera_relative_points, calibration_parameters.intrinsics
    )
