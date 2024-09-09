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

from deep_attention_visual_odometry.utils import (
    sin_x_on_x,
    one_minus_cos_x_on_x_squared,
)


def rotate_vector_axis_angle(
    vector: torch.Tensor, axis_angle: torch.Tensor
) -> torch.Tensor:
    """
    Rotate a 3-vector by sn axis-angle rotation expressed as \theta * n.
    :param vector: The vector to rotate
    :param axis_angle: A 3-vector, where the length of the vector is the angle to rotate and the direction is the axis.
    :return: The vector rotated around the axis by the angle
    """
    # This is based on maths that combines a conversion from axis-angle to quaternion
    # with the hamilton product to rotate the vector.
    # Produces a variant of Rodrigues' rotation formula
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    dot_product = (vector * axis_angle).sum(dim=-1, keepdims=True)
    cross_product = torch.linalg.cross(axis_angle, vector, dim=-1)
    cos_theta = torch.cos(angle)
    sin_theta_on_theta = sin_x_on_x(angle)
    one_minus_cos_term = one_minus_cos_x_on_x_squared(angle)
    out_vector = (
        vector * cos_theta
        + one_minus_cos_term * dot_product * axis_angle
        + cross_product * sin_theta_on_theta
    )
    return out_vector
