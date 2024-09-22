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


def projective_plane_angle_distance(
    projective_points_a: torch.Tensor,
    projective_points_b: torch.Tensor,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    The distance between two sets of homogeneous coordinates, defined as the angle between them in R^3.
    Ranges from 0 to 2pi.

    This will have divide-by-zero errors if given (0:0:0), which is an invalid homogeneous coordinate.

    :param projective_points_a: A (B...)x3 vector of homogeneous-coordinates
    :param projective_points_b: A (B...)x3 vector of homogeneous-coordinates
    :return: A (B...) or (B...)x1 tensor of the distances between each pair of coordinates
    """
    projective_points_a = projective_points_a / torch.linalg.vector_norm(
        projective_points_a, dim=-1, keepdim=True
    )
    projective_points_b = projective_points_b / torch.linalg.vector_norm(
        projective_points_b, dim=-1, keepdim=True
    )
    cosine = (projective_points_a * projective_points_b).sum(dim=-1, keepdims=keepdim)
    return torch.acos(cosine.clamp(min=-1.0, max=1.0))
