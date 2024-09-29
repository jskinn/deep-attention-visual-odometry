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
from torch.nn.functional import cosine_similarity


def projective_plane_cosine_distance(
    projective_points_a: torch.Tensor,
    projective_points_b: torch.Tensor,
) -> torch.Tensor:
    """
    A cosine distance between pairs of homogeneous coordinates.
    Since homogeneous coordinates are equivalent up to scale, we measure distance as
    the angle between their unit vectors in R^3 rather than R^2.
    The cosine of the angle is easy to compute as the dot product of the unit vectors.
    We therefore define the distance as:
    $1 - (a / |a|) \cdot (b / |b|)$
    which ranges from 0 to 2.

    This will have divide-by-zero errors if given (0:0:0), which is an invalid homogeneous coordinate.

    :param projective_points_a: A (B...)x3 vector of homogeneous-coordinates
    :param projective_points_b: A (B...)x3 vector of homogeneous-coordinates
    :param keepdim: Whether to retain the final dimension of size 1
    :return: A (B...) or (B...)x1 tensor of the distances between each pair of coordinates
    """
    return 1.0 - cosine_similarity(projective_points_a, projective_points_b, dim=-1)
