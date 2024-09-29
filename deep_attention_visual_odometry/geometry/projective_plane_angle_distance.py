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

    Uses Kahan's formulation from "Miscalculating area and angles of a Needle-Like Triangle",
    which is more numerically stable than the arccos or arctan approaches.

    This will produce NaN if given (0:0:0), which is an invalid homogeneous coordinate.

    :param projective_points_a: A (B...)x3 vector of homogeneous-coordinates
    :param projective_points_b: A (B...)x3 vector of homogeneous-coordinates
    :param keepdim: Whether to retain the final dimension as size 1.
    :return: A (B...) or (B...)x1 tensor of the distances between each pair of coordinates
    """
    # Calculate angle as 2 arctan2(|v1 / |v1| - v2 / |v2||, |v1 / |v1| + v2 / |v2||
    # Comes from W. Kahan "Computing Cross-Products and Rotations in 2- and 3-Dimensional Euclidean Spaces".
    # See https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors
    # Consider the parallelogram formed by adding the two vectors in different orders.
    # The angle we are interested in is one of the internal angles of the parallelogram.
    # Normalizing the two vectors doesn't change the angle and addresses issues with extremely large
    # and extremely small vectors.
    # One diagonal of the parallelogram is the sum of the two vectors, the other is the difference.
    # The two diagonals form a right-angled triangle, with side lengths half the diagonals and internal
    # angle half the angle between the vectors.
    # tan(theta / 2) = (|v1 - v2| / 2) / (|v1 + v2| / 2)
    # Rearranging gives the above equation.
    projective_points_a = projective_points_a / torch.linalg.vector_norm(
        projective_points_a, dim=-1, keepdim=True
    )
    projective_points_b = projective_points_b / torch.linalg.vector_norm(
        projective_points_b, dim=-1, keepdim=True
    )
    unit_vector_sum = torch.linalg.vector_norm(
        projective_points_a + projective_points_b, dim=-1, keepdim=keepdim
    )
    unit_vector_diff = torch.linalg.vector_norm(
        projective_points_a - projective_points_b, dim=-1, keepdim=keepdim
    )
    result = 2.0 * torch.atan2(unit_vector_diff, unit_vector_sum)
    return result
