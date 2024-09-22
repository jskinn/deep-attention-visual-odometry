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


def project_points_pinhole_homogeneous(
    points: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Project camera-relative 3D points to homogeneous coordinates on the euclidian projective plane.
    See https://en.wikipedia.org/wiki/Homogeneous_coordinates
    Note that even though they are 3-vectors, homogeneous coordinates are considered equivalent up to scale,
    that is (x:y:z) == (2x:2y:2z).
    This function does not do any normalization.

    Uses a 3-parameter pinhole camera model, with intrinsics:
    [[f, 0, cx], [0, f, cy], [0, 0, 1]]

    :param points: A (B...)x3 vector of world points to project
    :param intrinsics: A (B...)x3 vector of camera intrinsics: (fx, cx, cy)
    :return: A (B..)x3 tensor of those world points projected to homogeneous coordinates
    """
    focal_length = intrinsics[..., 0:1]
    principal_point = intrinsics[..., 1:3]
    xy = points[..., 0:2]
    z = points[..., 2:3]
    projected_points = focal_length * xy + z * principal_point

    # The point (0, 0, 0) is invalid in the projective plane.
    # Instead, we convert such points to (0, 0, 1)
    is_zero = torch.logical_and(
        z == 0, (projected_points == 0).all(dim=-1, keepdim=True)
    )
    z = torch.where(is_zero, 1.0, z)

    return torch.cat([projected_points, z], dim=-1)
