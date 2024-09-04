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


def project_points_basic_pinhole(
    points: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Basic pinhole camera projection, of 3D points using three intrinsics: f, cx, and cy.
    This function assumes the final dimension
    :param points: A (Bx)
    :param intrinsics:
    :return:
    """
    focal_length = intrinsics[..., 0:1]
    principal_point = intrinsics[..., 1:3]
    xy = points[..., 0:2]
    z = points[..., 2:3]
    return focal_length * xy / z + principal_point
