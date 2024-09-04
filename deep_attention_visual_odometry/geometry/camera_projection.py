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
from typing import Self
import torch


class PinholeCameraProjection:
    """
    Traditional pinhole camera projection, with no edge-cases.
    Returns u = f X / Z + cx and v = f Y / Z + cy in all cases
    """

    def __init__(self, camera_parameters: torch.Tensor):
        self._parameters = camera_parameters

    @property
    def focal_length(self) -> torch.Tensor:
        return self._parameters[..., 0:1]

    @property
    def principal_point(self) -> torch.Tensor:
        return self._parameters[..., 1:3]

    def project_points(self, points: torch.Tensor) -> torch.Tensor:
        xy = points[..., 0:2]
        z = points[..., 2:3]
        return self.focal_length * xy / z + self.principal_point

    def parameter_gradient(self, points: torch.Tensor) -> torch.Tensor:
        """
        Get the gradient of the projected points w.r.t. the camera parameters.
        That is, du/dfx, dv/dfx, du/dcx, dv/dcx, du/dcy, dv/dcy
        :returns: A (Bx)2x3 tensor, du gradients in the first row, dv gradients in the second
        """
        z = points[..., 2:3]
        f_on_z = self.focal_length / z

    def vector_gradient(self) -> torch.Tensor:
        """
        Get the gradient of the projected points w.r.t. the point coordinates
        That is, du/dx, dv/dx, du/dy, dv/dy, du/z, and dv/dz
        :return: A (Bx)2x3 tensor, du gradients along the top, and dv gradients in the second row
        """
        pass

    def add(self, parameters: torch.Tensor) -> Self:
        return type(self)(self._parameters + parameters)

    def masked_update(self, other: Self, mask: torch.Tensor) -> Self:
        new_parameters = torch.where(mask, other._parameters, self._parameters)
        return type(self)(new_parameters)


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
