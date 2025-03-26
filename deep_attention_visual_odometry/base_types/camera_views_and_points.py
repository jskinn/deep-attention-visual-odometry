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
from torch import Tensor
from typing import NamedTuple


class CameraViewsAndPoints(NamedTuple):
    """
    A set of world points, camera parameters, and those points as viewed by those cameras.
    Designed for M views of N points.
    Used as a batch type from the data loaders, so may gain a batch dimension
    """

    projected_points: Tensor  # (Bx)MxNx2
    visibility_mask: Tensor  # (Bx)MxN
    camera_intrinsics: Tensor  # (Bx)3
    camera_orientations: Tensor # (Bx)(M - 1)x3
    camera_translations: Tensor  # (Bx)(M - 1)x3
    world_points: Tensor  # (Bx)Nx3
