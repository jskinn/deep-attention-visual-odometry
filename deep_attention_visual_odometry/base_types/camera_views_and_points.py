from torch import Tensor
from typing import NamedTuple


class CameraViewsAndPoints(NamedTuple):
    """
    A set of world points, camera parameters, and those points as viewed by those cameras.
    Designed for M views of N points.
    Used as a batch type from the data loaders, so may gain a batch dimension
    """
    projected_points: Tensor    # (Bx)MxNx2
    visibility_mask: Tensor     # (Bx)MxN
    camera_intrinsics: Tensor   # (Bx)Mx3x4
    camera_extrinsics: Tensor   # (Bx)Mx4x4
    world_points: Tensor        # (Bx)Nx3
