from torch import Tensor
from typing import NamedTuple


class CameraViewsAndPoints(NamedTuple):
    projected_points: Tensor
    projection_weights: Tensor
    camera_intrinsics: Tensor
    camera_extrinsics: Tensor
    world_points: Tensor
