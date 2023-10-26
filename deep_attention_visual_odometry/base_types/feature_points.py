from torch import Tensor
from typing import NamedTuple


class FeaturePoints(NamedTuple):
    points: Tensor
    descriptors: Tensor
