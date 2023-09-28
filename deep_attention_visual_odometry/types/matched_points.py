from torch import Tensor
from typing import NamedTuple


class MatchedPoints(NamedTuple):
    points_a: Tensor
    points_b: Tensor
