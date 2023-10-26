from typing import NamedTuple
from torch import Tensor


class PointsAndJacobian(NamedTuple):
    """
    A set of estimated 2D points and the corresponding Jacobian for the parameters used to estiamte them.
    Points should have shape (batch x )num_frames x num_points x 2.
    Jacobian should have shape (batch x )num_frames x num_points x 2 x num_parameters.
    If one has a batch dimension, both should.
    """
    points: Tensor
    jacobian: Tensor
