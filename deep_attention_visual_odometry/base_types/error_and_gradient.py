from typing import NamedTuple
from torch import Tensor


class ErrorAndGradient(NamedTuple):
    """
    A combined error value as a batch-sided float,
    and partial derivatives for the gradient of the error w.r.r each parameter. Size BxP
    """
    error: Tensor
    jacobian: Tensor
