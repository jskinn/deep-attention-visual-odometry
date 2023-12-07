from typing import Self
from abc import ABC, abstractmethod
from torch import Tensor


class ILazyErrorAndGradient(ABC):
    """
    Lazy evaluation for the error and gradient.
    Evaluating the gradient of a function can require
    intermediate values that are also used when computing the error, or vice versa.
    To minimise computation, we might want to evaluate only one of those,
    reduce which items in the batch we are
    """

    @abstractmethod
    def get_error(self) -> Tensor:
        pass

    @abstractmethod
    def get_gradient(self) -> Tensor:
        pass

    @abstractmethod
    def reduce(self: Self, mask: Tensor) -> Self:
        """Slice along the batch dimension, returning a subset of the values."""
        pass
