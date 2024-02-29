from torch import Module, Tensor
from abc import ABC, abstractmethod


class IModifySearchDirections(ABC, Module):
    """
    A kind of torch module that adjusts the search direction of a quasi-newton optimisation.

    (The point of this interface is to standardise the forward arguments, allowing Liskov substitution,
    which normal torch Modules violate.)
    """

    @abstractmethod
    def forward(
        self, search_direction: Tensor, parameters: Tensor, error: Tensor, step_idx: int
    ) -> Tensor:
        pass
