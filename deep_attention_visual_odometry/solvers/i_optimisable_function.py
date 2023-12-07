from typing import Self
from abc import ABC, abstractmethod
import torch


class IOptimisableFunction(ABC):
    """
    Represents evaluation of an optimisable function for a particular set of arguments.
    Specifically supposed to support lazy evaluation of the error and gradient.
    Should also support batching of the parameters, and multiple simultaneous estimates
    """
    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def num_estimates(self) -> int:
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @abstractmethod
    def get_error(self) -> torch.Tensor:
        """Get the error of the function at the current point. BxE"""
        pass

    @abstractmethod
    def get_gradient(self) -> torch.Tensor:
        """Get the gradient of the function w.r.t. each of the parameters. BxExP"""
        pass

    @abstractmethod
    def add(self, parameters: torch.Tensor) -> Self:
        """Return a new instance of this function at a new set of parameters.
        Used for the optimisation step.
        Input tensor should be BxExN, the same shape as the output of 'get_gradient'
        """
        pass

    @abstractmethod
    def masked_add(self, parameters: torch.Tensor, mask: torch.Tensor) -> Self:
        """Return a new instance of this function, updating only a subset of the estimates/batches.
        The mask parameter should be a BxE tensor indicating which elements to update.
        The parameters then should be a MxP tensor containing new values that are added where the mask is true.
        """
        pass

    @abstractmethod
    def masked_update(self, other: Self, parameters: torch.Tensor, mask: torch.Tensor):
        """
        The operation we need to be able to do is to keep values from the current parameters where false,
        and where true set them to the values from another instance, plus a delta.

        This might be able to be expressed as two functions, add(..) and then update(...) to merge.

        We want to retain already-computed gradients and errors as much as possible.
        """
        pass
