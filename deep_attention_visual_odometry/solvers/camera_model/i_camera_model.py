from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import torch
from .i_lazy_error_and_gradient import ILazyErrorAndGradient


TParameters = TypeVar("TParameters")


class ICameraModel(ABC, Generic[TParameters]):
    @abstractmethod
    def make_parameters(self) -> TParameters:
        pass

    @abstractmethod
    def find_error_and_gradient(
        self, parameters: TParameters, points_2d: torch.Tensor, weights: torch.Tensor
    ) -> ILazyErrorAndGradient:
        pass
