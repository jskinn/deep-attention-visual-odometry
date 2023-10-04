from abc import ABC, abstractmethod
from typing import TypeVar, Generic
import torch
from deep_attention_visual_odometry.types import PointsAndJacobian


TParameters = TypeVar("TParameters")


class ICameraModel(ABC, Generic[TParameters]):
    @abstractmethod
    def make_parameters(self) -> TParameters:
        pass

    @abstractmethod
    def forward_model(self, camera_parameters: TParameters) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_model_and_jacobian(
        self, camera_parameters: TParameters
    ) -> PointsAndJacobian:
        pass
