from typing import Self
import torch
from deep_attention_visual_odometry.solvers.i_optimisable_function import (
    IOptimisableFunction,
)


class ExampleFunctionDistanceToPoint(IOptimisableFunction):
    def __init__(self, parameters: torch.Tensor, target_points: torch.Tensor):
        self.parameters = parameters
        self.target_points = target_points

    @property
    def batch_size(self) -> int:
        return self.parameters.size(0)

    @property
    def num_estimates(self) -> int:
        return self.parameters.size(1)

    @property
    def num_parameters(self) -> int:
        return self.parameters.size(2)

    @property
    def device(self) -> torch.device:
        return self.parameters.device

    def get_error(self) -> torch.Tensor:
        """Get the error of the function at the current point. BxE"""
        return (self.parameters - self.target_points).square().sum(dim=2)

    def get_gradient(self) -> torch.Tensor:
        """Get the gradient of the function w.r.t. each of the parameters. BxExP"""
        return 2.0 * (self.parameters - self.target_points)

    def add(self, parameters: torch.Tensor) -> Self:
        """Return a new instance of this function at a new set of parameters.
        Used for the optimisation step.
        Input tensor should be BxExN, the same shape as the output of 'get_gradient'
        """
        return ExampleFunctionDistanceToPoint(
            parameters=self.parameters + parameters, target_points=self.target_points
        )

    def masked_update(self, other: Self, mask: torch.Tensor) -> Self:
        """
        The operation we need to be able to do is to keep values from the current parameters where false,
        and where true set them to the values from another instance, plus a delta.

        This might be able to be expressed as two functions, add(..) and then update(...) to merge.

        We want to retain already-computed gradients and errors as much as possible.
        """
        mask = mask[:, :, None].tile(1, 1, self.num_parameters)
        new_parameters = torch.where(mask, other.parameters, self.parameters)
        return ExampleFunctionDistanceToPoint(
            parameters=new_parameters, target_points=self.target_points
        )
