import math
from typing import Iterable, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as fn
from .i_optimisable_function import IOptimisableFunction
from .least_squares_utils import find_residuals, find_error, find_error_gradient


class LearnedBFGSSolver(nn.Module):
    """
    Use the Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS) to iteratively optimise the various parameters.
    There are two learned heuristics modifying the algorithm, one as a warp on the search direction,
    and the other to resample the current

    The idea is that the learned modules can both warp the search directions to exclude certain directions,
    updating only a few at a time.
    Also, by running multiple estimates simultaneously from different initial guesses,
    we can combine and resample the estimate points st each step, focusing on those that have produced
    a low error.
    """

    def __init__(
        self,
        num_parameters: int,
        max_iterations: int,
        epsilon: float,
        line_search: nn.Module,
        search_direction_heuristic: nn.Module,
        transform_estimates: nn.Module,
    ):
        super().__init__()
        self.max_iterations = int(max_iterations)
        self.epsilon = torch.tensor(float(epsilon))
        self.line_search = line_search
        self.search_direction_heuristic = search_direction_heuristic
        self.transform_estimates = transform_estimates

        # Initial values of the inverse hessian
        self.inv_hessian = nn.Parameter(torch.eye(num_parameters))

    def forward(
        self,
        function: IOptimisableFunction,
    ) -> IOptimisableFunction:
        """
        Iterate
        :param function: An IOptimisableFunction, bundling multiple tensors together.
        :return: An instance of the same IOptimisableFunction, with optimised parameters.
        """
        batch_size = function.batch_size
        num_estimates = function.num_estimates

        # Initialise the inverse hessian
        inverse_hessian = self.inv_hessian.tile(batch_size, num_estimates, 1, 1)

        # TODO: Use updating as a mask to reduce computation
        updating = torch.ones(batch_size, device=function.device)
        for step in range(self.max_iterations):
            # Compute a search direction as -H \delta f
            gradient = function.get_gradient()
            search_direction = -1 * torch.matmul(inverse_hessian, gradient)
            # Use a heuristic to adjust the search direction, to stabilise the search
            search_direction = self.search_direction_heuristic(search_direction, step)
            # Line search for an update step that satisfies the wolfe conditions
            next_function_point, step = self.line_search(function, search_direction)
            # Use a sub-network to resample the next points after optimisation
            next_function_point, step = self.transform_estimates(
                next_function_point, step
            )
            # Cross all the different estimates, to produce
            # Update the inverse hessian based on the next chosen point
            # This is expressed as one update equation in the BFGS algorithm,
            # but it's got too many terms for one line.
            delta_gradient = next_function_point.get_gradient() - gradient
            step_dot_delta_grad = torch.sum(step * delta_gradient, dim=-1)
            inv_hessian_times_delta_grad = (
                delta_gradient[:, :, None, :]
                @ inverse_hessian
                @ delta_gradient[:, :, :, None]
            )
            step_outer_product = step[:, :, :, None] @ step[:, :, None, :]
            inv_hessian_delta_grad_step = (
                inverse_hessian @ delta_gradient[:, :, :, None] @ step[:, :, None, :]
            )
            step_delta_grad_inv_hessian = (
                step[:, :, :, None] @ delta_gradient[:, :, None, :] @ inverse_hessian
            )
            delta_inv_hessian = torch.zeros_like(inverse_hessian)
            delta_inv_hessian[step_dot_delta_grad.squeeze(2, 3) != 0] = (
                step_outer_product
                * (step_dot_delta_grad + inv_hessian_times_delta_grad)
                / step_dot_delta_grad.square()
                - (inv_hessian_delta_grad_step + step_delta_grad_inv_hessian)
                / step_dot_delta_grad
            )
            inverse_hessian = inverse_hessian + delta_inv_hessian
            # Set the current evaluation point to the next one
            # Batch elements with an error less than the configured epsilon stop updating.
            function = function.masked_update(
                next_function_point, updating[:, None].tile(1, num_estimates)
            )
            error = function.get_error()
            error = error.min(dim=1).values
            updating = torch.logical_and(updating, torch.greater(error, self.epsilon))
        return function
