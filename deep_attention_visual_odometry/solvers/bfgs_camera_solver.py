import math
from typing import Iterable, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as fn
from .i_optimisable_function import IOptimisableFunction
from .least_squares_utils import find_residuals, find_error, find_error_gradient


class BFGSCameraSolver(nn.Module):
    """
    Use the Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS) to iteratively optimise an underlying function.
    Relies on a separate line search algorithm that must satisfy the wolfe conditions.

    Why BFGS and not levenberg-marquard?
    Due to optimising the 3D positions of the points, the number of parameters and
    the size of the Jacobian matrix can get very large.
    The Levenberg-Marquardt algorithm requires either computing and inverting J^T J or solving the normal equations for
    Jp = r, both of which (according to "The Levenberg-Marquardt algorithm: implementation and theory" by Jorge More)
    are numerically unstable.
    BFGS allows us to avoid inverting the Jacobian, which is an O(n^3) operation.
    """

    def __init__(
        self,
        num_parameters: int,
        max_iterations: int,
        epsilon: float,
        line_search: nn.Module,
    ):
        super().__init__()
        self.line_search = line_search
        self.max_iterations = int(max_iterations)
        self.epsilon = torch.tensor(float(epsilon))

        # Initial values of the inverse hessian
        self.inv_hessian = nn.Parameter(torch.eye(num_parameters))

    def forward(
        self,
        function: IOptimisableFunction,
    ) -> IOptimisableFunction:
        """

        :param function: An IOptimisableFunction, bundling multiple tensors together.
        :return:
        """
        batch_size = function.batch_size
        num_estimates = function.num_estimates

        # Initialise the inverse hessian
        inverse_hessian = self.inv_hessian.tile(batch_size, num_estimates, 1, 1)

        # TODO: Use updating as a mask to reduce computation
        updating = torch.ones(batch_size, num_estimates, device=function.device)
        for step in range(self.max_iterations):
            # Compute a search direction as -H \delta f
            gradient = function.get_gradient()
            search_direction = -1 * torch.matmul(inverse_hessian, gradient)
            # Line search for an update step that satisfies the wolfe conditions
            next_function_point, step = self.line_search(function, search_direction)
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
            delta_inv_hessian[step_dot_delta_grad.squeeze(dim=(2, 3)) != 0] = (
                step_outer_product
                * (step_dot_delta_grad + inv_hessian_times_delta_grad)
                / step_dot_delta_grad.square()
                - (inv_hessian_delta_grad_step + step_delta_grad_inv_hessian)
                / step_dot_delta_grad
            )
            inverse_hessian = inverse_hessian + delta_inv_hessian
            # Set the current evaluation point to the next one
            function = function.masked_update(next_function_point, updating)
            updating = torch.logical_and(
                updating, torch.greater(function.get_error(), self.epsilon)
            )
        return function
