import math
from typing import Iterable, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as fn
from .i_optimisable_function import IOptimisableFunction
from .least_squares_utils import find_residuals, find_error, find_error_gradient


class _StepResidualsGradients(NamedTuple):
    step: torch.Tensor
    error: torch.Tensor
    gradient: torch.Tensor


class BFGSCameraSolver(nn.Module):
    """
    Given a set of points across multiple views,
    jointly optimise for the 3D positions of the points, the camera intrinsics, and extrinsics.

    Based on
    "More accurate pinhole camera calibration with imperfect planar target" by
    Klaus H Strobl and Gerd Hirzinger in ICCV 2011

    Uses the Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS) to iteratively optimise the various parameters.
    There are two learned heuristics modifying the algorithm, one as a warp on the search direction,
    and the other when searching for the step size.
    Changing the search direction in particular lets the optimiser specialise to only optimising some of the parameters,
    and leaving the others to the prior.

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
        search_direction_heuristic: nn.Module | None = None,
        line_search_sufficient_decrease: float = 1e-4,
        line_search_curvature: float = 0.9,
        line_search_max_step_size: float = 16.0,
        line_search_zoom_iterations: int = 20,
    ):
        super().__init__()
        self.search_direction_heuristic = search_direction_heuristic
        self.max_iterations = int(max_iterations)
        self.line_search_max_step_size = int(line_search_max_step_size)
        self.line_search_zoom_iterations = int(line_search_zoom_iterations)

        # line search parameters (c_1 and c_2).
        self.line_search_sufficient_decrease = line_search_sufficient_decrease
        self.line_search_curvature = line_search_curvature

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

        # TODO: Keep track of which ones are updating, and set the step for the others to zero
        updating = torch.ones(batch_size, num_estimates, device=function.device)
        for step in range(self.max_iterations):
            # Compute a search direction as -H \delta f
            gradient = function.get_gradient()
            search_direction = -1 * torch.matmul(inverse_hessian, gradient)
            # Use a heuristic to adjust the search direction, to stabilise the search
            search_direction = self.search_direction_heuristic(search_direction, step)
            # Line search for an update step that satisfies the wolfe conditions
            next_function_point, step = self.line_search(function, search_direction)
            # Update the inverse hessian based on the next chosen point
            # This is expressed as one update equation in the BFGS algorithm,
            # but it's got too many terms for one line.
            delta_gradient = next_function_point.get_gradient() - gradient
            step_dot_delta_grad = torch.sum(step * delta_gradient, dim=-1)
            inv_hessian_times_delta_grad = (delta_gradient[:, :, None, :] @ inverse_hessian @ delta_gradient[:, :, :, None])
            step_outer_product = step[:, :, :, None] @ step[:, :, None, :]
            inv_hessian_delta_grad_step = inverse_hessian @ delta_gradient[:, :, :, None] @ step[:, :, None, :]
            step_delta_grad_inv_hessian = step[:, :, :, None] @ delta_gradient[:, :, None, :] @ inverse_hessian
            delta_inv_hessian = torch.zeros_like(inverse_hessian)
            delta_inv_hessian[step_dot_delta_grad.squeeze(2, 3) != 0] = (
                    step_outer_product * (
                        step_dot_delta_grad + inv_hessian_times_delta_grad) / step_dot_delta_grad.square()
                    - (inv_hessian_delta_grad_step + step_delta_grad_inv_hessian) / step_dot_delta_grad
            )
            inverse_hessian = inverse_hessian + delta_inv_hessian
            # Set the current evaluation point to the next one
            gradient = next_function_point.get_gradient()
            function = next_function_point
        return function

    def line_search(
        self,
        function: IOptimisableFunction,
        search_direction: torch.Tensor,
    ) -> tuple[IOptimisableFunction, torch.Tensor]:
        """
        Line search for a step that satisfies the strong wolfe conditions.
        Implements algorithms 3.5 and 3.6 from Numerical Optimisation by Nocedal and Wright

        :param parameters:
        :param search_direction:
        :param initial_points_and_jacobian:
        :return:
        """
        batch_size = function.batch_size
        num_estimates = function.num_estimates
        lower_alpha = torch.zeros(batch_size, num_estimates, dtype=search_direction.dtype, device=search_direction.device)
        upper_alpha = torch.ones(batch_size, num_estimates, dtype=search_direction.dtype, device=search_direction.device)

        base_error = function.get_error()
        base_gradient = torch.sum(function.get_gradient() * search_direction, dim=-1)

        lower_error = function.get_error()
        upper_error = function.get_error()
        lower_gradient = function.get_gradient()
        upper_gradient = function.get_gradient()

        prev_error = base_error
        line_base_gradient = (base_gradient * search_direction).sum(dim=-1)

        candidate_function = function
        widening = torch.ones(batch_size, num_estimates, dtype=torch.bool, device=search_direction.device)
        zooming = torch.zeros(batch_size, num_estimates, dtype=torch.bool, device=search_direction.device)
        # Algorithm 3.5: Widen the search until the upper bound doesn't satisfy one of the conditions
        max_widen_iterations = int(math.ceil(math.log2(self.line_search_max_step_size)))
        for idx in range(max_widen_iterations):
            candidate_function = function.masked_add(upper_alpha[widening] * search_direction[widening], widening)
            error = candidate_function.get_error()
            upper_error[widening] = error[widening]
            # First conditional, check if the error has started increasing
            increasing_error = torch.greater(error[widening], base_error[widening] + self.line_search_sufficient_decrease * upper_alpha[widening] * base_gradient)
            if idx > 0:
                increasing_error = torch.logical_or(
                    increasing_error,
                    torch.greater_equal(error, prev_error)
                )
            zooming[widening] = torch.logical_or(zooming[widening], increasing_error)
            still_widening = torch.logical_not(increasing_error)
            # Second conditional, check if we've actually already met the conditions
            gradient = candidate_function.get_gradient()
            gradient = (gradient[still_widening] * search_direction[still_widening]).sum(dim=-1)
            met_conditions = torch.less_equal(gradient.abs(), -self.line_search_curvature * line_base_gradient[still_widening])
            still_widening[still_widening.clone()] = torch.logical_not(met_conditions)
            # Third conditional, positive gradient
            # TODO: The widening indices here are not quite right, still widening should update widening and reset
            #       to ~met_conditions
            positive_gradient = torch.greater_equal(gradient[still_widening], 0.0)
            zooming[still_widening] = positive_gradient
            temp = upper_alpha[positive_gradient]
            upper_alpha[positive_gradient] = lower_alpha[positive_gradient]
            lower_alpha[positive_gradient] = temp
            # TODO: Also swap stored errors
            widening[widening.clone()] = still_widening
            # Increase step size for anything still widening
            lower_alpha[widening] = upper_alpha[widening]
            upper_alpha[widening] = 2.0 * upper_alpha[widening]

        # Algorithm 3.6: Zoom on bounded ranges to find a point that satisfies the conditions
        for idx in range(self.line_search_zoom_iterations):
            # Linearly interpolate the high and low gradients to pick a new alpha
            gradient_diff = upper_gradient[zooming] - lower_gradient[zooming]
            candidate_alpha = lower_alpha[zooming] - lower_gradient[zooming] * (upper_alpha[zooming] - lower_alpha[zooming]) / (gradient_diff + 1e-8)
            # Fall back to bisection if the gradients were the same, or the candidate is outside our range
            non_linear_alpha = torch.logical_or(torch.eq(gradient_diff, 0.0), torch.logical_or(torch.less(candidate_alpha, torch.min(lower_alpha, upper_alpha)), torch.greater()))
            candidate_alpha[non_linear_alpha] = (lower_alpha[zooming] + upper_alpha[zooming]) / 2.0
            # Evaluate the gradient at the candidate
            candidate_function = function.masked_add()
            candidate_parameters = parameters[zooming] + candidate_alpha * search_direction[zooming]
            error_and_gradient = self.camera_model.find_error_and_gradient(candidate_parameters, points_2d[zooming], point_weights[zooming])
            error = error_and_gradient.get_error()

            error_and_gradient.reduce()

        return candidate_function


