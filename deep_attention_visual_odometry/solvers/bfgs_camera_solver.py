import torch
import torch.nn as nn
from deep_attention_visual_odometry.utils import inverse_curvature
from .i_optimisable_function import IOptimisableFunction


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
        max_iterations: int,
        epsilon: float,
        max_step_distance: float,
        line_search: nn.Module,
    ):
        super().__init__()
        self.line_search = line_search
        self.max_iterations = int(max_iterations)
        self.epsilon = torch.tensor(float(epsilon))
        self.max_step_distance = float(max_step_distance)

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
        inverse_hessian: torch.Tensor = torch.tensor([])

        # TODO: Use updating as a mask to reduce computation
        updating = torch.ones(
            batch_size, num_estimates, device=function.device, dtype=torch.bool
        )
        for step_idx in range(self.max_iterations):
            # Compute a search direction as -H \delta f
            gradient = function.get_gradient()
            if step_idx == 0:
                search_direction = -1.0 * gradient
            else:
                search_direction = -1.0 * torch.matmul(
                    inverse_hessian, gradient.unsqueeze(-1)
                )
                search_direction = search_direction.squeeze(-1)
            # Clamp the search direction. In practice, sometimes there are extreme gradients,
            # And if the inverse hessian is not sufficiently converged yet, we may end up stepping much too far
            search_direction = clamp_search_direction(search_direction, self.max_step_distance)
            # Line search for an update step that satisfies the wolfe conditions
            next_function_point, step = self.line_search(function, search_direction)
            # Update the inverse hessian based on the next chosen point
            # This is expressed as one update equation in the BFGS algorithm,
            # but it's got too many terms for one line.
            delta_gradient = next_function_point.get_gradient() - gradient
            if step_idx == 0:
                # For the first step, initialize the inverse hessian
                inverse_hessian = estimate_initial_inverse_hessian(
                    gradient.size(2), step, delta_gradient
                )
            inverse_hessian = update_inverse_hessian(
                inverse_hessian, step, delta_gradient
            )
            # Set the current evaluation point to the next one
            function = function.masked_update(next_function_point, updating)
            updating = torch.logical_and(
                updating, torch.greater(function.get_error(), self.epsilon)
            )
        return function


def clamp_search_direction(search_direction: torch.Tensor, max_step_length: float) -> torch.Tensor:
    """
    Make sure the search direction is not too large.
    """
    largest_step = search_direction.abs().max(dim=-1).values.clamp(min=1e-8)
    is_too_large = (largest_step > max_step_length)
    scale = torch.where(is_too_large, max_step_length / largest_step, torch.ones_like(largest_step))
    scale = scale.clamp(min=1e-16)
    search_direction = scale[:, :, None] * search_direction
    return search_direction


def estimate_initial_inverse_hessian(
    ndim: int, step: torch.Tensor, delta_gradient: torch.Tensor
) -> torch.Tensor:
    """
    Choose an initial value for the inverse hessian approximation.
    H_0 = \frac{y^T s}{y^T y} I
    This is equation 6.20 from "Numerical Optimisation" by Nocedal and Wright 2009.
    :param ndim: The number of parameters
    :param step: The first chosen step, satisfying the wolffe conditions.
    :param delta_gradient: The change in gradient between the current and next iterate points.
    :return: An initial guess for the inverse hessian H_0, to be updated to H_1.
    """
    denominator = delta_gradient.square().sum(dim=-1).clamp(min=1e-5)
    scale = (step * delta_gradient).sum(dim=-1)
    scale = scale / denominator
    return scale[:, :, None, None] * torch.eye(
        ndim, device=step.device, dtype=step.dtype
    ).reshape(1, 1, ndim, ndim)


def update_inverse_hessian(
    inverse_hessian: torch.Tensor, step: torch.Tensor, delta_gradient: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the BFGS update to the inverse hessian:
    H_{+} = (I - \frac{s y^T}{y^T s})H(I - frac{y s^T}{y^T s^T}) + frac{s s^T}{y^T s}
    (from equation 6.17, "Numerical Optimisation", Nocedal and Wright 2009)

    :param inverse_hessian: The current inverse hessian estimate H
    :param step: The change in parameters, x_{t+1} - x_{t}
    :param delta_gradient: The change in gradient f(x_{t+1}) - f(x_{t})
    :return: The updated inverse hessian estimate, H_{+}
    """
    # = (H - \frac{1}{y^T s} s y^T H)(I - frac{y s^T}{y^T s^T}) + frac{s s^T}{y^T s}
    # = H - frac{1}{y^T s} H y s^T - \frac{1}{y^T s} s y^T H + \frac{1}{(y^T s)^2} s y^T H y s^T + frac{s s^T}{y^T s}
    # = H - frac{1}{y^T s} (H y s^T + s y^T H) + \frac{1}{y^T s} (\frac{y^T H y}{y^T s} + 1.0) s s^T
    # y^T s (curvature):
    # Curvature (this product) _should_ be strictly positive.
    # If it isn't, set this scale factor to 0, so that the update is skipped
    # That is all being handled by an autograd function, to avoid nan gradients.
    inv_curvature = inverse_curvature(step, delta_gradient)
    # y^T H y / (y^T s):
    # For numerical stability, we assume |y^T H| < |y| and |y / (y^T s)| < |y|
    # So we calculate those ratios first before multiplying
    # To avoid an intermediate scale ~|y|^2
    gradient_postmultiply_hessian = torch.matmul(
        delta_gradient[:, :, None, :], inverse_hessian
    )
    gradient_on_curvature = delta_gradient * inv_curvature[:, :, None]
    gradient_inner_product = (
        gradient_postmultiply_hessian * gradient_on_curvature[:, :, None, :]
    ).sum(dim=-1)
    # s s^T / (y^T s):
    # Again, to avoid a matrix scaled ~|s^2|, we scale first by 1/|y^T s|
    step_on_curvature = step * inv_curvature[:, :, None]
    step_outer_product = torch.matmul(
        step_on_curvature[:, :, :, None], step[:, :, None, :]
    )
    # \frac{1}{y^T s} (\frac{y^T H y}{y^T s} + 1.0) s s^T
    step_outer_product = step_outer_product * (
        1.0 + gradient_inner_product[:, :, :, None]
    )

    # \frac{1}{y^T s} s y^T H = \frac{s}{y^T s} (y^T H)
    step_gradient_product = torch.matmul(
        step_on_curvature[:, :, :, None], gradient_postmultiply_hessian
    )
    # frac{1}{y^T s} H y s^T = (H y) frac{s^T}{y^T s}
    gradient_premultiply_hessian = torch.matmul(
        inverse_hessian, delta_gradient[:, :, :, None]
    )
    gradient_step_product = torch.matmul(
        gradient_premultiply_hessian, step_on_curvature[:, :, None, :]
    )

    return (
        inverse_hessian
        + step_outer_product
        - step_gradient_product
        - gradient_step_product
    )
