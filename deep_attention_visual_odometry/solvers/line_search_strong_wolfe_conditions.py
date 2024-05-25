import warnings
import math
import torch
import torch.nn as nn
from deep_attention_visual_odometry.utils import interpolate_alpha
from .i_optimisable_function import IOptimisableFunction


class LineSearchStrongWolfeConditions(nn.Module):
    """
    Line search for a point satisfying the strong wolfe conditions.

    Implements algorithms 3.5 and 3.6 from Numerical Optimisation by Nocedal and Wright
    """

    def __init__(
        self,
        max_step_size: float,
        zoom_iterations: int,
        sufficient_decrease: float = 1e-4,
        curvature: float = 0.9,
    ):
        super().__init__()
        if not 0.0 < sufficient_decrease < curvature < 1.0:
            warnings.warn(
                f"Line search conditions should satisfy 0 < c1 < c2 < 1. "
                f"Got c1={sufficient_decrease} and c2={curvature}"
            )
        self.max_step_size = torch.tensor(float(max_step_size))
        self.widen_iterations = int(math.ceil(math.log2(max_step_size)))
        self.zoom_iterations = int(zoom_iterations)
        self.sufficient_decrease = torch.tensor(float(sufficient_decrease))
        self.curvature = torch.tensor(float(curvature))

    def forward(
        self, function: IOptimisableFunction, search_direction: torch.Tensor
    ) -> tuple[IOptimisableFunction, torch.Tensor]:
        batch_size = function.batch_size
        num_estimates = function.num_estimates
        lower_alpha = torch.zeros(
            batch_size,
            num_estimates,
            dtype=search_direction.dtype,
            device=search_direction.device,
        )
        upper_alpha = torch.ones(
            batch_size,
            num_estimates,
            dtype=search_direction.dtype,
            device=search_direction.device,
        )

        error_scale = torch.tensor(
            1.0 / function.num_parameters, device=function.device, dtype=torch.float
        ).sqrt()
        base_error = error_scale * function.get_error()
        base_gradient = torch.sum(
            error_scale * function.get_gradient() * search_direction, dim=-1
        )

        lower_candidate_function = function
        upper_candidate_function = function
        output_function = function
        output_step = torch.zeros_like(search_direction)

        prev_error = base_error

        widening = torch.ones(
            batch_size, num_estimates, dtype=torch.bool, device=search_direction.device
        )
        zooming = torch.zeros(
            batch_size, num_estimates, dtype=torch.bool, device=search_direction.device
        )
        # Algorithm 3.5: Widen the search until the upper bound doesn't satisfy one of the conditions
        for idx in range(self.widen_iterations):
            candidate_step = upper_alpha.unsqueeze(-1) * search_direction
            candidate_function = function.add(candidate_step)
            upper_candidate_function = upper_candidate_function.masked_update(
                candidate_function, widening
            )
            error = error_scale * upper_candidate_function.get_error()
            # First conditional, check if the error has started increasing
            increasing_error = torch.zeros_like(widening)
            increasing_error[widening] = torch.greater(
                error[widening],
                base_error[widening]
                + self.sufficient_decrease
                * upper_alpha[widening]
                * base_gradient[widening],
            )
            if idx > 0:
                increasing_error[widening] = torch.logical_or(
                    increasing_error[widening],
                    torch.greater_equal(error[widening], prev_error[widening]),
                )
            zooming = torch.logical_or(zooming, increasing_error)
            widening = torch.logical_and(widening, torch.logical_not(increasing_error))
            # Second conditional, check if we've actually already met the conditions
            gradient = upper_candidate_function.get_gradient()
            gradient = (
                error_scale * gradient[widening] * search_direction[widening]
            ).sum(dim=-1)
            met_conditions = torch.zeros_like(widening)
            met_conditions[widening] = torch.less_equal(
                gradient.abs(), -self.curvature * base_gradient[widening]
            )
            output_function = output_function.masked_update(
                upper_candidate_function, met_conditions
            )
            output_step = torch.where(
                met_conditions[:, :, None], candidate_step, output_step
            )
            not_met_conditions = torch.logical_not(met_conditions)
            gradient = gradient[not_met_conditions[widening]]
            widening = torch.logical_and(widening, not_met_conditions)
            # Third conditional, positive gradient
            positive_gradient = torch.greater_equal(gradient, 0.0)
            swapping = torch.zeros_like(widening)
            swapping[widening] = positive_gradient
            zooming = torch.logical_or(zooming, swapping)
            new_upper_alpha = torch.where(swapping, lower_alpha, upper_alpha)
            new_lower_alpha = torch.where(swapping, upper_alpha, lower_alpha)
            upper_alpha = new_upper_alpha
            lower_alpha = new_lower_alpha
            new_lower_candidate_function = lower_candidate_function.masked_update(
                upper_candidate_function, swapping
            )
            new_upper_candidate_function = upper_candidate_function.masked_update(
                lower_candidate_function, swapping
            )
            lower_candidate_function = new_lower_candidate_function
            upper_candidate_function = new_upper_candidate_function
            widening = torch.logical_and(widening, torch.logical_not(swapping))
            # Increase step size for anything still widening
            lower_alpha[widening] = upper_alpha[widening]
            upper_alpha[widening] = torch.minimum(
                2.0 * upper_alpha[widening], self.max_step_size
            )
            lower_candidate_function = lower_candidate_function.masked_update(
                upper_candidate_function, widening
            )
            prev_error = error

        # Algorithm 3.6: Zoom on bounded ranges to find a point that satisfies the conditions
        for idx in range(self.zoom_iterations):
            # Linearly interpolate the high and low gradients to pick a new alpha
            lower_gradient = lower_candidate_function.get_gradient()
            upper_gradient = upper_candidate_function.get_gradient()
            lower_gradient = (
                error_scale * lower_gradient[zooming] * search_direction[zooming]
            ).sum(dim=-1)
            upper_gradient = (
                error_scale * upper_gradient[zooming] * search_direction[zooming]
            ).sum(dim=-1)
            candidate_alpha = interpolate_alpha(lower_alpha[zooming], upper_alpha[zooming], lower_gradient, upper_gradient)
            candidate_alpha_wide = torch.zeros_like(upper_alpha)
            candidate_alpha_wide[zooming] = candidate_alpha
            # Evaluate the function at the chosen candidate points
            candidate_step = torch.zeros_like(search_direction)
            candidate_step[zooming] = (
                candidate_alpha[:, None] * search_direction[zooming]
            )
            candidate_function = function.add(candidate_step)

            # Evaluate the error
            lower_error = error_scale * lower_candidate_function.get_error()
            error = error_scale * candidate_function.get_error()
            # Conditional one: Has the error at this intermediate increased from the low point
            #                  or a linear interpolation from the base?
            increasing_error = torch.zeros_like(zooming)
            increasing_error[zooming] = torch.greater(
                error[zooming],
                base_error[zooming]
                + self.sufficient_decrease * candidate_alpha * base_gradient[zooming],
            )
            increasing_error = torch.logical_and(
                zooming,
                torch.logical_or(
                    torch.greater_equal(error, lower_error), increasing_error
                ),
            )
            # Where true, a_upper = a_j
            upper_alpha = torch.where(
                increasing_error, candidate_alpha_wide, upper_alpha
            )
            upper_candidate_function = upper_candidate_function.masked_update(
                candidate_function, increasing_error
            )
            # Everything after this point should be else with the above condition
            non_increasing_error = torch.logical_and(
                zooming, torch.logical_not(increasing_error)
            )
            # Compute the gradient
            gradient = candidate_function.get_gradient()
            gradient = (
                error_scale
                * gradient[non_increasing_error]
                * search_direction[non_increasing_error]
            ).sum(dim=-1)
            # Conditional 2: Does the candidate meet the conditions
            met_conditions = torch.zeros_like(non_increasing_error)
            met_conditions[non_increasing_error] = torch.less_equal(
                gradient.abs(), -self.curvature * base_gradient[non_increasing_error]
            )
            # Where true, a* = a_j and stop
            output_function = output_function.masked_update(
                candidate_function, met_conditions
            )
            output_step = torch.where(
                met_conditions[:, :, None], candidate_step, output_step
            )
            zooming = torch.logical_and(zooming, torch.logical_not(met_conditions))
            # Conditional 3: Set high to low if the gradient becomes positive
            reversed_gradient = torch.greater_equal(
                gradient
                * (
                    upper_alpha[non_increasing_error]
                    - lower_alpha[non_increasing_error]
                ),
                0.0,
            )
            swapping = torch.zeros_like(zooming)
            swapping[non_increasing_error] = torch.logical_and(
                reversed_gradient, zooming[non_increasing_error]
            )
            upper_alpha = torch.where(swapping, lower_alpha, upper_alpha)
            upper_candidate_function = upper_candidate_function.masked_update(
                lower_candidate_function, swapping
            )
            # Update the lower alpha for the else case
            non_increasing_error = torch.logical_and(non_increasing_error, zooming)
            lower_alpha = torch.where(
                non_increasing_error, candidate_alpha_wide, lower_alpha
            )
            lower_candidate_function = lower_candidate_function.masked_update(
                candidate_function, non_increasing_error
            )

        # If either algorithm didn't finish, we want to pick a non-zero step anyway
        unfinished = torch.logical_or(zooming, widening)
        output_function = output_function.masked_update(
            upper_candidate_function, unfinished
        )
        output_step = torch.where(
            unfinished[:, :, None],
            upper_alpha[:, None] * search_direction,
            output_step,
        )
        return output_function, output_step
