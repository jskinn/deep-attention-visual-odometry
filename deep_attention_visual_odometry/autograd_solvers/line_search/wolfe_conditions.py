# Copyright (C) 2024  John Skinner
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
from typing import Callable
import warnings
import torch
import torch.autograd


def line_search_wolfe_conditions(
    parameters: torch.Tensor,
    search_direction: torch.Tensor,
    base_error: torch.Tensor,
    base_gradient: torch.Tensor,
    error_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sufficient_decrease: float = 1e-4,
    curvature: float = 0.9,
    strong: bool = False,
) -> torch.Tensor:
    """
    Search for a scaling factor (alpha) on the search direction,
    that produces a point satisfying the wolfe conditions. That is:
    1. f(x + a s) <= f(x) + c_1 a s f'(x)
    2. -s f'(x + a s) <= - c_2 s f'(x)
    with 0 < c_1 < c_2 < 1

    The first condition checks that the error has decreased by at least a certain amount,
    and the second condition ensures that the slope of the error has also decreased.

    Optionally, the second condition can be strengthened to
    2. | s f'(x + a s) | <= c_2 | s f'(x) |
    which is referred to as the strong wolfe conditions.
    See also https://en.wikipedia.org/wiki/Wolfe_conditions

    Implements algorithms 3.5 and 3.6 from Numerical Optimisation by Nocedal and Wright.

    This function does not propagate gradients, since the only output is "a",
    which is determined by search rather than calculated from the inputs.

    This function synchronises the GPU.

    :param parameters: The existing state, x, shape (B..)xN
    :param search_direction: The direction of the line search, s, shape (B..)xN
    :param base_error: The error at the existing state, f(x), shape (B..)x1
    :param base_gradient: The gradient at the existing state, f'(x), shape (B..)xN
    :param error_function: The error function to minimise, f, mapping (B..)xN to (B..)
    :param sufficient_decrease: Parameter c_1, specifying how much the error should decrease. 0 < c_1 < c_2
    :param curvature: Parameter c_2, specifying the change in curvature. c_1 < c_2 < 1.
    :param strong: Whether to apply the alternative "strong" condition on the curvature
    :return: A scalar a such that x + a * s satisfies the (strong) wolfe conditions.  Shape (B..)x1
    """
    if not 0.0 < sufficient_decrease < curvature < 1.0:
        warnings.warn(
            f"Line search conditions should satisfy 0 < c1 < c2 < 1. "
            f"Got c1={sufficient_decrease} and c2={curvature}"
        )
    parameters = parameters.detach()
    search_direction = search_direction.detach()
    base_error = base_error.detach()
    base_gradient = base_gradient.detach()

    # The gradient of the line search (w.r.t. alpha) is the product of the
    # error function gradient and the search direction.
    base_gradient = (search_direction * base_gradient).sum(dim=-1)

    # The algorithm has two phases: widening (algorithm 3.5), and then zooming (3.6)
    batch_dimensions = parameters.shape[:-1]
    widening = torch.ones(batch_dimensions, dtype=torch.bool, device=parameters.device)
    zooming = torch.zeros(batch_dimensions, dtype=torch.bool, device=parameters.device)

    # Each algorithm checks three conditions, which we store in three more masks across the batch dimensions.
    decrease_condition = torch.zeros(
        batch_dimensions, dtype=torch.bool, device=parameters.device
    )
    curvature_condition = torch.zeros(
        batch_dimensions, dtype=torch.bool, device=parameters.device
    )
    gradient_condition = torch.zeros(
        batch_dimensions, dtype=torch.bool, device=parameters.device
    )

    # Track three values for the search.
    # When widening, these are a_i and a_(i - 1). We use 'upper_alpha' as a_(i-1).
    # When zooming, they are a_lo and a_hi, and a_j
    # Since none of these tensors require gradients,
    # we're going to make significant use of in-place operations.
    lower_alpha = torch.zeros(
        batch_dimensions,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    upper_alpha = lower_alpha.clone()
    candidate_alpha = torch.ones(
        batch_dimensions,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    lower_error = base_error.clone()
    upper_error = base_error.clone()
    candidate_error = base_error.clone()
    candidate_gradient = base_gradient.clone()

    for step_idx in range(1000):
        # Stop once all batch elements have finished.
        # This conditional synchronises the GPU.
        updating = widening | zooming
        if not updating.any():
            break
        # Choose the next candidate alpha values
        # For the first step, candidate alpha = 1, and zooming = False.
        if step_idx > 0:
            upper_alpha[widening] = candidate_alpha[widening]
            upper_error[widening] = candidate_error[widening]
            candidate_alpha[widening] = 2.0 * candidate_alpha[widening]
            candidate_alpha[zooming] = _zoom_alpha(
                lower_alpha=lower_alpha[zooming],
                upper_alpha=upper_alpha[zooming],
            )

        # Evaluate the error and gradient for the new candidate
        updating_alpha = (
            candidate_alpha[updating].clone().unsqueeze(-1).requires_grad_(True)
        )
        updating_error = error_function(
            parameters[updating] + updating_alpha * search_direction[updating], updating
        )
        updating_gradient = torch.autograd.grad(updating_error.sum(), updating_alpha)
        candidate_error[updating] = updating_error.detach()
        candidate_gradient[updating] = updating_gradient[0].squeeze(-1).detach()

        # Check the sufficient decrease condition (the armijo condition)
        decrease_condition[updating] = torch.greater(
            candidate_error[updating],
            base_error[updating]
            + sufficient_decrease * candidate_alpha[updating] * base_gradient[updating],
        )
        decrease_condition[zooming] |= torch.greater_equal(
            candidate_error[zooming], lower_error[zooming]
        )
        if step_idx > 0:
            decrease_condition[widening] |= torch.greater_equal(
                candidate_error[widening], upper_error[widening]
            )

        # Check the curvature condition, depending on whether the strong wolfe conditions are desired
        if strong:
            curvature_condition[updating] = torch.less_equal(
                candidate_gradient[updating].abs(),
                -1.0 * curvature * base_gradient[updating],
            )
        else:
            curvature_condition[updating] = torch.less_equal(
                -1.0 * candidate_gradient[updating],
                -1.0 * curvature * base_gradient[updating],
            )

        # Evaluate a gradient condition, different for each algorithm
        # When widening, simply check if f'(x + a_j s) >= 0
        # For zooming, check f'(x + a_j s)(a_hi - a_lo) >= 0
        gradient_condition[widening] = torch.greater_equal(
            candidate_gradient[widening], 0.0
        )
        gradient_condition[zooming] = torch.greater_equal(
            candidate_gradient[zooming] * (upper_alpha[zooming] - lower_alpha[zooming]),
            0.0,
        )

        # Update the zooming points first so that widening can modify the 'zooming' mask
        # - Points that meet both conditions are done, set a_hi = a_lo = a_j and stop zooming
        # - Points that fail the sufficient decrease condition (decrease_condition == True), set a_hi = a_j
        # - Points that pass sufficient decrease but fail gradient condition and curvature, set a_hi = a_lo
        # - Points that pass sufficient decrease and fail curvature (independent of gradient condition), set a_lo = a_j
        set_high_mask = zooming & decrease_condition
        done_mask = zooming & ~decrease_condition & curvature_condition
        flip_mask = (
            zooming & ~decrease_condition & ~curvature_condition & gradient_condition
        )
        set_low_mask = zooming & ~decrease_condition & ~curvature_condition
        upper_alpha[set_high_mask | done_mask] = candidate_alpha[
            set_high_mask | done_mask
        ]
        upper_error[set_high_mask | done_mask] = candidate_error[
            set_high_mask | done_mask
        ]
        upper_alpha[flip_mask] = lower_alpha[flip_mask]
        upper_error[flip_mask] = lower_error[flip_mask]
        lower_alpha[set_low_mask | done_mask] = candidate_alpha[
            set_low_mask | done_mask
        ]
        lower_error[set_low_mask | done_mask] = candidate_error[
            set_low_mask | done_mask
        ]
        zooming &= ~done_mask

        # Update the widening points
        # - If a_i fails the sufficient decrease condition (decrease_condition = True),
        #   zoom with a_lo = a_{i-1} and a_hi = a_i
        # - If a_i passes both the sufficient decrease condition and the curvature condition,
        #   set a_hi = a_lo = a_i and stop
        # - If a_i passes the sufficient decrease, fails the curvature condition, and fails the gradient condition,
        #   zoom with a_lo = a_i and a_hi = a_{i-1}
        zoom_ordered_mask = widening & decrease_condition
        done_mask = widening & ~decrease_condition & curvature_condition
        zoom_flipped_mask = (
            widening & ~decrease_condition & ~curvature_condition & gradient_condition
        )
        lower_alpha[zoom_ordered_mask] = upper_alpha[zoom_ordered_mask]
        lower_error[zoom_ordered_mask] = upper_error[zoom_ordered_mask]
        upper_alpha[zoom_ordered_mask | done_mask] = candidate_alpha[
            zoom_ordered_mask | done_mask
        ]
        upper_error[zoom_ordered_mask | done_mask] = candidate_error[
            zoom_ordered_mask | done_mask
        ]
        lower_alpha[done_mask | zoom_flipped_mask] = candidate_alpha[
            done_mask | zoom_flipped_mask
        ]
        lower_error[done_mask | zoom_flipped_mask] = candidate_error[
            done_mask | zoom_flipped_mask
        ]
        zooming |= zoom_ordered_mask | zoom_flipped_mask
        zooming &= (lower_alpha != upper_alpha)     # Simple failure case, where the bounds converge quickly
        widening &= ~(zoom_ordered_mask | done_mask | zoom_flipped_mask)

    return upper_alpha


def _zoom_alpha(
    lower_alpha: torch.Tensor,
    upper_alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Pick a new candidate alpha between two bounds.

    :param lower_alpha: Lower bound
    :param upper_alpha: Upper bound
    """
    # TODO: Add quadratic or other interpolation
    return 0.5 * (lower_alpha + upper_alpha)
