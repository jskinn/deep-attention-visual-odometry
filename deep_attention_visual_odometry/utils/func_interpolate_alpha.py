from typing import Any
import torch


class InterpolateAlpha(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
        ctx: Any,
        alpha_1: torch.Tensor,
        alpha_2: torch.Tensor,
        value_1: torch.Tensor,
        value_2: torch.Tensor,
    ) -> torch.Tensor:
        min_alpha = torch.minimum(alpha_1, alpha_2)
        max_alpha = torch.maximum(alpha_1, alpha_2)
        # Linearly interpolate the high and low gradients to pick a new alpha
        value_diff = value_2 - value_1
        inv_gradient = (alpha_2 - alpha_1) / value_diff
        candidate_alpha = alpha_1 - value_1 * inv_gradient
        # Fall back to bisection if the gradients were the same, or the candidate is outside our range
        non_linear_alpha = torch.logical_or(
            torch.eq(value_diff, 0.0),
            torch.logical_or(
                torch.less(candidate_alpha, min_alpha),
                torch.greater(candidate_alpha, max_alpha),
            ),
        )
        candidate_alpha[non_linear_alpha] = (
            alpha_1[non_linear_alpha] + alpha_2[non_linear_alpha]
        ) / 2.0
        # Keep the mask of which
        one_on_value_diff = 1.0 / value_diff
        one_on_value_diff[non_linear_alpha] = 0.0
        ctx.save_for_backward(
            value_1, value_2, one_on_value_diff, inv_gradient, non_linear_alpha
        )
        ctx.set_materialize_grads(False)
        return candidate_alpha

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if grad_output is None:
            return None, None, None, None
        (
            value_1,
            value_2,
            one_on_value_diff,
            inv_gradient,
            non_linear_alpha,
        ) = ctx.saved_tensors
        grad_alpha_1 = grad_alpha_2 = grad_value_1 = grad_value_2 = None
        if ctx.needs_input_grad[0]:
            grad_alpha_1 = torch.where(
                non_linear_alpha,
                0.5 * grad_output,
                one_on_value_diff * value_2 * grad_output,
            )
        if ctx.needs_input_grad[1]:
            grad_alpha_2 = torch.where(
                non_linear_alpha,
                0.5 * grad_output,
                -1.0 * one_on_value_diff * value_1 * grad_output,
            )
        if ctx.needs_input_grad[2]:
            grad_value_1 = torch.where(
                non_linear_alpha,
                torch.zeros_like(grad_output),
                -1.0 * value_2 * inv_gradient * one_on_value_diff * grad_output,
            )
        if ctx.needs_input_grad[3]:
            grad_value_2 = torch.where(
                non_linear_alpha,
                torch.zeros_like(grad_output),
                value_1 * inv_gradient * one_on_value_diff * grad_output,
            )
        return grad_alpha_1, grad_alpha_2, grad_value_1, grad_value_2


def interpolate_alpha(
    alpha_1: torch.Tensor,
    alpha_2: torch.Tensor,
    value_1: torch.Tensor,
    value_2: torch.Tensor,
) -> torch.Tensor:
    """
    For a line search, produce a new candidate alpha between two given alpha values.
    Initially tries to do linear interpolation to find the zero of a related function
    (such as the error or the gradient of the error).
    If that fails because the value is constant, or if the resulting candidate
    is outside the range

    :param alpha_1:
    :param alpha_2:
    :return:
    """
    result = InterpolateAlpha.apply(alpha_1, alpha_2, value_1, value_2)
    return result
