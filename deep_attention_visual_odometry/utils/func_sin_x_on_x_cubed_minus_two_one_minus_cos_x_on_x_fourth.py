from typing import Any, Final
import torch
from .func_sin_x_on_x import cos_x_on_x_squared_minus_sin_x_on_x_cubed


class SinXonXCubedMinusTwoOneMinusCosXonXFourth(torch.autograd.Function):
    _taylor_threshold: Final[float] = 0.25

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_near_zero = torch.less(
            x.abs(), SinXonXCubedMinusTwoOneMinusCosXonXFourth._taylor_threshold
        )
        not_near_zero = torch.logical_not(is_near_zero)
        result = torch.empty_like(x)
        x_squared = x.square()
        x_fourth = x_squared.square()
        # Below the threshold, use a taylor expansion. This should have
        x_sixth = x_fourth[is_near_zero] * x_squared[is_near_zero]
        result[is_near_zero] = (
            -1.0 / 12.0
            + x_squared[is_near_zero] / 180.0
            - x_fourth[is_near_zero] / 6720.0
            + x_sixth / 362880.0
        )
        # For larger x, use the builtin sin and cos functions
        sin_x = torch.sin(x[not_near_zero])
        cos_x = torch.cos(x[not_near_zero])
        x_cubed = x[not_near_zero] * x_squared[not_near_zero]
        result[not_near_zero] = (
            sin_x / x_cubed - 2.0 * (1.0 - cos_x) / x_fourth[not_near_zero]
        )
        # To calculate the derivative, we need 1/x
        reciprocal = 1 / x
        reciprocal[x == 0] = 0.0
        return result, reciprocal

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor], output) -> Any:
        ctx.save_for_backward(inputs[0], output[0], output[1])
        ctx.set_materialize_grads(False)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, grad_reciprocal: torch.Tensor
    ) -> Any:
        if grad_output is None:
            return None
        (x, result, reciprocal) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad = 0.0
            if grad_output is not None:
                sin_and_cos_term = cos_x_on_x_squared_minus_sin_x_on_x_cubed(x)
                grad = grad_output * reciprocal * (sin_and_cos_term - 4.0 * result)
            if grad_reciprocal is not None:
                grad = grad - grad_reciprocal * reciprocal * reciprocal
            return grad
        return None


def sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth(
    x: torch.Tensor,
) -> torch.Tensor:
    result, _ = SinXonXCubedMinusTwoOneMinusCosXonXFourth.apply(x)
    return result
