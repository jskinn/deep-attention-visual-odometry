from typing import Any, Final
import torch
from .func_sin_x_on_x import sin_x_on_x


class OneMinusCosXonXsquared(torch.autograd.Function):
    _taylor_threshold: Final[float] = 0.05

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_near_zero = torch.less(x.abs(), OneMinusCosXonXsquared._taylor_threshold)
        not_near_zero = torch.logical_not(is_near_zero)
        result = torch.empty_like(x)
        x_squared = x.square()
        # Below the threshold, use a taylor expansion. This should have
        x_fourth = x_squared[is_near_zero].square()
        x_sixth = x_fourth * x_squared[is_near_zero]
        result[is_near_zero] = (
            0.5 - x_squared[is_near_zero] / 24 + x_fourth / 720 - x_sixth / 40320
        )
        # For larger x, use the builtin cos function
        cos_x = torch.cos(x[not_near_zero])
        result[not_near_zero] = (1.0 - cos_x) / x_squared[not_near_zero]
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
                sin_term = sin_x_on_x(x)
                grad = grad_output * reciprocal * (sin_term - 2.0 * result)
            if grad_reciprocal is not None:
                grad = grad - grad_reciprocal * reciprocal * reciprocal
            return grad
        return None


def one_minus_cos_x_on_x_squared(x: torch.Tensor) -> torch.Tensor:
    result, _ = OneMinusCosXonXsquared.apply(x)
    return result
