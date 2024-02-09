from typing import Any
import torch


class OneMinusCosXonXsquared(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        is_zero = torch.eq(x, 0.0)
        cos_x = torch.cos(x)
        result = (1.0 - cos_x) / x.square()
        return torch.where(is_zero, 0.5 * torch.ones_like(x), result)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor], output) -> Any:
        ctx.save_for_backward(inputs[0])
        ctx.set_materialize_grads(False)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if grad_output is None:
            return None
        (x,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            is_zero = torch.eq(x, 0.0)
            sin_x = torch.sin(x)
            cos_x = torch.cos(x)
            x_cubed = x * x.square()
            grad = x * sin_x + 2 * cos_x - 2
            grad = grad / x_cubed
            grad = torch.where(is_zero, torch.zeros_like(x), grad)
            grad = grad * grad_output
            return grad
        return None


def one_minus_cos_x_on_x_squared(x: torch.Tensor) -> torch.Tensor:
    return OneMinusCosXonXsquared.apply(x)
