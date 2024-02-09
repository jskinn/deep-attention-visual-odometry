from typing import Any
import torch


class CosXonXsquaredMinusSinXonXcubed(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        is_zero = torch.eq(x, 0.0)
        cos_x = torch.cos(x)
        sin_x = torch.sin(x)
        x_squared = x.square()
        x_cubed = x * x_squared
        result = cos_x / x_squared - sin_x / x_cubed
        return torch.where(is_zero, (-1.0 / 3.0) * torch.ones_like(x), result)

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
            x_squared = x.square()
            x_fourth = x_squared.square()
            grad = -(x_squared - 3.0) * sin_x - 3 * x * cos_x
            grad = grad / x_fourth
            grad = torch.where(is_zero, torch.zeros_like(x), grad)
            grad = grad * grad_output
            return grad
        return None


def cos_x_on_x_squared_minus_sin_x_on_x_cubed(x: torch.Tensor) -> torch.Tensor:
    return CosXonXsquaredMinusSinXonXcubed.apply(x)
