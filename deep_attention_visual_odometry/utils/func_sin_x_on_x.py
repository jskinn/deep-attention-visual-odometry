from typing import Any
import torch


class SinXonX(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        is_zero = torch.eq(x, 0.0)
        sin_x = torch.sin(x)
        return torch.where(is_zero, torch.ones_like(x), sin_x / x)

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
            grad = sin_x / x
            grad = cos_x - grad
            grad = grad / x
            grad = torch.where(is_zero, torch.zeros_like(x), grad)
            grad = grad * grad_output
            return grad
        return None


def sin_x_on_x(x: torch.Tensor) -> torch.Tensor:
    return SinXonX.apply(x)
