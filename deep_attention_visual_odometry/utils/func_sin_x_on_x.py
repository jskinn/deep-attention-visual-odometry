from typing import Any, Final
import torch


class SinXonX(torch.autograd.Function):
    _taylor_threshold: Final[float] = 0.01

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        is_near_zero = torch.less(x.abs(), SinXonX._taylor_threshold)
        not_near_zero = torch.logical_not(is_near_zero)
        result = torch.empty_like(x)
        # For values near zero, use three terms of the taylor expansion for sin(x) / x.
        # Since |x| will be small, higher powers will be even smaller, so less relevant
        x_squared = x[is_near_zero].square()
        x_fourth = x_squared.square()
        x_sixth = x_fourth * x_squared
        result[is_near_zero] = 1.0 - x_squared / 6.0 + x_fourth / 120 - x_sixth / 5040
        # For more distant values, use the builtin sin function
        result[not_near_zero] = torch.sin(x[not_near_zero]) / x[not_near_zero]
        return result

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor],
        outputs: tuple[torch.Tensor, torch.Tensor],
    ) -> Any:
        ctx.save_for_backward(inputs[0])
        ctx.set_materialize_grads(False)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if grad_output is None:
            return None
        (x,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            return grad_output * x * cos_x_on_x_squared_minus_sin_x_on_x_cubed(x)
        return None


class CosXonXSquaredMinusSinXonXCubed(torch.autograd.Function):
    _taylor_threshold: Final[float] = 0.01

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_near_zero = torch.less(
            x.abs(), CosXonXSquaredMinusSinXonXCubed._taylor_threshold
        )
        not_near_zero = torch.logical_not(is_near_zero)
        result = torch.empty_like(x)
        x_squared = x.square()
        # For values near zero, use a taylor expansion
        x_fourth = x_squared[is_near_zero].square()
        x_sixth = x_fourth * x_squared[is_near_zero]
        result[is_near_zero] = (
            -1.0 / 3.0
            + x_squared[is_near_zero] / 30.0
            - x_fourth / 840
            + x_sixth / 45360
        )
        # For more distant values, use the builtin cos and sin
        cos_x = torch.cos(x[not_near_zero])
        sin_x = torch.sin(x[not_near_zero])
        x_cubed = x[not_near_zero] * x_squared[not_near_zero]
        result[not_near_zero] = cos_x / x_squared[not_near_zero] - sin_x / x_cubed
        # To compute the derivative, we also need 1/x.
        reciprocal = 1.0 / x
        reciprocal[x == 0.0] = 0.0
        return result, reciprocal

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor], output: tuple[torch.Tensor, torch.Tensor]
    ) -> Any:
        ctx.save_for_backward(inputs[0], output[0], output[1])
        ctx.set_materialize_grads(False)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor | None, grad_reciprocal: torch.Tensor | None
    ) -> Any:
        if grad_output is None and grad_reciprocal is None:
            return None
        (x, result, reciprocal) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad = 0.0
            if grad_output is not None:
                sin_term = sin_x_on_x(x)
                grad = -1.0 * grad_output * reciprocal * (sin_term + 3.0 * result)
            if grad_reciprocal is not None:
                grad = grad - grad_reciprocal * reciprocal * reciprocal
            return grad
        return None


def sin_x_on_x(x: torch.Tensor) -> torch.Tensor:
    result = SinXonX.apply(x)
    return result


def cos_x_on_x_squared_minus_sin_x_on_x_cubed(x: torch.Tensor) -> torch.Tensor:
    result, _ = CosXonXSquaredMinusSinXonXCubed.apply(x)
    return result
