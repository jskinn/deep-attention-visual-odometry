from typing import Any
import torch


class ProjectVectorOntoAxis(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(
        vector: torch.Tensor, axis: torch.Tensor, axis_square_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_zero = torch.eq(axis_square_norm, 0.0)
        dot_product = (vector * axis).sum(dim=-1, keepdims=True)
        reciprocal = 1 / axis_square_norm
        reciprocal[is_zero] = 0.0
        scale = dot_product / axis_square_norm
        scale[is_zero] = 0.0
        return scale, reciprocal

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> Any:
        ctx.save_for_backward(inputs[0], inputs[1], output[0], output[1])
        ctx.set_materialize_grads(False)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(
        ctx: Any, scale_grad: torch.Tensor, reciprocal_grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if scale_grad is None and reciprocal_grad is None:
            return None, None, None
        vector, axis, scale, reciprocal = ctx.saved_tensors
        grad_vector = grad_axis = grad_norm = None
        if ctx.needs_input_grad[0] and scale_grad is not None:
            # Gradients for the vector is simply
            # a / theta^2, b / theta^2, c / theta^2,
            # since they don't appear in the denominator
            grad_vector = scale_grad * reciprocal * axis
        if ctx.needs_input_grad[1] and scale_grad is not None:
            # Same thing for the axis, the relationship between the axis and the reciprocal
            # is passed through the length gradient instead.
            grad_axis = scale_grad * reciprocal * vector
        if ctx.needs_input_grad[2]:
            grad_norm = 0.0
            if scale_grad is not None:
                # The square norm is just the denominator in all the maths\
                # so the derivatives just divide by it again and multiply by -1
                grad_norm = -1.0 * scale_grad * scale * reciprocal
            if reciprocal_grad is not None:
                grad_norm = grad_norm - reciprocal_grad * reciprocal * reciprocal
        return grad_vector, grad_axis, grad_norm


def project_vector_onto_axis(
    vector: torch.Tensor, axis: torch.Tensor, axis_square_norm: torch.Tensor = None
) -> torch.Tensor:
    """
    Project a vector onto an axis.
    The vector dimension is assumed to be the final dimension of both tensors.
    Both tensors should have the same shape.
    Neither the vector nor axis should be normalised,
    optionally takes the square norm of the axis as an argument if it has already been computed.
    """
    if axis_square_norm is None:
        axis_square_norm = axis.square().sum(dim=-1, keepdims=True)
    if axis_square_norm.ndim < axis.ndim:
        axis_square_norm = axis_square_norm.unsqueeze(-1)
    result, _ = ProjectVectorOntoAxis.apply(vector, axis, axis_square_norm)
    return result
