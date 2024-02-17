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
        result = dot_product * reciprocal * axis
        return result, reciprocal

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
        ctx: Any, result_grad: torch.Tensor, reciprocal_grad: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if result_grad is None and reciprocal_grad is None:
            return None, None, None
        vector, axis, result, reciprocal = ctx.saved_tensors
        grad_vector = grad_axis = grad_norm = None
        if ctx.needs_input_grad[0] and result_grad is not None:
            # Gradients for the vector are the outer pr
            # a^2 / theta^2, ab / theta^2, ac / theta^2 for x
            # since they don't appear in the denominator
            grad_vector = axis.unsqueeze(-2) * axis.unsqueeze(-1)
            grad_vector = reciprocal.unsqueeze(-1) * grad_vector
            grad_vector = (grad_vector * result_grad.unsqueeze(-2)).sum(dim=-1)
        if ctx.needs_input_grad[1] and result_grad is not None:
            # For the axis, it is the outer product of the vector and the axis
            dot_product = (vector * axis).sum(dim=-1, keepdims=True)
            grad_axis = axis.unsqueeze(-2) * vector.unsqueeze(-1)
            grad_axis = (grad_axis * result_grad.unsqueeze(-2)).sum(dim=-1)
            grad_axis = reciprocal * (grad_axis + dot_product * result_grad)
        if ctx.needs_input_grad[2]:
            grad_norm = 0.0
            if result_grad is not None:
                # The square norm is just the denominator in all the maths\
                # so the derivatives just divide by it again and multiply by -1
                grad_norm = -1.0 * result_grad * result * reciprocal
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
