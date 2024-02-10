from typing import Any
import torch


class ProjectVectorOntoAxis(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(vector: torch.Tensor, axis: torch.Tensor, axis_square_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        is_zero = torch.eq(axis_square_norm, 0.0)
        dot_product = (vector * axis).sum(dim=-1, keepdims=True)
        scale = dot_product / axis_square_norm
        scale = torch.where(is_zero, torch.zeros_like(scale), scale)
        return scale * axis, scale

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], output: tuple[torch.Tensor, torch.Tensor]) -> Any:
        ctx.save_for_backward(inputs[0], inputs[1], inputs[2], output[0], output[1])
        ctx.set_materialize_grads(False)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if grad_output is None:
            return None, None, None
        vector, axis, axis_square_norm, output, scale = ctx.saved_tensors
        projection_grad, scale_grad = grad_output
        grad_vector = grad_axis = grad_norm = None
        is_zero = torch.eq(axis_square_norm, 0.0)
        if ctx.needs_input_grad[0]:
            axis_self_outer = axis.unsqueeze(-2) * vector.unsqueeze(-1)
            grad_vector = axis_self_outer / axis_square_norm
            grad_vector = torch.where(is_zero, torch.zeros_like(grad_vector), grad_vector)
            grad_vector = grad_vector * projection_grad.unsqueeze(-1)
            grad_vector = grad_vector.sum(dim=-2)
        if ctx.needs_input_grad[1]:
            #
            axis_vector_outer = axis.unsqueeze(-2) * vector.unsqueeze(-1)
            axis_output_outer = axis.unsqueeze(-2) * output.unsqueeze(-1)
            grad_axis = axis_vector_outer - 2.0 * axis_output_outer
            grad_axis = grad_axis / axis_square_norm
            grad_axis = torch.where(is_zero.expand_as(grad_axis), torch.zeros_like(grad_axis), grad_axis)
            grad_axis = grad_axis * projection_grad.unsqueeze(-2)
            grad_axis = grad_axis.sum(dim=-1)
            # variables that are matched in the dot product have an additional term
            grad_axis = grad_axis + scale_grad * projection_grad
            # Add the scale component
            grad_scale_axis = (vector - 2.0 * output) / axis_square_norm
            grad_scale_axis = torch.where(is_zero.expand_as(grad_scale_axis), torch.zeros_like(grad_scale_axis), grad_scale_axis)
            grad_axis = grad_axis + grad_scale_axis
        if ctx.needs_input_grad[2]:
            pass
        return grad_vector, grad_axis, grad_norm


def project_vector_onto_axis(vector: torch.Tensor, axis: torch.Tensor, axis_square_norm: torch.Tensor = None) -> torch.Tensor:
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
    result = ProjectVectorOntoAxis.apply(vector, axis, axis_square_norm)
    return result
