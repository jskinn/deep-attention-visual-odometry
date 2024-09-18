# Copyright (C) 2024  John Skinner
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
from typing import Any
import torch


class InverseCurvature(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    def forward(step: torch.Tensor, delta_gradient: torch.Tensor) -> torch.Tensor:
        curvature = torch.sum(step * delta_gradient, dim=-1, keepdim=True)
        inv_curvature = 1.0 / curvature
        inv_curvature[curvature <= 0.0] = 0.0
        return inv_curvature

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor], output: torch.Tensor
    ) -> Any:
        ctx.save_for_backward(inputs[0], inputs[1], output)
        ctx.set_materialize_grads(False)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if grad_output is None or (
            not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
        ):
            return None, None
        step, delta_gradient, inv_curvature = ctx.saved_tensors
        grad_step = grad_delta_gradient = None
        grad_output = -1.0 * inv_curvature * inv_curvature * grad_output
        if ctx.needs_input_grad[0]:
            grad_step = delta_gradient * grad_output
        if ctx.needs_input_grad[1]:
            grad_delta_gradient = step * grad_output
        return grad_step, grad_delta_gradient


def inverse_curvature(step: torch.Tensor, delta_gradient: torch.Tensor) -> torch.Tensor:
    """
    In the BFGS update to the inverse hessian, everything is scaled by the inverse of the curvature
    1 / y^T s.
    The curvature _should_ be strictly positive, if our updates satisfy the wolffe conditions,
    however in practice we want to guard against it being zero or negative

    :param step:
    :param delta_gradient:
    :return:
    """
    result = InverseCurvature.apply(step, delta_gradient)
    return result
