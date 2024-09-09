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
from typing import Callable
import torch
import torch.autograd
from torch.nn import Module

from deep_attention_visual_odometry.utils import inverse_curvature
from .line_search import line_search_wolfe_conditions


class BFGSSolver(Module):
    """
    Use the Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS) to iteratively optimise an underlying function.
    Relies on a separate line search algorithm that must satisfy the wolfe conditions.

    Why BFGS and not levenberg-marquard?
    Due to optimising the 3D positions of the points, the number of parameters and
    the size of the Jacobian matrix can get very large.
    The Levenberg-Marquardt algorithm requires either computing and inverting J^T J or solving the normal equations for
    Jp = r, both of which (according to "The Levenberg-Marquardt algorithm: implementation and theory" by Jorge More)
    are numerically unstable.
    BFGS allows us to avoid inverting the Jacobian, which is an O(n^3) operation.
    """

    def __init__(
        self,
        sufficient_decrease: float = 1e-4,
        curvature: float = 0.9,
        error_threshold: float = 1e-4,
        iterations: int = 1000,
        minimum_step: float = 1e-8
    ):
        super().__init__()
        self.sufficient_decrease = float(sufficient_decrease)
        self.curvature = float(curvature)
        self.error_threshold = float(error_threshold)
        self.iterations = int(iterations)
        self.minimum_step = float(minimum_step)

    def forward(
        self,
        parameters: torch.Tensor,
        error_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        if parameters.requires_grad:
            create_graph = True
        else:
            create_graph = False
            parameters = parameters.requires_grad_()
        batch_dimensions = parameters.shape[:-1]
        parameter_dim = parameters.size(-1)
        updating = torch.ones(
            batch_dimensions, dtype=torch.bool, device=parameters.device
        )
        step = torch.zeros_like(parameters)
        error = torch.empty(
            batch_dimensions, dtype=parameters.dtype, device=parameters.device
        )
        gradient = torch.empty_like(parameters)
        prev_gradient: torch.Tensor
        inverse_hessian = torch.zeros(
            batch_dimensions + (parameter_dim, parameter_dim),
            dtype=parameters.dtype,
            device=parameters.device,
        )
        inverse_hessian[..., range(parameter_dim), range(parameter_dim)] = 1.0
        for step_idx in range(self.iterations):
            prev_gradient = gradient

            # Compute the error and gradient for the updating parameters
            updating_parameters = parameters[updating]
            updating_error = error_function(updating_parameters, updating)
            updating_gradient = torch.autograd.grad(
                updating_error.sum(), updating_parameters, create_graph=create_graph
            )
            error = error.masked_scatter(updating, updating_error)
            gradient = gradient.masked_scatter(
                updating.unsqueeze(-1).expand_as(gradient), updating_gradient[0]
            )

            # Stop updating when the error falls below a threshold
            # This synchronises the GPU.
            updating = updating & torch.greater(error.detach(), self.error_threshold)
            if not torch.any(updating):
                break

            # For the still updating elements,
            # update the estimate of the inverse hessian and find a new step
            updating_parameters = parameters[updating]
            updating_error = error[updating]
            updating_gradient = gradient[updating]
            if step_idx == 0:
                # Following the heuristic discussed on page 142 of Numerical Optimization
                # For the very first step we don't use the inverse hessian (effectively, it is I)
                search_direction = -1.0 * gradient[updating]
            else:
                delta_gradient = updating_gradient - prev_gradient[updating]
                updating_inverse_hessian = inverse_hessian[updating]
                if step_idx == 1:
                    # After the first step has been computed, but before the first update,
                    # We re-scale H_0 by a constant factor determined by equation 6.20
                    inverse_hessian_scale = self.scale_initial_inverse_hessian(
                        step=step[updating], delta_gradient=delta_gradient
                    )
                    updating_inverse_hessian = (
                        inverse_hessian_scale.unsqueeze(-1) * updating_inverse_hessian
                    )
                updating_inverse_hessian = self.update_inverse_hessian(
                    inverse_hessian=updating_inverse_hessian,
                    step=step[updating],
                    delta_gradient=delta_gradient,
                )
                search_direction = -1.0 * torch.matmul(
                    updating_inverse_hessian, updating_gradient.unsqueeze(-1)
                )
                search_direction = search_direction.squeeze(-1)
                inverse_hessian = inverse_hessian.masked_scatter(
                    updating.unsqueeze(-1).unsqueeze(-1).expand_as(inverse_hessian),
                    updating_inverse_hessian,
                )
            step_size = line_search_wolfe_conditions(
                parameters=updating_parameters,
                search_direction=search_direction,
                base_error=updating_error,
                base_gradient=updating_gradient,
                error_function=error_function,
                sufficient_decrease=self.sufficient_decrease,
                curvature=self.curvature,
                strong=True,
            )
            updating_step = step_size.unsqueeze(-1) * search_direction
            updating_parameters = updating_parameters + updating_step
            step = step.masked_scatter(
                updating.unsqueeze(-1).expand_as(step), updating_step
            )
            parameters = parameters.masked_scatter(
                updating.unsqueeze(-1).expand_as(parameters), updating_parameters
            )

            # Stop updating if the step distance is basically zero
            # This synchronises the GPU.
            updating = updating & torch.greater(torch.linalg.vector_norm(step, dim=-1), self.minimum_step)
            if not torch.any(updating):
                break
        if not create_graph:
            parameters = parameters.detach()
        return parameters

    @staticmethod
    def scale_initial_inverse_hessian(
        step: torch.Tensor, delta_gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Choose an initial value for the inverse hessian approximation.
        H_0 = \frac{y^T s}{y^T y} I
        This is equation 6.20 from "Numerical Optimisation" by Nocedal and Wright 2009.
        :param step: The first chosen step, satisfying the wolffe conditions.
        :param delta_gradient: The change in gradient between the current and next iterate points.
        :return: An initial guess for the inverse hessian H_0, to be updated to H_1.
        """
        denominator = delta_gradient.square().sum(dim=-1, keepdims=True).clamp(min=1e-5)
        scale = (step * delta_gradient).sum(dim=-1, keepdims=True)
        scale = scale / denominator
        scale = scale.clip(min=1e-4)
        return scale

    @staticmethod
    def update_inverse_hessian(
        inverse_hessian: torch.Tensor, step: torch.Tensor, delta_gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the BFGS update to the inverse hessian:
        H_{+} = (I - \frac{s y^T}{y^T s})H(I - frac{y s^T}{y^T s^T}) + frac{s s^T}{y^T s}
        (from equation 6.17, "Numerical Optimisation", Nocedal and Wright 2009)

        :param inverse_hessian: The current inverse hessian estimate H
        :param step: The change in parameters, x_{t+1} - x_{t}
        :param delta_gradient: The change in gradient f(x_{t+1}) - f(x_{t})
        :return: The updated inverse hessian estimate, H_{+}
        """
        # = (H - \frac{1}{y^T s} s y^T H)(I - frac{y s^T}{y^T s^T}) + frac{s s^T}{y^T s}
        # = H - frac{1}{y^T s} H y s^T - \frac{1}{y^T s} s y^T H +
        #   \frac{1}{(y^T s)^2} s y^T H y s^T + frac{s s^T}{y^T s}
        # = H - frac{1}{y^T s} (H y s^T + s y^T H) + \frac{1}{y^T s} (\frac{y^T H y}{y^T s} + 1.0) s s^T
        # y^T s (curvature):
        # Curvature (this product) _should_ be strictly positive.
        # If it isn't, set this scale factor to 0, so that the update is skipped
        # That is all being handled by an autograd function, to avoid nan gradients.
        inv_curvature = inverse_curvature(step, delta_gradient)
        # y^T H y / (y^T s):
        # For numerical stability, we assume |y^T H| < |y| and |y / (y^T s)| < |y|
        # So we calculate those ratios first before multiplying
        # To avoid an intermediate scale ~|y|^2
        gradient_postmultiply_hessian = torch.matmul(
            delta_gradient.unsqueeze(-2), inverse_hessian
        )
        gradient_on_curvature = delta_gradient * inv_curvature
        gradient_inner_product = (
            gradient_postmultiply_hessian * gradient_on_curvature.unsqueeze(-2)
        ).sum(dim=-1)
        # s s^T / (y^T s):
        # Again, to avoid a matrix scaled ~|s^2|, we scale first by 1/|y^T s|
        step_on_curvature = step * inv_curvature
        step_outer_product = torch.matmul(
            step_on_curvature.unsqueeze(-1), step.unsqueeze(-2)
        )
        # \frac{1}{y^T s} (\frac{y^T H y}{y^T s} + 1.0) s s^T
        step_outer_product = step_outer_product * (
            1.0 + gradient_inner_product.unsqueeze(-1)
        )

        # \frac{1}{y^T s} s y^T H = \frac{s}{y^T s} (y^T H)
        step_gradient_product = torch.matmul(
            step_on_curvature.unsqueeze(-1), gradient_postmultiply_hessian
        )
        # frac{1}{y^T s} H y s^T = (H y) frac{s^T}{y^T s}
        gradient_premultiply_hessian = torch.matmul(
            inverse_hessian, delta_gradient.unsqueeze(-1)
        )
        gradient_step_product = torch.matmul(
            gradient_premultiply_hessian, step_on_curvature.unsqueeze(-2)
        )

        return (
            inverse_hessian
            + step_outer_product
            - step_gradient_product
            - gradient_step_product
        )
