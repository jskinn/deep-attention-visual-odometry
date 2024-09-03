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
import pytest
from unittest.mock import Mock
import math
import numpy as np
import torch
from torch.nn import Module
import torch.autograd
from deep_attention_visual_odometry.autograd_solvers.line_search.wolfe_conditions import (
    line_search_wolfe_conditions,
)


@pytest.fixture(params=[False, True])
def strong_conditions(request) -> bool:
    return bool(request.param)


def test_returns_step_length_that_reduces_error(strong_conditions: bool) -> None:
    target_point = torch.tensor([2.8 - 1.0, 1.0 + 0.2])
    parameters = torch.tensor([2.0, -3.0], requires_grad=True)
    search_direction = torch.tensor([0.2, 1.0])

    def error_function(x: torch.Tensor, _) -> torch.Tensor:
        return torch.linalg.vector_norm(x - target_point, dim=-1)

    base_error = error_function(parameters, None)
    base_gradient = torch.autograd.grad(base_error, parameters)
    result = line_search_wolfe_conditions(
        parameters,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert result.shape == torch.Size([])
    assert torch.greater(result, 0.0).all()
    error_after_step = error_function(parameters + result * search_direction, None)
    assert torch.less(error_after_step, base_error)


def test_handles_multiple_batch_dimensions(strong_conditions: bool) -> None:
    target_points = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[4.3, -10], [-9.7, 2.2], [2.8, -8.2]],
        ]
    )
    parameters = torch.zeros_like(target_points, requires_grad=True)
    search_direction = torch.tensor(
        [
            [[0.1, 1.0], [10.0, -0.1], [0.1, 0.1]],
            [[1.0, -1.0], [-1.0, -0.1], [2.0, 0.1]],
        ]
    )

    def error_function(
        x: torch.Tensor, batch_mask: torch.Tensor = None
    ) -> torch.Tensor:
        target = target_points[batch_mask] if batch_mask is not None else target_points
        return (torch.linalg.vector_norm(x - target, dim=-1) - 0.5).square()

    base_error = error_function(parameters)
    base_gradient = torch.autograd.grad(base_error.sum(), parameters)

    result = line_search_wolfe_conditions(
        parameters,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert result.shape == (2, 3)
    assert torch.greater(result, 0.0).all()

    error_after_step = error_function(
        parameters + result.unsqueeze(-1) * search_direction
    )
    assert torch.less_equal(error_after_step, base_error).all()


def test_searches_in_desired_direction(
    fixed_random_seed: int, strong_conditions: bool
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    target_points = torch.tensor(rng.normal(0.0, 2.0, size=(3, 4, 2)))
    parameters = torch.tensor(rng.normal(0.0, 1.0, size=(3, 4, 2)), requires_grad=True)
    search_skew = torch.tensor(rng.uniform(-0.2, 0.2, size=(3, 4, 1)))

    def error_function(
        x: torch.Tensor, batch_mask: torch.Tensor = None
    ) -> torch.Tensor:
        target = target_points[batch_mask] if batch_mask is not None else target_points
        return (
            (torch.linalg.vector_norm(x - target, dim=-1) - 0.5).square() + 1.0
        ).log()

    base_error = error_function(parameters, None)
    base_gradient = torch.autograd.grad(base_error.sum(), parameters)

    # Rotate the search direction away from the negative gradient by a little bit
    search_direction = torch.cat(
        [
            -1.0 * search_skew.cos() * base_gradient[0][:, :, 0:1]
            + search_skew.sin() * base_gradient[0][:, :, 1:2],
            -1.0 * search_skew.sin() * base_gradient[0][:, :, 0:1]
            - search_skew.cos() * base_gradient[0][:, :, 1:2],
        ],
        dim=-1,
    )

    mock_error_function = Mock(spec_set=["__call__"], side_effect=error_function)

    _ = line_search_wolfe_conditions(
        parameters,
        search_direction,
        base_error,
        base_gradient[0],
        mock_error_function,
        strong=strong_conditions,
    )

    assert mock_error_function.called
    for call_args in mock_error_function.call_args_list:
        mask = call_args.args[1]
        step = call_args.args[0] - parameters[mask]
        ratio = step / search_direction[mask]
        assert torch.isclose(
            ratio[..., 0], ratio[..., 1]
        ).all(), f"{step} is not a multiple of {search_direction[mask]}"


def test_chosen_point_satisfies_wolffe_conditions(
    fixed_random_seed: int, strong_conditions: bool
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    target_points = torch.tensor(rng.normal(0.0, 2.0, size=(3, 4, 2)))
    parameters = torch.tensor(rng.normal(0.0, 1.0, size=(3, 4, 2)), requires_grad=True)
    search_skew = torch.tensor(rng.uniform(-0.2, 0.2, size=(3, 4, 1)))
    c1 = 0.1
    c2 = 0.6

    def error_function(
        x: torch.Tensor, batch_mask: torch.Tensor = None
    ) -> torch.Tensor:
        target = target_points[batch_mask] if batch_mask is not None else target_points
        return (
            (torch.linalg.vector_norm(x - target, dim=-1) - 0.5).square() + 1.0
        ).log()

    base_error = error_function(parameters, None)
    base_gradient = torch.autograd.grad(base_error.sum(), parameters)

    # Rotate the search direction away from the negative gradient by a little bit
    search_direction = torch.cat(
        [
            -1.0 * search_skew.cos() * base_gradient[0][:, :, 0:1]
            + search_skew.sin() * base_gradient[0][:, :, 1:2],
            -1.0 * search_skew.sin() * base_gradient[0][:, :, 0:1]
            - search_skew.cos() * base_gradient[0][:, :, 1:2],
        ],
        dim=-1,
    )
    base_line_gradient = (search_direction * base_gradient[0]).sum(dim=-1)

    result = line_search_wolfe_conditions(
        parameters,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        sufficient_decrease=c1,
        curvature=c2,
        strong=strong_conditions,
    )

    result_parameters = parameters + result.unsqueeze(-1) * search_direction
    result_error = error_function(result_parameters, None)
    result_gradient = torch.autograd.grad(result_error.sum(), result_parameters)
    result_line_gradient = (search_direction * result_gradient[0]).sum(dim=-1)

    assert torch.less_equal(
        result_error, base_error + c1 * result * base_line_gradient
    ).all()
    if strong_conditions:
        assert torch.less_equal(
            result_line_gradient.abs(), c2 * base_line_gradient.abs()
        ).all()
    else:
        assert torch.less_equal(
            -1.0 * result_line_gradient, -1.0 * c2 * base_line_gradient
        ).all()


def test_scales_down_too_large_step_size(strong_conditions: bool = True) -> None:
    target_point = torch.tensor([1.0, 1.0])
    initial_guess = torch.zeros_like(target_point, requires_grad=True)
    search_direction = torch.tensor([10.0, 0.0])

    def error_function(x: torch.Tensor, _) -> torch.Tensor:
        return torch.linalg.vector_norm(x - target_point, dim=-1)

    base_error = error_function(initial_guess, True)
    base_gradient = torch.autograd.grad(base_error.sum(), initial_guess)

    result = line_search_wolfe_conditions(
        initial_guess,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert result < 1.0


def test_widens_for_too_small_search_direction(strong_conditions: bool = True) -> None:
    target_points = torch.tensor([10.0, 10.0])
    initial_guess = torch.zeros_like(target_points, requires_grad=True)
    search_direction = torch.tensor([0.2, 0.1])

    def error_function(x: torch.Tensor, _) -> torch.Tensor:
        return torch.linalg.vector_norm(x - target_points, dim=-1)

    base_error = error_function(initial_guess, True)
    base_gradient = torch.autograd.grad(base_error.sum(), initial_guess)

    result = line_search_wolfe_conditions(
        initial_guess,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert result > 1.0


def test_finds_easy_solution(strong_conditions: bool) -> None:
    target_points = torch.tensor([0.25, 0.25])
    initial_guess = torch.zeros_like(target_points, requires_grad=True)
    search_direction = torch.tensor([1.0, 1.0])

    def error_function(x: torch.Tensor, _) -> torch.Tensor:
        return torch.linalg.vector_norm(x - target_points, dim=-1)

    base_error = error_function(initial_guess, True)
    base_gradient = torch.autograd.grad(base_error.sum(), initial_guess)

    result = line_search_wolfe_conditions(
        initial_guess,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert result == 0.25


def test_produces_small_change_when_search_direction_is_wrong(
    strong_conditions: bool,
) -> None:
    target_points = torch.tensor([-9.7, 2.2])
    initial_guess = torch.zeros_like(target_points, requires_grad=True)
    search_direction = torch.tensor([1.0, 0.0])

    def error_function(x: torch.Tensor, _) -> torch.Tensor:
        return torch.linalg.vector_norm(x - target_points, dim=-1)

    base_error = error_function(initial_guess, True)
    base_gradient = torch.autograd.grad(base_error.sum(), initial_guess)

    result = line_search_wolfe_conditions(
        initial_guess,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert torch.isclose(result, torch.tensor(0.0))


def test_produces_a_change_even_if_in_a_local_minima(strong_conditions: bool):
    # Trig functions give repeating local minima
    # The minima here is taken from wolfram alpha
    initial_guess = torch.tensor([-13.9922246512961], requires_grad=True)

    def error_function(x: torch.Tensor, _) -> torch.Tensor:
        distance = torch.linalg.vector_norm(x, dim=-1)
        return distance.square() * (distance.sin() + 2.0)

    base_error = error_function(initial_guess, True)
    base_gradient = torch.autograd.grad(base_error.sum(), initial_guess)
    search_direction = -1.0 * base_gradient[0]

    result = line_search_wolfe_conditions(
        initial_guess,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert torch.greater(result, torch.tensor(0.01))


def test_does_not_propagate_gradients(strong_conditions: bool) -> None:
    target_points = torch.tensor([0.25, 0.25], requires_grad=True)
    initial_guess = torch.zeros_like(target_points, requires_grad=True)
    search_direction = torch.tensor([1.0, -0.1], requires_grad=True)

    def error_function(x: torch.Tensor, _) -> torch.Tensor:
        return torch.linalg.vector_norm(x - target_points, dim=-1)

    base_error = error_function(initial_guess, True)
    base_gradient = torch.autograd.grad(base_error.sum(), initial_guess)

    result = line_search_wolfe_conditions(
        initial_guess,
        search_direction,
        base_error,
        base_gradient[0],
        error_function,
        strong=strong_conditions,
    )

    assert not result.requires_grad


class DemoWolfeConditionsModule(Module):
    def __init__(self, target: torch.Tensor, strong: bool):
        super().__init__()
        self.target = target
        self.strong = bool(strong)

    def forward(self, x: torch.Tensor, search_direction: torch.Tensor):
        base_error = self.error(x, None)
        base_gradient = torch.autograd.grad(base_error.sum(), x)
        alpha = line_search_wolfe_conditions(
            x,
            search_direction,
            base_error,
            base_gradient[0],
            self.error,
            strong=self.strong,
        )
        return x + alpha.unsqueeze(-1) * search_direction

    def error(self, x: torch.Tensor, batch_mask: torch.Tensor | None) -> torch.Tensor:
        target = self.target[batch_mask] if batch_mask is not None else self.target
        return ((x - target).square().sum(dim=-1) + 1.0).log()


def test_can_be_compiled(fixed_random_seed: int, strong_conditions: bool) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    target_points = torch.tensor(rng.normal(0.0, 3.0, size=(3, 4, 2)))
    parameters = torch.tensor(rng.normal(0.0, 1.0, size=(3, 4, 2)), requires_grad=True)
    search_skew = torch.tensor(rng.uniform(-0.2, 0.2, size=(3, 4, 1)))

    # Rotate the search direction away from the true direction by a little
    search_direction = target_points - parameters
    search_direction = torch.cat(
        [
            search_skew.cos() * search_direction[:, :, 0:1]
            - search_skew.sin() * search_direction[:, :, 1:2],
            search_skew.sin() * search_direction[:, :, 0:1]
            + search_skew.cos() * search_direction[:, :, 1:2],
        ],
        dim=-1,
    )
    subject = DemoWolfeConditionsModule(target_points, strong_conditions)

    compiled_subject = torch.compile(
        subject
    )  # TODO: fullgraph=True after a pytorch update
    result = compiled_subject(parameters, search_direction)

    assert torch.less(
        torch.linalg.vector_norm(result - target_points, dim=-1),
        torch.linalg.vector_norm(parameters - target_points, dim=-1),
    ).all()
