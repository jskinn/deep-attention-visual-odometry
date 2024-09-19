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
import math

import numpy as np
import pytest
import torch

from deep_attention_visual_odometry.autograd_solvers.bfgs_solver import BFGSSolver

from .reference_functions import (
    square_error,
    log_square_error,
    rosenbrock_function,
    rastrigrin_function,
    beale_function,
    bukin_function_6,
    easom_function,
)


def offset_square_error(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return square_error(x) + 10.0


def cosine_error(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    x = x / norm
    return (1.0 - x[..., 0]) + (1.0 - norm).square()


def x_squared_sine_x_error(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    # This distance function has a lot of local minima
    distance = torch.linalg.vector_norm(x, dim=-1)
    return distance.square() * (distance.sin() + 2.0)


def test_optimizes_simple_function() -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor([1.1, 2.3])
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, square_error)
    final_error = log_square_error(result, torch.tensor(True))
    assert torch.isclose(final_error, torch.tensor(0.0), atol=error_threshold).all()


def test_optimizes_simple_function_with_offset() -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor([1.1, 2.3])
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, offset_square_error)
    final_error = square_error(result, torch.tensor(True))
    assert torch.isclose(final_error, torch.tensor(0.0), atol=error_threshold).all()


def test_optimizes_log_function() -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor([-0.3, 0.8])
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, log_square_error)
    final_error = log_square_error(result, torch.tensor(True))
    assert torch.isclose(final_error, torch.tensor(0.0), atol=error_threshold).all()


def test_optimizes_log_function_from_large_estimate() -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor([-1700.3, 24942.8], dtype=torch.float64)
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, log_square_error)
    final_error = log_square_error(result, torch.tensor(True))
    assert torch.isclose(
        final_error, torch.tensor(0.0, dtype=torch.float64), atol=error_threshold
    ).all()


def test_optimizes_cosine_function() -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor([0.03, -18.8, 23.8, 19.0])
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, cosine_error)
    final_error = cosine_error(result, torch.tensor(True))
    assert torch.isclose(final_error, torch.tensor(0.0), atol=error_threshold).all()


def test_can_get_caught_in_local_minima() -> None:
    error_threshold = 1e-6
    minima_1 = -10.8060458497138
    minima_2 = 14.5496166081312
    initial_guess = torch.tensor(
        [
            [17.8885, 35.7771],
            [minima_1 * math.sqrt(2) / 2, minima_1 * math.sqrt(2) / 2],
            [minima_2 * math.sqrt(2) / 2, -1 * minima_2 * math.sqrt(2) / 2],
        ]
    )
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, x_squared_sine_x_error)
    assert torch.isclose(result, initial_guess, rtol=0.2).all()


def test_can_pass_over_local_minima() -> None:
    error_threshold = 1e-6
    minima_1 = 10.8060458497138 + 2.25
    minima_2 = 14.5496166081312 + 3.0
    initial_guess = torch.tensor(
        [
            [minima_1 * math.sqrt(2) / 2, minima_1 * math.sqrt(2) / 2],
            [minima_2 * math.sqrt(2) / 2, -1 * minima_2 * math.sqrt(2) / 2],
            [-18.025, 6.0083],
        ]
    )
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, x_squared_sine_x_error)
    assert torch.isclose(
        result, torch.zeros_like(result), atol=math.sqrt(error_threshold)
    ).all()


def test_moves_toward_solution_if_iterations_too_low() -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor([27.7, -4.8])
    initial_error = cosine_error(initial_guess, torch.tensor(True))
    solver = BFGSSolver(error_threshold=error_threshold, iterations=3)
    result = solver(initial_guess, cosine_error)
    final_error = cosine_error(result, torch.tensor(True))
    assert not torch.isclose(
        final_error, torch.zeros_like(final_error), atol=error_threshold
    ).all()
    assert torch.less(
        final_error,
        initial_error,
    ).all()


def test_handles_batch_dimensions(fixed_random_seed: int) -> None:
    error_threshold = 1e-6
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(3, 8, 4)))
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, log_square_error)
    final_error = log_square_error(result, torch.ones(3, 8, dtype=torch.bool))
    assert result.shape == initial_guess.shape
    assert torch.isclose(
        final_error, torch.zeros_like(final_error), atol=error_threshold
    ).all()


def test_passes_updating_mask_to_error_function(fixed_random_seed: int) -> None:
    error_threshold = 1e-6
    rng = np.random.default_rng(fixed_random_seed)
    shifts = torch.tensor(rng.normal(0.0, 3.0, size=(3, 8, 2)))
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(3, 8, 2)))
    got_partial_mask = False

    def error_function(x: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        nonlocal got_partial_mask
        assert update_mask.shape == (3, 8)
        assert update_mask.sum() == x.size(0)
        assert update_mask.any()
        got_partial_mask |= update_mask.sum() < 3 * 8
        x = x + shifts[update_mask]
        num_group_1 = update_mask[0, :].sum()
        num_group_2 = update_mask[1, :].sum()
        error_1 = bukin_function_6(x[0:num_group_1, :])
        error_2 = rosenbrock_function(x[num_group_1 : num_group_1 + num_group_2, :])
        error_3 = square_error(x[num_group_1 + num_group_2 :])
        return torch.cat([error_1, error_2, error_3], dim=0)

    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, error_function)
    assert result.shape == initial_guess.shape
    assert got_partial_mask


def test_extra_iterations_only_decrease_error(fixed_random_seed: int) -> None:
    error_threshold = 1e-6
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(5,)))
    prev_error = log_square_error(initial_guess, torch.tensor(True))
    for num_iterations in range(1, 15):
        solver = BFGSSolver(error_threshold=error_threshold, iterations=num_iterations)
        result = solver(initial_guess, log_square_error)
        result_error = log_square_error(result, torch.tensor(True))
        assert torch.less_equal(result_error, prev_error)
        prev_error = result_error


@pytest.mark.parametrize(
    "function,minima,tolerance",
    [
        (square_error, torch.zeros(2, dtype=torch.float64), 1e-6),
        (log_square_error, torch.zeros(2, dtype=torch.float64), 1e-4),
        (rosenbrock_function, torch.ones(2, dtype=torch.float64), 0.02),
        # TODO: These don't quite work
        # (beale_function, torch.tensor([3, 0.5], dtype=torch.float64), 1e-4),
        # (bukin_function_6, torch.tensor([-10.0, 1.0], dtype=torch.float64), 1e-4),
        # (easom_function, torch.tensor([torch.pi, torch.pi], dtype=torch.float64), 1e-4),
        # gets caught in local minima
        # (rastrigrin_function, torch.zeros(2, dtype=torch.float64), 0.001),
    ],
)
def test_optimises_reference_functions(
    fixed_random_seed: int,
    function: Callable[[torch.Tensor], torch.Tensor],
    minima: torch.Tensor,
    tolerance: float,
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    scale = max(float(minima.abs().max()), 1.0)
    initial_guess = torch.tensor(rng.normal(0.0, scale, size=(16, minima.size(-1))))
    solver = BFGSSolver(iterations=2000, error_threshold=1e-8)
    result = solver(initial_guess, function)
    assert torch.isclose(
        result, minima.unsqueeze(0).expand_as(result), atol=tolerance
    ).all()


def test_fit_plane_with_noise(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    num_points = 128
    true_plane = rng.normal(0.0, 1.0, size=(4,))
    true_plane = true_plane / np.linalg.norm(true_plane[0:3], keepdims=True)
    np_points = rng.normal(0.0, 15.0, size=(num_points, 3))
    origin = -1.0 * true_plane[3] * true_plane[0:3]
    np_points = np_points - origin[None, :]
    np_points = (
        np_points - np.dot(np_points, true_plane[0:3])[:, None] * true_plane[None, 0:3]
    )
    np_points = np_points + origin[None, :]
    np_points = np_points + rng.normal(0.0, 0.01, size=(num_points, 3))
    points = torch.tensor(np_points.reshape(1, num_points, 3))

    def error_function(x: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        error = (x[..., 0:3].unsqueeze(-2) * points).sum(dim=-1) + x[..., 3:4]
        return error.square().sum(dim=-1)

    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(4,)))
    solver = BFGSSolver(error_threshold=1e-10, minimum_step=1e-8, iterations=500)
    result = solver(initial_guess, error_function)
    result = result / torch.linalg.vector_norm(result[..., 0:3], dim=-1, keepdim=True)
    assert result.shape == initial_guess.shape
    assert (
        torch.isclose(result, torch.tensor(true_plane), atol=0.1).all()
        or torch.isclose(-1.0 * result, torch.tensor(true_plane), atol=0.1).all()
    )


def test_passes_through_gradients(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(3, 4)), requires_grad=True)
    solver = BFGSSolver(error_threshold=1e-6)
    result = solver(initial_guess, log_square_error)
    assert result.requires_grad is True
    assert result.grad_fn is not None
    loss = result.square().sum()
    loss.backward()
    assert initial_guess.grad is not None
    assert torch.all(torch.greater(torch.abs(initial_guess.grad), 0))


@pytest.mark.parametrize("requires_grad", [True, False])
def test_result_requires_grad_matches_input(requires_grad) -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor([1.1, 2.3], requires_grad=requires_grad)
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, square_error)
    assert result.requires_grad == requires_grad


def test_can_be_compiled(fixed_random_seed) -> None:
    error_threshold = 1e-6
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(
        rng.normal(0.0, 1.0, size=(3, 8, 4)), requires_grad=True
    )
    solver = BFGSSolver(error_threshold=error_threshold)
    complied_solver = torch.compile(solver)
    result = complied_solver(initial_guess, square_error)
    assert torch.isclose(result, torch.zeros_like(result), atol=error_threshold).all()


def test_update_inverse_hessian_matches_bfgs_formula() -> None:
    # The desired formula is:
    # H_{k+1} = (I - (s_k y^T_k) / (s^T_k y_k)) H_k (I - (y_k s^T_k) / (s^T_k y_k)) + (s_k s^T_k) / (s^T_k y_k)
    # Where H_k is the inverse hessian,
    step = torch.tensor([-1.26262069, -0.78272035, 0.98543104], dtype=torch.float64)
    delta_gradient = torch.tensor(
        [0.15339519, -0.28944666, 0.54194925], dtype=torch.float64
    )
    inverse_hessian = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float64,
    )

    curvature = (step * delta_gradient).sum()
    assert curvature > 0.0
    left_matrix = torch.eye(3) - (step[:, None] * delta_gradient[None, :]) / curvature
    right_matrix = torch.eye(3) - (delta_gradient[:, None] * step[None, :]) / curvature
    step_outer_product = step[:, None] * step[None, :] / curvature
    expected_result = left_matrix @ inverse_hessian @ right_matrix + step_outer_product

    result = BFGSSolver.update_inverse_hessian(inverse_hessian, step, delta_gradient)
    assert torch.isclose(expected_result, result).all()


@pytest.mark.parametrize(
    "delta_gradient",
    [
        # Curvature of zero
        torch.tensor([0.0, 0.0, -3.0], dtype=torch.float64),
        # Negative curvature
        torch.tensor([0.0, -1.0, -3.0], dtype=torch.float64),
    ],
)
def test_update_inverse_hessian_does_nothing_when_curvature_is_or_negative(
    delta_gradient: torch.Tensor,
) -> None:
    # Curvature is s^T_k y_k, that is, the dot product of the step and delta_gradient.
    # BFGS requires that this value remains positive to ensure that the inverse hessian
    # remains positive-definite.
    # At this stage, when the curvature is non-positive, we skip the update.
    step = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
    inverse_hessian = torch.tensor(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    result = BFGSSolver.update_inverse_hessian(inverse_hessian, step, delta_gradient)
    assert torch.equal(result, inverse_hessian)
