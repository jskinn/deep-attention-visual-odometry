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
import numpy as np
import pytest
import torch

from deep_attention_visual_odometry.autograd_solvers.bfgs_solver import BFGSSolver


def square_error(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return x.square().sum(dim=-1)


def log_square_error(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return (x.square().sum(dim=-1) + 1.0).log()


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


@pytest.mark.skip("Gets caught")
def test_optimizes_function_with_many_local_minima() -> None:
    error_threshold = 1e-6
    initial_guess = torch.tensor(
        [
            [-22.2, -15.8],
            [17.8885, 35.7771],
            [-18.025, 6.0083],
        ]
    )
    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, x_squared_sine_x_error)
    final_error = x_squared_sine_x_error(result, torch.tensor(True))
    assert torch.isclose(
        final_error, torch.zeros_like(final_error), atol=error_threshold
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


def test_passes_updating_mask_to_error_fuction(fixed_random_seed: int) -> None:
    error_threshold = 1e-6
    rng = np.random.default_rng(fixed_random_seed)
    target_points = torch.tensor(rng.normal(0.0, 3.0, size=(3, 8, 4)))
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(3, 8, 4)))

    def error_function(x: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        if update_mask.shape != target_points.shape[:-1]:
            update_mask = update_mask.reshape(target_points.shape[:-1])
        targets = target_points[update_mask]
        return torch.linalg.vector_norm(x - targets, dim=-1)

    solver = BFGSSolver(error_threshold=error_threshold)
    result = solver(initial_guess, error_function)
    assert result.shape == initial_guess.shape
    assert torch.isclose(result, target_points, atol=error_threshold).all()


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
