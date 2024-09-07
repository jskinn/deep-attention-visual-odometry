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
import pytest
import numpy as np
import torch

from deep_attention_visual_odometry.autograd_solvers.sgd_solver import SGDSolver

from .reference_functions import (
    square_error,
    log_square_error,
    rosenbrock_function,
    rastrigrin_function,
    # beale_function,
    # bukin_function_6,
    # easom_function,
)


def test_optimizes_simple_function() -> None:
    initial_guess = torch.tensor([1.1, 2.3])
    solver = SGDSolver(iterations=100, learning_rate=1e-1)
    result = solver(initial_guess, square_error)
    assert torch.isclose(result, torch.zeros_like(initial_guess)).all()


def test_moves_toward_solution_if_learning_rate_too_low() -> None:
    initial_guess = torch.tensor([-0.3, 0.8])
    solver = SGDSolver(iterations=100, learning_rate=1e-3)
    result = solver(initial_guess, log_square_error)
    assert not torch.isclose(result, torch.zeros_like(initial_guess)).all()
    assert torch.less(
        torch.linalg.vector_norm(result),
        torch.linalg.vector_norm(initial_guess),
    ).all()


def test_moves_toward_solutiion_if_iterations_too_low() -> None:
    initial_guess = torch.tensor([0.7, -0.8])
    solver = SGDSolver(iterations=10, learning_rate=1e-1)
    result = solver(initial_guess, log_square_error)
    assert not torch.isclose(result, torch.zeros_like(initial_guess)).all()
    assert torch.less(
        torch.linalg.vector_norm(result),
        torch.linalg.vector_norm(initial_guess),
    ).all()


def test_handles_batch_dimensions(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(3, 8, 4)))
    solver = SGDSolver(iterations=100, learning_rate=1e-1)
    result = solver(initial_guess, log_square_error)
    assert result.shape == initial_guess.shape
    assert torch.isclose(result, torch.zeros_like(initial_guess), atol=1e-6).all()


@pytest.mark.parametrize(
    "function,minima,iterations,learning_rate,tolerance",
    [
        (square_error, torch.zeros(2, dtype=torch.float64), 100, 0.1, 1e-6),
        (log_square_error, torch.zeros(2, dtype=torch.float64), 100, 0.1, 1e-6),
        (rosenbrock_function, torch.ones(2, dtype=torch.float64), 24000, 5e-4, 0.01),
        # These functions don't really work:
        # (beale_function, torch.tensor([3, 0.5], dtype=torch.float64), 10000, 5e-5),
        # (bukin_function_6, torch.tensor([-10.0, 1.0], dtype=torch.float64), 10000, 1e-3),
        # (easom_function, torch.tensor([torch.pi, torch.pi], dtype=torch.float64), 10000, 1e-4),
    ],
)
def test_optimises_reference_functions(
    fixed_random_seed: int,
    function: Callable[[torch.Tensor], torch.Tensor],
    minima: torch.Tensor,
    iterations: int,
    learning_rate: float,
        tolerance: float,
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    scale = max(float(minima.abs().max()), 1.0)
    initial_guess = torch.tensor(rng.normal(0.0, scale, size=(16, minima.size(-1))))
    solver = SGDSolver(iterations=iterations, learning_rate=learning_rate)
    result = solver(initial_guess, function)
    assert torch.isclose(result, minima.unsqueeze(0).expand_as(result), atol=tolerance).all()


def test_gets_caught_in_local_minima_of_rastrigrin_function(
    fixed_random_seed: int,
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(rng.normal(0.0, 3.0, size=(16, 2)))
    solver = SGDSolver(iterations=10000, learning_rate=1e-3)
    result = solver(initial_guess, rastrigrin_function)
    assert not torch.isclose(result, torch.zeros_like(result), atol=0.1).all()
    assert not torch.isclose(result, initial_guess, atol=0.001).any()
    # Local minima are around the integers, so they should all be close to an integer index
    assert torch.isclose(result, torch.round(result), atol=0.05).all()


def test_passes_through_gradients(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(3, 4)), requires_grad=True)
    solver = SGDSolver(iterations=100, learning_rate=1e-3)
    result = solver(initial_guess, log_square_error)
    assert result.requires_grad is True
    assert result.grad_fn is not None
    loss = result.square().sum()
    loss.backward()
    assert initial_guess.grad is not None
    assert torch.all(torch.greater(torch.abs(initial_guess.grad), 0))


def test_can_be_compiled() -> None:
    initial_guess = torch.tensor([1.1, 2.3], requires_grad=True)
    solver = SGDSolver(iterations=100, learning_rate=1e-1)
    complied_solver = torch.compile(solver)
    result = complied_solver(initial_guess, square_error)
    assert torch.isclose(result, torch.zeros_like(initial_guess)).all()
