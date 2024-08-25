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
import torch

from deep_attention_visual_odometry.autograd_solvers.sgd_solver import SGDSolver


def square_error(x: torch.Tensor):
    return x.square().sum(dim=-1)


def log_square_error(x: torch.Tensor):
    return (x.square().sum(dim=-1) + 1.0).log()


def test_optimizes_simple_function() -> None:
    initial_guess = torch.tensor([1.1, 2.3])
    solver = SGDSolver(iterations=100, learning_rate=1e-1)
    result = solver(initial_guess, square_error)
    assert torch.isclose(result, torch.zeros_like(initial_guess)).all()


def test_optimizes_log_function() -> None:
    initial_guess = torch.tensor([-0.3, 0.8])
    solver = SGDSolver(iterations=100, learning_rate=1e-1)
    result = solver(initial_guess, log_square_error)
    assert torch.isclose(result, torch.zeros_like(initial_guess)).all()


def test_moves_toward_solutiion_if_learning_rate_too_low() -> None:
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


def test_passes_through_gradients(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    initial_guess = torch.tensor(
        rng.normal(0.0, 1.0, size=(3, 4)), requires_grad=True
    )
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
