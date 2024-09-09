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
import math

import pytest
import numpy as np
import torch
from sympy import conjugate
from torch.nn import Module

from deep_attention_visual_odometry.autograd_solvers import SGDSolver, BFGSSolver
from deep_attention_visual_odometry.geometry.axis_angle_rotation import (
    rotate_vector_axis_angle,
)


@pytest.mark.parametrize(
    "vector,axis,angle,expected_result",
    [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), math.pi / 2, (0.0, 0.0, -1.0)),
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            math.pi / 4,
            (0.0, -math.sqrt(2) / 2, math.sqrt(2) / 2),
        ),
        (
            (1.1, 0.233, -3.3),
            (0.833, -0.22, -0.19),
            3 * math.pi / 8,
            (2.2132368, 2.48854705, -1.03102158),
        ),
    ],
)
def test_rotate_vector(
    vector: tuple[
        float,
        float,
        float,
    ],
    axis: tuple[float, float, float],
    angle: float,
    expected_result: tuple[float, float, float],
) -> None:
    vector = torch.tensor(vector)
    axis = torch.tensor(axis)
    expected_result = torch.tensor(expected_result)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1)
    result = rotate_vector_axis_angle(vector, axis * angle)
    assert torch.isclose(result, expected_result, atol=1e-7).all()


def test_rotation_by_zeros_is_unchanged():
    vector = torch.tensor([2.352, 6.33, 28.3338])
    result = rotate_vector_axis_angle(vector, torch.zeros(3))
    assert torch.equal(result, vector)


def test_handles_batch():
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.1, 0.233, -3.3],
        ]
    )
    axes = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.833, -0.22, -0.19],
        ]
    )
    axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    angles = torch.tensor([math.pi / 2, math.pi / 4, 3 * math.pi / 8]).unsqueeze(-1)
    axis_angles = angles * axes
    expected_results = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, -math.sqrt(2) / 2, math.sqrt(2) / 2],
            [2.2132368, 2.48854705, -1.03102158],
        ]
    )
    result = rotate_vector_axis_angle(vectors, axis_angles)
    assert result.shape == expected_results.shape
    assert torch.isclose(result, expected_results, atol=1e-7).all()


def test_handles_batch_and_broadcasts() -> None:
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.1, 0.233, -3.3],
        ]
    ).reshape(3, 1, 3)
    axes = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.833, -0.22, -0.19],
        ]
    ).reshape(1, 3, 3)
    axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    angles = torch.tensor([math.pi / 2, math.pi / 4, 3 * math.pi / 8]).unsqueeze(-1)
    axis_angles = angles * axes
    expected_results = torch.tensor(
        [
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.93298563, -0.34430013, 0.10485819],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, -math.sqrt(2) / 2, math.sqrt(2) / 2],
                [-0.35589641, -0.83914192, 0.41131324],
            ],
            [
                [-3.3, 0.233, -1.1],
                [1.1, 2.49820826, -2.1686965],
                [2.2132368, 2.48854705, -1.03102158],
            ],
        ],
    )
    result = rotate_vector_axis_angle(vectors, axis_angles)
    assert result.shape == expected_results.shape
    assert torch.isclose(result, expected_results, atol=1e-7).all()


def test_rotation_preserves_length(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(rng.normal(0.0, 1.0, size=(16, 1, 3)))
    axis_angles = torch.tensor(rng.normal(0.0, 1.0, size=(1, 16, 3)))
    vector_lengths = torch.linalg.vector_norm(vectors, dim=-1)
    result = rotate_vector_axis_angle(vectors, axis_angles)
    result_lengths = torch.linalg.vector_norm(result, dim=-1)
    assert torch.isclose(result_lengths, vector_lengths.expand_as(result_lengths)).all()


def test_vector_can_be_optimised():
    true_vector = torch.tensor([-0.844, 1.33, -2.6663])
    axis_angle = torch.tensor([0.36066463, 1.22801072, 1.23066714])
    rotated_true_vector = rotate_vector_axis_angle(true_vector, axis_angle)
    initial_guess = torch.tensor([0.60757815, 0.0427342 , 0.14293731])
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        rotated_vector = rotate_vector_axis_angle(x, axis_angle.unsqueeze(0))
        return torch.linalg.vector_norm(rotated_vector - rotated_true_vector, dim=-1)

    result = solver(initial_guess, error_function)
    assert torch.isclose(result, true_vector, atol=1e-6, rtol=1e-4).all()


def test_rotation_can_be_optimised():
    vectors = torch.tensor([
        [-0.82346044,  0.62842781,  0.25994188],
        [1.87352045, -0.25291732, -0.49509109]
    ])
    true_axis_angle = torch.tensor([[3.25343624, -1.05529992,  0.0097902]])
    true_rotated_vectors = rotate_vector_axis_angle(vectors, true_axis_angle)
    initial_guess = torch.tensor([0.18063841, -2.42946046,  1.12667155])
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        rotated_vectors = rotate_vector_axis_angle(vectors, x)
        return torch.linalg.vector_norm(rotated_vectors - true_rotated_vectors, dim=-1).sum(dim=-1, keepdims=True)

    result = solver(initial_guess, error_function)
    assert torch.isclose(result, true_axis_angle, atol=2e-5, rtol=1e-4).all()


@pytest.mark.parametrize("fixed_vectors,fixed_axis_angles", [
    (torch.tensor([[
        [-1.6513702 , -0.69378276, -0.49310085],
        [1.10361151, -0.17410359, -0.71963057],
    ]]), torch.empty(0, 1, 3)),
    (torch.tensor([[[-0.833546  ,  0.98379094,  0.72699827]]]), torch.tensor([[[-0.21144761,  1.23383457, -0.97610318]]])),
    (torch.empty(1, 0, 3), torch.tensor([
        [[-0.4407212 , -0.71397581, -0.63261585]],
        [[1.14472996, -0.54887293,  0.37555611]]
    ]))
])
def test_rotation_and_vector_can_be_jointly_optimised_if_six_degrees_of_freedom_are_fixed(fixed_random_seed: int, fixed_vectors: torch.Tensor, fixed_axis_angles: torch.Tensor):
    rng = np.random.default_rng(fixed_random_seed)
    num_vectors = 3
    num_rotations = 1
    true_vectors = torch.tensor(rng.normal(0.0, 1.0, size=(1, num_vectors, 3)))
    true_angles = torch.tensor(rng.uniform(0.0, np.pi, size=(num_rotations, 1, 1)))
    true_axes = torch.tensor(rng.normal(0.0, 1.0, size=(num_rotations, 1, 3)))
    true_axes = true_axes / torch.linalg.vector_norm(true_axes, dim=-1, keepdims=True)
    true_axis_angles = true_angles * true_axes
    true_rotated_vectors = rotate_vector_axis_angle(torch.cat([fixed_vectors, true_vectors], dim=1), torch.cat([fixed_axis_angles, true_axis_angles], dim=0))
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(num_vectors * 3 + num_vectors * 3)))
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        vectors = x[0, 0:3 * num_vectors].reshape(1, num_vectors, 3)
        axis_angles = x[0, 3 * num_vectors:3 * (num_vectors + num_rotations)].reshape(num_rotations, 1, 3)
        rotated_vectors = rotate_vector_axis_angle(torch.cat([fixed_vectors, vectors], dim=1), torch.cat([fixed_axis_angles, axis_angles], dim=0))
        return torch.linalg.vector_norm(rotated_vectors - true_rotated_vectors, dim=-1).sum()

    result = solver(initial_guess, error_function)
    result_vectors = result[0:3 * num_vectors].reshape(1, num_vectors, 3)
    result_axis_angles = result[3 * num_vectors:3 * (num_vectors + num_rotations)].reshape(num_rotations, 1, 3)
    result_angles = torch.linalg.vector_norm(result_axis_angles, dim=-1, keepdim=True)
    result_axes = result_axis_angles / result_angles
    # Rotations are equivalent if we negate the axis and subtract the angle from 2pi
    invert_axis = (result_angles > torch.pi).squeeze(-1)
    result_axes[invert_axis] = -1.0 * result_axes[invert_axis]
    result_angles[invert_axis] = 2 * torch.pi - result_angles[invert_axis]
    assert torch.isclose(result_vectors, true_vectors, atol=5e-4, rtol=1e-4).all()
    assert torch.isclose(result_angles.squeeze(-1), true_angles, atol=5e-4, rtol=1e-4).all()
    assert torch.isclose(result_axes, true_axes, atol=5e-4, rtol=1e-4).all()


@pytest.mark.skip("Failing")
def test_rotation_and_vector_can_be_jointly_optimised_if_vector_and_rotation_norms_are_zero(fixed_random_seed: int):
    rng = np.random.default_rng(fixed_random_seed)
    num_vectors = 4
    num_rotations = 4
    true_vectors = torch.tensor(rng.normal(0.0, 1.0, size=(1, num_vectors, 3)))
    true_vectors = true_vectors - true_vectors.mean(dim=1, keepdim=True)
    true_angles = torch.tensor(rng.uniform(0.0, np.pi, size=(num_rotations, 1, 1)))
    true_axes = torch.tensor(rng.normal(0.0, 1.0, size=(num_rotations, 1, 3)))
    true_axes = true_axes / torch.linalg.vector_norm(true_axes, dim=-1, keepdims=True)
    true_axis_angles = true_angles * true_axes
    true_axis_angles = true_axis_angles - true_axis_angles.mean(dim=0, keepdim=True)
    true_rotated_vectors = rotate_vector_axis_angle(true_vectors, true_axis_angles)
    initial_guess = torch.tensor(rng.normal(0.0, 1.0, size=(num_vectors * 3 + num_vectors * 3)))
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        vectors = x[0, 0:3 * num_vectors].reshape(1, num_vectors, 3)
        axis_angles = x[0, 3 * num_vectors:3 * (num_vectors + num_rotations)].reshape(num_rotations, 1, 3)
        vectors = vectors - vectors.mean(dim=1, keepdim=True)
        axis_angles = axis_angles - axis_angles.mean(dim=0, keepdim=True)
        rotated_vectors = rotate_vector_axis_angle(vectors, axis_angles)
        return torch.linalg.vector_norm(rotated_vectors - true_rotated_vectors, dim=-1).sum()

    result = solver(initial_guess, error_function)
    result_vectors = result[0:3 * num_vectors].reshape(1, num_vectors, 3)
    result_vectors = result_vectors - result_vectors.mean(dim=1, keepdim=True)
    result_axis_angles = result[3 * num_vectors:3 * (num_vectors + num_rotations)].reshape(num_rotations, 1, 3)
    result_axis_angles = result_axis_angles - result_axis_angles.mean(dim=0, keepdim=True)
    result_angles = torch.linalg.vector_norm(result_axis_angles, dim=-1, keepdim=True)
    result_axes = result_axis_angles / result_angles
    # Rotations are equivalent if we negate the axis and subtract the angle from 2pi
    invert_axis = (result_angles > torch.pi).squeeze(-1)
    result_axes[invert_axis] = -1.0 * result_axes[invert_axis]
    result_angles[invert_axis] = 2 * torch.pi - result_angles[invert_axis]
    assert torch.isclose(result_vectors, true_vectors, atol=5e-4, rtol=1e-4).all()
    assert torch.isclose(result_angles.squeeze(-1), true_angles, atol=5e-4, rtol=1e-4).all()
    assert torch.isclose(result_axes, true_axes, atol=5e-4, rtol=1e-4).all()



class CompileModule(Module):
    def forward(self, vector: torch.Tensor, axis_angle: torch.Tensor) -> torch.Tensor:
        return rotate_vector_axis_angle(vector, axis_angle)


def test_can_be_compiled(fixed_random_seed: int):
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(rng.normal(0.0, 1.0, size=(8, 3)))
    axis_angles = torch.tensor(rng.normal(0.0, 1.0, size=(8, 3)))
    intrinsics = torch.tensor([0.787, -0.13, -0.02])
    point = torch.tensor([1.1, 0.8, 17.3])
    projection = rotate_vector_axis_angle(point, intrinsics)
    module = CompileModule()
    complied_module = torch.compile(module)
    result = complied_module(point, intrinsics)
    assert torch.isclose(result, projection).all()
