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
from torch.nn import Module
import torch.autograd

from deep_attention_visual_odometry.geometry.projective_plane_angle_distance import (
    projective_plane_angle_distance,
)


@pytest.mark.parametrize("angle", [math.pi * idx / 36 for idx in range(37)])
def test_returns_angle_between_unit_vectors(angle: float) -> None:
    reference_vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    vector = torch.tensor([math.sin(angle), 0.0, math.cos(angle)], dtype=torch.float32)
    result = projective_plane_angle_distance(reference_vector, vector)
    assert float(result) == pytest.approx(angle, rel=0.0, abs=1.5e-6)


@pytest.mark.parametrize("angle", [math.pi * idx / 39 for idx in range(40)])
def test_returns_angle_between_vectors_of_arbitrary_length(
    fixed_random_seed: int, angle: float
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    lengths = rng.uniform(0.0, 10000.0, size=2)
    vector_1 = torch.tensor([0.0, lengths[0], 0.0], dtype=torch.float32)
    vector_2 = torch.tensor(
        [0.0, lengths[1] * math.cos(angle), lengths[1] * math.sin(angle)],
        dtype=torch.float32,
    )
    result = projective_plane_angle_distance(vector_1, vector_2)
    assert float(result) == pytest.approx(angle, rel=0.0, abs=5e-6)


def test_returns_nan_if_given_zeros() -> None:
    zeros = torch.zeros(3)
    other_vector = torch.tensor([1.0, 2.0, 4.0])
    result = projective_plane_angle_distance(zeros, other_vector)
    assert torch.isnan(result).all()
    result = projective_plane_angle_distance(other_vector, zeros)
    assert torch.isnan(result).all()
    result = projective_plane_angle_distance(zeros, zeros)
    assert torch.isnan(result).all()


def test_returns_zero_if_given_parallel_vectors() -> None:
    vector_1 = torch.tensor([3.0, 0.5, -0.4096])
    vector_2 = 2.125 * vector_1
    result = projective_plane_angle_distance(vector_1, vector_2)
    assert result == 0.0  # pytest.approx(0.0, rel=0, abs=1e-7)


def test_supports_batch_and_broadcasts(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vector_1 = torch.tensor(rng.normal(size=(4, 5, 1, 3)))
    vector_2 = torch.tensor(rng.normal(size=(4, 1, 6, 3)))
    result = projective_plane_angle_distance(vector_1, vector_2)
    assert result.shape == (4, 5, 6)


def test_keepdim_retains_final_dimension(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vector_1 = torch.tensor(rng.normal(size=(2, 3, 1, 3)))
    vector_2 = torch.tensor(rng.normal(size=(2, 1, 6, 3)))
    result = projective_plane_angle_distance(vector_1, vector_2, keepdim=True)
    assert result.shape == (2, 3, 6, 1)


def test_is_commutative(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vector_1 = torch.tensor(rng.normal(size=(1, 2, 3, 3)))
    vector_2 = torch.tensor(rng.normal(size=(4, 1, 3, 3)))
    result_1 = projective_plane_angle_distance(vector_1, vector_2)
    result_2 = projective_plane_angle_distance(vector_2, vector_1)
    assert torch.equal(result_1, result_2)


def test_angle_between_vectors_is_zero_for_identical_vectors(
    fixed_random_seed: int,
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float64)
    result = projective_plane_angle_distance(vectors, vectors)
    assert torch.equal(result, torch.zeros_like(result))


def test_angle_between_vectors_is_pi_for_negative_vectors(
    fixed_random_seed: int,
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(rng.normal(size=(7, 3)))
    result = projective_plane_angle_distance(vectors, -1.0 * vectors)
    assert torch.isclose(
        result, math.pi * torch.ones_like(result), rtol=0.0, atol=1e-7
    ).all()


def test_supports_needle_like_triangles():
    # Some example "needle-like" triangles, based on
    # "Miscalculating the Area and Angles of Needle-like triangles" by Kahan
    scale = 100000.0
    angle = 1.145915591e-7
    vector_1 = torch.tensor([scale, 0.0, 0.0], dtype=torch.float32)
    vector_2 = torch.tensor(
        [scale * math.cos(angle), 0.0, scale * math.sin(angle)], dtype=torch.float32
    )
    result = projective_plane_angle_distance(vector_1, vector_2)
    torch.equal(result, torch.tensor(angle, dtype=torch.float32))

    angle = 48.18968510
    vector_1 = torch.tensor([0.0, 0.0, 100000.0], dtype=torch.float32)
    vector_2 = torch.tensor(
        [0.0, 0.0001 * math.sin(angle), 0.0001 * math.cos(angle)], dtype=torch.float32
    )
    result = projective_plane_angle_distance(vector_1, vector_2)
    torch.equal(result, torch.tensor(angle, dtype=torch.float32))

    angle = math.radians(179.9985965)
    vector_1 = torch.tensor([0.0, 10000.0, 0.0], dtype=torch.float32)
    vector_2 = torch.tensor(
        [5000 * math.sin(angle), 5000 * math.cos(angle), 0.0], dtype=torch.float32
    )
    result = projective_plane_angle_distance(vector_1, vector_2)
    torch.equal(result, torch.tensor(angle, dtype=torch.float32))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("change_direction", [10000.0, -10000.0])
def test_is_relatively_stable_for_small_changes(
    axis: int, change_direction: float
) -> None:
    vector_values = (-0.84278762, 3.2473695, -1.18345099)
    base_vector = torch.tensor(vector_values, dtype=torch.float64)

    def step_vector(values: tuple[float, float, float]) -> tuple[float, float, float]:
        if axis == 0:
            return np.nextafter(values[0], change_direction), values[1], values[2]
        elif axis == 1:
            return values[0], np.nextafter(values[1], change_direction), values[2]
        return values[0], values[1], np.nextafter(values[2], change_direction)

    results = []
    for step_idx in range(100):
        vector_values = step_vector(vector_values)
        vector = torch.tensor(vector_values, dtype=torch.float64)
        result = projective_plane_angle_distance(base_vector, vector)
        results.append(float(result))
    assert (
        sum(result_2 >= result_1 for result_1, result_2 in zip(results, results[1:]))
        > 90
    )
    assert len(set(results)) > 92


@pytest.mark.parametrize("angle", [math.pi * idx / 38 for idx in range(39)])
def test_can_pass_through_gradients(fixed_random_seed: int, angle: float) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    lengths = rng.uniform(0.0, 10000.0, size=2)
    vector_1 = torch.tensor(
        [lengths[0], 0.0, 0.0], dtype=torch.float32, requires_grad=True
    )
    vector_2 = torch.tensor(
        [lengths[1] * math.cos(angle), 0.0, lengths[1] * math.sin(angle)],
        dtype=torch.float32,
        requires_grad=True,
    )
    distance = projective_plane_angle_distance(vector_1, vector_2)
    gradients = torch.autograd.grad(distance, (vector_1, vector_2))
    assert len(gradients) == 2
    assert torch.isfinite(gradients[0]).all()
    if angle > 0:
        assert not torch.eq(gradients[0], torch.zeros_like(gradients[0])).all()
    assert torch.isfinite(gradients[1]).all()
    if angle > 0:
        assert not torch.eq(gradients[1], torch.zeros_like(gradients[1])).all()


def test_gradients_at_parallel_are_zero(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(rng.normal(size=3), dtype=torch.float64, requires_grad=True)
    distance = projective_plane_angle_distance(vectors, vectors.detach())
    gradient = torch.autograd.grad(distance, vectors)
    assert torch.isfinite(gradient[0]).all()
    assert torch.equal(gradient[0], torch.zeros_like(gradient[0]))

    distance = projective_plane_angle_distance(vectors.detach(), vectors)
    gradient = torch.autograd.grad(distance, vectors)
    assert torch.isfinite(gradient[0]).all()
    assert torch.equal(gradient[0], torch.zeros_like(gradient[0]))


def test_gradients_are_valid_near_parallel() -> None:
    vector_1 = torch.tensor(
        [0.16747557919176526, 1.411436462001734, 17.299996837614568],
        dtype=torch.float64,
        requires_grad=True,
    )
    vector_2 = torch.tensor(
        [0.1674757565885585, 1.411436913882588, 17.3], dtype=torch.float64
    )
    distance = projective_plane_angle_distance(vector_1, vector_2)
    gradient = torch.autograd.grad(distance, vector_1)
    assert torch.isfinite(gradient[0]).all()

    vector_1 = torch.tensor(
        [1.2658736503683827, 1.411436462001734, 17.299996837614568], dtype=torch.float64
    )
    vector_2 = torch.tensor(
        [1.265873827765176, 1.411436913882588, 17.3],
        dtype=torch.float64,
        requires_grad=True,
    )
    distance = projective_plane_angle_distance(vector_1, vector_2)
    gradient = torch.autograd.grad(distance, vector_2)
    assert torch.isfinite(gradient[0]).all()


def test_gradients_are_valid_for_right_angles(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors_1 = torch.tensor(
        rng.normal(size=(7, 3)), dtype=torch.float64, requires_grad=True
    )
    vectors_2 = torch.tensor(rng.normal(size=(7, 3)), dtype=torch.float64)
    unit_1 = vectors_1.detach() / torch.linalg.vector_norm(
        vectors_1.detach(), dim=-1, keepdim=True
    )
    vectors_2 = vectors_2 - unit_1 * (unit_1 * vectors_2).sum(dim=-1, keepdims=True)

    distance = projective_plane_angle_distance(vectors_1, vectors_2)
    gradient = torch.autograd.grad(distance.sum(), vectors_1)
    assert torch.isfinite(gradient[0]).all()
    assert torch.greater(gradient[0].abs(), torch.zeros_like(gradient[0])).all()

    distance = projective_plane_angle_distance(vectors_2, vectors_1)
    gradient = torch.autograd.grad(distance.sum(), vectors_1)
    assert torch.isfinite(gradient[0]).all()
    assert torch.greater(gradient[0].abs(), torch.zeros_like(gradient[0])).all()


def test_gradients_are_zero_for_negative_vectors(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(
        rng.normal(size=(7, 3)), dtype=torch.float64, requires_grad=True
    )
    distance = projective_plane_angle_distance(vectors, -1.0 * vectors.detach())
    gradient = torch.autograd.grad(distance.sum(), vectors)
    assert torch.isfinite(gradient[0]).all()
    assert torch.isclose(gradient[0], torch.zeros_like(gradient[0])).all()

    distance = projective_plane_angle_distance(-1.0 * vectors.detach(), vectors)
    gradient = torch.autograd.grad(distance.sum(), vectors)
    assert torch.isfinite(gradient[0]).all()
    assert torch.isclose(gradient[0], torch.zeros_like(gradient[0])).all()


@pytest.mark.parametrize(
    "dtype,eps, atol, rtol",
    [
        (
            torch.float64,
            1e-6,
            1e-05,
            0.001,
        ),
        (
            torch.float32,
            1e-3,
            1e-04,
            0.005,
        ),
    ],
)
def test_gradients_gradcheck(
    fixed_random_seed: int, dtype: torch.dtype, eps: float, atol: float, rtol: float
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors_1 = torch.tensor(rng.normal(size=(48, 3)), dtype=dtype, requires_grad=True)
    vectors_2 = torch.tensor(rng.normal(size=(48, 3)), dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(
        projective_plane_angle_distance,
        (vectors_1, vectors_2),
        eps=eps,
        atol=atol,
        rtol=rtol,
    )


class CompileModule(Module):
    def forward(self, points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
        return projective_plane_angle_distance(points_a, points_b)


def test_can_be_compiled() -> None:
    point_a = torch.tensor([0.787, -0.13, -0.02])
    point_b = torch.tensor([1.1, 0.8, 17.3])
    projection = projective_plane_angle_distance(point_a, point_b)
    module = CompileModule()
    complied_module = torch.compile(module)
    result = complied_module(point_a, point_b)
    assert torch.isclose(result, projection).all()
