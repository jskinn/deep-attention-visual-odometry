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

from deep_attention_visual_odometry.geometry.projective_plane_cosine_distance import (
    projective_plane_cosine_distance,
)


def test_returns_one_minus_cosine_angle_between_vectors() -> None:
    reference_vector = torch.tensor([0.0, 0.0, 1.0])

    angle = 7 * math.pi / 16
    vector = torch.tensor([math.sin(angle), 0.0, math.cos(angle)])
    result = projective_plane_cosine_distance(reference_vector, vector)
    assert result == pytest.approx(1.0 - math.cos(angle), rel=0, abs=1e-7)

    angle = 11 * math.pi / 12
    vector = torch.tensor([0.0, math.sin(angle), math.cos(angle)])
    result = projective_plane_cosine_distance(reference_vector, vector)
    assert result == pytest.approx(1.0 - math.cos(angle), rel=0, abs=1e-7)


def test_normalizes_vectors() -> None:
    reference_vector = torch.tensor([0.0, 0.0, 2.233])

    angle = 5 * math.pi / 12
    length = 5.325
    vector = torch.tensor([length * math.sin(angle), 0.0, length * math.cos(angle)])
    result = projective_plane_cosine_distance(reference_vector, vector)
    assert result == pytest.approx(1.0 - math.cos(angle), rel=0, abs=1e-7)

    angle = 4 * math.pi / 15
    length = 0.0214
    vector = torch.tensor([0.0, length * math.sin(angle), length * math.cos(angle)])
    result = projective_plane_cosine_distance(reference_vector, vector)
    assert result == pytest.approx(1.0 - math.cos(angle), rel=0, abs=1e-7)


def test_returns_nan_if_given_zeros() -> None:
    zeros = torch.zeros(3)
    other_vector = torch.tensor([1.0, 2.0, 4.0])
    result = projective_plane_cosine_distance(zeros, other_vector)
    assert torch.isnan(result).all()
    result = projective_plane_cosine_distance(other_vector, zeros)
    assert torch.isnan(result).all()
    result = projective_plane_cosine_distance(zeros, zeros)
    assert torch.isnan(result).all()


def test_supports_batch_and_broadcasts(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vector_1 = torch.tensor(rng.normal(size=(4, 5, 1, 3)))
    vector_2 = torch.tensor(rng.normal(size=(4, 1, 6, 3)))
    result = projective_plane_cosine_distance(vector_1, vector_2)
    assert result.shape == (4, 5, 6)


def test_keepdim_retains_final_dimension(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vector_1 = torch.tensor(rng.normal(size=(2, 3, 1, 3)))
    vector_2 = torch.tensor(rng.normal(size=(2, 1, 6, 3)))
    result = projective_plane_cosine_distance(vector_1, vector_2, keepdim=True)
    assert result.shape == (2, 3, 6, 1)


def test_is_commutative(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vector_1 = torch.tensor(rng.normal(size=(1, 2, 3, 3)))
    vector_2 = torch.tensor(rng.normal(size=(4, 1, 3, 3)))
    result_1 = projective_plane_cosine_distance(vector_1, vector_2)
    result_2 = projective_plane_cosine_distance(vector_2, vector_1)
    assert torch.equal(result_1, result_2)


def test_angle_between_vectors_is_zero_for_identical_vectors(
    fixed_random_seed: int,
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(rng.normal(size=(7, 3)))
    result = projective_plane_cosine_distance(vectors, vectors)
    assert torch.isclose(result, torch.zeros_like(result), rtol=0.0, atol=1e-7).all()


def test_angle_between_vectors_is_two_for_negative_vectors(
    fixed_random_seed: int,
) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    vectors = torch.tensor(rng.normal(size=(7, 3)))
    result = projective_plane_cosine_distance(vectors, -1.0 * vectors)
    assert torch.isclose(
        result, 2.0 * torch.ones_like(result), rtol=0.0, atol=1e-7
    ).all()


class CompileModule(Module):
    def forward(self, points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
        return projective_plane_cosine_distance(points_a, points_b)


def test_can_be_compiled() -> None:
    point_a = torch.tensor([0.787, -0.13, -0.02])
    point_b = torch.tensor([1.1, 0.8, 17.3])
    projection = projective_plane_cosine_distance(point_a, point_b)
    module = CompileModule()
    complied_module = torch.compile(module)
    result = complied_module(point_a, point_b)
    assert torch.isclose(result, projection).all()
