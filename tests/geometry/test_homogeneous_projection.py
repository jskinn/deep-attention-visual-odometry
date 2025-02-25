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

from deep_attention_visual_odometry.geometry.homogeneous_projection import (
    pixel_coordinates_to_homogeneous,
    project_points_pinhole_homogeneous,
)


def test_pixel_coordinates_to_homogeneous() -> None:
    f = 0.0899
    cx = -0.1
    cy = 0.15
    u = 0.448
    v = -0.334
    projected_points = torch.tensor([u, v], dtype=torch.float64)
    intrinsics = torch.tensor([f, cx, cy], dtype=torch.float64)
    result = pixel_coordinates_to_homogeneous(projected_points, intrinsics)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)
    assert result[0] == u - cx
    assert result[1] == v - cy
    assert result[2] == f + 1.0


def test_pixel_coordinates_to_homogeneous_handles_batch_and_broadcasts(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    projected_points = torch.tensor(rng.normal(size=(1, 7, 2)))
    intrinsics = torch.tensor(rng.normal(size=(4, 1, 3)))
    result = pixel_coordinates_to_homogeneous(projected_points, intrinsics)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (4, 7, 3)

def test_pixel_coordinates_to_homogeneous_never_produces_negative_z(fixed_random_seed: int) -> None:
    rng = np.random.default_rng(fixed_random_seed)
    projected_points = torch.tensor(rng.normal(size=(65, 2)))
    intrinsics = torch.tensor(rng.normal(size=(65, 3)))
    intrinsics[:, 0] = torch.linspace(-32.0, 32.0, 65)
    result = pixel_coordinates_to_homogeneous(projected_points, intrinsics)
    assert result.shape == (65, 3)
    assert (result[:, 2] >= 0).all()
    assert torch.equal(result[intrinsics[:, 0] >= 0, 2], intrinsics[intrinsics[:, 0] >= 0, 0] + 1.0)
    assert torch.allclose(result[intrinsics[:, 0] < 0, 2], torch.exp(intrinsics[intrinsics[:, 0] < 0, 0]))


@pytest.mark.parametrize("focal_length", [-0.0899, 0.0899])
def test_project_single_point_inside_image(focal_length: float):
    f = focal_length + 1 if focal_length >= 0.0 else math.exp(focal_length)
    cx = -0.1
    cy = 0.15
    x = 1.0
    y = -2.0
    z = 15.0
    u = f * x / z + cx
    v = f * y / z + cy
    camera_intrinsics = torch.tensor([focal_length, cx, cy], dtype=torch.float64)
    point = torch.tensor([x, y, z], dtype=torch.float64)
    pixel = project_points_pinhole_homogeneous(point, camera_intrinsics)
    assert pixel.shape == (3,)
    assert float(pixel[0] / pixel[2]) == pytest.approx(u, rel=0, abs=2e-17)
    assert float(pixel[1] / pixel[2]) == pytest.approx(v, rel=0, abs=2e-17)
    assert pixel[2] == z


def test_project_point_on_image_plane():
    f = -2.0899
    cx = -0.1
    cy = 0.15
    x = 12.0
    y = -2.0
    z = 0.0
    expected_result = torch.tensor([math.exp(f) * x, math.exp(f) * y, 0.0], dtype=torch.float64)
    camera_intrinsics = torch.tensor([f, cx, cy], dtype=torch.float64)
    point = torch.tensor([x, y, z], dtype=torch.float64)
    pixel = project_points_pinhole_homogeneous(point, camera_intrinsics)
    assert torch.allclose(pixel, expected_result)


def test_project_origin():
    f = 1.5799
    cx = -0.1
    cy = 0.15
    camera_intrinsics = torch.tensor([f, cx, cy])
    point = torch.zeros(3)
    pixel = project_points_pinhole_homogeneous(point, camera_intrinsics)
    assert torch.equal(pixel, torch.tensor((0.0, 0.0, 1.0)))


def test_project_point_behind_camera():
    f = -0.0799
    cx = -0.1
    cy = 0.15
    x = 1.0
    y = -2.0
    z = -0.125
    u = math.exp(f) * x + cx * z
    v = math.exp(f) * y + cy * z
    expected_result = torch.tensor([u, v, z], dtype=torch.float64)
    camera_intrinsics = torch.tensor([f, cx, cy], dtype=torch.float64)
    point = torch.tensor([x, y, z], dtype=torch.float64)
    pixel = project_points_pinhole_homogeneous(point, camera_intrinsics)
    assert torch.equal(pixel, expected_result)


def test_project_point_close_to_camera():
    f = 0.122
    cx = -0.1
    cy = 0.15
    x = -0.0000012
    y = -0.0000023
    z = 0.000007
    u = (f + 1.0) * x + cx * z
    v = (f + 1.0) * y + cy * z
    expected_result = torch.tensor([u, v, z], dtype=torch.float64)
    camera_intrinsics = torch.tensor([f, cx, cy], dtype=torch.float64)
    point = torch.tensor([x, y, z], dtype=torch.float64)
    pixel = project_points_pinhole_homogeneous(point, camera_intrinsics)
    assert torch.equal(pixel, expected_result)


def test_project_batch_of_points_and_views(fixed_random_seed: int):
    rng = np.random.default_rng(fixed_random_seed)
    num_views = 7
    num_points = 5
    focal_length = torch.tensor(rng.uniform(1e-8, 4, size=(num_views, 1, 1)))
    principal_point = torch.tensor(rng.normal(1e-8, 4, size=(num_views, 1, 2)))
    camera_intrinsics = torch.concatenate([focal_length, principal_point], dim=-1)
    points = torch.tensor(rng.normal(0, 15, size=(1, num_points, 3)))
    pixels = project_points_pinhole_homogeneous(points, camera_intrinsics)
    assert pixels.shape == (num_views, num_points, 3)
    assert torch.isfinite(pixels).all()


class CompileModule(Module):
    def forward(self, points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        return project_points_pinhole_homogeneous(points, intrinsics)


def test_can_be_compiled():
    intrinsics = torch.tensor([0.787, -0.13, -0.02])
    point = torch.tensor([1.1, 0.8, 17.3])
    projection = project_points_pinhole_homogeneous(point, intrinsics)
    module = CompileModule()
    complied_module = torch.compile(module)
    result = complied_module(point, intrinsics)
    assert torch.isclose(result, projection).all()
