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
import torch
from torch.nn import Module

from deep_attention_visual_odometry.autograd_solvers import SGDSolver, BFGSSolver
from deep_attention_visual_odometry.geometry.camera_projection import (
    project_points_basic_pinhole,
)


def test_project_single_point():
    f = 0.899
    cx = -0.1
    cy = 0.15
    x = 1.0
    y = -2.0
    z = 15.0
    u = f * x / z + cx
    v = f * y / z + cy
    camera_intrinsics = torch.tensor([f, cx, cy], dtype=torch.float64)
    point = torch.tensor([x, y, z], dtype=torch.float64)
    pixel = project_points_basic_pinhole(point, camera_intrinsics)
    assert pixel.shape == (2,)
    assert pixel[0] == u
    assert pixel[1] == v


def test_project_batch_of_points_and_views():
    camera_intrinsics = torch.tensor(
        [
            [0.899, 0.1, -0.15],
            [1.0101, -0.1, 0.08],
        ],
        dtype=torch.float64,
    ).unsqueeze(1)
    points = torch.tensor(
        [
            [1.0, 1.0, 14.0],
            [1.0, -1.0, 14.0],
            [-1.0, 1.0, 14.0],
            [-1.0, -1.0, 14.0],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    expected_pixels = torch.tensor(
        [
            [
                [0.899 * 1.0 / 14.0 + 0.1, 0.899 * 1.0 / 14.0 - 0.15],
                [0.899 * 1.0 / 14.0 + 0.1, 0.899 * -1.0 / 14.0 - 0.15],
                [0.899 * -1.0 / 14.0 + 0.1, 0.899 * 1.0 / 14.0 - 0.15],
                [0.899 * -1.0 / 14.0 + 0.1, 0.899 * -1.0 / 14.0 - 0.15],
            ],
            [
                [1.0101 * 1.0 / 14.0 - 0.1, 1.0101 * 1.0 / 14.0 + 0.08],
                [1.0101 * 1.0 / 14.0 - 0.1, 1.0101 * -1.0 / 14.0 + 0.08],
                [1.0101 * -1.0 / 14.0 - 0.1, 1.0101 * 1.0 / 14.0 + 0.08],
                [1.0101 * -1.0 / 14.0 - 0.1, 1.0101 * -1.0 / 14.0 + 0.08],
            ],
        ],
        dtype=torch.float64,
    )
    pixels = project_points_basic_pinhole(points, camera_intrinsics)
    assert torch.equal(pixels, expected_pixels)


def test_intrinsics_can_be_optimised_by_sgd():
    points = torch.tensor(
        [
            [1.0, 1.0, 14.0],
            [1.0, -1.0, 14.0],
            [-1.0, 1.0, 14.0],
            [-1.0, -1.0, 14.0],
        ],
        dtype=torch.float64,
    )
    true_intrinsics = torch.tensor([[0.899, -0.15, 0.08]], dtype=torch.float64)
    intrinsics = torch.tensor([[0.989, -0.1, 0.15]], dtype=torch.float64)
    true_projection = project_points_basic_pinhole(points, true_intrinsics)
    solver = SGDSolver(learning_rate=0.0002, iterations=2000)

    def error_function(x: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_basic_pinhole(
            points[None, :, :], x[:, None, :]
        )
        return torch.linalg.vector_norm(
            projected_points - true_projection[None, :, :], dim=-1
        ).sum(dim=-1)

    result = solver(intrinsics, error_function)
    assert torch.isclose(result, true_intrinsics, atol=6e-5, rtol=1e-4).all()


def test_intrinsics_can_be_optimised_from_far_away_by_bfgs():
    points = torch.tensor(
        [
            [1.0, 0.0, 14.0],
            [1.0, 1.0, 15.0],
            [1.0, -1.0, 16.0],
            [0.0, 0.0, 17.0],
            [0.0, 1.0, 13.0],
            [0.0, -1.0, 14.0],
            [-1.0, 0.0, 12.0],
            [-1.0, 1.0, 17.0],
            [-1.0, -1.0, 15.0],
        ],
        dtype=torch.float64,
    )
    true_intrinsics = torch.tensor([[2.333, -0.35, 0.08]], dtype=torch.float64)
    intrinsics = torch.tensor([[0.0125, 0.81, -1.75]], dtype=torch.float64)
    true_projection = project_points_basic_pinhole(points, true_intrinsics)
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_basic_pinhole(
            points[None, :, :], x[:, None, :]
        )
        return torch.linalg.vector_norm(
            projected_points - true_projection[None, :, :], dim=-1
        ).sum(dim=-1)

    result = solver(intrinsics, error_function)
    assert torch.isclose(result, true_intrinsics, atol=1e-6, rtol=1e-4).all()


def test_world_points_can_be_optimised_up_to_scale():
    intrinsics = torch.tensor(
        [
            [0.33, 0.1, -0.8],
            [2.73, -0.333, -0.18],
            [1.28, 0.015, -0.09],
            [0.989, -0.08, 0.02],
        ],
        dtype=torch.float64,
    )
    true_point = torch.tensor([1.1, 0.8, 17.3], dtype=torch.float64)
    point = torch.tensor([-2.333, 0.35, 3.5], dtype=torch.float64)
    true_projection = project_points_basic_pinhole(true_point[None, :], intrinsics)
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_basic_pinhole(x, intrinsics)
        return torch.linalg.vector_norm(
            projected_points - true_projection[None, :, :], dim=-1
        ).sum(dim=-1)

    result = solver(point, error_function)
    scale = result / true_point
    assert not torch.isclose(result, true_point, atol=0.5).all()
    assert torch.isclose(scale, scale.mean(), atol=1e-4, rtol=1e-4).all()


def test_world_points_can_be_optimised_with_stereo_offset():
    intrinsics = torch.tensor([0.787, -0.13, -0.02], dtype=torch.float64)
    stereo_offset = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    true_point = torch.tensor([1.1, 0.8, 17.3], dtype=torch.float64)
    point = torch.tensor([-2.333, 0.35, 3.5], dtype=torch.float64)
    true_projection_left = project_points_basic_pinhole(true_point, intrinsics)
    true_projection_right = project_points_basic_pinhole(
        true_point + stereo_offset, intrinsics
    )
    solver = BFGSSolver(iterations=500, error_threshold=1e-6)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points_left = project_points_basic_pinhole(x, intrinsics)
        projected_points_right = project_points_basic_pinhole(
            x + stereo_offset, intrinsics
        )
        left_error = torch.linalg.vector_norm(
            projected_points_left - true_projection_left, dim=-1
        ).sum(dim=-1)
        rignt_error = torch.linalg.vector_norm(
            projected_points_right - true_projection_right, dim=-1
        ).sum(dim=-1)
        return left_error + rignt_error

    result = solver(point, error_function)
    assert torch.isclose(result, true_point, atol=1e-6, rtol=1e-4).all()


def test_intrinsics_can_be_optimised_from_negative_focal_length_by_bfgs():
    points = torch.tensor(
        [
            [1.0, 0.0, 14.0],
            [1.0, 1.0, 15.0],
            [1.0, -1.0, 16.0],
            [0.0, 0.0, 17.0],
            [0.0, 1.0, 13.0],
            [0.0, -1.0, 14.0],
            [-1.0, 0.0, 12.0],
            [-1.0, 1.0, 17.0],
            [-1.0, -1.0, 15.0],
        ],
        dtype=torch.float64,
    )
    true_intrinsics = torch.tensor([[2.333, -0.35, 0.08]], dtype=torch.float64)
    intrinsics = torch.tensor([[-3.0125, 0.81, -1.75]], dtype=torch.float64)
    true_projection = project_points_basic_pinhole(points, true_intrinsics)
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_basic_pinhole(
            points[None, :, :], x[:, None, :]
        )
        return torch.linalg.vector_norm(
            projected_points - true_projection[None, :, :], dim=-1
        ).sum(dim=-1)

    result = solver(intrinsics, error_function)
    assert torch.isclose(result, true_intrinsics, atol=1e-6, rtol=1e-4).all()


def test_world_points_can_be_optimised_close_to_the_camera_up_to_scale():
    intrinsics = torch.tensor(
        [
            [0.33, 0.1, -0.8],
            [2.73, -0.333, -0.18],
            [1.28, 0.015, -0.09],
            [0.989, -0.08, 0.02],
        ],
        dtype=torch.float64,
    )
    true_point = torch.tensor([-0.0011, 0.08, 0.0003], dtype=torch.float64)
    point = torch.tensor([-2.333, 0.35, 3.5], dtype=torch.float64)
    true_projection = project_points_basic_pinhole(true_point[None, :], intrinsics)
    solver = BFGSSolver(iterations=500)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_basic_pinhole(x, intrinsics)
        return torch.linalg.vector_norm(
            projected_points - true_projection[None, :, :], dim=-1
        ).sum(dim=-1)

    result = solver(point, error_function)
    scale = result / true_point
    assert not torch.isclose(result, true_point, atol=0.5).all()
    assert torch.isclose(scale, scale.mean(), atol=1e-4, rtol=1e-4).all()


def test_world_points_cannot_be_optimised_from_behind_the_camera():
    intrinsics = torch.tensor([0.787, -0.13, -0.02], dtype=torch.float64)
    stereo_offset = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    true_point = torch.tensor([1.1, 0.8, 17.3], dtype=torch.float64)
    point = torch.tensor([-2.333, 0.35, -3.5], dtype=torch.float64)
    true_projection_left = project_points_basic_pinhole(true_point, intrinsics)
    true_projection_right = project_points_basic_pinhole(
        true_point + stereo_offset, intrinsics
    )
    solver = BFGSSolver(iterations=700)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points_left = project_points_basic_pinhole(x, intrinsics)
        projected_points_right = project_points_basic_pinhole(
            x + stereo_offset, intrinsics
        )
        left_error = torch.linalg.vector_norm(
            projected_points_left - true_projection_left, dim=-1
        ).sum(dim=-1)
        rignt_error = torch.linalg.vector_norm(
            projected_points_right - true_projection_right, dim=-1
        ).sum(dim=-1)
        return left_error + rignt_error

    result = solver(point, error_function)
    scale = result / true_point
    # The result should still be negative in the Z axis, but approximately on the same ray as the true point.
    assert result[2] < 0.0
    assert torch.less(scale, torch.zeros_like(scale)).all()
    assert torch.isclose(scale, scale.mean(), atol=0.05, rtol=0.01).all()


class CompileModule(Module):
    def forward(self, points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        return project_points_basic_pinhole(points, intrinsics)


def test_can_be_compiled():
    intrinsics = torch.tensor([0.787, -0.13, -0.02])
    point = torch.tensor([1.1, 0.8, 17.3])
    projection = project_points_basic_pinhole(point, intrinsics)
    module = CompileModule()
    complied_module = torch.compile(module)
    result = complied_module(point, intrinsics)
    assert torch.isclose(result, projection).all()
