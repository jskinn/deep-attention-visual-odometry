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

from deep_attention_visual_odometry.autograd_solvers import SGDSolver
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


def test_intrinsics_can_be_optimised():
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
