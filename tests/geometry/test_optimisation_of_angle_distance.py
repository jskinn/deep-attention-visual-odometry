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

from deep_attention_visual_odometry.autograd_solvers import SGDSolver, BFGSSolver
from deep_attention_visual_odometry.geometry.homogeneous_projection import (
    project_points_pinhole_homogeneous,
)
from deep_attention_visual_odometry.geometry.projective_plane_angle_distance import (
    projective_plane_angle_distance,
)


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
    true_projection = project_points_pinhole_homogeneous(points, true_intrinsics)
    solver = SGDSolver(learning_rate=0.0002, iterations=4000)

    def error_function(x: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_pinhole_homogeneous(
            points[None, :, :], x[:, None, :]
        )
        return projective_plane_angle_distance(true_projection, projected_points)

    result = solver(intrinsics, error_function)
    assert torch.isclose(result, true_intrinsics, atol=5e-4, rtol=1e-4).all()


def test_intrinsics_can_be_optimised_by_bfgs():
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
    true_intrinsics = torch.tensor([[1.333, -0.35, 0.08]], dtype=torch.float64)
    intrinsics = torch.tensor([[-2.0125, 0.81, -1.75]], dtype=torch.float64)
    true_projection = project_points_pinhole_homogeneous(points, true_intrinsics)
    solver = BFGSSolver(iterations=500, error_threshold=1e-8)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_pinhole_homogeneous(
            points[None, :, :], x[:, None, :]
        )
        return projective_plane_angle_distance(true_projection[None, :, :], projected_points).sum(dim=-1)

    result = solver(intrinsics, error_function)
    assert torch.isclose(result, true_intrinsics, atol=1e-8, rtol=0.0).all()


def test_world_points_can_be_optimised_up_to_scale():
    intrinsics = torch.tensor(
        [
            [-0.66, 0.1, -0.8],
            [1.43, -0.333, -0.18],
            [0.78, 0.015, -0.09],
            [-1.989, -0.08, 0.02],
        ],
        dtype=torch.float64,
    )
    true_point = torch.tensor([1.1, 0.8, 17.3], dtype=torch.float64)
    point = torch.tensor([-2.333, 0.35, 3.5], dtype=torch.float64)
    true_projection = project_points_pinhole_homogeneous(true_point[None, :], intrinsics)
    solver = BFGSSolver(iterations=500, error_threshold=1e-8)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_pinhole_homogeneous(x, intrinsics)
        return projective_plane_angle_distance(true_projection, projected_points).sum(dim=-1)

    result = solver(point, error_function)
    scale = result / true_point
    assert not torch.isclose(result, true_point, atol=0.5).all()
    assert torch.isclose(scale, scale.mean(), atol=1e-8).all()


def test_world_points_can_be_optimised_with_stereo_offset():
    intrinsics = torch.tensor([0.787, -0.13, -0.02], dtype=torch.float64)
    stereo_offset = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
    true_point = torch.tensor([1.1, 0.8, 17.3], dtype=torch.float64)
    point = torch.tensor([-2.333, 0.35, 3.5], dtype=torch.float64)
    true_projection_left = project_points_pinhole_homogeneous(true_point, intrinsics)
    true_projection_right = project_points_pinhole_homogeneous(
        true_point + stereo_offset, intrinsics
    )
    solver = BFGSSolver(iterations=500, error_threshold=1e-8)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points_left = project_points_pinhole_homogeneous(x, intrinsics)
        projected_points_right = project_points_pinhole_homogeneous(
            x + stereo_offset, intrinsics
        )
        left_error = projective_plane_angle_distance(projected_points_left, true_projection_left)
        right_error = projective_plane_angle_distance(projected_points_right, true_projection_right)
        return left_error + right_error

    result = solver(point, error_function)
    assert torch.isclose(result, true_point, atol=1e-6, rtol=1e-4).all()


def test_world_points_can_be_optimised_from_behind_the_camera():
    intrinsics = torch.tensor([0.0787, -0.13, -0.02], dtype=torch.float64)
    view_offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.1, -0.1],
            [-1.5, -0.3, 0.11],
        ],
        dtype=torch.float64,
    )
    true_point = torch.tensor([1.1, 0.8, 17.3], dtype=torch.float64)
    point = torch.tensor([-2.333, 0.35, -3.5], dtype=torch.float64)
    true_projections = project_points_pinhole_homogeneous(
        true_point[None, :] + view_offsets, intrinsics
    )
    solver = BFGSSolver(iterations=500, error_threshold=1e-8, minimum_step=1e-10)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_pinhole_homogeneous(
            x[:, None, :] + view_offsets[None, :, :], intrinsics
        )
        error = projective_plane_angle_distance(projected_points, true_projections)
        return error.sum(dim=-1)

    result = solver(point, error_function)
    assert torch.isclose(result, true_point, atol=1e-8, rtol=1e-7).all()


def test_world_points_can_be_optimised_from_far_in_front():
    intrinsics = torch.tensor([0.0787, -0.13, -0.02], dtype=torch.float64)
    view_offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.1, -0.1],
            [-1.5, -0.3, 0.11],
            [-0.1, 1.2, 0.2],
            [0.2, -0.38, 12.0],
            [0.32, 0.41, -2.0],
        ],
        dtype=torch.float64,
    )
    true_point = torch.tensor([1.1, 0.8, 17.3], dtype=torch.float64)
    point = torch.tensor([-0.337, 0.14, 14159.0], dtype=torch.float64)
    true_projections = project_points_pinhole_homogeneous(
        true_point[None, :] + view_offsets, intrinsics
    )
    solver = BFGSSolver(iterations=500, error_threshold=1e-8, minimum_step=1e-10)

    def error_function(x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_pinhole_homogeneous(
            x[:, None, :] + view_offsets[None, :, :], intrinsics
        )
        error = projective_plane_angle_distance(projected_points, true_projections)
        return error.sum(dim=-1)

    result = solver(point, error_function)
    assert torch.isclose(result, true_point, atol=1e-8, rtol=1e-8).all()


def test_world_points_can_be_optimised_from_unit_normal_initial_guess(
    fixed_random_seed: int,
):
    rng = np.random.default_rng(fixed_random_seed)
    num_views = 4
    num_points = 5
    np_true_xy = rng.normal(0.0, 1.0, size=(num_points, 1, 2))
    np_true_z = rng.gamma(3.0, 5.0, size=(num_points, 1, 1))
    intrinsics = torch.tensor(rng.uniform(0, 1, size=3))
    view_offsets = torch.tensor(rng.normal(0.0, 1.0, size=(1, num_views, 3)))
    true_points = torch.tensor(np.concatenate([np_true_xy, np_true_z], axis=-1))
    true_projections = project_points_pinhole_homogeneous(
        true_points + view_offsets, intrinsics
    )
    points = torch.tensor(rng.normal(0.0, 1.0, size=(num_points, 3)))
    solver = BFGSSolver(iterations=500, error_threshold=1e-8, minimum_step=1e-10)

    def error_function(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected_points = project_points_pinhole_homogeneous(
            x[:, None, :] + view_offsets, intrinsics
        )
        targets = true_projections[mask]
        error = projective_plane_angle_distance(projected_points, targets)
        return error.sum(dim=-1)

    result = solver(points, error_function)
    assert torch.isclose(true_points.squeeze(1), result, atol=1e-8, rtol=1e-8).all()
