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
from transforms3d.axangles import mat2axangle
import torch
from torch.utils.data import Dataset

from deep_attention_visual_odometry.base_types import CameraViewsAndPoints
from deep_attention_visual_odometry.camera_model import get_camera_relative_points
from deep_attention_visual_odometry.geometry import (
    rotate_vector_axis_angle,
project_points_pinhole_homogeneous
)


class CameraAndParametersDataset(Dataset[CameraViewsAndPoints]):
    """
    A dataset that randomly generates 3D points, and projects them through a random transform and camera matrix
    """

    def __init__(
        self,
        epoch_length: int,
        num_points: int,
        num_views: int,
        min_camera_distance: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        self.epoch_length = int(epoch_length)
        self._num_points = int(num_points)
        self._num_views = int(num_views)
        self._min_camera_distance = float(min_camera_distance)
        self._dtype = dtype

        self._camera_distance_mean = 20.0
        self._camera_distance_std = 5.0
        self._camera_location_spread = torch.tensor(
            [3.0, 3.0, 3.0], dtype=self._dtype
        )
        self._camera_target_spread = torch.tensor([3.0, 3.0, 3.0], dtype=self._dtype)
        self._camera_up_spread = torch.tensor([3.0, 3.0, 3.0], dtype=self._dtype)
        self._transform_std = torch.tensor(
            [5.0, 5.0, 5.0, torch.pi / 3, torch.pi / 3, torch.pi / 3],
            dtype=self._dtype,
        )
        self._points_std = torch.tensor([3.0, 3.0, 3.0], dtype=self._dtype).reshape(
            1, 3
        )

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, item: int) -> CameraViewsAndPoints:
        if not 0 <= item < len(self):
            raise IndexError(f"Index {item} out of range")

        world_points = self._generate_world_points()
        camera_translations, camera_rotations = self._generate_camera_extrinisics(world_points)
        camera_intrinsics = self._generate_camera_intrinsics()
        projected_points, visibility_mask = self._project_points(
            world_points, camera_translations, camera_rotations, camera_intrinsics
        )
        return CameraViewsAndPoints(
            world_points=world_points,
            camera_intrinsics=camera_intrinsics,
            camera_orientations=camera_rotations,
            camera_translations=camera_translations,
            projected_points=projected_points,
            visibility_mask=visibility_mask,
        )

    def _generate_world_points(self) -> torch.Tensor:
        """
        Generate world points relative to the first view
        """
        # All points should be in front of the first camera, so Z should be positive.
        z_points = self._camera_distance_mean + self._camera_distance_std * torch.randn(self._num_points, 1, dtype=self._dtype)
        xy_points = self._points_std * torch.randn(
            self._num_points, 2, dtype=self._dtype
        )
        return torch.concatenate([xy_points, z_points.abs()], dim=-1)

    def _generate_camera_extrinisics(self, world_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # - Pick three points: A location, a view target, and an up direction
        # - Perturb each of those base points for each camera
        # - Orthonomralise (target - location) and (up - location) to produce a rotation matrix
        # This should give us n views with similar but spread out locations,
        # looking approximately through the centre of mass of the points
        up_distance = (
            self._camera_distance_mean
            + self._camera_distance_std * torch.randn(1, dtype=self._dtype)
        ).abs()
        camera_target_base = world_points.mean(dim=0) * (1.0 + torch.rand()) + (self._points_std / 2.0) * torch.randn(3, dtype=self._dtype)
        up_base = up_distance * torch.tensor([0.0, -1.0, 0.0], dtype=self._dtype)

        # Generate different reference points for each camera
        camera_locations = self._camera_location_spread[None, :] * torch.randn(self._num_views, 3, dtype=self._dtype)
        camera_targets = camera_target_base[None, :] + self._camera_target_spread[
            None, :
        ] * torch.randn(self._num_views, 3, dtype=self._dtype)
        camera_up = up_base[None, :] + self._camera_up_spread[None, :] * torch.randn(
            self._num_views, 3, dtype=self._dtype
        )

        # Orthonormalise to produce a rotation matrix
        forward = camera_targets - camera_locations
        up = camera_up - camera_locations
        forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
        up = up - forward * (forward * up).sum(dim=-1, keepdim=True)
        up = up / torch.linalg.vector_norm(up, dim=-1, keepdim=True)
        left = torch.linalg.cross(forward, up)

        # Make rotation matrices in numpy, and convert them to axis-angle form.
        rotation_matrix = torch.stack([-left, -up, forward], dim=-2).numpy()
        axis, angle = mat2axangle(rotation_matrix)
        angle = torch.tensor(angle)
        axis = torch.tensor(axis)
        camera_rotations = angle * axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True)

        # Check that the points are all at least the minimum distance away
        z_distances = (
            forward[:, None, :]
            * (world_points[None, :, :] - camera_locations[:, None, :])
        ).sum(dim=-1)
        z_distances = z_distances - self._min_camera_distance
        z_distances, _ = torch.min(z_distances, dim=-1)
        z_distances = torch.where(
            z_distances < 1e-3, z_distances, torch.zeros_like(z_distances)
        )
        camera_locations = camera_locations - z_distances[:, None] * forward

        return camera_locations, camera_rotations

    def _generate_camera_intrinsics(self) -> torch.Tensor:
        angle = 3 * torch.pi / 18 + (9 * torch.pi / 18) * torch.rand(1)
        image_centre = (0.2 * torch.randn(2)).clamp(min=-0.5, max=0.5)
        focal_length = 1.0 / torch.tan(angle / 2.0)
        return torch.tensor([focal_length, image_centre[0], image_centre[1]])

    def _project_points(
        self,
        world_points: torch.Tensor,
        camera_translations: torch.Tensor,
        camera_rotations: torch.Tensor,
        camera_intrinisics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_translations = torch.cat([
            torch.zeros()
            camera_translations,
        ], dim=0)
        camera_rotations = torch.cat()
        camera_relative_points = get_camera_relative_points(
            world_points=world_points[],
            camera_translations=camera_translations,
            camera_rotations=camera_rotations,
        )
        homogen

        augmented_points = torch.cat(
            [world_points, torch.ones_like(world_points[:, 0:1])], dim=1
        )
        # This only works because there is no scale component to the extrinsics
        inv_rotations = camera_extrinisics[:, 0:3, 0:3].transpose(-2, -1)
        inv_translations = -1.0 * torch.matmul(
            inv_rotations, camera_extrinisics[:, 0:3, 3:4]
        )
        inv_transforms = torch.cat([inv_rotations, inv_translations], dim=-1)
        # inv_transforms = torch.linalg.inv(camera_extrinisics)
        camera_relative_points = torch.matmul(
            inv_transforms[:, None, :, :], augmented_points[None, :, :, None]
        )
        z_distances = camera_relative_points[:, :, 2, 0].clamp(min=1e-8)
        projected_u = (
            camera_intrinisics[0] * camera_relative_points[:, :, 0, 0] / z_distances
            + camera_intrinisics[1]
        )
        projected_v = (
            camera_intrinisics[0] * camera_relative_points[:, :, 1, 0] / z_distances
            + camera_intrinisics[2]
        )
        visibility = torch.logical_and(
            torch.logical_and(projected_u > -1.0, projected_u < 1.0),
            torch.logical_and(projected_v > -1.0, projected_v < 1.0),
        )
        return (
            torch.cat([projected_u.unsqueeze(-1), projected_v.unsqueeze(-1)], dim=-1),
            visibility,
        )
