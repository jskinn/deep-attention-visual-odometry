import torch
from torch.utils.data import Dataset
from deep_attention_visual_odometry.base_types import CameraViewsAndPoints


class CameraAndParametersDataset(Dataset[CameraViewsAndPoints]):
    """
    A dataset that randomly generates 3D points, and projects them through a random transform and camera matrix
    """

    def __init__(self, epoch_length: int, num_points: int, num_views: int):
        self.epoch_length = int(epoch_length)
        self._num_points = int(num_points)
        self._num_views = int(num_views)
        self._camera_parameter_means = torch.tensor(
            [640.0, 320.0, 240.0], dtype=torch.float32
        )
        self._camera_parameter_std = torch.tensor(
            [100.0, 100.0, 100.0], dtype=torch.float32
        )
        self._camera_distance_mean = 15.0
        self._camera_distance_std = 5.0
        self._camera_location_spread = torch.tensor(
            [3.0, 3.0, 3.0], dtype=torch.float32
        )
        self._camera_target_spread = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
        self._camera_up_spread = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
        self._transform_std = torch.tensor(
            [5.0, 5.0, 5.0, torch.pi / 3, torch.pi / 3, torch.pi / 3],
            dtype=torch.float32,
        )
        self._points_std = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32).reshape(
            1, 3, 1
        )

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, item: int) -> CameraViewsAndPoints:
        if not 0 <= item < len(self):
            raise IndexError(f"Index {item} out of range")

        world_points = self._generate_world_points()
        camera_extrinsics = self._generate_camera_extrinisics(world_points.mean(dim=0))
        camera_intrinsics = self._generate_camera_intrinsics()
        projected_points, projection_weights = self._project_points(
            world_points, camera_extrinsics, camera_intrinsics
        )
        return CameraViewsAndPoints(
            world_points=world_points,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics,
            projected_points=projected_points,
            projection_weights=projection_weights,
        )

    def _generate_world_points(self) -> torch.Tensor:
        world_points = self._points_std * torch.randn(
            self._num_points - 1, 3, dtype=torch.float32
        )
        # Setting zeros in the first 3 points constrains the origin and orientation of the points
        world_points[0, 1] = 0.0
        world_points[0, 2] = 0.0
        world_points[1, 2] = 0.0
        return torch.cat([torch.zeros(1, 3, dtype=torch.float32), world_points])

    def _generate_camera_extrinisics(self, world_centre: torch.Tensor) -> torch.Tensor:
        # TODO:
        # - Pick three points: A location, a view target, and an up direction
        # - Perturb each of those base points for each camera
        # - Orthonomralise (target - location) and (up - location) to produce a rotation matrix
        # This should give us n views with similar but spread out locations,
        # looking approximately through the centre of mass of the points
        camera_direction = torch.randn(3, dtype=torch.float32)
        camera_direction = camera_direction / torch.linalg.vector_norm(
            camera_direction, keep_dim=True
        )
        up_direction = torch.randn(3, dtype=torch.float32)
        up_direction = up_direction / torch.linalg.vector_norm(
            up_direction, keep_dim=True
        )
        camera_distance = (
            self._camera_distance_mean
            + self._camera_distance_std * torch.randn(1, dtype=torch.float32)
        ).abs()
        view_distance = (
            self._camera_distance_mean
            + self._camera_distance_std * torch.randn(1, dtype=torch.float32)
        ).abs()
        up_distance = (
            self._camera_distance_mean
            + self._camera_distance_std * torch.randn(1, dtype=torch.float32)
        ).abs()
        camera_location_base = world_centre + camera_distance * camera_direction
        camera_target_base = world_centre - view_distance * camera_direction
        up_base = camera_location_base + up_distance * up_direction

        # Generate different reference points for each camera
        camera_locations = camera_location_base[None, :] + self._camera_location_spread[
            None, :
        ] * torch.randn(self._num_views, 3, dtype=torch.float32)
        camera_targets = camera_target_base[None, :] + self._camera_target_spread[
            None, :
        ] * torch.randn(self._num_views, 3, dtype=torch.float32)
        camera_up = up_base[None, :] + self._camera_up_spread[None, :] * torch.rand(
            self._num_views, 3, dtype=torch.float32
        )

        # Orthonormalise to produce a rotation matrix
        forward = camera_targets - camera_locations
        up = camera_up - camera_locations
        forward = forward / torch.linalg.vector_norm(forward, keep_dim=True)
        up = up - forward * torch.dot(forward, up).unsqueeze(-1)
        up = up / torch.linalg.vector_norm(up, keep_dim=True)
        left = torch.linalg.cross(forward, up)

        # Assemble 4x4 extrinisics matrices
        transforms = torch.zeros(self._num_views, 4, 4, dtype=torch.float32)
        transforms[:, 3, 3] = 1.0
        transforms[:, 0:3, 0] = -left
        transforms[:, 0:3, 1] = -up
        transforms[:, 0:3, 2] = forward
        transforms[:, 0:3, 3] = camera_locations

        # TODO: This should be easier than linalg.inv,
        #       since we know that R^-1 = R^T
        # Assemble inverse matrices

        return transforms

    def _generate_camera_intrinsics(self) -> torch.Tensor:
        camera_parameters = (
            self._camera_parameter_means + torch.randn(3) * self._camera_parameter_std
        )
        return torch.tensor(
            [
                [
                    [camera_parameters[0], 0.0, camera_parameters[1], 0.0],
                    [0.0, camera_parameters[0], camera_parameters[2], 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        )

    def _project_points(
        self,
        world_points: torch.Tensor,
        camera_extrinisics,
        camera_intrinisics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        augmented_points = torch.cat(
            [world_points, torch.ones_like(world_points[:, 0:1])]
        )
        inv_transforms = torch.linalg.inv(camera_extrinisics)
        camera_relative_points = torch.matmul(inv_transforms, augmented_points)
        projected_points = torch.matmul(camera_intrinisics, camera_relative_points)
        projected_points = projected_points[:, :, 0:2] / projected_points[:, :, 2:3]
        # weights = torch.where(projected_points[])
        return projected_points, torch.ones_like(projected_points[:, :, 0])

    def _generate_transform(self) -> torch.Tensor:
        transform_parameters = torch.randn(6) * self._transform_std
        cos_angles = torch.cos(transform_parameters[3:6])
        sin_angles = torch.sin(transform_parameters[3:6])
        rot_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_angles[0], -sin_angles[0]],
                [0.0, sin_angles[0], cos_angles[0]],
            ]
        )
        rot_y = torch.tensor(
            [
                [cos_angles[1], 0.0, -sin_angles[1]],
                [0, 1.0, 0.0],
                [sin_angles[1], 0.0, cos_angles[1]],
            ]
        )
        rot_z = torch.tensor(
            [
                [cos_angles[1], -sin_angles[1], 0.0],
                [sin_angles[1], cos_angles[1], 0.0],
                [0.0, 0, 1.0],
            ]
        )
        transform = torch.zeros(1, 4, 4)
        transform[0, 0:3, 0:3] = rot_x @ rot_y @ rot_z
        transform[0, 0:3, 3] = transform_parameters[0:3]
        transform[0, 3, 3] = 1.0
        return transform
