import torch
from deep_attention_visual_odometry.geometry.camera_projection import PinholeCameraProjection


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
    projector = PinholeCameraProjection(camera_intrinsics)
    pixel = projector.project_points(point)
    assert pixel.shape == (2,)
    assert pixel[0] == u
    assert pixel[1] == v


def test_project_batch_of_points_and_views():
    camera_intrinsics = torch.tensor([
        [0.899, 0.1, -0.15],
        [1.0101, -0.1, 0.08],
    ], dtype=torch.float64).unsqueeze(1)
    points = torch.tensor([
        [1.0, 1.0, 14.0],
        [1.0, -1.0, 14.0],
        [-1.0, 1.0, 14.0],
        [-1.0, -1.0, 14.0],
    ], dtype=torch.float64).unsqueeze(0)
    expected_pixels = torch.tensor([
        [
            [0.899 * 1.0 / 14.0 + 0.1, 0.899 * 1.0 / 14.0 - 0.15],
            [0.899 * 1.0 / 14.0 + 0.1, 0.899 * -1.0 / 14.0 - 0.15],
            [0.899 * -1.0 / 14.0 + 0.1, 0.899 * 1.0 / 14.0 - 0.15],
            [0.899 * -1.0 / 14.0 + 0.1, 0.899 * -1.0 / 14.0 - 0.15],
        ],[
            [1.0101 * 1.0 / 14.0 - 0.1, 1.0101 * 1.0 / 14.0 + 0.08],
            [1.0101 * 1.0 / 14.0 - 0.1, 1.0101 * -1.0 / 14.0 + 0.08],
            [1.0101 * -1.0 / 14.0 - 0.1, 1.0101 * 1.0 / 14.0 + 0.08],
            [1.0101 * -1.0 / 14.0 - 0.1, 1.0101 * -1.0 / 14.0 + 0.08],
        ]
    ], dtype=torch.float64)
    projector = PinholeCameraProjection(camera_intrinsics)
    pixels = projector.project_points(points)
    assert torch.equal(pixels, expected_pixels)




