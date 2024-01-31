import math
import pytest
import torch
import torch.nn as nn
from deep_attention_visual_odometry.solvers.camera_model import SimpleCameraModel
from deep_attention_visual_odometry.solvers.camera_model.lie_rotation import LieRotation


@pytest.fixture()
def models_and_mask() -> (
    tuple[SimpleCameraModel, SimpleCameraModel, torch.Tensor, SimpleCameraModel]
):
    focal_length = torch.tensor([340, 600])
    cx = torch.tensor([320, 420])
    cy = torch.tensor([240, 300])
    angle = math.pi / 180
    axis = torch.tensor(
        [[0.01, 0.01, math.sqrt(1.0 - 2e-4)], [0.01, math.sqrt(1.0 - 3e-4), 0.02]]
    )
    translation = torch.tensor([[-0.1, 0.3, 3.0], [0.2, -0.2, 7.0]])
    orientation = LieRotation((axis * angle))
    point = torch.tensor([[0.2, -0.2, 0.1], [-0.1, -0.1, -0.6]])
    camera_relative_point = orientation.rotate_vector(point) + translation
    expected_u = (
        focal_length * camera_relative_point[:, 0] / camera_relative_point[:, 2] + cx
    )
    expected_v = (
        focal_length * camera_relative_point[:, 1] / camera_relative_point[:, 2] + cy
    )
    camera_model_1 = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length[0]], [focal_length[0]]]),
        cx=torch.tensor([[cx[0]], [cx[0]]]),
        cy=torch.tensor([[cy[0]], [cy[0]]]),
        translation=translation[0, :].reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        orientation=LieRotation(
            (axis[0, :] * angle).reshape(1, 1, 1, 1, 3).tile(2, 1, 1, 1, 1)
        ),
        world_points=point[0, :].reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        true_projected_points=torch.tensor(
            [
                [expected_u[0] + 5.0, expected_v[0] - 7.0],
                [expected_u[0] - 8.0, expected_v[0] + 3.0],
            ]
        ).reshape(2, 1, 1, 1, 2),
    )
    camera_model_2 = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length[1]], [focal_length[1]]]),
        cx=torch.tensor([[cx[1]], [cx[1]]]),
        cy=torch.tensor([[cy[1]], [cy[1]]]),
        translation=translation[1, :].reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        orientation=LieRotation(
            (axis[1, :] * angle).reshape(1, 1, 1, 1, 3).tile(2, 1, 1, 1, 1)
        ),
        world_points=point[1, :].reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        true_projected_points=torch.tensor(
            [
                [expected_u[1] + 2.1, expected_v[1] + 4.6],
                [expected_u[1] + 9.0, expected_v[1] + 2.0],
            ]
        ).reshape(2, 1, 1, 1, 2),
    )
    update_mask = torch.tensor([[False], [True]])
    camera_model_3 = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length[0]], [focal_length[1]]]),
        cx=torch.tensor([[cx[0]], [cx[1]]]),
        cy=torch.tensor([[cy[0]], [cy[1]]]),
        translation=translation.reshape(2, 1, 1, 3),
        orientation=LieRotation((axis * angle).reshape(2, 1, 1, 1, 3)),
        world_points=point.reshape(2, 1, 1, 3),
        true_projected_points=torch.tensor(
            [
                [expected_u[0] + 5.0, expected_v[0] - 7.0],
                [expected_u[1] + 9.0, expected_v[1] + 2.0],
            ]
        ).reshape(2, 1, 1, 1, 2),
    )
    return camera_model_1, camera_model_2, update_mask, camera_model_3


def test_num_parameters_scales_with_num_views_and_num_points():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    camera_model = SimpleCameraModel(
        focal_length=340 * torch.ones(batch_size, num_estimates),
        cx=320 * torch.ones(batch_size, num_estimates),
        cy=240 * torch.ones(batch_size, num_estimates),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        orientation=LieRotation(torch.zeros(batch_size, num_estimates, num_views, 3)),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        ).reshape(batch_size, num_estimates, num_points, 3),
        true_projected_points=torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        ).reshape(batch_size, num_estimates, num_views, num_points, 2),
    )
    assert camera_model.num_parameters == 3 + 6 * num_views + 3 * num_points


def test_get_error_return_shape():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    camera_model = SimpleCameraModel(
        focal_length=340 * torch.ones(batch_size, num_estimates),
        cx=320 * torch.ones(batch_size, num_estimates),
        cy=240 * torch.ones(batch_size, num_estimates),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        orientation=LieRotation(
            torch.zeros(batch_size, num_estimates, num_views, 1, 3)
        ),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        )
        .reshape(batch_size, num_estimates, num_points, 3)
        .to(torch.float),
        true_projected_points=torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        ).reshape(batch_size, num_estimates, num_views, num_points, 2),
    )
    error = camera_model.get_error()
    assert error.shape == (batch_size, num_estimates)


def test_get_error_returns_square_error_between_estimated_and_true_projected_points():
    focal_length = 340
    cx = 320
    cy = 240
    translation = torch.tensor([-0.1, 0.3, 3.0])
    axis = torch.tensor([0.5, -0.3, 0.5])
    axis = axis / torch.linalg.norm(axis)
    angle = math.pi / 16
    orientation = LieRotation(axis * angle)
    point = torch.tensor([0.2, -0.2, 0.1])
    camera_relative_point = orientation.rotate_vector(point) + translation
    expected_u = focal_length * camera_relative_point[0] / camera_relative_point[2] + cx
    expected_v = focal_length * camera_relative_point[1] / camera_relative_point[2] + cy
    expected_error_u = 10.0
    expected_error_v = 7.0
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        translation=translation.reshape(1, 1, 1, 3),
        orientation=LieRotation(angle * axis.reshape(1, 1, 1, 1, 3)),
        world_points=point.reshape(1, 1, 1, 3),
        true_projected_points=torch.tensor(
            [expected_u + expected_error_u, expected_v + expected_error_v]
        ).reshape(1, 1, 1, 1, 2),
    )
    error = camera_model.get_error()
    assert error.shape == (1, 1)
    assert torch.isclose(
        error[0, 0],
        torch.tensor(
            expected_error_u * expected_error_u + expected_error_v * expected_error_v
        ),
    )


def test_camera_projection_clips_negative_z_to_minimum_distance():
    minimum_distance = 2.5e-3
    focal_length = 340
    cx = 320
    cy = 240
    translation = torch.tensor([-0.1, 0.3, -3.0])
    point = torch.tensor([0.2, -0.2, 0.1])
    camera_relative_point = point + translation
    expected_u = focal_length * camera_relative_point[0] / minimum_distance + cx
    expected_v = focal_length * camera_relative_point[1] / minimum_distance + cy
    expected_error_u = 8.0
    expected_error_v = 7.0
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        translation=translation.reshape(1, 1, 1, 3),
        orientation=LieRotation(torch.zeros(1, 1, 1, 1, 3)),
        world_points=point.reshape(1, 1, 1, 3),
        true_projected_points=torch.tensor(
            [expected_u + expected_error_u, expected_v + expected_error_v]
        ).reshape(1, 1, 1, 1, 2),
        minimum_distance=minimum_distance,
    )
    error = camera_model.get_error()
    assert error.shape == (1, 1)
    assert torch.isclose(
        error[0, 0],
        torch.tensor(
            expected_error_u * expected_error_u + expected_error_v * expected_error_v
        ),
    )


def test_get_error_caches_returned_tensor():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    camera_model = SimpleCameraModel(
        focal_length=340 * torch.ones(batch_size, num_estimates),
        cx=320 * torch.ones(batch_size, num_estimates),
        cy=240 * torch.ones(batch_size, num_estimates),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        orientation=LieRotation(
            torch.tensor([0.01, -0.02, 0.1])
            .reshape(1, 1, 1, 1, 3)
            .tile(batch_size, num_estimates, num_views, 1, 1)
        ),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        )
        .reshape(batch_size, num_estimates, num_points, 3)
        .to(torch.float),
        true_projected_points=torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        ).reshape(batch_size, num_estimates, num_views, num_points, 2),
    )
    error = camera_model.get_error()
    error2 = camera_model.get_error()
    assert error is error2


def test_masked_update_produces_same_error_with_no_pre_computation(models_and_mask):
    (
        camera_model_1,
        camera_model_2,
        update_mask,
        camera_model_3,
    ) = models_and_mask

    updated_camera_model = camera_model_1.masked_update(camera_model_2, update_mask)

    updated_error = updated_camera_model.get_error()
    raw_error = camera_model_3.get_error()
    assert torch.equal(updated_error, raw_error)


def test_masked_update_produces_same_error_with_model_1_pre_computed(models_and_mask):
    (
        camera_model_1,
        camera_model_2,
        update_mask,
        camera_model_3,
    ) = models_and_mask
    _ = camera_model_1.get_error()

    updated_camera_model = camera_model_1.masked_update(camera_model_2, update_mask)

    updated_error = updated_camera_model.get_error()
    raw_error = camera_model_3.get_error()
    assert torch.equal(updated_error, raw_error)


def test_masked_update_produces_same_error_with_model_2_pre_computed(models_and_mask):
    (
        camera_model_1,
        camera_model_2,
        update_mask,
        camera_model_3,
    ) = models_and_mask
    _ = camera_model_2.get_error()

    updated_camera_model = camera_model_1.masked_update(camera_model_2, update_mask)

    updated_error = updated_camera_model.get_error()
    raw_error = camera_model_3.get_error()
    assert torch.equal(updated_error, raw_error)


def test_masked_update_produces_same_error_with_both_models_pre_computed(
    models_and_mask,
):
    (
        camera_model_1,
        camera_model_2,
        update_mask,
        camera_model_3,
    ) = models_and_mask
    _ = camera_model_1.get_error()
    _ = camera_model_2.get_error()

    updated_camera_model = camera_model_1.masked_update(camera_model_2, update_mask)

    updated_error = updated_camera_model.get_error()
    raw_error = camera_model_3.get_error()
    assert torch.equal(updated_error, raw_error)


def test_get_gradient_return_shape():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    camera_model = SimpleCameraModel(
        focal_length=340 * torch.ones(batch_size, num_estimates),
        cx=320 * torch.ones(batch_size, num_estimates),
        cy=240 * torch.ones(batch_size, num_estimates),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        orientation=LieRotation(
            torch.tensor([0.2, -0.02, -0.004])
            .reshape(1, 1, 1, 1, 3)
            .tile(batch_size, num_estimates, num_views, 1, 1)
        ),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        )
        .reshape(batch_size, num_estimates, num_points, 3)
        .to(torch.float),
        true_projected_points=torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        ).reshape(batch_size, num_estimates, num_views, num_points, 2),
    )
    gradient = camera_model.get_gradient()
    assert gradient.shape == (batch_size, num_estimates, camera_model.num_parameters)


def test_gradient_is_zero_for_correct_estimate():
    focal_length = 340
    cx = 320
    cy = 240
    axis = torch.tensor([0.5, -0.3, 0.5])
    axis = axis / torch.linalg.norm(axis)
    angle = math.pi / 16
    translation = torch.tensor([-0.1, 0.3, 3.0])
    orientation = LieRotation(angle * axis)
    point = torch.tensor([0.2, -0.2, 0.1])
    camera_relative_point = orientation.rotate_vector(point) + translation
    expected_u = focal_length * camera_relative_point[0] / camera_relative_point[2] + cx
    expected_v = focal_length * camera_relative_point[1] / camera_relative_point[2] + cy
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        translation=translation.reshape(1, 1, 1, 3),
        orientation=LieRotation(angle * axis.reshape(1, 1, 1, 1, 3)),
        world_points=point.reshape(1, 1, 1, 3),
        true_projected_points=torch.tensor([expected_u, expected_v]).reshape(
            1, 1, 1, 1, 2
        ),
    )
    gradient = camera_model.get_gradient()
    assert gradient.shape == (1, 1, 12)
    assert torch.all(gradient.abs() < 1e-8)


def test_cx_gradient_is_high_if_incorrect():
    num_points = 5
    num_views = 3
    focal_length = 340
    cx = 320
    true_cx = 300
    cy = 240
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    angles = torch.tensor([[math.pi / 36], [0.0], [-math.pi / 36]])
    orientations = LieRotation((angles * axis).reshape(3, 1, 3))
    translations = torch.tensor(
        [
            [-0.1, 0.3, 8.0],
            [0.2, 0.2, 8.0],
            [0.3, -0.1, 8.2],
        ]
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.1],
            [0.2, -0.2, 0.1],
            [-0.2, 0.2, 0.1],
            [-0.2, -0.2, 0.1],
        ]
    )
    camera_relative_points = (
        orientations.rotate_vector(points[None, :, :]) + translations[:, None, :]
    )
    expected_u = (
        focal_length * camera_relative_points[:, :, 0] / camera_relative_points[:, :, 2]
        + true_cx
    )
    expected_v = (
        focal_length * camera_relative_points[:, :, 1] / camera_relative_points[:, :, 2]
        + cy
    )
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        translation=translations.reshape(1, 1, num_views, 3),
        orientation=LieRotation((angles * axis).reshape(1, 1, num_views, 1, 3)),
        world_points=points.reshape(1, 1, num_points, 3),
        true_projected_points=torch.cat(
            [expected_u[:, :, None], expected_v[:, :, None]], dim=2
        ).reshape(1, 1, num_views, num_points, 2),
    )
    gradient = camera_model.get_gradient()
    assert gradient.shape == (1, 1, 3 + 6 * num_views + 3 * num_points)
    assert gradient[0, 0, 0] > 1.0


def test_cy_gradient_is_high_if_incorrect():
    num_points = 5
    num_views = 3
    focal_length = 340
    cx = 320
    cy = 240
    true_cy = 260
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    angles = torch.tensor([[math.pi / 36], [0.0], [-math.pi / 36]])
    orientations = LieRotation((angles * axis).reshape(3, 1, 3))
    translations = torch.tensor(
        [
            [-0.1, 0.3, 8.0],
            [0.2, 0.2, 8.0],
            [0.3, -0.1, 8.2],
        ]
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.1],
            [0.2, -0.2, 0.1],
            [-0.2, 0.2, 0.1],
            [-0.2, -0.2, 0.1],
        ]
    )
    camera_relative_points = (
        orientations.rotate_vector(points[None, :, :]) + translations[:, None, :]
    )
    expected_u = (
        focal_length * camera_relative_points[:, :, 0] / camera_relative_points[:, :, 2]
        + cx
    )
    expected_v = (
        focal_length * camera_relative_points[:, :, 1] / camera_relative_points[:, :, 2]
        + true_cy
    )
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        translation=translations.reshape(1, 1, num_views, 3),
        orientation=LieRotation((angles * axis).reshape(1, 1, num_views, 1, 3)),
        world_points=points.reshape(1, 1, num_points, 3),
        true_projected_points=torch.cat(
            [expected_u[:, :, None], expected_v[:, :, None]], dim=2
        ).reshape(1, 1, num_views, num_points, 2),
    )
    gradient = camera_model.get_gradient()
    assert gradient.shape == (1, 1, 3 + 6 * num_views + 3 * num_points)
    assert gradient[0, 0, 1] < -1.0


def test_fx_gradient_is_high_if_incorrect():
    num_points = 5
    num_views = 3
    focal_length = 340
    true_focal_length = 400
    cx = 320
    cy = 240
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    angles = torch.tensor([[math.pi / 36], [0.0], [-math.pi / 36]])
    orientations = LieRotation((angles * axis).reshape(3, 1, 3))
    translations = torch.tensor(
        [
            [-0.1, 0.3, 8.0],
            [0.2, 0.2, 8.0],
            [0.3, -0.1, 8.2],
        ]
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.1],
            [0.2, -0.2, 0.1],
            [-0.2, 0.2, 0.1],
            [-0.2, -0.2, 0.1],
        ]
    )
    camera_relative_points = (
        orientations.rotate_vector(points[None, :, :]) + translations[:, None, :]
    )
    expected_u = (
        true_focal_length
        * camera_relative_points[:, :, 0]
        / camera_relative_points[:, :, 2]
        + cx
    )
    expected_v = (
        true_focal_length
        * camera_relative_points[:, :, 1]
        / camera_relative_points[:, :, 2]
        + cy
    )
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        translation=translations.reshape(1, 1, num_views, 3),
        orientation=LieRotation((angles * axis).reshape(1, 1, num_views, 1, 3)),
        world_points=points.reshape(1, 1, num_points, 3),
        true_projected_points=torch.cat(
            [expected_u[:, :, None], expected_v[:, :, None]], dim=2
        ).reshape(1, 1, num_views, num_points, 2),
    )
    gradient = camera_model.get_gradient()
    assert gradient.shape == (1, 1, 3 + 6 * num_views + 3 * num_points)
    assert gradient[0, 0, 2] < -1.0


def test_orientation_x_gradient_is_high_if_incorrect():
    num_points = 5
    num_views = 3
    focal_length = 340
    cx = 320
    cy = 240
    true_orientations = torch.tensor(
        [
            [-0.1, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
    )
    estimated_orientations = torch.tensor(
        [
            [-0.15, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    orientations = LieRotation(true_orientations.reshape(3, 1, 3))
    translations = torch.tensor(
        [
            [-0.1, 0.3, 8.0],
            [0.2, 0.2, 8.0],
            [0.3, -0.1, 8.2],
        ]
    )
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.2, 0.1],
            [0.2, -0.2, 0.1],
            [-0.2, 0.2, 0.1],
            [-0.2, -0.2, 0.1],
        ]
    )
    camera_relative_points = (
        orientations.rotate_vector(points[None, :, :]) + translations[:, None, :]
    )
    expected_u = (
        focal_length * camera_relative_points[:, :, 0] / camera_relative_points[:, :, 2]
        + cx
    )
    expected_v = (
        focal_length * camera_relative_points[:, :, 1] / camera_relative_points[:, :, 2]
        + cy
    )
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        translation=translations.reshape(1, 1, num_views, 3),
        orientation=LieRotation(estimated_orientations.reshape(1, 1, num_views, 1, 3)),
        world_points=points.reshape(1, 1, num_points, 3),
        true_projected_points=torch.cat(
            [expected_u[:, :, None], expected_v[:, :, None]], dim=2
        ).reshape(1, 1, num_views, num_points, 2),
    )
    gradient = camera_model.get_gradient()
    assert gradient.shape == (1, 1, 3 + 6 * num_views + 3 * num_points)
    assert gradient[0, 0, 3] < -1.0


def test_tensor_gradient_passes_from_error_to_inputs():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    focal_length = (340 * torch.ones(batch_size, num_estimates)).requires_grad_()
    cx = (320 * torch.ones(batch_size, num_estimates)).requires_grad_()
    cy = (240 * torch.ones(batch_size, num_estimates)).requires_grad_()
    translation = (
        torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1)
        .requires_grad_()
    )
    orientation = (
        torch.tensor([0.2, -0.02, -0.004])
        .reshape(1, 1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1, 1)
        .requires_grad_()
    )
    world_points = (
        torch.tensor(list(range(batch_size * num_estimates * num_points * 3)))
        .reshape(batch_size, num_estimates, num_points, 3)
        .to(torch.float)
        .requires_grad_()
    )
    true_projected_points = (
        torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        )
        .reshape(batch_size, num_estimates, num_views, num_points, 2)
        .to(torch.float)
        .requires_grad_()
    )
    camera_model = SimpleCameraModel(
        focal_length=focal_length,
        cx=cx,
        cy=cy,
        translation=translation,
        orientation=LieRotation(orientation),
        world_points=world_points,
        true_projected_points=true_projected_points,
    )
    error = camera_model.get_error()
    assert error.requires_grad is True
    assert error.grad_fn is not None
    loss = error.square().sum()
    loss.backward()
    assert focal_length.grad is not None
    assert torch.all(torch.greater(torch.abs(focal_length.grad), 0))
    assert focal_length.grad is not None
    assert torch.all(torch.greater(torch.abs(focal_length.grad), 0))
    assert cx.grad is not None
    assert torch.all(torch.greater(torch.abs(cx.grad), 0))
    assert cy.grad is not None
    assert torch.all(torch.greater(torch.abs(cy.grad), 0))
    assert translation.grad is not None
    assert torch.all(torch.greater(torch.abs(translation.grad), 0))
    assert orientation.grad is not None
    assert torch.all(torch.greater(torch.abs(orientation.grad), 0))
    assert world_points.grad is not None
    assert torch.all(torch.greater(torch.abs(true_projected_points.grad), 0))


def test_tensor_gradient_passes_from_gradient_to_inputs():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    focal_length = (340 * torch.ones(batch_size, num_estimates)).requires_grad_()
    cx = (320 * torch.ones(batch_size, num_estimates)).requires_grad_()
    cy = (240 * torch.ones(batch_size, num_estimates)).requires_grad_()
    translation = (
        torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1)
        .requires_grad_()
    )
    orientation = (
        torch.tensor([0.2, -0.02, -0.004])
        .reshape(1, 1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1, 1)
        .requires_grad_()
    )
    world_points = (
        torch.tensor(list(range(batch_size * num_estimates * num_points * 3)))
        .reshape(batch_size, num_estimates, num_points, 3)
        .to(torch.float)
        .requires_grad_()
    )
    true_projected_points = (
        torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        )
        .reshape(batch_size, num_estimates, num_views, num_points, 2)
        .to(torch.float)
        .requires_grad_()
    )
    camera_model = SimpleCameraModel(
        focal_length=focal_length,
        cx=cx,
        cy=cy,
        translation=translation,
        orientation=LieRotation(orientation),
        world_points=world_points,
        true_projected_points=true_projected_points,
    )
    gradient = camera_model.get_gradient()
    assert gradient.requires_grad is True
    assert gradient.grad_fn is not None
    loss = gradient.square().sum()
    loss.backward()
    assert focal_length.grad is not None
    assert torch.all(torch.greater(torch.abs(focal_length.grad), 0))
    assert focal_length.grad is not None
    assert torch.all(torch.greater(torch.abs(focal_length.grad), 0))
    assert cx.grad is not None
    assert torch.all(torch.greater(torch.abs(cx.grad), 0))
    assert cy.grad is not None
    assert torch.all(torch.greater(torch.abs(cy.grad), 0))
    assert translation.grad is not None
    assert torch.all(torch.greater(torch.abs(translation.grad), 0))
    assert orientation.grad is not None
    assert torch.all(torch.greater(torch.abs(orientation.grad), 0))
    assert world_points.grad is not None
    assert torch.all(torch.greater(torch.abs(true_projected_points.grad), 0))


class CameraModule(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        num_estimates = x.size(1)
        num_views = 5
        num_points = 11
        camera_model = SimpleCameraModel(
            focal_length=x,
            cx=320 * torch.ones(batch_size, num_estimates),
            cy=240 * torch.ones(batch_size, num_estimates),
            translation=torch.tensor([-1.0, 2.0, -0.3])
            .reshape(1, 1, 1, 3)
            .tile(batch_size, num_estimates, num_views, 1),
            orientation=LieRotation(
                torch.tensor([0.2, -0.02, -0.004])
                .reshape(1, 1, 1, 1, 3)
                .tile(batch_size, num_estimates, num_views, 1, 1)
            ),
            world_points=torch.tensor(
                list(range(batch_size * num_estimates * num_points * 3))
            )
            .reshape(batch_size, num_estimates, num_points, 3)
            .to(torch.float),
            true_projected_points=torch.tensor(
                list(range(batch_size * num_estimates * num_views * num_points * 2))
            ).reshape(batch_size, num_estimates, num_views, num_points, 2),
        )
        error = camera_model.get_error()
        gradient = camera_model.get_gradient()
        return error, gradient


def test_can_be_compiled():
    focal_length = 340 * torch.ones(3, 7)
    rotation_comp = torch.compile(CameraModule())
    error, gradient = rotation_comp(focal_length)
    assert error.shape == (3, 7)
    assert gradient.shape == (3, 7, 3 + 6 * 5 + 3 * 11)
