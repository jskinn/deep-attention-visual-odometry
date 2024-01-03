import pytest
import torch
from deep_attention_visual_odometry.solvers.camera_model import SimpleCameraModel


@pytest.fixture()
def models_and_mask(
) -> tuple[SimpleCameraModel, SimpleCameraModel, torch.Tensor, SimpleCameraModel]:
    focal_length = torch.tensor([340, 600])
    cx = torch.tensor([320, 420])
    cy = torch.tensor([240, 300])
    right = torch.tensor([1.0, 0.0, 0.0])
    down = torch.tensor([0.0, 1.0, 0.0])
    translation = torch.tensor([[-0.1, 0.3, 3.0], [0.2, -0.2, 7.0]])
    point = torch.tensor([[0.2, -0.2, 0.1], [-0.1, -0.1, -0.6]])
    camera_relative_point = point + translation
    expected_u = (
            focal_length * camera_relative_point[:, 0] / camera_relative_point[:, 2]
            + cx
    )
    expected_v = (
            focal_length * camera_relative_point[:, 1] / camera_relative_point[:, 2]
            + cy
    )
    camera_model_1 = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length[0]], [focal_length[0]]]),
        cx=torch.tensor([[cx[0]], [cx[0]]]),
        cy=torch.tensor([[cy[0]], [cy[0]]]),
        a=right.reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        b=down.reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        translation=translation[0, :].reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
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
        a=right.reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        b=down.reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        translation=translation[1, :].reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
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
        a=right.reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        b=down.reshape(1, 1, 1, 3).tile(2, 1, 1, 1),
        translation=translation.reshape(2, 1, 1, 3),
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
        a=torch.tensor([1.0, 0.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        b=torch.tensor([0.0, 1.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        ).reshape(batch_size, num_estimates, num_points, 3),
        true_projected_points=torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        ).reshape(batch_size, num_estimates, num_views, num_points, 2),
    )
    assert camera_model.num_parameters == 3 + 9 * num_views + 3 * num_points


def test_get_error_return_shape():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    camera_model = SimpleCameraModel(
        focal_length=340 * torch.ones(batch_size, num_estimates),
        cx=320 * torch.ones(batch_size, num_estimates),
        cy=240 * torch.ones(batch_size, num_estimates),
        a=torch.tensor([1.0, 0.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        b=torch.tensor([0.0, 1.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        ).reshape(batch_size, num_estimates, num_points, 3),
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
    right = torch.tensor([1.0, 0.0, 0.0])
    down = torch.tensor([0.0, 1.0, 0.0])
    translation = torch.tensor([-0.1, 0.3, 3.0])
    point = torch.tensor([0.2, -0.2, 0.1])
    camera_relative_point = point + translation
    expected_u = (
            focal_length * camera_relative_point[0] / camera_relative_point[2] + cx
    )
    expected_v = (
            focal_length * camera_relative_point[1] / camera_relative_point[2] + cy
    )
    expected_error_u = 10.0
    expected_error_v = 7.0
    camera_model = SimpleCameraModel(
        focal_length=torch.tensor([[focal_length]]),
        cx=torch.tensor([[cx]]),
        cy=torch.tensor([[cy]]),
        a=right.reshape(1, 1, 1, 3),
        b=down.reshape(1, 1, 1, 3),
        translation=translation.reshape(1, 1, 1, 3),
        world_points=point.reshape(1, 1, 1, 3),
        true_projected_points=torch.tensor(
            [expected_u + expected_error_u, expected_v + expected_error_v]
        ).reshape(1, 1, 1, 1, 2),
    )
    error = camera_model.get_error()
    assert error.shape == (1, 1)
    assert error[0, 0] == expected_error_u * expected_error_u + expected_error_v * expected_error_v


def test_camera_projection_clips_negative_z_to_minimum_distance(
):
    minimum_distance = 2.5e-3
    focal_length = 340
    cx = 320
    cy = 240
    right = torch.tensor([1.0, 0.0, 0.0])
    down = torch.tensor([0.0, 1.0, 0.0])
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
        a=right.reshape(1, 1, 1, 3),
        b=down.reshape(1, 1, 1, 3),
        translation=translation.reshape(1, 1, 1, 3),
        world_points=point.reshape(1, 1, 1, 3),
        true_projected_points=torch.tensor(
            [expected_u + expected_error_u, expected_v + expected_error_v]
        ).reshape(1, 1, 1, 1, 2),
        minimum_distance=minimum_distance,
    )
    error = camera_model.get_error()
    assert error.shape == (1, 1)
    assert error[0, 0] == expected_error_u * expected_error_u + expected_error_v * expected_error_v


def test_get_error_caches_returned_tensor():
    batch_size = 3
    num_estimates = 7
    num_views = 5
    num_points = 11
    camera_model = SimpleCameraModel(
        focal_length=340 * torch.ones(batch_size, num_estimates),
        cx=320 * torch.ones(batch_size, num_estimates),
        cy=240 * torch.ones(batch_size, num_estimates),
        a=torch.tensor([1.0, 0.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        b=torch.tensor([0.0, 1.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        ).reshape(batch_size, num_estimates, num_points, 3),
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


def test_masked_update_produces_same_error_with_both_models_pre_computed(models_and_mask):
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
        a=torch.tensor([1.0, 0.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        b=torch.tensor([0.0, 1.0, 0.0])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        translation=torch.tensor([-1.0, 2.0, -0.3])
        .reshape(1, 1, 1, 3)
        .tile(batch_size, num_estimates, num_views, 1),
        world_points=torch.tensor(
            list(range(batch_size * num_estimates * num_points * 3))
        ).reshape(batch_size, num_estimates, num_points, 3),
        true_projected_points=torch.tensor(
            list(range(batch_size * num_estimates * num_views * num_points * 2))
        ).reshape(batch_size, num_estimates, num_views, num_points, 2),
    )
    gradient = camera_model.get_gradient()
    assert gradient.shape == (batch_size, num_estimates, camera_model.num_parameters)
