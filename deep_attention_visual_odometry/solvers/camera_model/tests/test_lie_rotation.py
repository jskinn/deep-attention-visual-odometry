import math
import pytest
import torch
import torch.nn as nn
from deep_attention_visual_odometry.solvers.camera_model.lie_rotation import LieRotation


@pytest.fixture(
    params=[
        (0.1, 0.1, 0.1),
        (-0.1, -0.1, -0.1),
        (0.1, 0.0, 0.0),
        (0.0, 0.1, 0.0),
        (0.0, 0.0, 0.1),
        (-0.1, 0.0, 0.0),
        (0.0, -0.1, 0.0),
        (0.0, 0.0, -0.1),
    ]
)
def delta(request) -> torch.Tensor:
    return torch.tensor(request.param)


def test_simple_x_rotation():
    vector = torch.tensor([-1.0, 2.0, 3.0])
    theta = math.pi / 6
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation = LieRotation(torch.tensor([theta, 0.0, 0.0]))
    result = rotation.rotate_vector(vector)
    assert torch.all(
        torch.isclose(
            result,
            torch.tensor(
                [
                    -1.0,
                    2.0 * cos_theta - 3.0 * sin_theta,
                    3.0 * cos_theta + 2.0 * sin_theta,
                ]
            ),
        )
    )


def test_simple_y_rotation():
    vector = torch.tensor([-1.0, 2.0, 3.0])
    theta = 5 * math.pi / 16
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation = LieRotation(torch.tensor([0.0, theta, 0.0]))
    result = rotation.rotate_vector(vector)
    assert torch.all(
        torch.isclose(
            result,
            torch.tensor(
                [
                    -1.0 * cos_theta + 3.0 * sin_theta,
                    2.0,
                    3.0 * cos_theta + 1.0 * sin_theta,
                ]
            ),
        )
    )


def test_simple_z_rotation():
    vector = torch.tensor([-1.0, 2.0, 3.0])
    theta = 3 * math.pi / 16
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation = LieRotation(torch.tensor([0.0, 0.0, theta]))
    result = rotation.rotate_vector(vector)
    assert torch.all(
        torch.isclose(
            result,
            torch.tensor(
                [
                    -1.0 * cos_theta - 2.0 * sin_theta,
                    2.0 * cos_theta - 1.0 * sin_theta,
                    3.0,
                ]
            ),
        )
    )


def test_zero_rotation_doesnt_change_vector():
    vector = torch.tensor([-2.1, 0.7, 1.87])
    rotation = LieRotation(torch.tensor([0.0, 0.0, 0.0]))
    result = rotation.rotate_vector(vector)
    assert torch.equal(vector, result)


def test_rotated_vector_parallel_to_axis_doesnt_change():
    vector = torch.tensor([1.0, 2.0, 3.0])
    rotation = LieRotation(torch.tensor([0.2, 0.4, 0.6]))
    result = rotation.rotate_vector(vector)
    assert torch.equal(vector, result)


def test_rotation_preserves_vector_length():
    vector = torch.tensor([-2.1, 0.7, 1.87])
    rotation = LieRotation(torch.tensor([0.5, -0.5, 0.25]))
    result = rotation.rotate_vector(vector)
    assert torch.abs(torch.linalg.norm(vector) - torch.linalg.norm(result)) < 1e-6


def test_from_quaternion_rotates_the_same():
    vector = torch.tensor([-6.5, 3.1, 0.8])
    quaternion = torch.tensor([math.sqrt(0.86), 0.1, -0.2, 0.3])
    # Calculated using transforms3d
    expected_result = torch.tensor([-6.90764883, -1.12108911, -1.87817646])
    rotation = LieRotation.from_quaternion(quaternion)
    result = rotation.rotate_vector(vector)
    assert torch.all(torch.isclose(expected_result, result))


def test_rotate_vector_handles_multiple_batch_dimensions():
    vectors = torch.linspace(-10, 10, 2 * 5 * 3).reshape(2, 5, 3)
    rotations = LieRotation(
        torch.linspace(-0.1, 0.1, 2 * 5 * 3).reshape(5, 3, 2).permute(2, 0, 1)
    )
    result = rotations.rotate_vector(vectors)
    assert result.shape == vectors.shape
    assert torch.all(
        torch.less(
            torch.abs(
                torch.linalg.norm(vectors, dim=-1) - torch.linalg.norm(result, dim=-1)
            ),
            2e-6,
        )
    )


def test_rotate_vector_broadcasts_correctly():
    vectors = torch.linspace(-10, 10, 5 * 7 * 3).reshape(5, 1, 7, 3)
    rotations = LieRotation(
        torch.linspace(-0.1, 0.1, 2 * 5 * 3).reshape(1, 3, 2, 5).permute(3, 2, 0, 1)
    )
    result = rotations.rotate_vector(vectors)
    assert result.shape == (5, 2, 7, 3)
    assert torch.all(
        torch.less(
            torch.abs(
                torch.linalg.norm(vectors, dim=-1).tile(1, 2, 1)
                - torch.linalg.norm(result, dim=-1)
            ),
            2e-6,
        )
    )


def test_gradient_is_correct_shape():
    vector = torch.tensor([1.6, -3.7, 2.8])
    rotation = LieRotation(torch.tensor([-0.25, 0.3, 0.1]))
    gradient = rotation.gradient(vector)
    assert gradient.shape == (3, 3)


def test_gradient_handles_batch_dimensions():
    vectors = torch.linspace(-10, 10, 2 * 7 * 3).reshape(2, 7, 3)
    axes = torch.linspace(-1, 1, 2 * 7 * 3).reshape(7, 3, 2).permute(2, 0, 1)
    rotation = LieRotation(axes)
    gradient = rotation.gradient(vectors)
    assert gradient.shape == (2, 7, 3, 3)


def test_gradient_broadcasts():
    vectors = torch.linspace(-10, 10, 2 * 5 * 3).reshape(2, 1, 5, 3)
    axes = torch.linspace(-1, 1, 2 * 7 * 3).reshape(1, 7, 3, 2).permute(3, 1, 0, 2)
    rotation = LieRotation(axes)
    gradient = rotation.gradient(vectors)
    assert gradient.shape == (2, 7, 5, 3, 3)


def test_gradient_passes_through_vector_gradients():
    vector = torch.tensor([1.6, -3.7, 2.8], requires_grad=True)
    rotation = LieRotation(torch.tensor([-0.25, 0.3, 0.1]))
    gradient = rotation.gradient(vector)
    assert gradient.requires_grad is True
    assert gradient.grad_fn is not None
    loss = gradient.square().sum()
    loss.backward()
    assert vector.grad is not None
    assert torch.all(torch.greater(torch.abs(vector.grad), 0))


def test_gradient_passes_through_axis_gradients():
    vector = torch.tensor([1.6, -3.7, 2.8])
    axis = torch.tensor([-0.25, 0.3, 0.1], requires_grad=True)
    rotation = LieRotation(axis)
    gradient = rotation.gradient(vector)
    assert gradient.requires_grad is True
    assert gradient.grad_fn is not None
    loss = gradient.square().sum()
    loss.backward()
    assert axis.grad is not None
    assert torch.all(torch.greater(torch.abs(axis.grad), 0))


def test_gradient_is_zero_along_diagonal_if_vector_is_zero():
    vector = torch.tensor([-4.8, -9.2, 2.2])
    rotation = LieRotation(torch.tensor([0.0, 0.0, 0.0]))
    gradient = rotation.gradient(vector)
    assert gradient.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert gradient[i, j] == 0
            else:
                assert torch.abs(gradient[i, j]) > 0.0


def test_gradient_is_zero_parallel_to_axis_when_vector_and_gradient_are_parallel():
    vector = torch.tensor([0.0, -9.2, 0.0])
    rotation = LieRotation(torch.tensor([0.0, 0.3, 0.0]))
    gradient = rotation.gradient(vector)
    assert gradient.shape == (3, 3)
    assert torch.abs(gradient[0, 0]) > 0
    assert gradient[0, 1] == 0.0
    assert torch.abs(gradient[0, 2]) > 0
    assert torch.all(gradient[1, :] == 0.0)
    assert torch.abs(gradient[2, 0]) > 0
    assert gradient[2, 1] == 0.0
    assert torch.abs(gradient[2, 2]) > 0


def test_gradient_converges_under_sgd_for_small_changes(delta: torch.Tensor):
    vector = torch.tensor([-6.1, 0.2, 1.7])
    true_axis_angle = (7 * math.pi / 16) * torch.tensor([0.3, -0.5, math.sqrt(0.66)])
    true_rotation = LieRotation(true_axis_angle)
    estimate = LieRotation(true_axis_angle + delta)
    true_rotated_vector = true_rotation.rotate_vector(vector)
    for step in range(100):
        estimated_rotated_vector = estimate.rotate_vector(vector)
        gradient = estimate.gradient(vector)
        diff = estimated_rotated_vector - true_rotated_vector
        


class RotationModule(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = LieRotation(torch.tensor([-0.25, 0.3, 0.1]))
        rotated_vector = rotation.rotate_vector(x)
        gradient = rotation.gradient(x)
        return rotated_vector, gradient


def test_can_be_compiled():
    vector = torch.tensor([1.6, -3.7, 2.8])
    rotation_comp = torch.compile(RotationModule())
    rotated_vector, gradient = rotation_comp(vector)
    assert rotated_vector.shape == (3,)
    assert gradient.shape == (3, 3)
