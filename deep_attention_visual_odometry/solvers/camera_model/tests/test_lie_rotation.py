import math
import hashlib
import pytest
import torch
import torch.nn as nn
from deep_attention_visual_odometry.solvers.camera_model.lie_rotation import LieRotation


@pytest.fixture(
    params=[
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, -1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    ]
)
def delta(request) -> torch.Tensor:
    return 0.01 * torch.tensor(request.param)


@pytest.fixture()
def random_generator(request) -> torch.Generator:
    name_hash = hashlib.md5(request.node.name.encode("utf-8"))
    generator = torch.Generator()
    generator.manual_seed(int.from_bytes(name_hash.digest()[:8], byteorder="big"))
    return generator


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


def test_gradient_x_matches_equation():
    angle = 3 * math.pi / 16
    axis = torch.tensor([-0.5, 0.8, 0.6])
    axis = axis / torch.linalg.norm(axis)
    vector = torch.tensor([-6.1, 0.2, 1.7])

    # Really raw calculations of the gradient
    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))
    a = angle * axis[0]
    b = angle * axis[1]
    c = angle * axis[2]
    x = vector[0]
    y = vector[1]
    z = vector[2]
    gamma = (a * x + b * y + c * z) / (a * a + b * b + c * c)
    dx_da = (1 - cos_theta) * (
        (a * x - 2 * a * a * gamma) / (angle * angle) + gamma
    ) + (a / angle) * (
        (a * gamma - x) * sin_theta
        + (b * z - c * y) * (cos_theta / angle - sin_theta / (angle * angle))
    )
    dx_db = (
        (1 - cos_theta) * ((a * y - 2 * a * b * gamma) / (angle * angle))
        + (b / angle)
        * (
            (a * gamma - x) * sin_theta
            + (b * z - c * y) * (cos_theta / angle - sin_theta / (angle * angle))
        )
        + z * sin_theta / angle
    )
    dx_dc = (
        (1 - cos_theta) * ((a * z - 2 * a * c * gamma) / (angle * angle))
        + (c / angle)
        * (
            (a * gamma - x) * sin_theta
            + (b * z - c * y) * (cos_theta / angle - sin_theta / (angle * angle))
        )
        - y * sin_theta / angle
    )

    rotation = LieRotation(angle * axis)
    gradient = rotation.gradient(vector)
    assert gradient.shape == (3, 3)
    assert torch.abs(gradient[0, 0] - dx_da) < 2e-7
    assert torch.abs(gradient[0, 1] - dx_db) < 2e-7
    assert torch.abs(gradient[0, 2] - dx_dc) < 2e-7


def test_gradient_y_matches_equation():
    angle = 3 * math.pi / 16
    axis = torch.tensor([-0.5, 0.8, 0.6])
    axis = axis / torch.linalg.norm(axis)
    vector = torch.tensor([-6.1, 0.2, 1.7])

    # Really raw calculations of the gradient
    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))
    a = angle * axis[0]
    b = angle * axis[1]
    c = angle * axis[2]
    x = vector[0]
    y = vector[1]
    z = vector[2]
    gamma = (a * x + b * y + c * z) / (a * a + b * b + c * c)
    dy_da = (
        (1 - cos_theta) * ((b * x - 2 * b * a * gamma) / (angle * angle))
        + (a / angle)
        * (
            (b * gamma - y) * sin_theta
            + (c * x - a * z) * (cos_theta / angle - sin_theta / (angle * angle))
        )
        - z * sin_theta / angle
    )
    dy_db = (1 - cos_theta) * (
        (b * y - 2 * b * b * gamma) / (angle * angle) + gamma
    ) + (b / angle) * (
        (b * gamma - y) * sin_theta
        + (c * x - a * z) * (cos_theta / angle - sin_theta / (angle * angle))
    )
    dy_dc = (
        (1 - cos_theta) * ((b * z - 2 * b * c * gamma) / (angle * angle))
        + (c / angle)
        * (
            (b * gamma - y) * sin_theta
            + (c * x - a * z) * (cos_theta / angle - sin_theta / (angle * angle))
        )
        + x * sin_theta / angle
    )

    rotation = LieRotation(angle * axis)
    gradient = rotation.gradient(vector)
    assert gradient.shape == (3, 3)
    assert gradient[1, 0] == dy_da
    assert gradient[1, 1] == dy_db
    assert gradient[1, 2] == dy_dc


def test_gradient_z_matches_equation():
    angle = 3 * math.pi / 16
    axis = torch.tensor([-0.5, 0.8, 0.6])
    axis = axis / torch.linalg.norm(axis)
    vector = torch.tensor([-6.1, 0.2, 1.7])

    # Really raw calculations of the gradient
    cos_theta = torch.cos(torch.tensor(angle))
    sin_theta = torch.sin(torch.tensor(angle))
    a = angle * axis[0]
    b = angle * axis[1]
    c = angle * axis[2]
    x = vector[0]
    y = vector[1]
    z = vector[2]
    gamma = (a * x + b * y + c * z) / (a * a + b * b + c * c)
    dz_da = (
        (1 - cos_theta) * ((c * x - 2 * c * a * gamma) / (angle * angle))
        + (a / angle)
        * (
            (c * gamma - z) * sin_theta
            + (a * y - b * x) * (cos_theta / angle - sin_theta / (angle * angle))
        )
        + y * sin_theta / angle
    )
    dz_db = (
        (1 - cos_theta) * ((c * y - 2 * c * b * gamma) / (angle * angle))
        + (b / angle)
        * (
            (c * gamma - z) * sin_theta
            + (a * y - b * x) * (cos_theta / angle - sin_theta / (angle * angle))
        )
        - y * sin_theta / angle
    )
    dy_dc = (1 - cos_theta) * (
        (c * z - 2 * c * c * gamma) / (angle * angle) + gamma
    ) + (c / angle) * (
        (c * gamma - z) * sin_theta
        + (a * y - b * x) * (cos_theta / angle - sin_theta / (angle * angle))
    )

    rotation = LieRotation(angle * axis)
    gradient = rotation.gradient(vector)
    assert gradient.shape == (3, 3)
    assert gradient[2, 0] == dz_da
    assert gradient[2, 1] == dz_db
    assert gradient[2, 2] == dz_dc


def test_sgd_is_stable_when_correct(random_generator):
    # do a fixed number of gradient descent steps
    learning_rate = 2e-2
    steps = 10
    vector = 10 * torch.randn(3, generator=random_generator)
    angle = (math.pi / 16) + (14 * math.pi / 16) * torch.rand(
        1, generator=random_generator
    )
    axis = torch.randn(3, generator=random_generator)
    true_axis_angle = angle * axis / torch.linalg.norm(axis)
    true_rotation = LieRotation(true_axis_angle)
    estimate = LieRotation(true_axis_angle)
    true_rotated_vector = true_rotation.rotate_vector(vector)
    for step_num in range(steps):
        estimated_rotated_vector = estimate.rotate_vector(vector)
        gradient = estimate.gradient(vector)
        diff = estimated_rotated_vector - true_rotated_vector
        step = -2.0 * (diff.unsqueeze(-1) * gradient).sum(dim=0)
        estimate = estimate.add_lie_parameters(learning_rate * step)

    # Check that the rotation hasn't changed
    estimated_rotated_vector = estimate.rotate_vector(vector)
    diff = estimated_rotated_vector - true_rotated_vector
    assert torch.equal(diff, torch.zeros_like(diff))


def test_converges_under_sgd_for_small_changes(delta: torch.Tensor, random_generator):
    # do a fixed number of gradient descent steps
    learning_rate = 4e-3
    steps = 200
    vector = 5 * torch.randn(3, generator=random_generator)
    angle = (math.pi / 16) + (14 * math.pi / 16) * torch.rand(
        1, generator=random_generator
    )
    axis = torch.randn(3, generator=random_generator)
    true_axis_angle = angle * axis / torch.linalg.norm(axis)
    true_rotation = LieRotation(true_axis_angle)
    estimate = LieRotation(true_axis_angle + delta)
    true_rotated_vector = true_rotation.rotate_vector(vector)
    for step_num in range(steps):
        estimated_rotated_vector = estimate.rotate_vector(vector)
        gradient = estimate.gradient(vector)
        diff = estimated_rotated_vector - true_rotated_vector
        step = -2.0 * (diff.unsqueeze(-1) * gradient).sum(dim=0)
        estimate = estimate.add_lie_parameters(learning_rate * step)

    estimated_rotated_vector = estimate.rotate_vector(vector)
    diff = estimated_rotated_vector - true_rotated_vector
    assert diff.abs().max() < 1e-5


def test_converges_under_sgd_near_zero(delta: torch.Tensor):
    # do a fixed number of gradient descent steps
    learning_rate = 7e-4
    beta = 0.9
    steps = 200
    vector = torch.tensor([-6.1, 0.2, 1.7])
    true_rotation = LieRotation(delta)
    estimate = LieRotation(torch.zeros(3))
    true_rotated_vector = true_rotation.rotate_vector(vector)
    step = torch.zeros(3)
    for step_num in range(steps):
        estimated_rotated_vector = estimate.rotate_vector(vector)
        gradient = estimate.gradient(vector)
        diff = estimated_rotated_vector - true_rotated_vector
        step = beta * step + 2.0 * (1.0 - beta) * (diff.unsqueeze(-1) * gradient).sum(
            dim=0
        )
        estimate = estimate.add_lie_parameters(-1.0 * learning_rate * step)

    estimated_rotated_vector = estimate.rotate_vector(vector)
    diff = estimated_rotated_vector - true_rotated_vector
    assert diff.abs().max() < 1e-6


def test_converges_under_sgd_near_2pi(delta: torch.Tensor):
    # do a fixed number of gradient descent steps
    learning_rate = 2e-2
    steps = 100
    vector = torch.tensor([-6.1, 0.2, 1.7])
    full_rotation = 2 * math.pi * torch.tensor([0.5, -math.sqrt(0.66), -0.3])
    true_rotation = LieRotation(full_rotation + delta)
    estimate = LieRotation(full_rotation)
    true_rotated_vector = true_rotation.rotate_vector(vector)
    for step_num in range(steps):
        estimated_rotated_vector = estimate.rotate_vector(vector)
        gradient = estimate.gradient(vector)
        diff = estimated_rotated_vector - true_rotated_vector
        step = -2.0 * (diff.unsqueeze(-1) * gradient).sum(dim=0)
        estimate = estimate.add_lie_parameters(learning_rate * step)

    estimated_rotated_vector = estimate.rotate_vector(vector)
    diff = estimated_rotated_vector - true_rotated_vector
    assert diff.abs().max() < 1e-6


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
