import math
import torch
from deep_attention_visual_odometry.solvers.camera_model.lie_rotation import LieRotation


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


def test_handles_multiple_batch_dimensions():
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


def test_broadcasts_correctly():
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
