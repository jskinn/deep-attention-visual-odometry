import unittest
import torch
import matplotlib.pyplot as plt
from spatial_maths import (
    compute_distorted_camera_model,
    compute_distorted_camera_model_and_jacobian,
    camera_model_parameters as pidx,
)


class TestDistortedCameraModel(unittest.TestCase):

    _parameter_names = [
                "cx",
                "cy",
                "k1",
                "k2",
                "k3",
                "p1",
                "p2",
                "fx",
                "s",
                "fy",
                "Rx",
                "Ry",
                "Rz",
                "Tx",
                "Ty",
                "Tz",
            ]
    def test_plot_residuals_additive(self):
        rng = torch.Generator()
        rng.manual_seed(0x932A5653720AF9BF)
        points_3d, parameters = get_random_points(100, rng=rng)
        points_3d = points_3d.unsqueeze(0)
        parameters = parameters.unsqueeze(0)
        u_prime, v_prime = compute_distorted_camera_model(points_3d, parameters)
        points_2d = torch.stack([u_prime, v_prime], dim=1)

        fig, axes = plt.subplots(4, 4)
        axes = [ax for inner in axes for ax in inner]
        for idx, parameter_name in enumerate(self._parameter_names):
            x = torch.linspace(-1.0, 1.0, 100) + parameters[0, idx]
            y = [
                float(find_residuals(parameters, points_3d, points_2d, x_val, idx))
                for x_val in x
            ]
            x = [float(x_val) for x_val in x]
            axes[idx].plot(x, y)
            axes[idx].set_title(parameter_name)
        plt.show()

    def test_plot_residuals_relative(self):
        rng = torch.Generator()
        rng.manual_seed(0x932A5653720AF9BF)
        points_3d, parameters = get_random_points(100, rng=rng)
        points_3d = points_3d.unsqueeze(0)
        parameters = parameters.unsqueeze(0)
        u_prime, v_prime = compute_distorted_camera_model(points_3d, parameters)
        points_2d = torch.stack([u_prime, v_prime], dim=1)

        fig, axes = plt.subplots(4, 4)
        axes = [ax for inner in axes for ax in inner]
        for idx, parameter_name in enumerate(self._parameter_names):
            x = torch.linspace(0.9, 1.1, 100) * parameters[0, idx]
            y = [
                float(find_residuals(parameters, points_3d, points_2d, x_val, idx))
                for x_val in x
            ]
            x = [float(x_val) for x_val in x]
            axes[idx].plot(x, y)
            axes[idx].set_title(parameter_name)
        plt.show()

    def test_plot_gradient_relative(self):
        rng = torch.Generator()
        rng.manual_seed(0x932A5653720AF9BF)
        points_3d, parameters = get_random_points(100, rng=rng)
        points_3d = points_3d.unsqueeze(0)
        parameters = parameters.unsqueeze(0)
        u_prime, v_prime = compute_distorted_camera_model(points_3d, parameters)
        points_2d = torch.stack([u_prime, v_prime], dim=1)

        fig, axes = plt.subplots(4, 4)
        axes = [ax for inner in axes for ax in inner]
        for idx, parameter_name in enumerate(self._parameter_names):
            x = torch.linspace(0.9, 1.1, 100) * parameters[0, idx]
            y = [
                float(find_residual_gradient(parameters, points_3d, points_2d, x_val, idx))
                for x_val in x
            ]
            x = [float(x_val) for x_val in x]
            axes[idx].plot(x, y)
            axes[idx].set_title(parameter_name)
        plt.show()

    def test_plot_jacobian_relative(self):
        rng = torch.Generator()
        rng.manual_seed(0xb238399434c23991)
        points_3d, parameters = get_random_points(100, rng=rng)
        points_3d = points_3d.unsqueeze(0)
        parameters = parameters.unsqueeze(0)
        u_prime, v_prime = compute_distorted_camera_model(points_3d, parameters)

        shift = 0.95
        alt_parameters = parameters.clone()
        alt_parameters[:, 0] = parameters[:, 0] * shift
        jacobian, u_prime2, v_prime2 = compute_distorted_camera_model_and_jacobian(points_3d, alt_parameters)

        fig, axes = plt.subplots(4, 4)
        axes = [ax for inner in axes for ax in inner]
        for idx, parameter_name in enumerate(self._parameter_names):
            axes[idx].scatter(torch.cat([u_prime[0, :], v_prime[0, :]], dim=0), jacobian[0, :, idx])
            axes[idx].set_title(parameter_name)
        plt.show()

    def test_plot_residual_gradient_relative(self):
        rng = torch.Generator()
        rng.manual_seed(0xb238399434c23991)
        points_3d, parameters = get_random_points(100, rng=rng)
        points_3d = points_3d.unsqueeze(0)
        parameters = parameters.unsqueeze(0)
        u_prime, v_prime = compute_distorted_camera_model(points_3d, parameters)

        shift = 0.95
        alt_parameters = parameters.clone()
        alt_parameters[:, 7] = parameters[:, 7] * shift
        jacobian, u_prime2, v_prime2 = compute_distorted_camera_model_and_jacobian(points_3d, alt_parameters)
        residual = torch.cat([(u_prime - u_prime2), (v_prime - v_prime2)], dim=1)
        gradient = residual[:, :, None] * jacobian

        fig, axes = plt.subplots(4, 4)
        axes = [ax for inner in axes for ax in inner]
        for idx, parameter_name in enumerate(self._parameter_names):
            axes[idx].scatter(torch.cat([u_prime[0, :], v_prime[0, :]], dim=0), gradient[0, :, idx])
            axes[idx].set_title(parameter_name)
        plt.show()

    def test_plot_each_parameter(self):
        num_points = 30
        rx = torch.linspace(-torch.pi / 6, torch.pi / 6, num_points)
        fx = torch.linspace(100, 200, num_points)
        tz = torch.linspace(-10, 10, num_points)
        parameters = pidx.make_camera_parameters(
            cx=180,
            cy=160,
            k1=0.0001,
            k2=-2e-7,
            k3=3e-9,
            p1=-0.0001,
            p2=0.000004,
            fx=180,
            s=0,
            fy=180,
            tx=0,
            ty=0,
            tz=-10,
            rx=0,
            ry=0,
            rz=0,
        )
        parameters = parameters.unsqueeze(0).tile(num_points, 1)
        parameters[:, 15] = tz
        points_3d = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ]
        )
        points_3d = points_3d.unsqueeze(0).tile(num_points, 1, 1)
        points_2d = torch.tensor([[[180, 160]]]).tile(num_points, 9, 1)
        jacobian = compute_distorted_camera_model(points_2d, points_3d, parameters)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        for idx in range(9):
            ax1.plot(rx.numpy(), jacobian[:, idx, 15].numpy())
        for idx in range(9, 18):
            ax2.plot(rx.numpy(), jacobian[:, idx, 15].numpy())
        plt.show()

    def test_taylors_approximation_one_dimension_at_a_time(self):
        rng = torch.Generator()
        rng.manual_seed(0x8EF04E69C0B61338)
        points_3d, parameters = get_random_points(100, rng=rng)
        u_prime, v_prime = compute_distorted_camera_model(points_3d, parameters)
        points_2d = torch.stack([u_prime, v_prime], dim=1)
        points_3d = points_3d.unsqueeze(0)
        parameters = parameters.unsqueeze(0)
        points_2d = torch.cat([points_2d[:, 0], points_2d[:, 1]], dim=0).unsqueeze(0)

        for index in range(16):
            alt_parameters = parameters.clone()
            alt_parameters[:, index] += 1e-4
            jacobian, u_prime, v_prime = compute_distorted_camera_model_and_jacobian(
                points_3d=points_3d, parameters=alt_parameters
            )
            taylor_approximation = torch.cat([u_prime, v_prime], dim=1) + torch.matmul(
                jacobian, (parameters - alt_parameters).unsqueeze(2)
            ).squeeze(2)
            error = (points_2d - taylor_approximation).abs()
            self.assertLess(error.max(), 1e-4, f"Error was too large at axis {index}")


def get_random_points(
    num_points: int, scale_3d: float = 100.0, rng: torch.Generator = None
):
    img_height = 100 * torch.rand(1, generator=rng) + 640
    img_width = 100 * torch.rand(1, generator=rng) + 480
    translation = scale_3d * torch.randn(1, 3, generator=rng)
    rotation = 2 * torch.pi * torch.rand(3, generator=rng)
    fov_x = torch.deg2rad(60.0 * torch.rand(1, generator=rng) + 30.0)
    fov_y = torch.deg2rad(60.0 * torch.rand(1, generator=rng) + 30.0)
    tan_fov_x = torch.tan(fov_x / 2.0)
    tan_fov_y = torch.tan(fov_y / 2.0)
    fx = 1 / tan_fov_x
    fy = img_height / (img_width * tan_fov_y)
    skew = 0.01 * torch.randn(1, generator=rng)
    cx = 0.2 * torch.rand(1, generator=rng) + 0.4
    cy = (img_height / img_width) * (0.2 * torch.rand(1, generator=rng) + 0.5)
    k1 = 1e-4 * torch.randn(1, generator=rng)
    k2 = 1e-8 * torch.randn(1, generator=rng)
    k3 = 1e-12 * torch.randn(1, generator=rng)
    p1 = 1e-8 * torch.randn(1, generator=rng)
    p2 = 1e-8 * torch.randn(1, generator=rng)

    # Generate points that are within the fov and have positive z
    camera_z = scale_3d * (torch.randn(num_points, 1) + 1.0).abs()
    camera_x = camera_z * tan_fov_x * (2.0 * torch.rand(num_points, 1) - 1.0)
    camera_y = camera_z * tan_fov_y * (2.0 * torch.rand(num_points, 1) - 1.0)

    points_3d = torch.cat([camera_x, camera_y, camera_z], dim=1)
    points_3d = points_3d - translation
    sin_rx = torch.sin(rotation[0])
    cos_rx = torch.cos(rotation[0])
    sin_ry = torch.sin(rotation[1])
    cos_ry = torch.cos(rotation[1])
    sin_rz = torch.sin(rotation[2])
    cos_rz = torch.cos(rotation[2])
    rot_matrix3 = torch.tensor([[
            cos_ry * cos_rz,
            sin_rx * sin_ry * cos_rz - cos_rx * sin_rz,
            cos_rx * sin_ry * cos_rz + sin_rx * sin_rz,
    ], [
        cos_ry * sin_rz,
        sin_rx * sin_ry * sin_rz + cos_rx * cos_rz,
        cos_rx * sin_ry * sin_rz - sin_rx * cos_rz,
    ], [

        -sin_ry,
        sin_rx * cos_ry,
        cos_rx * cos_ry
    ]])
    rot_matrix3_inv = torch.linalg.inv(rot_matrix3)
    points_3d = torch.matmul(rot_matrix3_inv.unsqueeze(0), points_3d.unsqueeze(2)).squeeze(2)
    return points_3d, pidx.make_camera_parameters(
        cx=cx,
        cy=cy,
        k1=k1,
        k2=k2,
        k3=k3,
        p1=p1,
        p2=p2,
        fx=fx,
        s=skew,
        fy=fy,
        tx=translation[0, 0],
        ty=translation[0, 1],
        tz=translation[0, 2],
        rx=rotation[0],
        ry=rotation[1],
        rz=rotation[2]
    )


def find_residuals(parameters, points_3d, points_2d, alt_value, alt_index):
    parameters = parameters.clone()
    parameters[:, alt_index] = alt_value
    u_prime, v_prime = compute_distorted_camera_model(points_3d, parameters)
    alt_points_2d = torch.stack([u_prime, v_prime], dim=1)
    return (alt_points_2d - points_2d).square().sum()


def find_residual_gradient(parameters, points_3d, points_2d, alt_value, alt_index):
    parameters = parameters.clone()
    parameters[:, alt_index] = alt_value
    jacobian, u_prime, v_prime = compute_distorted_camera_model_and_jacobian(points_3d, parameters)
    residuals = torch.cat([points_2d[:, 0, :] - u_prime, points_2d[:, 1, :] - v_prime], dim=1)
    gradients = torch.matmul(residuals, jacobian)
    return gradients[:, 0, alt_index]
