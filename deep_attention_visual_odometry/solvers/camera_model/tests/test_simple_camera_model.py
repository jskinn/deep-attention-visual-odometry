import unittest
import torch
import matplotlib.pyplot as pyplot
import spatial_maths.simple_camera_model_parameters as pidx
import spatial_maths.simple_camera_model as cam


class TestSimpleCameraModel(unittest.TestCase):

    def test_simple_camera_model_produces_correct_shape(self):
        batch_size = 14
        num_points = 97
        parameters = torch.rand(batch_size, 12)
        points = 100 * torch.randn(batch_size, num_points, 3)
        points[:, :, 0] = points[:, :, 0].abs()
        u, v = cam.compute_simple_camera_model(parameters=parameters, points_3d=points)
        self.assertEqual((batch_size, num_points), u.shape)
        self.assertEqual((batch_size, num_points), v.shape)

    def test_simple_camera_model_jacobian_returns_correct_shape(self):
        batch_size = 11
        num_points = 67
        parameters = torch.rand(batch_size, 12)
        points = 100 * torch.randn(batch_size, num_points, 3)
        points[:, :, 0] = points[:, :, 0].abs()
        jacobian, u, v = cam.compute_simple_camera_model_and_jacobian(parameters=parameters, points_3d=points)
        self.assertEqual((batch_size, num_points), u.shape)
        self.assertEqual((batch_size, num_points), v.shape)
        self.assertEqual((batch_size, 2 * num_points, 12), jacobian.shape)

    def test_simple_camera_model_jacobian_converges_under_gradient_descent(self):
        rng = torch.Generator()
        rng.manual_seed(0x649eb9c5a8a2284c)
        learning_rate = 0.1
        num_points = 35
        true_parameters = pidx.make_camera_parameters(
            cx=torch.rand(1, generator=rng),
            cy=torch.rand(1, generator=rng),
            f=torch.randn(1, generator=rng).abs(),
            a1=torch.randn(1, generator=rng),
            a2=torch.randn(1, generator=rng),
            a3=torch.randn(1, generator=rng),
            b1=torch.randn(1, generator=rng),
            b2=torch.randn(1, generator=rng),
            b3=torch.randn(1, generator=rng),
            tx=100 * torch.randn(1, generator=rng),
            ty=100 * torch.randn(1, generator=rng),
            tz=100 * torch.randn(1, generator=rng),
        )
        true_parameters = true_parameters[None, :]
        points_3d = 100 * torch.randn(1, num_points, 3, generator=rng)
        true_u, true_v = cam.compute_simple_camera_model(points_3d=points_3d, parameters=true_parameters)

        # parameters = pidx.make_camera_parameters(
        #     cx=torch.rand(1, generator=rng),
        #     cy=torch.rand(1, generator=rng),
        #     f=torch.randn(1, generator=rng).abs(),
        #     a1=torch.randn(1, generator=rng),
        #     a2=torch.randn(1, generator=rng),
        #     a3=torch.randn(1, generator=rng),
        #     b1=torch.randn(1, generator=rng),
        #     b2=torch.randn(1, generator=rng),
        #     b3=torch.randn(1, generator=rng),
        #     tx=100 * torch.randn(1, generator=rng),
        #     ty=100 * torch.randn(1, generator=rng),
        #     tz=100 * torch.randn(1, generator=rng),
        # )
        # parameters = parameters[None, :]
        parameters = true_parameters.clone()
        parameters[:, pidx.TZ] = 100 * torch.randn(1, generator=rng)
        parameters_history = parameters.clone()
        residuals_history = []
        step = torch.zeros_like(parameters)
        for _ in range(500):
            jacobian, u, v = cam.compute_simple_camera_model_and_jacobian(points_3d, parameters)
            residuals = torch.cat([
                u - true_u,
                v - true_v
            ], dim=1)
            residuals_history.append(torch.max(residuals))
            if residuals.max() < 1e-5:
                break
            gradients = (residuals[:, :, None] * jacobian).sum(dim=1)
            step[:, pidx.TZ] = - learning_rate * gradients[:, pidx.TZ]
            parameters = parameters + step
            parameters_history = torch.cat([parameters_history, parameters], dim=0)

        fig, axes = pyplot.subplots(1, 1)
        axes.plot(list(range(len(residuals_history))), residuals_history)
        axes.set_yscale("log")

        fig, axes = pyplot.subplots(1, 1)
        steps = list(range(parameters_history.size(0)))
        axes.plot(steps, parameters_history[:, pidx.TZ])
        axes.plot(steps, true_parameters[:, pidx.TZ].tile(len(steps)))

        # fig, axes = pyplot.subplots(4, 3)
        # steps = list(range(parameters_history.size(0)))
        # diffs = parameters_history - true_parameters
        # axes[0][0].plot(steps, diffs[:, 0])
        # axes[0][1].plot(steps, diffs[:, 1])
        # axes[0][2].plot(steps, diffs[:, 2])
        # axes[1][0].plot(steps, diffs[:, 3])
        # axes[1][1].plot(steps, diffs[:, 4])
        # axes[1][2].plot(steps, diffs[:, 5])
        # axes[2][0].plot(steps, diffs[:, 6])
        # axes[2][1].plot(steps, diffs[:, 7])
        # axes[2][2].plot(steps, diffs[:, 8])
        # axes[3][0].plot(steps, diffs[:, 9])
        # axes[3][1].plot(steps, diffs[:, 10])
        # axes[3][2].plot(steps, diffs[:, 11])
        # fig.suptitle("Difference from true")

        pyplot.show()
