import unittest
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import spatial_maths.vectors_to_rotation_matrix as vec


class TestMakeRotationMatrix(unittest.TestCase):
    def test_returns_same_size_as_input(self):
        batch_size = 11
        vectors = torch.randn(batch_size, 6)
        matrix = vec.make_rotation_matrix(
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            vectors[:, 3],
            vectors[:, 4],
            vectors[:, 5],
        )
        self.assertEqual(len(matrix), 9)
        self.assertEqual((batch_size,), matrix.r1.shape)
        self.assertEqual((batch_size,), matrix.r2.shape)
        self.assertEqual((batch_size,), matrix.r3.shape)
        self.assertEqual((batch_size,), matrix.r4.shape)
        self.assertEqual((batch_size,), matrix.r5.shape)
        self.assertEqual((batch_size,), matrix.r6.shape)
        self.assertEqual((batch_size,), matrix.r7.shape)
        self.assertEqual((batch_size,), matrix.r8.shape)
        self.assertEqual((batch_size,), matrix.r9.shape)

    def test_rotation_matrices_are_orthogonal(self):
        batch_size = 17
        vectors = torch.randn(batch_size, 6)
        matrix_data = vec.make_rotation_matrix(
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            vectors[:, 3],
            vectors[:, 4],
            vectors[:, 5],
        )
        matrix_tensor = torch.stack(
            [
                torch.stack([matrix_data.r1, matrix_data.r2, matrix_data.r3], dim=1),
                torch.stack([matrix_data.r4, matrix_data.r5, matrix_data.r6], dim=1),
                torch.stack([matrix_data.r7, matrix_data.r8, matrix_data.r9], dim=1),
            ],
            dim=1,
        )
        determinants = torch.linalg.det(matrix_tensor)
        self.assertTrue(torch.all(torch.isclose(determinants, torch.ones(batch_size))))
        transpose = matrix_tensor.transpose(1, 2)
        self.assertTrue(
            torch.all(
                torch.isclose(
                    torch.bmm(matrix_tensor, transpose),
                    torch.eye(3).unsqueeze(0).tile(batch_size, 1, 1),
                    atol=1e-6,
                )
            )
        )

    def test_jacobians_are_the_right_shape(self):
        batch_size = 11
        vectors = torch.randn(batch_size, 6)
        matrix = vec.make_rotation_matrix_and_derivatives(
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            vectors[:, 3],
            vectors[:, 4],
            vectors[:, 5],
        )
        self.assertEqual(len(matrix), 9 + 3 * 9 + 3 * 6)
        self.assertEqual((batch_size,), matrix.r1.shape)
        self.assertEqual((batch_size,), matrix.r2.shape)
        self.assertEqual((batch_size,), matrix.r3.shape)
        self.assertEqual((batch_size,), matrix.r4.shape)
        self.assertEqual((batch_size,), matrix.r5.shape)
        self.assertEqual((batch_size,), matrix.r6.shape)
        self.assertEqual((batch_size,), matrix.r7.shape)
        self.assertEqual((batch_size,), matrix.r8.shape)
        self.assertEqual((batch_size,), matrix.r9.shape)

        self.assertEqual((batch_size,), matrix.dr1_da1.shape)
        self.assertEqual((batch_size,), matrix.dr1_da2.shape)
        self.assertEqual((batch_size,), matrix.dr1_da3.shape)
        self.assertEqual((batch_size,), matrix.dr2_da1.shape)
        self.assertEqual((batch_size,), matrix.dr2_da2.shape)
        self.assertEqual((batch_size,), matrix.dr2_da3.shape)
        self.assertEqual((batch_size,), matrix.dr3_da1.shape)
        self.assertEqual((batch_size,), matrix.dr3_da2.shape)
        self.assertEqual((batch_size,), matrix.dr3_da3.shape)
        self.assertEqual((batch_size,), matrix.dr4_da1.shape)
        self.assertEqual((batch_size,), matrix.dr4_da2.shape)
        self.assertEqual((batch_size,), matrix.dr4_da3.shape)
        self.assertEqual((batch_size,), matrix.dr5_da1.shape)
        self.assertEqual((batch_size,), matrix.dr5_da2.shape)
        self.assertEqual((batch_size,), matrix.dr5_da3.shape)
        self.assertEqual((batch_size,), matrix.dr6_da1.shape)
        self.assertEqual((batch_size,), matrix.dr6_da2.shape)
        self.assertEqual((batch_size,), matrix.dr6_da3.shape)
        self.assertEqual((batch_size,), matrix.dr7_da1.shape)
        self.assertEqual((batch_size,), matrix.dr7_da2.shape)
        self.assertEqual((batch_size,), matrix.dr7_da3.shape)
        self.assertEqual((batch_size,), matrix.dr8_da1.shape)
        self.assertEqual((batch_size,), matrix.dr8_da2.shape)
        self.assertEqual((batch_size,), matrix.dr8_da3.shape)
        self.assertEqual((batch_size,), matrix.dr9_da1.shape)
        self.assertEqual((batch_size,), matrix.dr9_da2.shape)
        self.assertEqual((batch_size,), matrix.dr9_da3.shape)

        self.assertEqual((batch_size,), matrix.dr2_db1.shape)
        self.assertEqual((batch_size,), matrix.dr2_db2.shape)
        self.assertEqual((batch_size,), matrix.dr2_db3.shape)
        self.assertEqual((batch_size,), matrix.dr3_db1.shape)
        self.assertEqual((batch_size,), matrix.dr3_db2.shape)
        self.assertEqual((batch_size,), matrix.dr3_db3.shape)
        self.assertEqual((batch_size,), matrix.dr5_db1.shape)
        self.assertEqual((batch_size,), matrix.dr5_db2.shape)
        self.assertEqual((batch_size,), matrix.dr5_db3.shape)
        self.assertEqual((batch_size,), matrix.dr6_db1.shape)
        self.assertEqual((batch_size,), matrix.dr6_db2.shape)
        self.assertEqual((batch_size,), matrix.dr6_db3.shape)
        self.assertEqual((batch_size,), matrix.dr8_db1.shape)
        self.assertEqual((batch_size,), matrix.dr8_db2.shape)
        self.assertEqual((batch_size,), matrix.dr8_db3.shape)
        self.assertEqual((batch_size,), matrix.dr9_db1.shape)
        self.assertEqual((batch_size,), matrix.dr9_db2.shape)
        self.assertEqual((batch_size,), matrix.dr9_db3.shape)

    def test_converges_under_gradient_descent(self):
        rng = torch.Generator()
        rng.manual_seed(0xA7D4F2E5A69AAB4E)
        learning_rate = 0.05
        true_a = torch.randn(3, generator=rng)
        true_b = torch.randn(3, generator=rng)
        true_norm_a = true_a / torch.linalg.vector_norm(true_a)
        true_norm_b = true_b / torch.linalg.vector_norm(true_b)
        true_matrix = vec.make_rotation_matrix(
            true_a[0], true_a[1], true_a[2], true_b[0], true_b[1], true_b[2]
        )
        true_matrix_tensor = _make_flat_matrix_tensor(true_matrix)
        a = torch.randn(3, generator=rng)
        b = torch.randn(3, generator=rng)
        norm_a = a / torch.linalg.vector_norm(a)
        norm_b = b / torch.linalg.vector_norm(b)
        matrix_residuals = torch.empty(0, 9)
        a_values = a.clone().unsqueeze(0)
        b_values = b.clone().unsqueeze(0)
        error_a1 = [true_a[0] - a[0]]
        error_a2 = [true_a[1] - a[1]]
        error_a3 = [true_a[2] - a[2]]
        error_b1 = [true_b[0] - b[0]]
        error_b2 = [true_b[1] - b[1]]
        error_b3 = [true_b[2] - b[2]]
        angle_a = [torch.arccos((norm_a * true_norm_a).sum())]
        angle_b = [torch.arccos((norm_b * true_norm_b).sum())]
        max_residuals = []
        for step in range(1000):
            matrix_and_derivatives = vec.make_rotation_matrix_and_derivatives(
                a[0], a[1], a[2], b[0], b[1], b[2]
            )
            matrix_tensor = _make_flat_matrix_tensor(matrix_and_derivatives)
            residuals = matrix_tensor - true_matrix_tensor
            matrix_residuals = torch.cat([matrix_residuals, residuals[None, :]], dim=0)
            a1_derivatives = torch.tensor(
                [
                    matrix_and_derivatives.dr1_da1,
                    matrix_and_derivatives.dr2_da1,
                    matrix_and_derivatives.dr3_da1,
                    matrix_and_derivatives.dr4_da1,
                    matrix_and_derivatives.dr5_da1,
                    matrix_and_derivatives.dr6_da1,
                    matrix_and_derivatives.dr7_da1,
                    matrix_and_derivatives.dr8_da1,
                    matrix_and_derivatives.dr9_da1,
                ]
            )
            a2_derivatives = torch.tensor(
                [
                    matrix_and_derivatives.dr1_da2,
                    matrix_and_derivatives.dr2_da2,
                    matrix_and_derivatives.dr3_da2,
                    matrix_and_derivatives.dr4_da2,
                    matrix_and_derivatives.dr5_da2,
                    matrix_and_derivatives.dr6_da2,
                    matrix_and_derivatives.dr7_da2,
                    matrix_and_derivatives.dr8_da2,
                    matrix_and_derivatives.dr9_da2,
                ]
            )
            a3_derivatives = torch.tensor(
                [
                    matrix_and_derivatives.dr1_da3,
                    matrix_and_derivatives.dr2_da3,
                    matrix_and_derivatives.dr3_da3,
                    matrix_and_derivatives.dr4_da3,
                    matrix_and_derivatives.dr5_da3,
                    matrix_and_derivatives.dr6_da3,
                    matrix_and_derivatives.dr7_da3,
                    matrix_and_derivatives.dr8_da3,
                    matrix_and_derivatives.dr9_da3,
                ]
            )
            b1_derivatives = torch.tensor(
                [
                    0.0,
                    matrix_and_derivatives.dr2_db1,
                    matrix_and_derivatives.dr3_db1,
                    0.0,
                    matrix_and_derivatives.dr5_db1,
                    matrix_and_derivatives.dr6_db1,
                    0.0,
                    matrix_and_derivatives.dr8_db1,
                    matrix_and_derivatives.dr9_db1,
                ]
            )
            b2_derivatives = torch.tensor(
                [
                    0.0,
                    matrix_and_derivatives.dr2_db2,
                    matrix_and_derivatives.dr3_db2,
                    0.0,
                    matrix_and_derivatives.dr5_db2,
                    matrix_and_derivatives.dr6_db2,
                    0.0,
                    matrix_and_derivatives.dr8_db2,
                    matrix_and_derivatives.dr9_db2,
                ]
            )
            b3_derivatives = torch.tensor(
                [
                    0.0,
                    matrix_and_derivatives.dr2_db3,
                    matrix_and_derivatives.dr3_db3,
                    0.0,
                    matrix_and_derivatives.dr5_db3,
                    matrix_and_derivatives.dr6_db3,
                    0.0,
                    matrix_and_derivatives.dr8_db3,
                    matrix_and_derivatives.dr9_db3,
                ]
            )
            a = a - learning_rate * torch.tensor(
                [
                    (a1_derivatives * residuals).sum(),
                    (a2_derivatives * residuals).sum(),
                    (a3_derivatives * residuals).sum(),
                ]
            )
            b = b - learning_rate * torch.tensor(
                [
                    (b1_derivatives * residuals).sum(),
                    (b2_derivatives * residuals).sum(),
                    (b3_derivatives * residuals).sum(),
                ]
            )
            norm_a = a / torch.linalg.vector_norm(a)
            norm_b = b / torch.linalg.vector_norm(b)
            error_a1.append(true_norm_a[0] - norm_a[0])
            error_a2.append(true_norm_a[1] - norm_a[1])
            error_a3.append(true_norm_a[2] - norm_a[2])
            error_b1.append(true_norm_b[0] - norm_b[0])
            error_b2.append(true_norm_b[1] - norm_b[1])
            error_b3.append(true_norm_b[2] - norm_b[2])
            angle_a.append(torch.arccos((norm_a * true_norm_a).sum()))
            angle_b.append(torch.arccos((norm_b * true_norm_b).sum()))
            a_values = torch.cat([a_values, a[None, :]], dim=0)
            b_values = torch.cat([b_values, b[None, :]], dim=0)
            max_residuals.append(residuals.max())
            if residuals.max() < 5e-6:
                break

        fig, axes = plt.subplots(3, 3)
        steps = list(range(matrix_residuals.size(0)))
        axes[0][0].plot(steps, matrix_residuals[:, 0])
        axes[0][1].plot(steps, matrix_residuals[:, 1])
        axes[0][2].plot(steps, matrix_residuals[:, 2])
        axes[1][0].plot(steps, matrix_residuals[:, 3])
        axes[1][1].plot(steps, matrix_residuals[:, 4])
        axes[1][2].plot(steps, matrix_residuals[:, 5])
        axes[2][0].plot(steps, matrix_residuals[:, 6])
        axes[2][1].plot(steps, matrix_residuals[:, 7])
        axes[2][2].plot(steps, matrix_residuals[:, 8])
        fig.suptitle("R1..9 residuals")

        fig, ax = plt.subplots(1, 1)
        steps = list(range(len(max_residuals)))
        ax.plot(steps, max_residuals)
        ax.set_yscale("log")
        ax.set_title("Max residuals")

        fig, axes = plt.subplots(1, 2)
        axes[0].plot(list(range(len(angle_a))), angle_a)
        axes[1].plot(list(range(len(angle_b))), angle_b)
        axes[0].set_title("a angle")
        axes[1].set_title("b angle")

        fig, axes = plt.subplots(2, 3)
        steps = list(range(a_values.size(0)))
        norm_a_values = a_values / torch.linalg.vector_norm(a_values, dim=1, keepdim=True)
        norm_b_values = b_values / torch.linalg.vector_norm(b_values, dim=1, keepdim=True)
        axes[0][0].plot(steps, norm_a_values[:, 0])
        axes[0][0].plot(steps, true_norm_a[0] * torch.ones(len(steps)))
        axes[0][1].plot(steps, norm_a_values[:, 1])
        axes[0][1].plot(steps, true_norm_a[1] * torch.ones(len(steps)))
        axes[0][2].plot(steps, norm_a_values[:, 2])
        axes[0][2].plot(steps, true_norm_a[2] * torch.ones(len(steps)))
        axes[1][0].plot(steps, norm_b_values[:, 0])
        axes[1][0].plot(steps, true_norm_b[0] * torch.ones(len(steps)))
        axes[1][1].plot(steps, norm_b_values[:, 1])
        axes[1][1].plot(steps, true_norm_b[1] * torch.ones(len(steps)))
        axes[1][2].plot(steps, norm_b_values[:, 2])
        axes[1][2].plot(steps, true_norm_b[2] * torch.ones(len(steps)))
        fig.suptitle("normed values")

        fig, axes = plt.subplots(2, 3)
        axes[0][0].plot(norm_a_values[:, 1], norm_a_values[:, 2])
        axes[0][0].scatter(true_norm_a[1], true_norm_a[2])
        axes[0][1].plot(norm_a_values[:, 0], norm_a_values[:, 1])
        axes[0][1].scatter(true_norm_a[0], true_norm_a[1])
        axes[0][2].plot(norm_a_values[:, 0], norm_a_values[:, 2])
        axes[0][2].scatter(true_norm_a[0], true_norm_a[2])
        axes[1][0].plot(norm_b_values[:, 1], norm_b_values[:, 2])
        axes[1][0].scatter(true_norm_b[1], true_norm_b[2])
        axes[1][1].plot(norm_b_values[:, 0], norm_b_values[:, 1])
        axes[1][1].scatter(true_norm_b[0], true_norm_b[1])
        axes[1][2].plot(norm_b_values[:, 0], norm_b_values[:, 2])
        axes[1][2].scatter(true_norm_b[0], true_norm_b[2])
        fig.suptitle("2d normed values")

        fig, axes = plt.subplots(1, 2)
        steps = list(range(a_values.size(0)))
        axes[0].plot(steps, torch.linalg.vector_norm(a_values, dim=-1))
        axes[0].set_title("A length")
        axes[1].plot(steps, torch.linalg.vector_norm(b_values, dim=-1))
        axes[1].set_title("B length")

        plt.show()


def _make_flat_matrix_tensor(matrix_data):
    return torch.tensor(
        [
            matrix_data.r1,
            matrix_data.r2,
            matrix_data.r3,
            matrix_data.r4,
            matrix_data.r5,
            matrix_data.r6,
            matrix_data.r7,
            matrix_data.r8,
            matrix_data.r9,
        ],
    )
