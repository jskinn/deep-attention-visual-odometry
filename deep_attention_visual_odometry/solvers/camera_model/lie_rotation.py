from typing import Self
import torch
from deep_attention_visual_odometry.utils import sin_x_on_x, one_minus_cos_x_on_x_squared, cos_x_on_x_squared_minus_sin_x_on_x_cubed


class LieRotation:
    """
    A differentiable rotation parameter based around the Lie theory for SO(3).

    There are multiple possible representations of a rotation in SO(3), including
    quaternions and 3x3 rotation matrices.
    Aside from constructor, this class should be largely opaque to the user as to which
    representation it uses internally.
    """

    def __init__(self, lie_vector):
        self._lie_vector = lie_vector

    def rotate_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Rotate a 3-vector by the current rotation
        """
        # This is based on maths that combines a conversion from axis-angle to quaternion
        # with the hamilton product to rotate the vector.
        # Produces a variant of Rodrigues' rotation formula
        angle_squared = self._lie_vector.square().sum(dim=-1, keepdims=True)
        angle = angle_squared.sqrt()
        is_angle_zero = angle == 0.0
        gamma = (vector * self._lie_vector).sum(dim=-1, keepdims=True) / angle_squared
        gamma = torch.where(is_angle_zero, torch.zeros_like(gamma), gamma)
        cos_theta = torch.cos(angle)
        # Lim x->0 sin(x)/x = 1.0
        sin_theta_on_theta = sin_x_on_x(angle)
        # Compute the sin coefficients for each output.
        sin_coefficients = torch.linalg.cross(self._lie_vector, vector, dim=-1)
        gamma_axis = self._lie_vector * gamma
        out_vector = (
            vector * cos_theta
            + (1 - cos_theta) * gamma_axis
            + sin_coefficients * sin_theta_on_theta
        )
        return out_vector

    def parameter_gradient(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Get the gradient of a rotated vector coordinates
        with respect to the three parameters of the rotation.

        :param vector: The vector being rotated, shape (Bx)3
        :returns: A matrix of gradients for each coordinate w.r.t. each parameter.
        Shape (Bx)3x3. The first axis is for the coordinate, second for the parameter.
        """
        angle_squared = (
            self._lie_vector.square().sum(dim=-1, keepdims=True).unsqueeze(-1)
        )
        angle = angle_squared.sqrt()
        is_angle_zero = angle == 0
        gamma = (vector * self._lie_vector).sum(dim=-1, keepdims=True).unsqueeze(
            -1
        ) / angle_squared
        gamma = torch.where(is_angle_zero, torch.zeros_like(gamma), gamma)
        cos_theta = torch.cos(angle)
        sin_theta_on_theta = sin_x_on_x(angle)
        one_minus_cos_theta = 1.0 - cos_theta
        cross_product = torch.linalg.cross(self._lie_vector, vector, dim=-1)

        # first term (1 - cos(theta)) ((outer(v, e) - 2 \gamma outer(e, e))/theta^2 + \gamma I)
        outer_product = vector.unsqueeze(-2) * self._lie_vector.unsqueeze(-1)
        term_1 = (
            outer_product
            - 2.0
            * self._lie_vector.unsqueeze(-2)
            * self._lie_vector.unsqueeze(-1)
            * gamma
        )
        term_1 = term_1 / angle_squared
        term_1 = torch.where(is_angle_zero, torch.zeros_like(term_1), term_1)
        term_1 = term_1 + gamma * torch.eye(3, device=vector.device).reshape(
            *(1 for _ in range(gamma.ndim - 2)), 3, 3
        )
        term_1 = term_1 * one_minus_cos_theta

        # Term 2: e ((e \gamma - v) \sin{theta}/theta + (e x v)(cos(theta) / theta^2 - sin(theta) / theta^3))
        dot_diff = self._lie_vector * gamma.squeeze(-1) - vector
        dot_diff = dot_diff * sin_theta_on_theta.squeeze(-1)
        cross_diff = cos_x_on_x_squared_minus_sin_x_on_x_cubed(angle)
        cross_diff = cross_diff.squeeze(-1) * cross_product
        term_2 = dot_diff + cross_diff
        term_2 = term_2.unsqueeze(-1) * self._lie_vector.unsqueeze(-2)

        # Term 3: derivatives from the cross product
        # (sin(theta) / theta) [[0, z, -y], [-z, 0, x], [y, -x, 0]]
        x = torch.index_select(
            vector, dim=-1, index=torch.tensor([0], device=vector.device)
        )
        y = torch.index_select(
            vector, dim=-1, index=torch.tensor([1], device=vector.device)
        )
        z = torch.index_select(
            vector, dim=-1, index=torch.tensor([2], device=vector.device)
        )
        zeros = torch.zeros_like(x)
        term_3 = torch.stack(
            [
                torch.cat([zeros, -z, y], dim=-1),
                torch.cat([z, zeros, -x], dim=-1),
                torch.cat([-y, x, zeros], dim=-1),
            ],
            dim=-1,
        )
        term_3 = term_3 * sin_theta_on_theta

        return term_1 + term_2 + term_3

    def vector_gradient(self) -> torch.Tensor:
        angle_squared = self._lie_vector.square().sum(dim=-1, keepdims=True)
        angle = angle_squared.sqrt()

        # Gradient is
        sin_theta_on_theta = sin_x_on_x(angle)
        cos_term = one_minus_cos_x_on_x_squared(angle)
        outer_product = self._lie_vector.unsqueeze(-2) * self._lie_vector.unsqueeze(-1)
        outer_product = outer_product * cos_term.unsqueeze(-1)

        a = torch.index_select(
            self._lie_vector,
            dim=-1,
            index=torch.tensor([0], device=self._lie_vector.device),
        )
        b = torch.index_select(
            self._lie_vector,
            dim=-1,
            index=torch.tensor([1], device=self._lie_vector.device),
        )
        c = torch.index_select(
            self._lie_vector,
            dim=-1,
            index=torch.tensor([2], device=self._lie_vector.device),
        )
        a = a * sin_theta_on_theta
        b = b * sin_theta_on_theta
        c = c * sin_theta_on_theta
        cross_product_derivative = torch.stack(
            [
                torch.cat([cos_theta, c, -b], dim=-1),
                torch.cat([-c, cos_theta, a], dim=-1),
                torch.cat([b, -a, cos_theta], dim=-1),
            ],
            dim=-1,
        )
        return outer_product + cross_product_derivative

    def slice(self, mask: torch.Tensor) -> Self:
        """Slice the batch dimensions (if any), returning a rotation for some of the values."""
        sliced_vector = self._lie_vector[mask]
        for _ in range(mask.ndim - 1):
            sliced_vector = sliced_vector.unsqueeze(1)
        return type(self)(sliced_vector)

    def add_lie_parameters(self, lie_vector) -> Self:
        """Add a set of parameters to the current values (such as from a gradient)"""
        return type(self)(self._lie_vector + lie_vector)

    def masked_update(self, other: Self, mask: torch.Tensor) -> Self:
        """
        Using a mask, update some of the values in this from another rotation.
        Mask should match the batch dimensions.
        """
        # mask = mask.unsqueeze(-1).tile(
        #     *(1 for _ in range(self._lie_vector.ndim - 1)), 3
        # )
        lie_vector = torch.where(mask, other._lie_vector, self._lie_vector)
        return type(self)(lie_vector)

    @classmethod
    def from_quaternion(cls: type[Self], quaternion: torch.Tensor) -> Self:
        """
        Construct a rotation object from a quaternion.
        :param quaternion: A quaternion WXYZ in the final dimension.
        :return: A LieRotation rotating as described by the quaternion
        """
        scalar = torch.index_select(
            quaternion, dim=-1, index=torch.tensor([0], device=quaternion.device)
        )
        vector = torch.index_select(
            quaternion, dim=-1, index=torch.tensor([1, 2, 3], device=quaternion.device)
        )
        vector_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
        half_angle = torch.atan2(vector_norm, scalar)
        sin_half_angle = torch.sin(half_angle)
        scale = torch.nan_to_num(2 * half_angle / sin_half_angle, nan=0.0)
        return cls(scale * vector)
