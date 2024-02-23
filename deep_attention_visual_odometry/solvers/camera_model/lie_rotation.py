from typing import Self
import torch
from deep_attention_visual_odometry.utils import (
    sin_x_on_x,
    one_minus_cos_x_on_x_squared,
    cos_x_on_x_squared_minus_sin_x_on_x_cubed,
    sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth,
)


class LieRotation:
    """
    A differentiable rotation parameter based around the Lie theory for SO(3).

    There are multiple possible representations of a rotation in SO(3), including
    quaternions and 3x3 rotation matrices.
    Aside from constructor, this class should be largely opaque to the user as to which
    representation it uses internally.

    TODO: This should probably be replaced with a pair of autograd functions.
    """

    def __init__(self, lie_vector):
        self._lie_vector = lie_vector
        self._angle_squared = None
        self._angle = None
        self._sin_theta_on_theta = None
        self._one_minus_cos_theta_on_theta_squared = None
        self._outer_product = None

    def rotate_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Rotate a 3-vector by the current rotation
        """
        # This is based on maths that combines a conversion from axis-angle to quaternion
        # with the hamilton product to rotate the vector.
        # Produces a variant of Rodrigues' rotation formula
        angle = self.angle()
        dot_product = (vector * self._lie_vector).sum(dim=-1, keepdims=True)
        cross_product = torch.linalg.cross(self._lie_vector, vector, dim=-1)
        cos_theta = torch.cos(angle)
        sin_theta_on_theta = sin_x_on_x(angle)
        one_minus_cos_term = one_minus_cos_x_on_x_squared(angle)
        out_vector = (
            vector * cos_theta
            + one_minus_cos_term * dot_product * self._lie_vector
            + cross_product * sin_theta_on_theta
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
        # Compute the derivative of a Rodrigues rotation w.r.t. the axis of rotation.
        # This has a bunch of divisions, which if you're not careful, end up dividing by zero.
        # It's important to correctly group the trigonometric terms, to avoid this.
        # Those key groupings have been delegated to other autograd functions.
        angle = self.angle()
        sin_theta_on_theta = self.sin_theta_on_theta()
        one_minus_cos_theta_on_theta_squared = (
            self.one_minus_cos_theta_on_theta_squared()
        )
        cos_theta_minus_sin_theta_term = cos_x_on_x_squared_minus_sin_x_on_x_cubed(
            angle
        )
        sin_x_on_x_cubed_term = sin_x_on_x_cubed_minus_two_one_minus_cos_x_on_x_fourth(
            angle
        )

        # We also need the dot and cross products
        dot_product = (vector * self._lie_vector).sum(dim=-1, keepdims=True)
        cross_product = torch.linalg.cross(self._lie_vector, vector, dim=-1)

        # The actual gradient matrices are built up from a series of outer products
        outer_product = self._lie_vector.unsqueeze(-2) * vector.unsqueeze(-1)
        axis_outer_product = self.axis_outer_product()
        axis_cross_outer_product = self._lie_vector.unsqueeze(
            -2
        ) * cross_product.unsqueeze(-1)
        outer_product_transpose = torch.transpose(outer_product, -2, -1)

        # first term: The gradients of x cos(theta)
        term_1 = -1.0 * outer_product * sin_theta_on_theta.unsqueeze(-1)

        # Second term: the first two terms of the gradient of the (1 - cos(x)) term
        term_2 = (dot_product * sin_x_on_x_cubed_term).unsqueeze(
            -1
        ) * axis_outer_product

        # Third term: The other (two) terms from the gradient of the (1-cos(x)) term
        term_3 = dot_product.unsqueeze(-1) * torch.eye(3, device=dot_product.device)
        term_3 = one_minus_cos_theta_on_theta_squared.unsqueeze(-1) * (
            outer_product_transpose + term_3
        )

        # Fourth term: derivatives from the cross product term
        term_4 = axis_cross_outer_product * cos_theta_minus_sin_theta_term.unsqueeze(-1)

        # Term 5: derivatives from the cross product
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
        term_5 = torch.stack(
            [
                torch.cat([zeros, -z, y], dim=-1),
                torch.cat([z, zeros, -x], dim=-1),
                torch.cat([-y, x, zeros], dim=-1),
            ],
            dim=-1,
        )
        term_5 = term_5 * sin_theta_on_theta.unsqueeze(-1)

        return term_1 + term_2 + term_3 + term_4 + term_5

    def vector_gradient(self) -> torch.Tensor:
        # Gradient is simpler for the vector parameters, partly because they are never
        # in the denominator or inside the sin/cos.
        # Gradient is then the outer product times (1 - cos(x)) / x^2
        # plus sin(x) / x times a skew-symmetric term for the cross product
        angle = self.angle()
        cos_theta = torch.cos(angle)
        sin_theta_on_theta = self.sin_theta_on_theta()
        one_minus_cos_theta_term = self.one_minus_cos_theta_on_theta_squared()

        outer_product = self._lie_vector.unsqueeze(-2) * self._lie_vector.unsqueeze(-1)
        outer_product = outer_product * one_minus_cos_theta_term.unsqueeze(-1)

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

    def angle_squared(self) -> torch.Tensor:
        if self._angle_squared is None:
            self._angle_squared = self._lie_vector.square().sum(dim=-1, keepdims=True)
        return self._angle_squared

    def angle(self) -> torch.Tensor:
        if self._angle is None:
            self._angle = torch.linalg.norm(self._lie_vector, dim=-1, keepdim=True)
        return self._angle

    def sin_theta_on_theta(self) -> torch.Tensor:
        if self._sin_theta_on_theta is None:
            self._sin_theta_on_theta = sin_x_on_x(self.angle())
        return self._sin_theta_on_theta

    def one_minus_cos_theta_on_theta_squared(self) -> torch.Tensor:
        if self._one_minus_cos_theta_on_theta_squared is None:
            self._one_minus_cos_theta_on_theta_squared = one_minus_cos_x_on_x_squared(
                self.angle()
            )
        return self._one_minus_cos_theta_on_theta_squared

    def axis_outer_product(self) -> torch.Tensor:
        if self._outer_product is None:
            self._outer_product = self._lie_vector.unsqueeze(
                -2
            ) * self._lie_vector.unsqueeze(-1)
        return self._outer_product
