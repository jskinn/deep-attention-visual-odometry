from typing import Self
import torch


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
        gamma = torch.nan_to_num(
            (vector * self._lie_vector).sum(dim=-1, keepdims=True) / angle_squared
        )
        cos_theta = torch.cos(angle)
        # Lim x->0 sin(x)/x = 1.0
        sin_theta_on_theta = torch.nan_to_num(torch.sin(angle) / angle, nan=1.0)
        # Compute the sin coefficients for each output.
        sin_coefficients = torch.cross(self._lie_vector, vector)
        gamma_axis = self._lie_vector * gamma
        out_vector = (
            vector * cos_theta
            + (1 - cos_theta) * gamma_axis
            + sin_coefficients * sin_theta_on_theta
        )
        return out_vector

    def gradient(self):
        """
        Get the gradient of the current
        """
        pass

    def add_lie_parameters(self, lie_vector) -> Self:
        pass

    def masked_update(self, other: Self, mask: torch.Tensor) -> Self:
        """ """
        lie_vector = torch.where(mask, other._lie_vector, self._lie_vector)
        return type(self)(lie_vector)

    @classmethod
    def from_quaternion(cls: type[Self], quaternion: torch.Tensor) -> Self:
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
