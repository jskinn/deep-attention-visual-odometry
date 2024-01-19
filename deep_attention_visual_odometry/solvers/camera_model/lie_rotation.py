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

    def __init__(self, quaternion):
        self._quaternion = quaternion

    def rotate_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Rotate a 3-vector by the current rotation
        """
        pass

    def gradient(self):
        """
        Get the gradient of the current
        """
        pass

    def add_lie_parameters(self, lie_vector) -> Self:
        pass

    def masked_update(self, other: Self) -> Self:
        """

        """
        pass
