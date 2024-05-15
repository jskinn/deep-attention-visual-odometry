from typing import NamedTuple, Self
import torch


class RotationMatrix(NamedTuple):
    r1: torch.Tensor
    r2: torch.Tensor
    r3: torch.Tensor
    r4: torch.Tensor
    r5: torch.Tensor
    r6: torch.Tensor
    r7: torch.Tensor
    r8: torch.Tensor
    r9: torch.Tensor


class RotationMatrixDerivatives(NamedTuple):
    dr1_da1: torch.Tensor
    dr1_da2: torch.Tensor
    dr1_da3: torch.Tensor
    dr2_da1: torch.Tensor
    dr2_da2: torch.Tensor
    dr2_da3: torch.Tensor
    dr3_da1: torch.Tensor
    dr3_da2: torch.Tensor
    dr3_da3: torch.Tensor
    dr4_da1: torch.Tensor
    dr4_da2: torch.Tensor
    dr4_da3: torch.Tensor
    dr5_da1: torch.Tensor
    dr5_da2: torch.Tensor
    dr5_da3: torch.Tensor
    dr6_da1: torch.Tensor
    dr6_da2: torch.Tensor
    dr6_da3: torch.Tensor
    dr7_da1: torch.Tensor
    dr7_da2: torch.Tensor
    dr7_da3: torch.Tensor
    dr8_da1: torch.Tensor
    dr8_da2: torch.Tensor
    dr8_da3: torch.Tensor
    dr9_da1: torch.Tensor
    dr9_da2: torch.Tensor
    dr9_da3: torch.Tensor

    dr2_db1: torch.Tensor
    dr2_db2: torch.Tensor
    dr2_db3: torch.Tensor
    dr3_db1: torch.Tensor
    dr3_db2: torch.Tensor
    dr3_db3: torch.Tensor
    dr5_db1: torch.Tensor
    dr5_db2: torch.Tensor
    dr5_db3: torch.Tensor
    dr6_db1: torch.Tensor
    dr6_db2: torch.Tensor
    dr6_db3: torch.Tensor
    dr8_db1: torch.Tensor
    dr8_db2: torch.Tensor
    dr8_db3: torch.Tensor
    dr9_db1: torch.Tensor
    dr9_db2: torch.Tensor
    dr9_db3: torch.Tensor


class _Intermediates(NamedTuple):
    a_squared: torch.Tensor
    a_length: torch.Tensor
    a_square_length: torch.Tensor
    project_b_onto_a: torch.Tensor
    b_prime: torch.Tensor
    b_prime_squared: torch.Tensor
    b_prime_length: torch.Tensor
    b_prime_square_length: torch.Tensor


class TwoVectorOrientation:
    """
    A rotation matrix made from two basis vectors, that are orthonormalised.
    Expects a batch dimension, a multiple estimates dimension, and a multiple views dimension,
    so the input points are BxExMx3
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        self._a = a
        self._b = b
        self._rotation_matrix = None
        self._gradients = None
        self._intermediates = None

    @property
    def a(self) -> torch.Tensor:
        return self._a

    @property
    def b(self) -> torch.Tensor:
        return self._b

    def get_rotation_matrix(self) -> RotationMatrix:
        """
        :returns: named tuple of BxExM tensors
        """
        if self._rotation_matrix is None:
            intermediates = self._get_intermediates()
            col_1 = self._a / intermediates.a_length.unsqueeze(-1)
            col_2 = intermediates.b_prime / intermediates.b_prime_length.unsqueeze(-1)
            # The final column is the cross product of the first two columns
            col_3 = torch.linalg.cross(col_1, col_2, dim=-1)
            self._rotation_matrix = RotationMatrix(
                r1=col_1[:, :, :, 0],
                r2=col_2[:, :, :, 0],
                r3=col_3[:, :, :, 0],
                r4=col_1[:, :, :, 1],
                r5=col_2[:, :, :, 1],
                r6=col_3[:, :, :, 1],
                r7=col_1[:, :, :, 2],
                r8=col_2[:, :, :, 2],
                r9=col_3[:, :, :, 2],
            )
        return self._rotation_matrix

    def get_derivatives(self) -> RotationMatrixDerivatives:
        """
        :returns: Named tuple of BxExM tensors
        """
        if self._gradients is None:
            intermediates = self._get_intermediates()
            rotation_matrix = self.get_rotation_matrix()
            self._gradients = _compute_derivatives(
                self._a, self._b, rotation_matrix, intermediates
            )
        return self._gradients

    def add(self, delta: torch.Tensor) -> Self:
        return TwoVectorOrientation(
            a=self._a + delta[:, :, :, 0:3], b=self._b + delta[:, :, :, 3:6]
        )

    def _get_intermediates(self) -> _Intermediates:
        if self._intermediates is None:
            # First we need to build the rotation matrix from the forward directions
            # This starts with two vectors a and b, we normalise a as the first column
            a_squared = self._a.square()
            a_square_length = torch.sum(a_squared, dim=-1)
            a_length = torch.sqrt(a_square_length)
            a_dot_b = (self._a * self._b).sum(dim=-1)
            # We subtract the component of b parallel to a, and renormalise as the second column
            project_b_onto_a = a_dot_b / a_square_length
            b_prime = self._b - self._a * project_b_onto_a.unsqueeze(-1)
            b_prime_squared = b_prime.square()
            b_prime_square_length = b_prime_squared.sum(dim=-1)
            b_prime_length = torch.sqrt(b_prime_square_length)
            self._intermediates = _Intermediates(
                a_squared=a_squared,
                a_length=a_length,
                a_square_length=a_square_length,
                project_b_onto_a=project_b_onto_a,
                b_prime=b_prime,
                b_prime_squared=b_prime_squared,
                b_prime_square_length=b_prime_square_length,
                b_prime_length=b_prime_length,
            )
        return self._intermediates


def _compute_derivatives(
    a_vec: torch.Tensor,
    b_vec: torch.Tensor,
    rotation_matrix: RotationMatrix,
    intermediates: _Intermediates,
) -> RotationMatrixDerivatives:
    a1 = a_vec[:, :, :, 0]
    a2 = a_vec[:, :, :, 1]
    a3 = a_vec[:, :, :, 2]

    b1 = b_vec[:, :, :, 0]
    b2 = b_vec[:, :, :, 1]
    b3 = b_vec[:, :, :, 2]

    a1_squared = intermediates.a_squared[:, :, :, 0]
    a2_squared = intermediates.a_squared[:, :, :, 1]
    a3_squared = intermediates.a_squared[:, :, :, 2]
    a_length = intermediates.a_length
    a_square_length = intermediates.a_square_length

    project_b_onto_a = intermediates.project_b_onto_a
    b_prime_1 = intermediates.b_prime[:, :, :, 0]
    b_prime_2 = intermediates.b_prime[:, :, :, 1]
    b_prime_3 = intermediates.b_prime[:, :, :, 2]
    b_prime_1_squared = intermediates.b_prime_squared[:, :, :, 0]
    b_prime_2_squared = intermediates.b_prime_squared[:, :, :, 1]
    b_prime_3_squared = intermediates.b_prime_squared[:, :, :, 2]
    b_prime_length = intermediates.b_prime_length
    b_prime_square_length = intermediates.b_prime_square_length

    r1 = rotation_matrix.r1
    r2 = rotation_matrix.r2
    r4 = rotation_matrix.r4
    r5 = rotation_matrix.r5
    r7 = rotation_matrix.r7
    r8 = rotation_matrix.r8

    a_length_cubed = a_length * a_square_length
    a1_a2 = a_vec[:, :, :, 0] * a_vec[:, :, :, 1]
    a1_a3 = a_vec[:, :, :, 0] * a_vec[:, :, :, 2]
    a2_a3 = a_vec[:, :, :, 1] * a_vec[:, :, :, 2]

    # The first column only vary with a, through the normalisation
    dr1_da1 = (a2_squared + a3_squared) / a_length_cubed
    dr1_da2 = -a1_a2 / a_length_cubed
    dr1_da3 = -a1_a3 / a_length_cubed
    dr4_da1 = dr1_da2
    dr4_da2 = (a1_squared + a3_squared) / a_length_cubed
    dr4_da3 = -a2_a3 / a_length_cubed
    dr7_da1 = dr1_da3
    dr7_da2 = dr4_da3
    dr7_da3 = (a1_squared + a2_squared) / a_length_cubed

    # The second column vary with both a and b, and are complex, particularly for a
    # We first build a bunch of intermediate derivatives, for the projection term and
    # the difference b_prime = b - proj(b, a) a
    dproj_da1 = (b1 - 2 * a1 * project_b_onto_a) / a_square_length
    dproj_da2 = (b2 - 2 * a2 * project_b_onto_a) / a_square_length
    dproj_da3 = (b3 - 2 * a3 * project_b_onto_a) / a_square_length
    dbprime1_da1 = -a1 * dproj_da1 - project_b_onto_a
    dbprime1_da2 = -a1 * dproj_da2
    dbprime1_da3 = -a1 * dproj_da3
    dbprime2_da1 = -a2 * dproj_da1
    dbprime2_da2 = -a2 * dproj_da2 - project_b_onto_a
    dbprime2_da3 = -a2 * dproj_da3
    dbprime3_da1 = -a3 * dproj_da1
    dbprime3_da2 = -a3 * dproj_da2
    dbprime3_da3 = -a3 * dproj_da3 - project_b_onto_a

    # Now we can express the gradients of the second column based on these intermediate gradients
    b_prime_length_cubed = b_prime_length * b_prime_square_length
    b_prime_1_2 = b_prime_1 * b_prime_2
    b_prime_1_3 = b_prime_1 * b_prime_3
    b_prime_2_3 = b_prime_2 * b_prime_3
    dr2_da1 = (
        dbprime1_da1 * (b_prime_2_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime2_da1
        - b_prime_1_3 * dbprime3_da1
    ) / b_prime_length_cubed
    dr2_da2 = (
        dbprime1_da2 * (b_prime_2_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime2_da2
        - b_prime_1_3 * dbprime3_da2
    ) / b_prime_length_cubed
    dr2_da3 = (
        dbprime1_da3 * (b_prime_2_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime2_da3
        - b_prime_1_3 * dbprime3_da3
    ) / b_prime_length_cubed
    dr5_da1 = (
        dbprime2_da1 * (b_prime_1_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime1_da1
        - b_prime_2_3 * dbprime3_da1
    ) / b_prime_length_cubed
    dr5_da2 = (
        dbprime2_da2 * (b_prime_1_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime1_da2
        - b_prime_2_3 * dbprime3_da2
    ) / b_prime_length_cubed
    dr5_da3 = (
        dbprime2_da3 * (b_prime_1_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime1_da3
        - b_prime_2_3 * dbprime3_da3
    ) / b_prime_length_cubed
    dr8_da1 = (
        dbprime3_da1 * (b_prime_1_squared + b_prime_2_squared)
        - b_prime_1_3 * dbprime1_da1
        - b_prime_2_3 * dbprime2_da1
    ) / b_prime_length_cubed
    dr8_da2 = (
        dbprime3_da2 * (b_prime_1_squared + b_prime_2_squared)
        - b_prime_1_3 * dbprime1_da2
        - b_prime_2_3 * dbprime2_da2
    ) / b_prime_length_cubed
    dr8_da3 = (
        dbprime3_da3 * (b_prime_1_squared + b_prime_2_squared)
        - b_prime_1_3 * dbprime1_da3
        - b_prime_2_3 * dbprime2_da3
    ) / b_prime_length_cubed

    # The gradients wrt b are simpler
    dbprime1_db1 = 1 + a1_squared / a_square_length
    dbprime1_db2 = a1_a2 / a_square_length
    dbprime1_db3 = a1_a3 / a_square_length
    dbprime2_db1 = dbprime1_db2
    dbprime2_db2 = 1 + a2_squared / a_square_length
    dbprime2_db3 = a2_a3 / a_square_length
    dbprime3_db1 = dbprime1_db3
    dbprime3_db2 = dbprime2_db3
    dbprime3_db3 = 1 + a3_squared / a_square_length

    dr2_db1 = (
        dbprime1_db1 * (b_prime_2_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime2_db1
        - b_prime_1_3 * dbprime3_db1
    ) / b_prime_length_cubed
    dr2_db2 = (
        dbprime1_db2 * (b_prime_2_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime2_db2
        - b_prime_1_3 * dbprime3_db2
    ) / b_prime_length_cubed
    dr2_db3 = (
        dbprime1_db3 * (b_prime_2_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime2_db3
        - b_prime_1_3 * dbprime3_db3
    ) / b_prime_length_cubed
    dr5_db1 = (
        dbprime2_db1 * (b_prime_1_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime1_db1
        - b_prime_2_3 * dbprime3_db1
    ) / b_prime_length_cubed
    dr5_db2 = (
        dbprime2_db2 * (b_prime_1_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime1_db2
        - b_prime_2_3 * dbprime3_db2
    ) / b_prime_length_cubed
    dr5_db3 = (
        dbprime2_db3 * (b_prime_1_squared + b_prime_3_squared)
        - b_prime_1_2 * dbprime1_db3
        - b_prime_2_3 * dbprime3_db3
    ) / b_prime_length_cubed
    dr8_db1 = (
        dbprime3_db1 * (b_prime_1_squared + b_prime_2_squared)
        - b_prime_1_3 * dbprime1_db1
        - b_prime_2_3 * dbprime2_db1
    ) / b_prime_length_cubed
    dr8_db2 = (
        dbprime3_db2 * (b_prime_1_squared + b_prime_2_squared)
        - b_prime_1_3 * dbprime1_db2
        - b_prime_2_3 * dbprime2_db2
    ) / b_prime_length_cubed
    dr8_db3 = (
        dbprime3_db3 * (b_prime_1_squared + b_prime_2_squared)
        - b_prime_1_3 * dbprime1_db3
        - b_prime_2_3 * dbprime2_db3
    ) / b_prime_length_cubed

    # The gradient of the final column comes from the cross product, and can be built from the existing values
    dr3_da1 = r4 * dr8_da1 + r8 * dr4_da1 - r5 * dr7_da1 - r7 * dr5_da1
    dr3_da2 = r4 * dr8_da2 + r8 * dr4_da2 - r5 * dr7_da2 - r7 * dr5_da2
    dr3_da3 = r4 * dr8_da3 + r8 * dr4_da3 - r5 * dr7_da3 - r7 * dr5_da3
    dr3_db1 = r4 * dr8_db1 - r7 * dr5_db1
    dr3_db2 = r4 * dr8_db2 - r7 * dr5_db2
    dr3_db3 = r4 * dr8_db3 - r7 * dr5_db3

    dr6_da1 = r2 * dr7_da1 + r7 * dr2_da1 - r1 * dr8_da1 - r8 * dr1_da1
    dr6_da2 = r2 * dr7_da2 + r7 * dr2_da2 - r1 * dr8_da2 - r8 * dr1_da2
    dr6_da3 = r2 * dr7_da3 + r7 * dr2_da3 - r1 * dr8_da3 - r8 * dr1_da3
    dr6_db1 = r7 * dr2_db1 - r1 * dr8_db1
    dr6_db2 = r7 * dr2_db2 - r1 * dr8_db2
    dr6_db3 = r7 * dr2_db3 - r1 * dr8_db3

    dr9_da1 = r1 * dr5_da1 + r5 * dr1_da1 - r2 * dr4_da1 - r4 * dr2_da1
    dr9_da2 = r1 * dr5_da2 + r5 * dr1_da2 - r2 * dr4_da2 - r4 * dr2_da2
    dr9_da3 = r1 * dr5_da3 + r5 * dr1_da3 - r2 * dr4_da3 - r4 * dr2_da3
    dr9_db1 = r1 * dr5_db1 - r4 * dr2_db1
    dr9_db2 = r1 * dr5_db2 - r4 * dr2_db2
    dr9_db3 = r1 * dr5_db3 - r4 * dr2_db3

    return RotationMatrixDerivatives(
        dr1_da1=dr1_da1,
        dr1_da2=dr1_da2,
        dr1_da3=dr1_da3,
        dr2_da1=dr2_da1,
        dr2_da2=dr2_da2,
        dr2_da3=dr2_da3,
        dr3_da1=dr3_da1,
        dr3_da2=dr3_da2,
        dr3_da3=dr3_da3,
        dr4_da1=dr4_da1,
        dr4_da2=dr4_da2,
        dr4_da3=dr4_da3,
        dr5_da1=dr5_da1,
        dr5_da2=dr5_da2,
        dr5_da3=dr5_da3,
        dr6_da1=dr6_da1,
        dr6_da2=dr6_da2,
        dr6_da3=dr6_da3,
        dr7_da1=dr7_da1,
        dr7_da2=dr7_da2,
        dr7_da3=dr7_da3,
        dr8_da1=dr8_da1,
        dr8_da2=dr8_da2,
        dr8_da3=dr8_da3,
        dr9_da1=dr9_da1,
        dr9_da2=dr9_da2,
        dr9_da3=dr9_da3,
        dr2_db1=dr2_db1,
        dr2_db2=dr2_db2,
        dr2_db3=dr2_db3,
        dr3_db1=dr3_db1,
        dr3_db2=dr3_db2,
        dr3_db3=dr3_db3,
        dr5_db1=dr5_db1,
        dr5_db2=dr5_db2,
        dr5_db3=dr5_db3,
        dr6_db1=dr6_db1,
        dr6_db2=dr6_db2,
        dr6_db3=dr6_db3,
        dr8_db1=dr8_db1,
        dr8_db2=dr8_db2,
        dr8_db3=dr8_db3,
        dr9_db1=dr9_db1,
        dr9_db2=dr9_db2,
        dr9_db3=dr9_db3,
    )
