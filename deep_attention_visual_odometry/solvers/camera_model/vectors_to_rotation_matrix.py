from typing import NamedTuple
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


class RotationMatrixAndJacobian(NamedTuple):
    r1: torch.Tensor
    r2: torch.Tensor
    r3: torch.Tensor
    r4: torch.Tensor
    r5: torch.Tensor
    r6: torch.Tensor
    r7: torch.Tensor
    r8: torch.Tensor
    r9: torch.Tensor

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


class _MatrixAndIntermediates(NamedTuple):
    r1: torch.Tensor
    r2: torch.Tensor
    r3: torch.Tensor
    r4: torch.Tensor
    r5: torch.Tensor
    r6: torch.Tensor
    r7: torch.Tensor
    r8: torch.Tensor
    r9: torch.Tensor
    a1_squared: torch.Tensor
    a2_squared: torch.Tensor
    a3_squared: torch.Tensor
    a_length: torch.Tensor
    a_square_length: torch.Tensor
    project_b_onto_a: torch.Tensor
    b_prime_1: torch.Tensor
    b_prime_2: torch.Tensor
    b_prime_3: torch.Tensor
    b_prime_1_squared: torch.Tensor
    b_prime_2_squared: torch.Tensor
    b_prime_3_squared: torch.Tensor
    b_prime_length: torch.Tensor
    b_prime_square_length: torch.Tensor


# @torch.jit.script
def _make_rotation_matrix_and_intermediates(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
) -> _MatrixAndIntermediates:
    """
    Assemble a pair of 3-vectors into a rotation matrix
    and some intermediate values necessary for computing the jacobians.

    :return:
    """
    # First we need to build the rotation matrix from the forward directions
    # This starts with two vectors a and b, we normalise a as the first column
    a1_squared = a1.square()
    a2_squared = a2.square()
    a3_squared = a3.square()
    a_square_length = a1_squared + a2_squared + a3_squared
    a_length = torch.sqrt(a_square_length)
    a_dot_b = a1 * b1 + a2 * b2 + a3 * b3
    # We subtract the component of b parallel to a, and renormalise as the second column
    project_b_onto_a = a_dot_b / a_square_length
    b_prime_1 = b1 - a1 * project_b_onto_a
    b_prime_2 = b2 - a2 * project_b_onto_a
    b_prime_3 = b3 - a3 * project_b_onto_a
    b_prime_1_squared = b_prime_1.square()
    b_prime_2_squared = b_prime_2.square()
    b_prime_3_squared = b_prime_3.square()
    b_prime_square_length = b_prime_1_squared + b_prime_2_squared + b_prime_3_squared
    b_prime_length = torch.sqrt(b_prime_square_length)
    r1 = a1 / a_length
    r4 = a2 / a_length
    r7 = a3 / a_length
    r2 = b_prime_1 / b_prime_length
    r5 = b_prime_2 / b_prime_length
    r8 = b_prime_3 / b_prime_length
    # The final column is the cross product of the first two columns
    r3 = r4 * r8 - r5 * r7
    r6 = r2 * r7 - r1 * r8
    r9 = r1 * r5 - r2 * r4
    return _MatrixAndIntermediates(
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        r5=r5,
        r6=r6,
        r7=r7,
        r8=r8,
        r9=r9,
        a1_squared=a1_squared,
        a2_squared=a2_squared,
        a3_squared=a3_squared,
        a_length=a_length,
        a_square_length=a_square_length,
        project_b_onto_a=project_b_onto_a,
        b_prime_1=b_prime_1,
        b_prime_2=b_prime_2,
        b_prime_3=b_prime_3,
        b_prime_1_squared=b_prime_1_squared,
        b_prime_2_squared=b_prime_2_squared,
        b_prime_3_squared=b_prime_3_squared,
        b_prime_length=b_prime_length,
        b_prime_square_length=b_prime_square_length,
    )


# @torch.jit.script
def make_rotation_matrix(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
) -> RotationMatrix:
    """
    Compute a rotation matrix from two vectors, a and b.
    A is normalised as the first column of R,
    b is orthogonalised as b - (a . b) a, and then normalised as the second column.
    The third column is the cross product of a and the orthonormalised b.

    :return:
    """
    matrix_and_intermediates = _make_rotation_matrix_and_intermediates(
        a1, a2, a3, b1, b2, b3
    )
    return RotationMatrix(
        r1=matrix_and_intermediates.r1,
        r2=matrix_and_intermediates.r2,
        r3=matrix_and_intermediates.r3,
        r4=matrix_and_intermediates.r4,
        r5=matrix_and_intermediates.r5,
        r6=matrix_and_intermediates.r6,
        r7=matrix_and_intermediates.r7,
        r8=matrix_and_intermediates.r8,
        r9=matrix_and_intermediates.r9,
    )


@torch.jit.script
def make_rotation_matrix_and_derivatives(
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
) -> RotationMatrixAndJacobian:
    """
    Build a rotation matrix from two vectors, and return both the matrix itself and it's derivatives
    :return: A named tuple with every element of the 3x3 rotation matrix, and its derivative w.r.t. each input.
    """
    # First make the rotation matrix
    matrix_and_intermediates = _make_rotation_matrix_and_intermediates(
        a1, a2, a3, b1, b2, b3
    )

    a1_squared = matrix_and_intermediates.a1_squared
    a2_squared = matrix_and_intermediates.a2_squared
    a3_squared = matrix_and_intermediates.a3_squared
    a_length = matrix_and_intermediates.a_length
    a_square_length = matrix_and_intermediates.a_square_length

    project_b_onto_a = matrix_and_intermediates.project_b_onto_a
    b_prime_1 = matrix_and_intermediates.b_prime_1
    b_prime_2 = matrix_and_intermediates.b_prime_2
    b_prime_3 = matrix_and_intermediates.b_prime_3
    b_prime_1_squared = matrix_and_intermediates.b_prime_1_squared
    b_prime_2_squared = matrix_and_intermediates.b_prime_2_squared
    b_prime_3_squared = matrix_and_intermediates.b_prime_3_squared
    b_prime_length = matrix_and_intermediates.b_prime_length
    b_prime_square_length = matrix_and_intermediates.b_prime_square_length

    r1 = matrix_and_intermediates.r1
    r2 = matrix_and_intermediates.r2
    r3 = matrix_and_intermediates.r3
    r4 = matrix_and_intermediates.r4
    r5 = matrix_and_intermediates.r5
    r6 = matrix_and_intermediates.r6
    r7 = matrix_and_intermediates.r7
    r8 = matrix_and_intermediates.r8
    r9 = matrix_and_intermediates.r9

    a_length_cubed = a_length * a_square_length
    a1_a2 = a1 * a2
    a1_a3 = a1 * a3
    a2_a3 = a2 * a3

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

    return RotationMatrixAndJacobian(
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        r5=r5,
        r6=r6,
        r7=r7,
        r8=r8,
        r9=r9,
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
