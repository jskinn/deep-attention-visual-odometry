from typing import NamedTuple, Tuple
import torch
import deep_attention_visual_odometry.solvers.camera_model.vectors_to_rotation_matrix as vec_to_rot
import deep_attention_visual_odometry.solvers.camera_model.simple_camera_model_parameters as pidx
from .i_camera_model import ICameraModel


class SimpleCameraModel(ICameraModel):

    def forward_model(self, camera_parameters) -> torch.Tensor:
        pass

    def forward_model_and_jacobian(self, camera_parameters) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class _PointsAndIntermediates(NamedTuple):
    u: torch.Tensor
    v: torch.Tensor
    x_prime: torch.Tensor
    y_prime: torch.Tensor
    z_prime: torch.Tensor


# @torch.jit.script
def _simple_forward_model(
    points_3d: torch.Tensor,
    parameters: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    r3: torch.Tensor,
    r4: torch.Tensor,
    r5: torch.Tensor,
    r6: torch.Tensor,
    r7: torch.Tensor,
    r8: torch.Tensor,
    r9: torch.Tensor,
) -> _PointsAndIntermediates:
    """
    The forward model part of the simple camera model
    :param points_3d:
    :param parameters:
    :return:
    """
    # Multiply by the assembled rotation matrix, and translate
    x_prime = (
        points_3d[:, :, 0] * r1
        + points_3d[:, :, 1] * r2
        + points_3d[:, :, 2] * r3
        + parameters[:, None, pidx.TX]
    )
    y_prime = (
        points_3d[:, :, 0] * r4
        + points_3d[:, :, 1] * r5
        + points_3d[:, :, 2] * r6
        + parameters[:, None, pidx.TY]
    )
    z_prime = (
        points_3d[:, :, 0] * r7
        + points_3d[:, :, 1] * r8
        + points_3d[:, :, 2] * r9
        + parameters[:, None, pidx.TZ]
    )
    # Add a little bit for Z' == 0 since it is mostly in the denominator
    z_prime[z_prime == 0] += 1e-8
    # Now use the camera intrinsics to compute the undistorted pixel coordinates
    u = parameters[:, None, pidx.F] * (x_prime / z_prime) + parameters[:, None, pidx.CX]
    v = parameters[:, None, pidx.F] * (y_prime / z_prime) + parameters[:, None, pidx.CY]
    return _PointsAndIntermediates(
        u=u,
        v=v,
        x_prime=x_prime,
        y_prime=y_prime,
        z_prime=z_prime,
    )


@torch.jit.script
def compute_simple_camera_model(
    points_3d: torch.Tensor, parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the forward pass of a simple camera model, with 3 intrinsic and 9 extrinsic parameters
    The model is first translation and rotation
    x' = r1 x + r2 y + r3 z + tx
    y' = r4 x + r5 y + r6 z + ty
    z' = r7 x + r8 y + r9 z + tz
    Then camera projection
    u = f * (x' / z') + c_x
    v = f * (y' / z') + c_y

    The rotation matrix r1..r9 is computed from two vectors a and b, that are orthononormalised.

    :param points_3d:
    :param parameters:
    :return:
    """
    matrix_data = vec_to_rot.make_rotation_matrix(
        a1=parameters[:, None, pidx.A1],
        a2=parameters[:, None, pidx.A2],
        a3=parameters[:, None, pidx.A3],
        b1=parameters[:, None, pidx.B1],
        b2=parameters[:, None, pidx.B2],
        b3=parameters[:, None, pidx.B3],
    )
    forward_model = _simple_forward_model(
        points_3d=points_3d,
        parameters=parameters,
        r1=matrix_data.r1,
        r2=matrix_data.r2,
        r3=matrix_data.r3,
        r4=matrix_data.r4,
        r5=matrix_data.r5,
        r6=matrix_data.r6,
        r7=matrix_data.r7,
        r8=matrix_data.r8,
        r9=matrix_data.r9,
    )
    return forward_model.u, forward_model.v


# @torch.jit.script
def compute_simple_camera_model_and_jacobian(
    points_3d: torch.Tensor, parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate both the forward camera model and the jacobian of the simple camera model.
    Reuses intermediate values from the forward pass to calculate the jacobian, so it's easier together.
    See `compute_simple_camera_model` for a description of the camera model.
    :param points_3d:
    :param parameters:
    :return:
    """
    # First handle computation of the rotation matrix and its derivatives
    matrix_data_and_derivatives = vec_to_rot.make_rotation_matrix_and_derivatives(
        a1=parameters[:, None, pidx.A1],
        a2=parameters[:, None, pidx.A2],
        a3=parameters[:, None, pidx.A3],
        b1=parameters[:, None, pidx.B1],
        b2=parameters[:, None, pidx.B2],
        b3=parameters[:, None, pidx.B3],
    )

    # Then the forward model, grabbing intermediate values as well
    forward_model = _simple_forward_model(
        points_3d=points_3d,
        parameters=parameters,
        r1=matrix_data_and_derivatives.r1,
        r2=matrix_data_and_derivatives.r2,
        r3=matrix_data_and_derivatives.r3,
        r4=matrix_data_and_derivatives.r4,
        r5=matrix_data_and_derivatives.r5,
        r6=matrix_data_and_derivatives.r6,
        r7=matrix_data_and_derivatives.r7,
        r8=matrix_data_and_derivatives.r8,
        r9=matrix_data_and_derivatives.r9,
    )
    u = forward_model.u
    v = forward_model.v
    x_prime = forward_model.x_prime
    y_prime = forward_model.y_prime
    z_prime = forward_model.z_prime

    f_on_z_prime = parameters[:, None, pidx.F] / z_prime
    x_on_z_prime = x_prime / z_prime
    y_on_z_prime = y_prime / z_prime
    du_dxprime = f_on_z_prime
    dv_dyprime = f_on_z_prime
    du_dzprime = -f_on_z_prime * x_on_z_prime  # = -f x / z^2
    dv_dzprime = -f_on_z_prime * y_on_z_prime

    # Camera parameter derivatives are fairly simple, there's only three of them
    partial_du_dcx = torch.ones_like(u)
    partial_dv_dcx = torch.zeros_like(v)
    partial_du_dcy = torch.zeros_like(u)
    partial_dv_dcy = torch.ones_like(v)
    partial_du_df = x_on_z_prime
    partial_dv_df = y_on_z_prime

    # Translation parameters are also fairly simple
    partial_du_dtx = du_dxprime
    partial_dv_dtx = torch.zeros_like(v)
    partial_du_dty = torch.zeros_like(u)
    partial_dv_dty = dv_dyprime
    partial_du_dtz = du_dzprime
    partial_dv_dtz = dv_dzprime

    # The direction vector derivatives are built from the rotation vector derivatives
    # Basically, we sum up du_dxprime * dxprime_dr1 * dr1_da1 for each coordinate and rotation matrix element
    # dxprime_dr1 = x, dxprime_dr2 = y, dxprime_dr3 = z, and so on
    partial_du_da1 = (
        du_dxprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr1_da1
        + du_dxprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr2_da1
        + du_dxprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr3_da1
        + du_dzprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr7_da1
        + du_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_da1
        + du_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_da1
    )
    partial_du_da2 = (
        du_dxprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr1_da2
        + du_dxprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr2_da2
        + du_dxprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr3_da2
        + du_dzprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr7_da2
        + du_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_da2
        + du_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_da2
    )
    partial_du_da3 = (
        du_dxprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr1_da3
        + du_dxprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr2_da3
        + du_dxprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr3_da3
        + du_dzprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr7_da3
        + du_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_da3
        + du_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_da3
    )
    partial_dv_da1 = (
        dv_dyprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr4_da1
        + dv_dyprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr5_da1
        + dv_dyprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr6_da1
        + dv_dzprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr7_da1
        + dv_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_da1
        + dv_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_da1
    )
    partial_dv_da2 = (
        dv_dyprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr4_da2
        + dv_dyprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr5_da2
        + dv_dyprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr6_da2
        + dv_dzprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr7_da2
        + dv_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_da2
        + dv_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_da2
    )
    partial_dv_da3 = (
        dv_dyprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr4_da3
        + dv_dyprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr5_da3
        + dv_dyprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr6_da3
        + dv_dzprime * points_3d[:, :, 0] * matrix_data_and_derivatives.dr7_da3
        + dv_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_da3
        + dv_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_da3
    )
    partial_du_db1 = (
        du_dxprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr2_db1
        + du_dxprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr3_db1
        + du_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_db1
        + du_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_db1
    )
    partial_du_db2 = (
        du_dxprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr2_db2
        + du_dxprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr3_db2
        + du_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_db2
        + du_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_db2
    )
    partial_du_db3 = (
        du_dxprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr2_db3
        + du_dxprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr3_db3
        + du_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_db3
        + du_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_db3
    )
    partial_dv_db1 = (
        dv_dyprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr5_db1
        + dv_dyprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr6_db1
        + dv_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_db1
        + dv_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_db1
    )
    partial_dv_db2 = (
        dv_dyprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr5_db2
        + dv_dyprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr6_db2
        + dv_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_db2
        + dv_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_db2
    )
    partial_dv_db3 = (
        dv_dyprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr5_db3
        + dv_dyprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr6_db3
        + dv_dzprime * points_3d[:, :, 1] * matrix_data_and_derivatives.dr8_db3
        + dv_dzprime * points_3d[:, :, 2] * matrix_data_and_derivatives.dr9_db3
    )

    # Build the jacobian, a 2Nx12 matrix measuring the rate of change of each residual w.r.t. each parameter
    # I'm hoping the compiler is smart enough to optimize down to the reduction
    jacobian = torch.stack(
        [
            torch.cat([partial_du_dcx, partial_dv_dcx], dim=1),
            torch.cat([partial_du_dcy, partial_dv_dcy], dim=1),
            torch.cat([partial_du_df, partial_dv_df], dim=1),
            torch.cat([partial_du_da1, partial_dv_da1], dim=1),
            torch.cat([partial_du_da2, partial_dv_da2], dim=1),
            torch.cat([partial_du_da3, partial_dv_da3], dim=1),
            torch.cat([partial_du_db1, partial_dv_db1], dim=1),
            torch.cat([partial_du_db2, partial_dv_db2], dim=1),
            torch.cat([partial_du_db3, partial_dv_db3], dim=1),
            torch.cat([partial_du_dtx, partial_dv_dtx], dim=1),
            torch.cat([partial_du_dty, partial_dv_dty], dim=1),
            torch.cat([partial_du_dtz, partial_dv_dtz], dim=1),
        ],
        dim=2,
    )
    return jacobian, u, v
