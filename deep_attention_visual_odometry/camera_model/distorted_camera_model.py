from typing import NamedTuple, Tuple
import torch
import spatial_maths.camera_model_parameters as pidx


class _PointsAndIntermediates(NamedTuple):
    u_prime: torch.Tensor
    v_prime: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    radius_squared: torch.Tensor
    radial_distortion: torch.Tensor
    x_prime: torch.Tensor
    y_prime: torch.Tensor
    z_prime: torch.Tensor
    sin_rx: torch.Tensor
    cos_rx: torch.Tensor
    sin_ry: torch.Tensor
    cos_ry: torch.Tensor
    sin_rz: torch.Tensor
    cos_rz: torch.Tensor


@torch.jit.script
def _full_forward_model(
    points_3d: torch.Tensor, parameters: torch.Tensor
) -> _PointsAndIntermediates:
    # Part 1: the forward model.
    sin_rx = torch.sin(parameters[:, None, pidx.RX])
    cos_rx = torch.cos(parameters[:, None, pidx.RX])
    sin_ry = torch.sin(parameters[:, None, pidx.RY])
    cos_ry = torch.cos(parameters[:, None, pidx.RY])
    sin_rz = torch.sin(parameters[:, None, pidx.RZ])
    cos_rz = torch.cos(parameters[:, None, pidx.RZ])
    # Run through a combined rotation matrix where the parametes are euler angles
    # See https://en.wikipedia.org/wiki/Rotation_matrix
    # This is the camera extrinsic transformation
    x_prime = (
        points_3d[:, :, 0] * cos_ry * cos_rz
        + points_3d[:, :, 1] * (sin_rx * sin_ry * cos_rz - cos_rx * sin_rz)
        + points_3d[:, :, 2] * (cos_rx * sin_ry * cos_rz + sin_rx * sin_rz)
        + parameters[:, None, pidx.TX]
    )
    y_prime = (
        points_3d[:, :, 0] * cos_ry * sin_rz
        + points_3d[:, :, 1] * (sin_rx * sin_ry * sin_rz + cos_rx * cos_rz)
        + points_3d[:, :, 2] * (cos_rx * sin_ry * sin_rz - sin_rx * cos_rz)
        + parameters[:, None, pidx.TY]
    )
    z_prime = (
        points_3d[:, :, 0] * -sin_ry
        + points_3d[:, :, 1] * sin_rx * cos_ry
        + points_3d[:, :, 2] * cos_rx * cos_ry
        + parameters[:, None, pidx.TZ]
    )
    # Add a little bit for Z' == 0 since it is mostly in the denominator
    z_prime[z_prime == 0] += 1e-8
    # Now use the camera intrinsics to compute the undistorted pixel coordinates
    u = parameters[:, None, pidx.FX] * (x_prime / z_prime) + parameters[
        :, None, pidx.S
    ] * (y_prime / z_prime)
    v = parameters[:, None, pidx.FY] * (y_prime / z_prime)
    # Now for the distortion model
    radius_squared = u * u + v * v
    uv = u * v
    radial_distortion = (
        1.0
        + parameters[:, None, pidx.K1] * radius_squared
        + parameters[:, None, pidx.K2] * radius_squared * radius_squared
        + parameters[:, None, pidx.K3]
        * radius_squared
        * radius_squared
        * radius_squared
    )
    u_prime = (
        u * radial_distortion
        + 2.0 * parameters[:, None, pidx.P1] * uv
        + parameters[:, None, pidx.P2] * (radius_squared + 2 * u * u)
        + parameters[:, None, pidx.CX]
    )
    v_prime = (
        v * radial_distortion
        + 2.0 * parameters[:, None, pidx.P2] * uv
        + parameters[:, None, pidx.P1] * (radius_squared + 2 * v * v)
        + parameters[:, None, pidx.CY]
    )
    return _PointsAndIntermediates(
        u_prime=u_prime,
        v_prime=v_prime,
        u=u,
        v=v,
        radius_squared=radius_squared,
        radial_distortion=radial_distortion,
        x_prime=x_prime,
        y_prime=y_prime,
        z_prime=z_prime,
        sin_rx=sin_rx,
        cos_rx=cos_rx,
        sin_ry=sin_ry,
        cos_ry=cos_ry,
        sin_rz=sin_rz,
        cos_rz=cos_rz,
    )


@torch.jit.script
def compute_distorted_camera_model(
    points_3d: torch.Tensor, parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    forward_model = _full_forward_model(points_3d, parameters)
    return forward_model.u_prime, forward_model.v_prime


@torch.jit.script
def compute_distorted_camera_model_and_jacobian(
    points_3d: torch.Tensor, parameters: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # First run the forward model, grabbing intermediate values as well
    forward_model = _full_forward_model(points_3d, parameters)
    u_prime = forward_model.u_prime
    v_prime = forward_model.v_prime
    u = forward_model.u
    v = forward_model.v
    radius_squared = forward_model.radius_squared
    radial_distortion = forward_model.radial_distortion
    x_prime = forward_model.x_prime
    y_prime = forward_model.y_prime
    z_prime = forward_model.z_prime
    sin_rx = forward_model.sin_rx
    cos_rx = forward_model.cos_rx
    sin_ry = forward_model.sin_ry
    cos_ry = forward_model.cos_ry
    sin_rz = forward_model.sin_rz
    cos_rz = forward_model.cos_rz

    # Now for the derivatives.
    # For gauss-newton, we need the pseudo-hamiltonian J^T.J
    # and the product of the jacobian with the residuals, J^T.r(\theta)
    # We can then find the update by solving for the
    # See https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
    radial_distortion_derivative = (
        parameters[:, None, pidx.K1]
        + 2 * parameters[:, None, pidx.K2] * radius_squared
        + 3 * parameters[:, None, pidx.K3] * radius_squared * radius_squared
    )

    # distortion parameters
    partial_du_dcx = torch.ones_like(u)
    partial_dv_dcx = torch.zeros_like(v)
    partial_du_dcy = torch.zeros_like(u)
    partial_dv_dcy = torch.ones_like(v)
    partial_du_dk1 = u * radius_squared
    partial_dv_dk1 = v * radius_squared
    partial_du_dk2 = u * radius_squared * radius_squared
    partial_dv_dk2 = v * radius_squared * radius_squared
    partial_du_dk3 = u * radius_squared * radius_squared * radius_squared
    partial_dv_dk3 = v * radius_squared * radius_squared * radius_squared
    partial_du_dp1 = 2 * u * v
    partial_dv_dp1 = radius_squared + 2 * v * v
    partial_du_dp2 = radius_squared + 2 * u * u
    partial_dv_dp2 = 2 * u * v

    # Camera parameters
    partial_du_dfx = (x_prime / z_prime) * (
        radial_distortion
        + 2 * u * u * radial_distortion_derivative
        + 2 * parameters[:, None, pidx.P1] * v
        + 6 * parameters[:, None, pidx.P2] * u
    )
    partial_dv_dfx = (
        2
        * (x_prime / z_prime)
        * (
            u * v * radial_distortion_derivative
            + parameters[:, None, pidx.P1] * v
            + parameters[:, None, pidx.P2] * u
        )
    )
    partial_du_ds = (y_prime / z_prime) * (
        radial_distortion
        + 2 * u * u * radial_distortion_derivative
        + 2 * parameters[:, None, pidx.P1] * v
        + 6 * parameters[:, None, pidx.P2] * u
    )
    partial_dv_ds = (
        2
        * (y_prime / z_prime)
        * (
            u * v * radial_distortion_derivative
            + parameters[:, None, pidx.P1] * v
            + parameters[:, None, pidx.P2] * u
        )
    )
    partial_du_dfy = (
        2
        * (y_prime / z_prime)
        * (
            u * v * radial_distortion_derivative
            + parameters[:, None, pidx.P1] * v
            + parameters[:, None, pidx.P2] * u
        )
    )
    partial_dv_dfy = (y_prime / z_prime) * (
        radial_distortion
        + 2 * v * v * radial_distortion_derivative
        + 6 * parameters[:, None, pidx.P1] * v
        + 2 * parameters[:, None, pidx.P2] * u
    )

    # Translation
    partial_du_dt1 = (parameters[:, None, pidx.FX] / z_prime) * (
        radial_distortion
        + 2 * u * u * radial_distortion_derivative
        + 2 * parameters[:, None, pidx.P1] * v
        + 6 * parameters[:, None, pidx.P2] * u
    )
    partial_dv_dt1 = (
        2
        * (parameters[:, None, pidx.FX] / z_prime)
        * (
            u * v * radial_distortion_derivative
            + parameters[:, None, pidx.P1] * v
            + parameters[:, None, pidx.P2] * u
        )
    )
    partial_du_dt2 = (parameters[:, None, pidx.S] / z_prime) * (
        radial_distortion
        + 2 * u * u * radial_distortion_derivative
        + 2 * parameters[:, None, pidx.P1] * v
        + 6 * parameters[:, None, pidx.P2] * u
    ) + 2 * (parameters[:, None, pidx.FY] / z_prime) * (
        u * v * radial_distortion_derivative
        + parameters[:, None, pidx.P1] * u
        + parameters[:, None, pidx.P2] * v
    )
    partial_dv_dt2 = (parameters[:, None, pidx.FY] / z_prime) * (
        radial_distortion
        + 2 * v * v * radial_distortion_derivative
        + 2 * parameters[:, None, pidx.P1] * u
        + 6 * parameters[:, None, pidx.P2] * v
    ) + 2 * (parameters[:, None, pidx.S] / z_prime) * (
        u * v * radial_distortion_derivative
        + parameters[:, None, pidx.P1] * u
        + parameters[:, None, pidx.P2] * v
    )
    partial_du_dt3 = (
        -(
            u * radial_distortion
            + 2 * u * radius_squared * radial_distortion_derivative
            + 4 * parameters[:, None, pidx.P1] * u * v
            + 2 * parameters[:, None, pidx.P2] * radius_squared
            + 4 * parameters[:, None, pidx.P2] * u * u
        )
        / z_prime
    )
    partial_dv_dt3 = (
        -(
            v * radial_distortion
            + 2 * v * radius_squared * radial_distortion_derivative
            + 2 * parameters[:, None, pidx.P1] * radius_squared
            + 4 * parameters[:, None, pidx.P1] * v * v
            + 4 * parameters[:, None, pidx.P2] * u * v
        )
        / z_prime
    )

    # Rotation
    # These equations are a pain due to the trigonometry, so we'll do them in stages
    # Hopefully the compiler will flatten them out anyway
    x = points_3d[:, :, 0]
    y = points_3d[:, :, 1]
    z = points_3d[:, :, 2]
    dxprime_drx = cos_rx * sin_ry * (y * cos_rz - z * sin_rz) + sin_rx * (
        y * sin_rz + z * cos_rz
    )
    dxprime_dry = -cos_rz * (x * sin_ry + cos_ry * (y * sin_rx - z * cos_rx))
    dxprime_drz = -sin_rz * (
        x * cos_ry + sin_ry * (y * sin_rx + z * cos_rx)
    ) + cos_rz * (z * sin_rx - y * cos_rx)
    dyprime_drx = sin_ry * sin_rz * (y * cos_rx - z * cos_rz) - cos_rz * (
        z * sin_rx - y * cos_rx
    )
    dyprime_dry = sin_rz * (cos_ry * (y * sin_rx + z * cos_rx) - x * sin_ry)
    dyprime_drz = cos_rz * (
        x * cos_ry + sin_ry * (y * sin_rx + z * cos_rx)
    ) - sin_rz * (y * cos_rx + z * sin_rx)
    dzprime_drx = cos_ry * (y * cos_rx - z * sin_rx)
    dzprime_dry = -x * cos_ry - sin_ry * (y * sin_rx + z * cos_rx)
    # dzprime_drz = 0
    # Note that these are the 'true' u and v, where all the others above are technically u_prime and v_prime
    du_drx = (parameters[:, None, pidx.FX] / (z_prime * z_prime)) * (
        z_prime * dxprime_drx - x_prime * dzprime_drx
    ) + (parameters[:, None, pidx.S] / (z_prime * z_prime)) * (
        z_prime * dyprime_drx - y_prime * dzprime_drx
    )
    dv_drx = (parameters[:, None, pidx.FY] / (z_prime * z_prime)) * (
        z_prime * dyprime_drx - y_prime * dzprime_drx
    )
    du_dry = (parameters[:, None, pidx.FX] / (z_prime * z_prime)) * (
        z_prime * dxprime_dry - x_prime * dzprime_dry
    ) + (parameters[:, None, pidx.S] / (z_prime * z_prime)) * (
        z_prime * dyprime_dry - y_prime * dzprime_dry
    )
    dv_dry = (parameters[:, None, pidx.FY] / (z_prime * z_prime)) * (
        z_prime * dyprime_dry - y_prime * dzprime_dry
    )
    du_drz = (
        parameters[:, None, pidx.FX] * dxprime_drz
        + parameters[:, None, pidx.S] * dyprime_drz
    ) / z_prime
    dv_drz = parameters[:, None, pidx.FY] * dyprime_drz / z_prime
    # Finally the distortion model
    dd_drx = 2 * u * du_drx + 2 * v * dv_drx
    dd_dry = 2 * u * du_dry + 2 * v * dv_dry
    dd_drz = 2 * u * du_drz + 2 * v * dv_drz
    db_drx = radial_distortion_derivative * dd_drx
    db_dry = radial_distortion_derivative * dd_dry
    db_drz = radial_distortion_derivative * dd_drz
    partial_du_drx = (
        du_drx * radial_distortion
        + db_drx * u
        + 2 * parameters[:, None, pidx.P1] * (v * du_drx + u * dv_drx)
        + parameters[:, None, pidx.P2] * dd_drx
        + 4 * parameters[:, None, pidx.P2] * du_drx
    )
    partial_dv_drx = (
        radial_distortion * dv_drx
        + v * db_drx
        + parameters[:, None, pidx.P1] * dd_drx
        + 4 * parameters[:, None, pidx.P1] * v * dv_drx
        + 2 * parameters[:, None, pidx.P1] * (v * du_drx + u * dv_drx)
    )
    partial_du_dry = (
        du_dry * radial_distortion
        + db_dry * u
        + 2 * parameters[:, None, pidx.P1] * (v * du_dry + u * dv_dry)
        + parameters[:, None, pidx.P2] * dd_dry
        + 4 * parameters[:, None, pidx.P2] * du_dry
    )
    partial_dv_dry = (
        radial_distortion * dv_dry
        + v * db_dry
        + parameters[:, None, pidx.P1] * dd_dry
        + 4 * parameters[:, None, pidx.P1] * v * dv_dry
        + 2 * parameters[:, None, pidx.P1] * (v * du_dry + u * dv_dry)
    )
    partial_du_drz = (
        du_drz * radial_distortion
        + db_drz * u
        + 2 * parameters[:, None, pidx.P1] * (v * du_drz + u * dv_drz)
        + parameters[:, None, pidx.P2] * dd_drz
        + 4 * parameters[:, None, pidx.P2] * du_drz
    )
    partial_dv_drz = (
        radial_distortion * dv_drz
        + v * db_drz
        + parameters[:, None, pidx.P1] * dd_drz
        + 4 * parameters[:, None, pidx.P1] * v * dv_drz
        + 2 * parameters[:, None, pidx.P1] * (v * du_drz + u * dv_drz)
    )

    # Build the jacobian, a 2Nx16 matrix measuring the rate of change of each residual w.r.t. each parameter
    # I'm hoping the compiler is smart enough to optimize down to the reduction
    jacobian = torch.stack(
        [
            torch.cat([partial_du_dcx, partial_dv_dcx], dim=1),
            torch.cat([partial_du_dcy, partial_dv_dcy], dim=1),
            torch.cat([partial_du_dk1, partial_dv_dk1], dim=1),
            torch.cat([partial_du_dk2, partial_dv_dk2], dim=1),
            torch.cat([partial_du_dk3, partial_dv_dk3], dim=1),
            torch.cat([partial_du_dp1, partial_dv_dp1], dim=1),
            torch.cat([partial_du_dp2, partial_dv_dp2], dim=1),
            torch.cat([partial_du_dfx, partial_dv_dfx], dim=1),
            torch.cat([partial_du_ds, partial_dv_ds], dim=1),
            torch.cat([partial_du_dfy, partial_dv_dfy], dim=1),
            torch.cat([partial_du_drx, partial_dv_drx], dim=1),
            torch.cat([partial_du_dry, partial_dv_dry], dim=1),
            torch.cat([partial_du_drz, partial_dv_drz], dim=1),
            torch.cat([partial_du_dt1, partial_dv_dt1], dim=1),
            torch.cat([partial_du_dt2, partial_dv_dt2], dim=1),
            torch.cat([partial_du_dt3, partial_dv_dt3], dim=1),
        ],
        dim=2,
    )
    return jacobian, u_prime, v_prime
