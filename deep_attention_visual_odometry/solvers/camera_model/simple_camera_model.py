from typing import Final, NamedTuple, Self
import torch

from deep_attention_visual_odometry.solvers import IOptimisableFunction
from .vectors_to_rotation_matrix import (
    TwoVectorOrientation,
    RotationMatrix,
    RotationMatrixDerivatives,
)


class SimpleCameraModel(IOptimisableFunction):
    """
    A simple camera model, projecting 3D world points to produce image coordinates in multiple views.
    Supports a batch dimension and multiple parallel estimates,
    so the image coordinates should be shape BxExMxNx2.

    Where B is batch size, E the number of estimates, M the number of camera views,
    and N the number of points
    """

    # Parameter indices in order
    CX: Final[int] = 0
    CY: Final[int] = 1
    F: Final[int] = 2
    A1: Final[int] = 3
    A2: Final[int] = 4
    A3: Final[int] = 5
    B1: Final[int] = 6
    B2: Final[int] = 7
    B3: Final[int] = 8
    TX: Final[int] = 9
    TY: Final[int] = 10
    TZ: Final[int] = 11
    FIRST_WORLD_POINT: Final[int] = 12

    def __init__(
        self,
        focal_length: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        translation: torch.Tensor,
        world_points: torch.Tensor,
        true_projected_points: torch.Tensor,
    ):
        """
        :param focal_length: BxE (assumed fixed for all views)
        :param cx: BxE (fixed for all views)
        :param cy: BxE (fixed for all views)
        :param a: BxExMx3
        :param b: BxExMx3
        :param translation: BxExMx3
        :param world_points: BxExNx3 (fixed for all views)
        :param true_projected_points: BxExMxNx2
        """
        self._num_views = true_projected_points.size(2)
        self._num_points = true_projected_points.size(3)
        self._focal_length = focal_length
        self._cx = cx
        self._cy = cy
        self._translation = translation
        self._world_points = world_points
        self._true_projected_points = true_projected_points
        self._orientation = TwoVectorOrientation(a, b)
        self._camera_relative_points = None
        self._u = None
        self._v = None
        self._residuals = None
        self._error = None
        self._gradient = None

    @property
    def batch_size(self) -> int:
        return self._true_projected_points.size(0)

    @property
    def num_estimates(self) -> int:
        return self._true_projected_points.size(1)

    @property
    def num_parameters(self) -> int:
        return 3 + 6 * self._num_views + 3 * self._num_points

    @property
    def device(self) -> torch.device:
        return self._true_projected_points.device

    def get_camera_intrinsics(self) -> torch.Tensor:
        return torch.stack()

    def get_error(self) -> torch.Tensor:
        """
        :returns: Total square error for each estimate, BxE.
        """
        if self._error is None:
            u = self._get_u()
            v = self._get_v()
            self._error = (u - self._true_projected_points[:, :, :, :, 0]).square().sum(
                dim=(-2, -1)
            ) + (v - self._true_projected_points[:, :, :, :, 1]).square().sum(
                dim=(-2, -1)
            )
        return self._error

    def get_gradient(self) -> torch.Tensor:
        if self._gradient is None:
            rotation_matrix = self._orientation.get_rotation_matrix()
            rotation_matrix_gradients = self._orientation.get_derivatives()
            camera_relative_points = self._camera_relative_points()
            u = self._get_u()
            v = self._get_v()
            residuals_u = u - self._true_projected_points[:, :, :, :, 0]
            residuals_v = v - self._true_projected_points[:, :, :, :, 1]
            partial_derivatives = _compute_gradient_from_intermediates(
                u=u,
                v=v,
                x=self._world_points[:, :, 0],
                y=self._world_points[:, :, 1],
                z=self._world_points[:, :, 2],
                x_prime=camera_relative_points[:, :0],
                y_prime=camera_relative_points[:, :, 1],
                z_prime=camera_relative_points[:, :, 2],
                focal_length=self._focal_length,
                rotation_matrix=rotation_matrix,
                orientation_derivatives=rotation_matrix_gradients,
            )
            # Note, since we're computing per-world-point gradients, we only sum along the views
            world_x_gradients = (
                residuals_u * partial_derivatives.partial_du_dx
            ) + residuals_v * partial_derivatives.partial_dv_dx
            world_y_gradients = (
                residuals_u * partial_derivatives.partial_du_dy
                + residuals_v * partial_derivatives.partial_dv_dy
            )
            world_z_gradients = (
                residuals_u * partial_derivatives.partial_du_dz
                + residuals_v * partial_derivatives.partial_dv_dz
            )
            self._gradient = torch.stack(
                [
                    # CX/CY derivatives are 1 for their coordinate, and 0 for the other
                    residuals_u.sum(dim=-1),
                    residuals_v.sum(dim=-1),
                    # The remaining derivatives are the residuals times the partials
                    (residuals_u * partial_derivatives.partial_du_df).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_df).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_da1).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_da1).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_da2).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_da2).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_da3).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_da3).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_db1).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_db1).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_db2).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_db2).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_db3).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_db3).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_dtx).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_dtx).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_dty).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_dtz).sum(dim=-1),
                    (residuals_u * partial_derivatives.partial_du_dtz).sum(dim=-1)
                    + (residuals_v * partial_derivatives.partial_dv_dtz).sum(dim=-1),
                    world_x_gradients,
                    world_y_gradients,
                    world_z_gradients,
                ],
                dim=-1,
            )
        return self._gradient

    def add(self, parameters: torch.Tensor) -> Self:
        num_world_points = self._world_points.size(2)
        world_point_params = torch.stack(
            [
                parameters[
                    :,
                    :,
                    self.FIRST_WORLD_POINT : self.FIRST_WORLD_POINT + num_world_points,
                ],
                parameters[
                    :,
                    :,
                    self.FIRST_WORLD_POINT
                    + num_world_points : self.FIRST_WORLD_POINT
                    + 2 * num_world_points,
                ],
                parameters[
                    :,
                    :,
                    self.FIRST_WORLD_POINT
                    + 2 * num_world_points : self.FIRST_WORLD_POINT
                    + 3 * num_world_points,
                ],
            ],
            dim=2,
        )
        return type(self)(
            focal_length=self._focal_length + parameters[:, :, self.F],
            cx=self._cx + parameters[:, :, self.CX],
            cy=self._cy + parameters[:, :, self.CY],
            a=self._orientation.a + parameters[:, :, self.A1 : self.A3],
            b=self._orientation.b + parameters[:, :, self.B1 : self.B3],
            translation=self._translation + parameters[:, :, self.TX : self.TZ],
            world_points=self._world_points + world_point_params,
            true_projected_points=self._true_projected_points,
        )

    def masked_add(self, parameters: torch.Tensor, mask: torch.Tensor) -> Self:

        num_world_points = self._world_points.size(2)
        world_point_params = torch.stack(
            [
                parameters[
                :,
                :,
                self.FIRST_WORLD_POINT: self.FIRST_WORLD_POINT + num_world_points,
                ],
                parameters[
                :,
                :,
                self.FIRST_WORLD_POINT
                + num_world_points: self.FIRST_WORLD_POINT
                                    + 2 * num_world_points,
                ],
                parameters[
                :,
                :,
                self.FIRST_WORLD_POINT
                + 2 * num_world_points: self.FIRST_WORLD_POINT
                                        + 3 * num_world_points,
                ],
            ],
            dim=2,
        )
        return type(self)(
            focal_length=self._focal_length + parameters[:, :, self.F],
            cx=self._cx + parameters[:, :, self.CX],
            cy=self._cy + parameters[:, :, self.CY],
            a=self._orientation.a + parameters[:, :, self.A1: self.A3],
            b=self._orientation.b + parameters[:, :, self.B1: self.B3],
            translation=self._translation + parameters[:, :, self.TX: self.TZ],
            world_points=self._world_points + world_point_params,
            true_projected_points=self._true_projected_points,
        )


    def _get_camera_relative_points(self) -> torch.Tensor:
        """
        :returns: 3d points relative to each camera, shape BxExMxNx3
        """
        if self._camera_relative_points is None:
            rotation_matrix = self._orientation.get_rotation_matrix()
            x_prime = (
                self._world_points[:, :, None, :, 0] * rotation_matrix.r1[:, :, :, None]
                + self._world_points[:, :, None, :, 1]
                * rotation_matrix.r2[:, :, :, None]
                + self._world_points[:, :, None, :, 2]
                * rotation_matrix.r3[:, :, :, None]
            )
            y_prime = (
                self._world_points[:, :, None, :, 0] * rotation_matrix.r4[:, :, :, None]
                + self._world_points[:, :, None, :, 1]
                * rotation_matrix.r5[:, :, :, None]
                + self._world_points[:, :, None, :, 2]
                * rotation_matrix.r6[:, :, :, None]
            )
            z_prime = (
                self._world_points[:, :, None, :, 0] * rotation_matrix.r7[:, :, :, None]
                + self._world_points[:, :, None, :, 1]
                * rotation_matrix.r8[:, :, :, None]
                + self._world_points[:, :, None, :, 2]
                * rotation_matrix.r9[:, :, :, None]
            )
            # Clamp the camera-relative z' to treat all points as "in front" of the camera,
            # Due to the division, the optmisation cannot cross through Z' = 0,
            # because the projected points go to infinity, and thus so does the error.
            z_prime = torch.clamp(z_prime, min=1e-8)
            rotated_points = torch.stack([x_prime, y_prime, z_prime], dim=-1)
            self._camera_relative_points = (
                rotated_points + self._translation[:, :, :, None]
            )
        return self._camera_relative_points

    def _get_u(self) -> torch.Tensor:
        """
        :returns: BxExMxN
        """
        if self._u is None:
            camera_relative_points = self._get_camera_relative_points()
            self._u = (
                self._focal_length
                * camera_relative_points[:, :, :, :, 0]
                / camera_relative_points[:, :, :, :, 2]
                + self._cx
            )
        return self._u

    def _get_v(self) -> torch.Tensor:
        """
        :returns: BxExMxN
        """
        if self._v is None:
            camera_relative_points = self._get_camera_relative_points()
            self._v = (
                self._focal_length
                * camera_relative_points[:, :, :, :, 1]
                / camera_relative_points[:, :, :, :, 2]
                + self._cy
            )
        return self._v


class _CameraGradients(NamedTuple):
    partial_du_df: torch.Tensor
    partial_dv_df: torch.Tensor
    partial_du_da1: torch.Tensor
    partial_dv_da1: torch.Tensor
    partial_du_da2: torch.Tensor
    partial_dv_da2: torch.Tensor
    partial_du_da3: torch.Tensor
    partial_dv_da3: torch.Tensor
    partial_du_db1: torch.Tensor
    partial_dv_db1: torch.Tensor
    partial_du_db2: torch.Tensor
    partial_dv_db2: torch.Tensor
    partial_du_db3: torch.Tensor
    partial_dv_db3: torch.Tensor
    partial_du_dtx: torch.Tensor
    partial_dv_dtx: torch.Tensor
    partial_du_dty: torch.Tensor
    partial_dv_dty: torch.Tensor
    partial_du_dtz: torch.Tensor
    partial_dv_dtz: torch.Tensor
    partial_du_dx: torch.Tensor
    partial_dv_dx: torch.Tensor
    partial_du_dy: torch.Tensor
    partial_dv_dy: torch.Tensor
    partial_du_dz: torch.Tensor
    partial_dv_dz: torch.Tensor


def _compute_gradient_from_intermediates(
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    x_prime: torch.Tensor,
    y_prime: torch.Tensor,
    z_prime: torch.Tensor,
    focal_length: torch.Tensor,
    rotation_matrix: RotationMatrix,
    orientation_derivatives: RotationMatrixDerivatives,
) -> _CameraGradients:
    """
    Compute the gradient of the error
    :param u: BxExMxN
    :param v: BxExMxN
    :param x: BxEx1xN
    :param y: BxEx1xN
    :param z: BxEx1xN
    :param x_prime: BxExMxN
    :param y_prime: BxExMxN
    :param z_prime: BxExMxN
    :param focal_length: BxE
    :param rotation_matrix:
    :param orientation_derivatives:
    :return:
    """

    f_on_z_prime = focal_length[:, :, None, None] / z_prime
    x_on_z_prime = x_prime / z_prime
    y_on_z_prime = y_prime / z_prime
    du_dxprime = f_on_z_prime
    dv_dyprime = f_on_z_prime
    du_dzprime = -f_on_z_prime * x_on_z_prime  # = -f x / z^2
    dv_dzprime = -f_on_z_prime * y_on_z_prime

    # Camera parameter derivatives are fairly simple, cx and cy are ones/zeros
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
        du_dxprime * x * orientation_derivatives.dr1_da1
        + du_dxprime * y * orientation_derivatives.dr2_da1
        + du_dxprime * z * orientation_derivatives.dr3_da1
        + du_dzprime * x * orientation_derivatives.dr7_da1
        + du_dzprime * y * orientation_derivatives.dr8_da1
        + du_dzprime * z * orientation_derivatives.dr9_da1
    )
    partial_du_da2 = (
        du_dxprime * x * orientation_derivatives.dr1_da2
        + du_dxprime * y * orientation_derivatives.dr2_da2
        + du_dxprime * z * orientation_derivatives.dr3_da2
        + du_dzprime * x * orientation_derivatives.dr7_da2
        + du_dzprime * y * orientation_derivatives.dr8_da2
        + du_dzprime * z * orientation_derivatives.dr9_da2
    )
    partial_du_da3 = (
        du_dxprime * x * orientation_derivatives.dr1_da3
        + du_dxprime * y * orientation_derivatives.dr2_da3
        + du_dxprime * z * orientation_derivatives.dr3_da3
        + du_dzprime * x * orientation_derivatives.dr7_da3
        + du_dzprime * y * orientation_derivatives.dr8_da3
        + du_dzprime * z * orientation_derivatives.dr9_da3
    )
    partial_dv_da1 = (
        dv_dyprime * x * orientation_derivatives.dr4_da1
        + dv_dyprime * y * orientation_derivatives.dr5_da1
        + dv_dyprime * z * orientation_derivatives.dr6_da1
        + dv_dzprime * x * orientation_derivatives.dr7_da1
        + dv_dzprime * y * orientation_derivatives.dr8_da1
        + dv_dzprime * z * orientation_derivatives.dr9_da1
    )
    partial_dv_da2 = (
        dv_dyprime * x * orientation_derivatives.dr4_da2
        + dv_dyprime * y * orientation_derivatives.dr5_da2
        + dv_dyprime * z * orientation_derivatives.dr6_da2
        + dv_dzprime * x * orientation_derivatives.dr7_da2
        + dv_dzprime * y * orientation_derivatives.dr8_da2
        + dv_dzprime * z * orientation_derivatives.dr9_da2
    )
    partial_dv_da3 = (
        dv_dyprime * x * orientation_derivatives.dr4_da3
        + dv_dyprime * y * orientation_derivatives.dr5_da3
        + dv_dyprime * z * orientation_derivatives.dr6_da3
        + dv_dzprime * x * orientation_derivatives.dr7_da3
        + dv_dzprime * y * orientation_derivatives.dr8_da3
        + dv_dzprime * z * orientation_derivatives.dr9_da3
    )
    partial_du_db1 = (
        du_dxprime * y * orientation_derivatives.dr2_db1
        + du_dxprime * z * orientation_derivatives.dr3_db1
        + du_dzprime * y * orientation_derivatives.dr8_db1
        + du_dzprime * z * orientation_derivatives.dr9_db1
    )
    partial_du_db2 = (
        du_dxprime * y * orientation_derivatives.dr2_db2
        + du_dxprime * z * orientation_derivatives.dr3_db2
        + du_dzprime * y * orientation_derivatives.dr8_db2
        + du_dzprime * z * orientation_derivatives.dr9_db2
    )
    partial_du_db3 = (
        du_dxprime * y * orientation_derivatives.dr2_db3
        + du_dxprime * z * orientation_derivatives.dr3_db3
        + du_dzprime * y * orientation_derivatives.dr8_db3
        + du_dzprime * z * orientation_derivatives.dr9_db3
    )
    partial_dv_db1 = (
        dv_dyprime * y * orientation_derivatives.dr5_db1
        + dv_dyprime * z * orientation_derivatives.dr6_db1
        + dv_dzprime * y * orientation_derivatives.dr8_db1
        + dv_dzprime * z * orientation_derivatives.dr9_db1
    )
    partial_dv_db2 = (
        dv_dyprime * y * orientation_derivatives.dr5_db2
        + dv_dyprime * z * orientation_derivatives.dr6_db2
        + dv_dzprime * y * orientation_derivatives.dr8_db2
        + dv_dzprime * z * orientation_derivatives.dr9_db2
    )
    partial_dv_db3 = (
        dv_dyprime * y * orientation_derivatives.dr5_db3
        + dv_dyprime * z * orientation_derivatives.dr6_db3
        + dv_dzprime * y * orientation_derivatives.dr8_db3
        + dv_dzprime * z * orientation_derivatives.dr9_db3
    )

    # Lastly, the gradients for the world points
    # These are again chain rule patterns
    partial_du_dx = du_dxprime * rotation_matrix.r1 + du_dzprime * rotation_matrix.r7
    partial_dv_dx = dv_dyprime * rotation_matrix.r4 + dv_dzprime * rotation_matrix.r7
    partial_du_dy = du_dxprime * rotation_matrix.r2 + dv_dzprime * rotation_matrix.r8
    partial_dv_dy = dv_dyprime * rotation_matrix.r5 + dv_dzprime * rotation_matrix.r8
    partial_du_dz = du_dxprime * rotation_matrix.r3 + du_dzprime * rotation_matrix.r9
    partial_dv_dz = dv_dyprime * rotation_matrix.r6 + dv_dzprime * rotation_matrix.r9

    return _CameraGradients(
        partial_du_df=partial_du_df,
        partial_dv_df=partial_dv_df,
        partial_du_da1=partial_du_da1,
        partial_dv_da1=partial_dv_da1,
        partial_du_da2=partial_du_da2,
        partial_dv_da2=partial_dv_da2,
        partial_du_da3=partial_du_da3,
        partial_dv_da3=partial_dv_da3,
        partial_du_db1=partial_du_db1,
        partial_dv_db1=partial_dv_db1,
        partial_du_db2=partial_du_db2,
        partial_dv_db2=partial_dv_db2,
        partial_du_db3=partial_du_db3,
        partial_dv_db3=partial_dv_db3,
        partial_du_dtx=partial_du_dtx,
        partial_dv_dtx=partial_dv_dtx,
        partial_du_dty=partial_du_dty,
        partial_dv_dty=partial_dv_dty,
        partial_du_dtz=partial_du_dtz,
        partial_dv_dtz=partial_dv_dtz,
        partial_du_dx=partial_du_dx,
        partial_dv_dx=partial_dv_dx,
        partial_du_dy=partial_du_dy,
        partial_dv_dy=partial_dv_dy,
        partial_du_dz=partial_du_dz,
        partial_dv_dz=partial_dv_dz,
    )
