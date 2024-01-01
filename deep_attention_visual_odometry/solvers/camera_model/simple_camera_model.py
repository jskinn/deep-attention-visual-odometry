from typing import Final, NamedTuple, Self
import torch

from deep_attention_visual_odometry.solvers import IOptimisableFunction
from deep_attention_visual_odometry.utils import masked_merge_tensors
from .vectors_to_rotation_matrix import (
    TwoVectorOrientation,
    RotationMatrix,
    RotationMatrixDerivatives,
)


class SimpleCameraModel(IOptimisableFunction):
    """
    Given a set of points across multiple views,
    jointly optimise for the 3D positions of the points, the camera intrinsics, and extrinsics.

    Based on
    "More accurate pinhole camera calibration with imperfect planar target" by
    Klaus H Strobl and Gerd Hirzinger in ICCV 2011

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
    VIEW_START: Final[int] = 3

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
        minimum_distance: float = 1e-5,
        _error: torch.Tensor | None = None,
        _gradient: torch.Tensor | None = None,
        _error_mask: torch.Tensor | None = None,
        _gradient_mask: torch.Tensor | None = None,
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
        :param minimum_distance: Minimum distance from the camera plane
        """
        self.minimum_distance = float(minimum_distance)
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
        self._error = _error
        self._gradient = _gradient

        self._error_mask = _error_mask
        self._gradient_mask = _gradient_mask

    @property
    def batch_size(self) -> int:
        return self._true_projected_points.size(0)

    @property
    def num_estimates(self) -> int:
        return self._true_projected_points.size(1)

    @property
    def num_parameters(self) -> int:
        return 3 + 9 * self._num_views + 3 * self._num_points

    @property
    def device(self) -> torch.device:
        return self._true_projected_points.device

    @property
    def focal_length(self) -> torch.Tensor:
        return self._focal_length

    @property
    def cx(self) -> torch.Tensor:
        return self._cx

    @property
    def cy(self) -> torch.Tensor:
        return self._cy

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
            self._error_mask = None
        elif self._error_mask is not None:
            u = self._get_u()
            v = self._get_v()
            to_update = torch.logical_not(self._error_mask)
            u = u[to_update]
            v = v[to_update]
            true_projected_points = self._true_projected_points[to_update]
            new_error = (u - true_projected_points[:, :, :, 0]).square().sum(
                dim=(-2, -1)
            ) + (v - true_projected_points[:, :, :, 1]).square().sum(dim=(-2, -1))
            self._error = self._error.masked_scatter(to_update, new_error)
            self._error_mask = None
        return self._error

    def get_gradient(self) -> torch.Tensor:
        """Get the gradient for each estimate, BxExP"""
        if self._gradient is None:
            rotation_matrix = self._orientation.get_rotation_matrix()
            rotation_matrix_gradients = self._orientation.get_derivatives()
            camera_relative_points = self._get_camera_relative_points()
            u = self._get_u()
            v = self._get_v()
            residuals_u = u - self._true_projected_points[:, :, :, :, 0]
            residuals_v = v - self._true_projected_points[:, :, :, :, 1]
            partial_derivatives = _compute_gradient_from_intermediates(
                x=self._world_points[:, :, None, :, 0],
                y=self._world_points[:, :, None, :, 1],
                z=self._world_points[:, :, None, :, 2],
                x_prime=camera_relative_points[:, :, :, :, 0],
                y_prime=camera_relative_points[:, :, :, :, 1],
                z_prime=camera_relative_points[:, :, :, :, 2],
                focal_length=self._focal_length,
                rotation_matrix=rotation_matrix,
                orientation_derivatives=rotation_matrix_gradients,
            )
            self._gradient = _stack_gradients(
                residuals_u, residuals_v, partial_derivatives
            )
            self._gradient_mask = None
        elif self._gradient_mask is not None:
            rotation_matrix = self._orientation.get_rotation_matrix()
            rotation_matrix = RotationMatrix(
                *(value[self._gradient_mask] for value in rotation_matrix)
            )
            rotation_matrix_gradients = self._orientation.get_derivatives()
            rotation_matrix_gradients = RotationMatrixDerivatives(
                *(value[self._gradient_mask] for value in rotation_matrix_gradients)
            )
            camera_relative_points = self._camera_relative_points()
            camera_relative_points = camera_relative_points[self._gradient_mask]
            u = self._get_u()
            u = u[self._gradient_mask]
            v = self._get_v()
            v = v[self._gradient_mask]
            focal_length = self._focal_length[self._gradient_mask]
            true_projected_points = self._true_projected_points[self._gradient_mask]
            world_points = self._world_points[self._gradient_mask]
            residuals_u = u - true_projected_points[:, :, :, 0]
            residuals_v = v - true_projected_points[:, :, :, 1]
            partial_derivatives = _compute_gradient_from_intermediates(
                x=world_points[:, None, :, 0],
                y=world_points[:, None, :, 1],
                z=world_points[:, None, :, 2],
                x_prime=camera_relative_points[:, :, :, 0],
                y_prime=camera_relative_points[:, :, :, 1],
                z_prime=camera_relative_points[:, :, :, 2],
                focal_length=focal_length,
                rotation_matrix=rotation_matrix,
                orientation_derivatives=rotation_matrix_gradients,
            )
            gradient = _stack_gradients(residuals_u, residuals_v, partial_derivatives)
            self._gradient = self._gradient.masked_scatter(
                self._gradient_mask, gradient
            )
            self._gradient_mask = None
        return self._gradient

    def add(self, parameters: torch.Tensor) -> Self:
        # Find the slice indices for the parameters, based on the number of views and world points
        a1_idx = self.VIEW_START
        a2_idx = a1_idx + self._num_views
        a3_idx = a2_idx + self._num_views
        b1_idx = a3_idx + self._num_views
        b2_idx = b1_idx + self._num_views
        b3_idx = b2_idx + self._num_views
        tx_idx = b3_idx + self._num_views
        ty_idx = tx_idx + self._num_views
        tz_idx = ty_idx + self._num_views
        x_idx = tz_idx + self._num_views
        y_idx = x_idx + self._num_points
        z_idx = y_idx + self._num_points
        end_idx = z_idx + self._num_points
        a1_params = parameters[:, :, a1_idx:a2_idx]
        a2_params = parameters[:, :, a2_idx:a3_idx]
        a3_params = parameters[:, :, a3_idx:b1_idx]
        b1_params = parameters[:, :, b1_idx:b2_idx]
        b2_params = parameters[:, :, b2_idx:b3_idx]
        b3_params = parameters[:, :, b3_idx:tx_idx]
        tx_params = parameters[:, :, tx_idx:ty_idx]
        ty_params = parameters[:, :, ty_idx:tz_idx]
        tz_params = parameters[:, :, tz_idx:x_idx]
        x_params = parameters[:, :, x_idx:y_idx]
        y_params = parameters[:, :, y_idx:z_idx]
        z_params = parameters[:, :, z_idx:end_idx]

        a_params = torch.stack([a1_params, a2_params, a3_params], dim=2)
        b_params = torch.stack([b1_params, b2_params, b3_params], dim=2)
        t_params = torch.stack([tx_params, ty_params, tz_params], dim=2)
        point_params = torch.stack([x_params, y_params, z_params], dim=2)

        return type(self)(
            focal_length=self._focal_length + parameters[:, :, self.F],
            cx=self._cx + parameters[:, :, self.CX],
            cy=self._cy + parameters[:, :, self.CY],
            a=self._orientation.a + a_params,
            b=self._orientation.b + b_params,
            translation=self._translation + t_params,
            world_points=self._world_points + point_params,
            true_projected_points=self._true_projected_points,
            minimum_distance=self.minimum_distance,
        )

    def masked_update(self, other: Self, mask: torch.Tensor) -> Self:
        focal_length = torch.where(mask, other._focal_length, self._focal_length)
        cx = torch.where(mask, other._cx, self._cx)
        cy = torch.where(mask, other._cy, self._cy)
        vector_mask = mask[:, :, None, None].tile(
            1, 1, self._translation.size(2), self._translation.size(3)
        )
        a = torch.where(vector_mask, other._orientation.a, self._orientation.a)
        b = torch.where(vector_mask, other._orientation.b, self._orientation.b)
        translation = torch.where(vector_mask, other._translation, self._translation)
        if other._world_points is self._world_points:
            # Simple shorthand equality check. If they happen to be the same tensor, we can just reuse it.
            # Should happen most of the time.
            world_points = self._world_points
        else:
            world_mask = mask[:, :, None, None].tile(
                1, 1, *self._world_points.shape[2:]
            )
            world_points = torch.where(
                world_mask, other._world_points, self._world_points
            )
        if other._true_projected_points is self._true_projected_points:
            true_projected_points = self._true_projected_points
        else:
            projected_points_mask = mask[:, :, None, None, None].tile(
                1, 1, *self._true_projected_points.shape[2:]
            )
            true_projected_points = torch.where(
                projected_points_mask,
                other._true_projected_points,
                self._true_projected_points,
            )
        error, error_mask = masked_merge_tensors(
            self._error, self._error_mask, other._error, other._error_mask, mask
        )
        gradient, gradient_mask = masked_merge_tensors(
            self._gradient,
            self._gradient_mask,
            other._gradient,
            other._gradient_mask,
            mask,
        )
        return type(self)(
            focal_length=focal_length,
            cx=cx,
            cy=cy,
            a=a,
            b=b,
            translation=translation,
            world_points=world_points,
            true_projected_points=true_projected_points,
            minimum_distance=self.minimum_distance,
            _error=error,
            _error_mask=error_mask,
            _gradient=gradient,
            _gradient_mask=gradient_mask,
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
                + self._translation[:, :, :, 0:1]
            )
            y_prime = (
                self._world_points[:, :, None, :, 0] * rotation_matrix.r4[:, :, :, None]
                + self._world_points[:, :, None, :, 1]
                * rotation_matrix.r5[:, :, :, None]
                + self._world_points[:, :, None, :, 2]
                * rotation_matrix.r6[:, :, :, None]
                + self._translation[:, :, :, 1:2]
            )
            z_prime = (
                self._world_points[:, :, None, :, 0] * rotation_matrix.r7[:, :, :, None]
                + self._world_points[:, :, None, :, 1]
                * rotation_matrix.r8[:, :, :, None]
                + self._world_points[:, :, None, :, 2]
                * rotation_matrix.r9[:, :, :, None]
                + self._translation[:, :, :, 2:3]
            )
            # Clamp the camera-relative z' to treat all points as "in front" of the camera,
            # Due to the division, the optmisation cannot cross through Z' = 0,
            # because the projected points go to infinity, and thus so does the error.
            z_prime = torch.clamp(z_prime, min=self.minimum_distance)
            rotated_points = torch.stack([x_prime, y_prime, z_prime], dim=-1)
            self._camera_relative_points = rotated_points
        return self._camera_relative_points

    def _get_u(self) -> torch.Tensor:
        """
        :returns: BxExMxN
        """

        if self._u is None:
            camera_relative_points = self._get_camera_relative_points()
            self._u = self._focal_length.view(
                *self._focal_length.shape, 1, 1
            ) * camera_relative_points[:, :, :, :, 0] / camera_relative_points[
                :, :, :, :, 2
            ] + self._cx.view(
                *self._cx.shape, 1, 1
            )
        return self._u

    def _get_v(self) -> torch.Tensor:
        """
        :returns: BxExMxN
        """
        if self._v is None:
            camera_relative_points = self._get_camera_relative_points()
            self._v = self._focal_length.view(
                *self._focal_length.shape, 1, 1
            ) * camera_relative_points[:, :, :, :, 1] / camera_relative_points[
                :, :, :, :, 2
            ] + self._cy.view(
                *self._cy.shape, 1, 1
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
    while focal_length.ndim < z_prime.ndim:
        focal_length = focal_length.unsqueeze(-1)
    f_on_z_prime = focal_length / z_prime
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
    # Note that ty does not affect u and tx does not affect v
    partial_du_dtx = du_dxprime
    partial_dv_dty = dv_dyprime
    partial_du_dtz = du_dzprime
    partial_dv_dtz = dv_dzprime

    # The direction vector derivatives are built from the rotation vector derivatives
    # Basically, we sum up du_dxprime * dxprime_dr1 * dr1_da1 for each coordinate and rotation matrix element
    # dxprime_dr1 = x, dxprime_dr2 = y, dxprime_dr3 = z, and so on
    partial_du_da1 = (
        du_dxprime * x * orientation_derivatives.dr1_da1.unsqueeze(-1)
        + du_dxprime * y * orientation_derivatives.dr2_da1.unsqueeze(-1)
        + du_dxprime * z * orientation_derivatives.dr3_da1.unsqueeze(-1)
        + du_dzprime * x * orientation_derivatives.dr7_da1.unsqueeze(-1)
        + du_dzprime * y * orientation_derivatives.dr8_da1.unsqueeze(-1)
        + du_dzprime * z * orientation_derivatives.dr9_da1.unsqueeze(-1)
    )
    partial_du_da2 = (
        du_dxprime * x * orientation_derivatives.dr1_da2.unsqueeze(-1)
        + du_dxprime * y * orientation_derivatives.dr2_da2.unsqueeze(-1)
        + du_dxprime * z * orientation_derivatives.dr3_da2.unsqueeze(-1)
        + du_dzprime * x * orientation_derivatives.dr7_da2.unsqueeze(-1)
        + du_dzprime * y * orientation_derivatives.dr8_da2.unsqueeze(-1)
        + du_dzprime * z * orientation_derivatives.dr9_da2.unsqueeze(-1)
    )
    partial_du_da3 = (
        du_dxprime * x * orientation_derivatives.dr1_da3.unsqueeze(-1)
        + du_dxprime * y * orientation_derivatives.dr2_da3.unsqueeze(-1)
        + du_dxprime * z * orientation_derivatives.dr3_da3.unsqueeze(-1)
        + du_dzprime * x * orientation_derivatives.dr7_da3.unsqueeze(-1)
        + du_dzprime * y * orientation_derivatives.dr8_da3.unsqueeze(-1)
        + du_dzprime * z * orientation_derivatives.dr9_da3.unsqueeze(-1)
    )
    partial_dv_da1 = (
        dv_dyprime * x * orientation_derivatives.dr4_da1.unsqueeze(-1)
        + dv_dyprime * y * orientation_derivatives.dr5_da1.unsqueeze(-1)
        + dv_dyprime * z * orientation_derivatives.dr6_da1.unsqueeze(-1)
        + dv_dzprime * x * orientation_derivatives.dr7_da1.unsqueeze(-1)
        + dv_dzprime * y * orientation_derivatives.dr8_da1.unsqueeze(-1)
        + dv_dzprime * z * orientation_derivatives.dr9_da1.unsqueeze(-1)
    )
    partial_dv_da2 = (
        dv_dyprime * x * orientation_derivatives.dr4_da2.unsqueeze(-1)
        + dv_dyprime * y * orientation_derivatives.dr5_da2.unsqueeze(-1)
        + dv_dyprime * z * orientation_derivatives.dr6_da2.unsqueeze(-1)
        + dv_dzprime * x * orientation_derivatives.dr7_da2.unsqueeze(-1)
        + dv_dzprime * y * orientation_derivatives.dr8_da2.unsqueeze(-1)
        + dv_dzprime * z * orientation_derivatives.dr9_da2.unsqueeze(-1)
    )
    partial_dv_da3 = (
        dv_dyprime * x * orientation_derivatives.dr4_da3.unsqueeze(-1)
        + dv_dyprime * y * orientation_derivatives.dr5_da3.unsqueeze(-1)
        + dv_dyprime * z * orientation_derivatives.dr6_da3.unsqueeze(-1)
        + dv_dzprime * x * orientation_derivatives.dr7_da3.unsqueeze(-1)
        + dv_dzprime * y * orientation_derivatives.dr8_da3.unsqueeze(-1)
        + dv_dzprime * z * orientation_derivatives.dr9_da3.unsqueeze(-1)
    )
    partial_du_db1 = (
        du_dxprime * y * orientation_derivatives.dr2_db1.unsqueeze(-1)
        + du_dxprime * z * orientation_derivatives.dr3_db1.unsqueeze(-1)
        + du_dzprime * y * orientation_derivatives.dr8_db1.unsqueeze(-1)
        + du_dzprime * z * orientation_derivatives.dr9_db1.unsqueeze(-1)
    )
    partial_du_db2 = (
        du_dxprime * y * orientation_derivatives.dr2_db2.unsqueeze(-1)
        + du_dxprime * z * orientation_derivatives.dr3_db2.unsqueeze(-1)
        + du_dzprime * y * orientation_derivatives.dr8_db2.unsqueeze(-1)
        + du_dzprime * z * orientation_derivatives.dr9_db2.unsqueeze(-1)
    )
    partial_du_db3 = (
        du_dxprime * y * orientation_derivatives.dr2_db3.unsqueeze(-1)
        + du_dxprime * z * orientation_derivatives.dr3_db3.unsqueeze(-1)
        + du_dzprime * y * orientation_derivatives.dr8_db3.unsqueeze(-1)
        + du_dzprime * z * orientation_derivatives.dr9_db3.unsqueeze(-1)
    )
    partial_dv_db1 = (
        dv_dyprime * y * orientation_derivatives.dr5_db1.unsqueeze(-1)
        + dv_dyprime * z * orientation_derivatives.dr6_db1.unsqueeze(-1)
        + dv_dzprime * y * orientation_derivatives.dr8_db1.unsqueeze(-1)
        + dv_dzprime * z * orientation_derivatives.dr9_db1.unsqueeze(-1)
    )
    partial_dv_db2 = (
        dv_dyprime * y * orientation_derivatives.dr5_db2.unsqueeze(-1)
        + dv_dyprime * z * orientation_derivatives.dr6_db2.unsqueeze(-1)
        + dv_dzprime * y * orientation_derivatives.dr8_db2.unsqueeze(-1)
        + dv_dzprime * z * orientation_derivatives.dr9_db2.unsqueeze(-1)
    )
    partial_dv_db3 = (
        dv_dyprime * y * orientation_derivatives.dr5_db3.unsqueeze(-1)
        + dv_dyprime * z * orientation_derivatives.dr6_db3.unsqueeze(-1)
        + dv_dzprime * y * orientation_derivatives.dr8_db3.unsqueeze(-1)
        + dv_dzprime * z * orientation_derivatives.dr9_db3.unsqueeze(-1)
    )

    # Lastly, the gradients for the world points
    # These are again chain rule patterns
    partial_du_dx = du_dxprime * rotation_matrix.r1.unsqueeze(
        -1
    ) + du_dzprime * rotation_matrix.r7.unsqueeze(-1)
    partial_dv_dx = dv_dyprime * rotation_matrix.r4.unsqueeze(
        -1
    ) + dv_dzprime * rotation_matrix.r7.unsqueeze(-1)
    partial_du_dy = du_dxprime * rotation_matrix.r2.unsqueeze(
        -1
    ) + dv_dzprime * rotation_matrix.r8.unsqueeze(-1)
    partial_dv_dy = dv_dyprime * rotation_matrix.r5.unsqueeze(
        -1
    ) + dv_dzprime * rotation_matrix.r8.unsqueeze(-1)
    partial_du_dz = du_dxprime * rotation_matrix.r3.unsqueeze(
        -1
    ) + du_dzprime * rotation_matrix.r9.unsqueeze(-1)
    partial_dv_dz = dv_dyprime * rotation_matrix.r6.unsqueeze(
        -1
    ) + dv_dzprime * rotation_matrix.r9.unsqueeze(-1)

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


def _stack_gradients(
    residuals_u: torch.Tensor,
    residuals_v: torch.Tensor,
    partial_derivatives: _CameraGradients,
) -> torch.Tensor:
    # CX/CY derivatives are 1 for their coordinate, and 0 for the other
    cx_gradients = residuals_u.sum(dim=(-2, -1)).unsqueeze(-1)
    cy_gradients = residuals_v.sum(dim=(-2, -1)).unsqueeze(-1)
    # FX derivatives are summed over both the views and the points
    f_gradients = (
        (
            (residuals_u * partial_derivatives.partial_du_df).sum(dim=(-2, -1))
            + (residuals_v * partial_derivatives.partial_dv_df).sum(dim=(-2, -1))
        ).unsqueeze(-1)
    )
    # Compute camera view gradients. Summed over all world points
    a1_gradients = (residuals_u * partial_derivatives.partial_du_da1).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_da1
    ).sum(dim=-1)
    a2_gradients = (residuals_u * partial_derivatives.partial_du_da2).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_da2
    ).sum(dim=-1)
    a3_gradients = (residuals_u * partial_derivatives.partial_du_da3).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_da3
    ).sum(dim=-1)
    b1_gradients = (residuals_u * partial_derivatives.partial_du_db1).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_db1
    ).sum(dim=-1)
    b2_gradients = (residuals_u * partial_derivatives.partial_du_db2).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_db2
    ).sum(dim=-1)
    b3_gradients = (residuals_u * partial_derivatives.partial_du_db3).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_db3
    ).sum(dim=-1)
    tx_gradients = (residuals_u * partial_derivatives.partial_du_dtx).sum(dim=-1)
    ty_gradients = (residuals_v * partial_derivatives.partial_dv_dty).sum(dim=-1)
    tz_gradients = (residuals_u * partial_derivatives.partial_du_dtz).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_dtz
    ).sum(dim=-1)
    # Compute world point gradients, summed over all views
    world_x_gradients = (residuals_u * partial_derivatives.partial_du_dx).sum(
        dim=-2
    ) + (residuals_v * partial_derivatives.partial_dv_dx).sum(dim=-2)
    world_y_gradients = (
        residuals_u * partial_derivatives.partial_du_dy
        + residuals_v * partial_derivatives.partial_dv_dy
    ).sum(dim=-2)
    world_z_gradients = (
        residuals_u * partial_derivatives.partial_du_dz
        + residuals_v * partial_derivatives.partial_dv_dz
    ).sum(dim=-2)
    return torch.cat(
        [
            cx_gradients,
            cy_gradients,
            f_gradients,
            # Stack the camera gradients, by parameter
            a1_gradients,
            a2_gradients,
            a3_gradients,
            b1_gradients,
            b2_gradients,
            b3_gradients,
            tx_gradients,
            ty_gradients,
            tz_gradients,
            # Stack the world point gradients, per axis
            world_x_gradients,
            world_y_gradients,
            world_z_gradients,
        ],
        dim=-1,
    )
