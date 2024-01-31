from typing import Final, NamedTuple, Self
import torch

from deep_attention_visual_odometry.solvers import IOptimisableFunction
from deep_attention_visual_odometry.utils import masked_merge_tensors
from .lie_rotation import LieRotation


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
        translation: torch.Tensor,
        orientation: LieRotation,
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
        :param translation: BxExMx3
        :param orientation: BxExMx3
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
        self._orientation = orientation
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
        return 3 + 6 * self._num_views + 3 * self._num_points

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
            orientation_gradients = self._orientation.parameter_gradient(
                self._world_points[:, :, None, :, :]
            )
            rotation_gradients = self._orientation.vector_gradient()
            camera_relative_points = self._get_camera_relative_points()
            u = self._get_u()
            v = self._get_v()
            residuals_u = u - self._true_projected_points[:, :, :, :, 0]
            residuals_v = v - self._true_projected_points[:, :, :, :, 1]
            partial_derivatives = _compute_gradient_from_intermediates(
                x_prime=camera_relative_points[:, :, :, :, 0],
                y_prime=camera_relative_points[:, :, :, :, 1],
                z_prime=camera_relative_points[:, :, :, :, 2],
                focal_length=self._focal_length,
                orientation_gradients=orientation_gradients,
                rotated_vector_gradients=rotation_gradients,
            )
            self._gradient = _stack_gradients(
                residuals_u, residuals_v, partial_derivatives
            )
            self._gradient_mask = None
        elif self._gradient_mask is not None:
            camera_relative_points = self._get_camera_relative_points()
            camera_relative_points = camera_relative_points[self._gradient_mask]
            camera_relative_points = camera_relative_points.unsqueeze(1)
            u = self._get_u()
            u = u[self._gradient_mask]
            u = u.unsqueeze(1)
            v = self._get_v()
            v = v[self._gradient_mask]
            v = v.unsqueeze(1)
            focal_length = self._focal_length[self._gradient_mask]
            focal_length = focal_length.unsqueeze(1)
            true_projected_points = self._true_projected_points[self._gradient_mask]
            true_projected_points = true_projected_points.unsqueeze(1)
            world_points = self._world_points[self._gradient_mask]
            world_points = world_points.unsqueeze(1)
            orientation = self._orientation.slice(self._gradient_mask)
            orientation_gradients = orientation.parameter_gradient(world_points[:, :, None, :, :])
            rotation_gradients = orientation.vector_gradient()
            residuals_u = u - true_projected_points[:, :, :, :, 0]
            residuals_v = v - true_projected_points[:, :, :, :, 1]
            partial_derivatives = _compute_gradient_from_intermediates(
                x_prime=camera_relative_points[:, :, :, :, 0],
                y_prime=camera_relative_points[:, :, :, :, 1],
                z_prime=camera_relative_points[:, :, :, :, 2],
                focal_length=focal_length,
                orientation_gradients=orientation_gradients,
                rotated_vector_gradients=rotation_gradients,
            )
            gradient = _stack_gradients(residuals_u, residuals_v, partial_derivatives)
            self._gradient = self._gradient.masked_scatter(
                self._gradient_mask.unsqueeze(-1), gradient
            )
            self._gradient_mask = None
        return self._gradient

    def add(self, parameters: torch.Tensor) -> Self:
        # Find the slice indices for the parameters, based on the number of views and world points
        a_idx = self.VIEW_START
        b_idx = a_idx + self._num_views
        c_idx = b_idx + self._num_views
        tx_idx = c_idx + self._num_views
        ty_idx = tx_idx + self._num_views
        tz_idx = ty_idx + self._num_views
        x_idx = tz_idx + self._num_views
        y_idx = x_idx + self._num_points
        z_idx = y_idx + self._num_points
        end_idx = z_idx + self._num_points
        a_params = parameters[:, :, a_idx:b_idx]
        b_params = parameters[:, :, b_idx:c_idx]
        c_params = parameters[:, :, c_idx:tx_idx]
        tx_params = parameters[:, :, tx_idx:ty_idx]
        ty_params = parameters[:, :, ty_idx:tz_idx]
        tz_params = parameters[:, :, tz_idx:x_idx]
        x_params = parameters[:, :, x_idx:y_idx]
        y_params = parameters[:, :, y_idx:z_idx]
        z_params = parameters[:, :, z_idx:end_idx]

        t_params = torch.stack([tx_params, ty_params, tz_params], dim=-1)
        point_params = torch.stack([x_params, y_params, z_params], dim=-1)
        new_orientation = self._orientation.add_lie_parameters(
            torch.stack([a_params, b_params, c_params], dim=-1).unsqueeze(-2)
        )
        return type(self)(
            focal_length=self._focal_length + parameters[:, :, self.F],
            cx=self._cx + parameters[:, :, self.CX],
            cy=self._cy + parameters[:, :, self.CY],
            translation=self._translation + t_params,
            orientation=new_orientation,
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
        orientation = self._orientation.masked_update(
            other._orientation, vector_mask.unsqueeze(-2)
        )
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
            translation=translation,
            orientation=orientation,
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
            rotated_points = self._orientation.rotate_vector(
                self._world_points[:, :, None, :, :]
            )
            rotated_points = rotated_points + self._translation[:, :, :, None, :]
            # Clamp the camera-relative z' to treat all points as "in front" of the camera,
            # Due to the division, the optmisation cannot cross through Z' = 0,
            # because the projected points go to infinity, and thus so does the error.
            # It is a relatively safe assumption that any point we can see, is in front of the camera.
            rotated_points = torch.cat(
                [
                    rotated_points[:, :, :, :, 0:2],
                    torch.clamp(
                        rotated_points[:, :, :, :, 2:3], min=self.minimum_distance
                    ),
                ],
                dim=-1,
            )
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
    partial_du_da: torch.Tensor
    partial_dv_da: torch.Tensor
    partial_du_db: torch.Tensor
    partial_dv_db: torch.Tensor
    partial_du_dc: torch.Tensor
    partial_dv_dc: torch.Tensor
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
    x_prime: torch.Tensor,
    y_prime: torch.Tensor,
    z_prime: torch.Tensor,
    focal_length: torch.Tensor,
    orientation_gradients: torch.Tensor,
    rotated_vector_gradients: torch.Tensor,
) -> _CameraGradients:
    """
    Compute the gradient of the error
    :param x_prime: BxExMxN
    :param y_prime: BxExMxN
    :param z_prime: BxExMxN
    :param focal_length: BxE
    :param orientation_gradients: BxExMxNx3x3
    :param rotated_vector_gradients: BxExMx3x3
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

    # The rotation derivatives
    partial_du_da = (
        du_dxprime * orientation_gradients[:, :, :, :, 0, 0]
        + du_dzprime * orientation_gradients[:, :, :, :, 2, 0]
    )
    partial_du_db = (
        du_dxprime * orientation_gradients[:, :, :, :, 0, 1]
        + du_dzprime * orientation_gradients[:, :, :, :, 2, 1]
    )
    partial_du_dc = (
        du_dxprime * orientation_gradients[:, :, :, :, 0, 2]
        + du_dzprime * orientation_gradients[:, :, :, :, 2, 2]
    )
    partial_dv_da = (
        dv_dyprime * orientation_gradients[:, :, :, :, 1, 0]
        + dv_dzprime * orientation_gradients[:, :, :, :, 2, 0]
    )
    partial_dv_db = (
        dv_dyprime * orientation_gradients[:, :, :, :, 1, 1]
        + dv_dzprime * orientation_gradients[:, :, :, :, 2, 1]
    )
    partial_dv_dc = (
        dv_dyprime * orientation_gradients[:, :, :, :, 1, 2]
        + dv_dzprime * orientation_gradients[:, :, :, :, 2, 2]
    )

    # Lastly, the gradients for the world points
    # These are again chain rule patterns
    partial_du_dx = (
        du_dxprime * rotated_vector_gradients[:, :, :, :, 0, 0]
        + du_dzprime * rotated_vector_gradients[:, :, :, :, 2, 0]
    )
    partial_dv_dx = (
        dv_dyprime * rotated_vector_gradients[:, :, :, :, 1, 0]
        + dv_dzprime * rotated_vector_gradients[:, :, :, :, 2, 0]
    )
    partial_du_dy = (
        du_dxprime * rotated_vector_gradients[:, :, :, :, 0, 1]
        + dv_dzprime * rotated_vector_gradients[:, :, :, :, 2, 1]
    )
    partial_dv_dy = (
        dv_dyprime * rotated_vector_gradients[:, :, :, :, 1, 1]
        + dv_dzprime * rotated_vector_gradients[:, :, :, :, 2, 1]
    )
    partial_du_dz = (
        du_dxprime * rotated_vector_gradients[:, :, :, :, 0, 2]
        + du_dzprime * rotated_vector_gradients[:, :, :, :, 2, 2]
    )
    partial_dv_dz = (
        dv_dyprime * rotated_vector_gradients[:, :, :, :, 1, 2]
        + dv_dzprime * rotated_vector_gradients[:, :, :, :, 2, 2]
    )

    return _CameraGradients(
        partial_du_df=partial_du_df,
        partial_dv_df=partial_dv_df,
        partial_du_da=partial_du_da,
        partial_dv_da=partial_dv_da,
        partial_du_db=partial_du_db,
        partial_dv_db=partial_dv_db,
        partial_du_dc=partial_du_dc,
        partial_dv_dc=partial_dv_dc,
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
        (residuals_u * partial_derivatives.partial_du_df).sum(dim=(-2, -1))
        + (residuals_v * partial_derivatives.partial_dv_df).sum(dim=(-2, -1))
    ).unsqueeze(-1)
    # Compute camera view gradients. Summed over all world points
    a_gradients = (residuals_u * partial_derivatives.partial_du_da).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_da
    ).sum(dim=-1)
    b_gradients = (residuals_u * partial_derivatives.partial_du_db).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_db
    ).sum(dim=-1)
    c_gradients = (residuals_u * partial_derivatives.partial_du_dc).sum(dim=-1) + (
        residuals_v * partial_derivatives.partial_dv_dc
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
            a_gradients,
            b_gradients,
            c_gradients,
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
