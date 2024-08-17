import torch

from .camera_projection import PinholeCameraProjection


class ClampedPinholeCameraProjection(PinholeCameraProjection):
    """
    An alternative to traditional pinhole camera projection that tries to ensure that:
    - projected coordinates are not too large
    - points behind the camera get projected to the edges of the image, rather than looping like the traditional model
    - Large points still have gradients, rather than saturating with a clamp
    We assume that the image bounds are -1, 1.

    The projection calculation becomes a four-part case statement, first depending on z
    if z < -1:
        u = 101 + log(abs(z)) + cx
    elif -1 < z < 1e-100:
        u = 100 - z + cx
    elif log(x) + log(fx) - log(z) > 0
        u = 1 + log(x) + log(fx) - log(z) + cx
    else:
        u = fx * x / z + cx
    """

    def __init__(self, parameters: torch.Tensor):
        super().__init__(parameters)
        self._log_focal_length = None

    @property
    def log_focal_length(self) -> torch.Tensor:
        if self._log_focal_length is None:
            self._log_focal_length = self.focal_length.abs().log()
        return self._log_focal_length

    def project_points(self, points: torch.Tensor) -> torch.Tensor:
        xy = points[..., 0:2]
        z = points[..., 2:3]
        is_z_large_negative = (z < -1.0)
        is_z_positive = (z > 1e-100)
        log_points = points.abs().log()
        sign_xy = xy.sign()
        negative_projected_points = sign_xy * torch.where(is_z_large_negative, 101 + log_points[..., 2:3], 100 + z)
        log_projection = self.log_focal_length + log_points[..., 0:2] - log_points[..., 2:3]
        projection = self.focal_length * xy / z
        is_in_bounds = (log_projection < 0.0)
        positive_projected_points = torch.where(is_in_bounds, projection, log_projection)
        return torch.where(is_z_positive, positive_projected_points, negative_projected_points)

    def parameter_gradient(self) -> torch.Tensor:
        pass

    def vector_gradient(self) -> torch.Tensor:
        """
        Get the gradient of the projected points w.r.t. the point coordinates
        That is, du/dx, dv/dx, du/dy, dv/dy, du/z, and dv/dz
        :return: A (Bx)2x3 tensor, du gradients along the top, and dv gradients in the second row
        """
        pass
