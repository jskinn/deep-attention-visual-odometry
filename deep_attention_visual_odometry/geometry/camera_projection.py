from typing import Self
import torch


class PinholeCameraProjection:
    """
    Traditional pinhole camera projection, with no edge-cases.
    Returns u = f X / Z + cx and v = f Y / Z + cy in all cases
    """

    def __init__(self, camera_parameters: torch.Tensor):
        self._parameters = camera_parameters

    @property
    def focal_length(self) -> torch.Tensor:
        return self._parameters[..., 0:1]

    @property
    def principal_point(self) -> torch.Tensor:
        return self._parameters[..., 1:3]

    def project_points(self, points: torch.Tensor) -> torch.Tensor:
        xy = points[..., 0:2]
        z = points[..., 2:3]
        return self.focal_length * xy / z + self.principal_point

    def parameter_gradient(self, points: torch.Tensor) -> torch.Tensor:
        """
        Get the gradient of the projected points w.r.t. the camera parameters.
        That is, du/dfx, dv/dfx, du/dcx, dv/dcx, du/dcy, dv/dcy
        :returns: A (Bx)2x3 tensor, du gradients in the first row, dv gradients in the second
        """
        z = points[..., 2:3]
        f_on_z = self.focal_length / z
        


    def vector_gradient(self) -> torch.Tensor:
        """
        Get the gradient of the projected points w.r.t. the point coordinates
        That is, du/dx, dv/dx, du/dy, dv/dy, du/z, and dv/dz
        :return: A (Bx)2x3 tensor, du gradients along the top, and dv gradients in the second row
        """
        pass

    def add(self, parameters: torch.Tensor) -> Self:
        return type(self)(self._parameters + parameters)

    def masked_update(self, other: Self, mask: torch.Tensor) -> Self:
        new_parameters = torch.where(mask, other._parameters, self._parameters)
        return type(self)(new_parameters)
