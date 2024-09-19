# Copyright (C) 2024  John Skinner
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA
import torch


def project_points_clamped_pinhole(
    points: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    An alternative to traditional pinhole camera projection that tries to ensure that:
    - projected coordinates are not too large
    - points behind the camera get projected to the edges of the image,
      rather than the middle like the traditional model
    - Large points still have gradients, rather than saturating with a clamp
    - Focal length should be strictly positive and small changes should have large effects.
      For that reason, e^(fx) is a more effective parametrisation than fx
    We assume that the image bounds are -1, 1.

    The projection calculation is a four-part if statement:
    if z < -1:
        u = 101 + log(abs(z)) + cx
    elif -1 < z < 1e-100:
        u = 100 - z + cx
    elif fx + log|x| - log|z| > 0
        u = sign(x) (1 + fx + log|x| - log|z|) + cx
    else:
        u = e^fx * x / z + cx

    This function is expected to work better with standard-normal distributed points and parameters,
    which occur more often when dealing with neural networks.

    :param points: A (B...)x3 vector of world points to project
    :param intrinsics: A (B...)x3 vector of camera intrinsics: (fx, cx, cy)
    :return: A (B..)x2 vector of pixel coordinates of the point in the view.
    """
    focal_length = intrinsics[..., 0:1]
    principal_point = intrinsics[..., 1:3]
    xy = points[..., 0:2]
    z = points[..., 2:3]

    is_z_large_negative = z < -1.0
    is_z_positive = z > 1e-100
    log_points = points.abs().log()
    log_xy = log_points[..., 0:2]
    log_z = log_points[..., 2:3]
    exp_focal_length = focal_length.exp()

    sign_xy = xy.sign()
    negative_projected_points = sign_xy * torch.where(
        is_z_large_negative, 101 + log_z, 100 - z
    )

    log_projection = focal_length + log_xy - log_z
    projection = exp_focal_length * xy / z
    is_in_bounds = log_projection < 0.0
    positive_projected_points = torch.where(
        is_in_bounds, projection, sign_xy * (log_projection + 1.0)
    )

    return principal_point + torch.where(
        is_z_positive, positive_projected_points, negative_projected_points
    )
