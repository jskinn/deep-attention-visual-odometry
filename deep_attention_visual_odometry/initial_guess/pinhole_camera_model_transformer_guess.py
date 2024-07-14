from typing import Literal
import logging
import torch
import torch.nn as nn
from deep_attention_visual_odometry.networks.weights import (
    get_kaiming_normal_init_function,
)
from deep_attention_visual_odometry.camera_model.pinhole_camera_model_l1 import (
    PinholeCameraModelL1,
)
from .base_pinhole_camera_model_guess import BasePinholeCameraModelGuess


class PinholeCameraModelTransformerGuess(BasePinholeCameraModelGuess):
    def __init__(
        self,
        num_views: int,
        num_points: int,
        embed_dim: int,
        num_estimates: int = 1,
        num_layers: int = 4,
        init_weights: bool = False,
        float_precision: Literal["32", "64"] = "32",
        constrain: bool = True,
        max_gradient: float = -1.0,
        minimum_z_distance: float = 1e-3,
        maximum_pixel_ratio: float = 5.0,
        enable_error_gradients: bool = True,
        enable_grad_gradients: bool = False,
    ):
        super().__init__(
            num_views=num_views,
            num_points=num_points,
            constrain=constrain,
            max_gradient=max_gradient,
            minimum_z_distance=minimum_z_distance,
            maximum_pixel_ratio=maximum_pixel_ratio,
            enable_error_gradients=enable_error_gradients,
            enable_grad_gradients=enable_grad_gradients,
        )
        dtype = torch.float64 if float_precision == "64" else torch.float32
        num_estimates = int(num_estimates)
        if num_estimates <= 0:
            num_estimates = 1
        elif num_estimates > num_views * num_points:
            logging.getLogger(__name__).warning(
                f"The transformer initial guess can only provide {num_views * num_points} estimates. "
                f"Reducing from {num_estimates} to that"
            )
            num_estimates = num_views * num_points
        self.num_estimates = num_estimates

        # Create a pair of orthonormal unit vectors that widen the u,v coordinates to the embed dim.
        u_embedding = torch.randn(embed_dim, dtype=dtype)
        u_embedding = u_embedding / torch.linalg.vector_norm(u_embedding)
        v_embedding = torch.randn(embed_dim, dtype=dtype)
        v_embedding = v_embedding - u_embedding * torch.dot(u_embedding, v_embedding)
        v_embedding = v_embedding / torch.linalg.norm(v_embedding)
        self.pixel_embedding = nn.Parameter(torch.stack([u_embedding, v_embedding]))

        # Create positional encoding for the views and point indices
        positional_encoding = torch.empty(num_views, num_points, embed_dim)
        positional_encoding[:, :, 0::4] = torch.sin().tile((1, num_points, 1))
        positional_encoding[:, :, 1::4] = torch.cos().tile((1, num_points, 1))
        positional_encoding[:, :, 2::4] = torch.sin().tile((num_views, 1, 1))
        positional_encoding[:, :, 3::4] = torch.cos().tile((num_views, 1, 1))
        self.positional_encoding = nn.Parameter(positional_encoding)

        # Build a transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True, dtype=dtype
        )
        self.estimator = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers),
            nn.Linear(
                embed_dim, self._get_num_model_parameters(), bias=True, dtype=dtype
            ),
        )
        if init_weights:
            self.init_weights()

    def init_weights(self):
        self.apply(
            get_kaiming_normal_init_function(
                {nn.Linear}, mode="fan_in", nonlinearity="relu"
            )
        )

    def forward(
        self, projected_points: torch.Tensor, visibility_mask: torch.Tensor
    ) -> PinholeCameraModelL1:
        batch_size = projected_points.size(0)
        x = torch.matmul(projected_points, self.pixel_embedding)
        x = x + self.positional_encoding[None, :, :, :]
        x = x.reshape(batch_size, -1, x.shape[-1])
        x = self.estimator(x)
        x = x[:, :self.num_estimates, :]
        return self._build_model_from_vector(
            x, batch_size, projected_points, visibility_mask
        )
