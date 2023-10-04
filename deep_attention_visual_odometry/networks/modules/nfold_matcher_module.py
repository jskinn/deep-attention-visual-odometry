from typing import Iterable
import torch.nn as nn
import torch.nn.functional as fn
from deep_attention_visual_odometry.types import FeaturePoints, MatchedPoints


class FeatureMatchModule(nn.Module):
    """
    Match points based on feature descriptors using scaled dot product attention.
    Given points and feature descriptors from two images, match those points based on the descriptors,
    and return pairs of points that are considered the "same" point.
    """

    def __init__(
        self,
        descriptor_size: int,
        embedding_size: int | None = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        descriptor_size = max(int(descriptor_size), 1)
        if embedding_size is None:
            embedding_size = descriptor_size
        self.key = nn.Linear(descriptor_size, embedding_size)
        self.query = nn.Linear(descriptor_size, embedding_size)
        self.dropout = float(dropout)

    def forward(
        self, image_features: Iterable[FeaturePoints]
    ) -> MatchedPoints:
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0.0
        for feature_points in image_features:
            
        key = self.key(features_a.descriptors)
        query = self.query(features_b.descriptors)
        matched_points = fn.scaled_dot_product_attention(
            query, key, features_b.points, dropout_p=dropout
        )
        return MatchedPoints(features_a.points, matched_points)
