from __future__ import annotations

import torch
from torch import nn
import timm

from .cbam import CBAM


class EfficientNetV2CBAM(nn.Module):
    def __init__(
        self,
        model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k",
        projection_dim: int = 128,
        reduction: int = 16,
        spatial_kernel: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(-1,))
        channels = self.backbone.feature_info.channels()[-1]
        self.cbam = CBAM(channels, reduction=reduction, spatial_kernel=spatial_kernel)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, projection_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, return_feature_map: bool = False):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        fmap = self.backbone(x)[-1]
        refined = self.cbam(fmap)
        z = self.pool(refined).flatten(1)
        z = self.proj(z)
        if return_feature_map:
            return z, refined
        return z
