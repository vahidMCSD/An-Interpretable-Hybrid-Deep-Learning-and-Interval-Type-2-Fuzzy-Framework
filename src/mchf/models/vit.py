from __future__ import annotations

import torch
from torch import nn
import timm


class ViTBranch(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
        projection_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = int(getattr(self.backbone, "num_features", 768))
        self.proj = nn.Linear(feat_dim, projection_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.backbone.forward_features(x)
        if feat.ndim == 3:
            cls = feat[:, 0]
        else:
            cls = feat
        return self.proj(cls)
