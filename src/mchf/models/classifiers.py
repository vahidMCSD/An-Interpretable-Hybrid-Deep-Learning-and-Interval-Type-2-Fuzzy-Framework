from __future__ import annotations
import torch
from torch import nn
from .effnet_cbam import EfficientNetV2CBAM
from .vit import ViTBranch

class EfficientNetCBAMClassifier(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__(); self.encoder=EfficientNetV2CBAM(**encoder_kwargs); self.head=nn.Linear(128,1)
    def forward(self,x): return self.head(self.encoder(x)).squeeze(-1)

class ViTClassifier(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__(); self.encoder=ViTBranch(**encoder_kwargs); self.head=nn.Linear(256,1)
    def forward(self,x): return self.head(self.encoder(x)).squeeze(-1)
