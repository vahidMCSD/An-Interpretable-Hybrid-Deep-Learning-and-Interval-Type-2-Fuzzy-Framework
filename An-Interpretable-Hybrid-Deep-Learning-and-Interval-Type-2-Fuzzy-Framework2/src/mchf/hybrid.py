from __future__ import annotations

import torch
from torch import nn

from .models.fusion import AttentionFusion
from .fuzzy.type2 import IntervalType2Fuzzy, SemanticRuleAdapter, Type2Config


class FeatureLevelHybrid(nn.Module):
    """Trainable fusion/fuzzy/calibration head over pre-extracted branch features."""
    def __init__(self, wave_dim: int, eff_dim: int = 128, vit_dim: int = 256, hidden_dim: int = 256, heads: int = 4, dropout: float = 0.1, fuzzy_cfg: Type2Config | None = None):
        super().__init__()
        self.fusion = AttentionFusion(wave_dim + eff_dim + vit_dim, hidden_dim, heads, dropout)
        self.semantic = SemanticRuleAdapter(eff_dim, vit_dim)
        self.fuzzy = IntervalType2Fuzzy(fuzzy_cfg)
        # Continuous probability used for AUC/calibration/Youden threshold.
        # The manuscript requires a continuous score in addition to the ART label.
        self.calibration_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, wave, eff, vit, wave_rule_values: dict[str, torch.Tensor], fusion_scaled: torch.Tensor | None = None):
        fusion_input = fusion_scaled if fusion_scaled is not None else torch.cat([wave, eff, vit], dim=-1)
        fused = self.fusion(fusion_input)
        semantic = self.semantic(eff, vit)
        values = dict(wave_rule_values)
        values.update(semantic)
        fuzzy_score, lower, upper, rule_l, rule_u = self.fuzzy(values)
        logits = self.calibration_head(torch.cat([fused, fuzzy_score[:, None]], dim=-1)).squeeze(-1)
        prob = torch.sigmoid(logits)
        return {
            "fused": fused,
            "fuzzy_score": fuzzy_score,
            "fuzzy_lower": lower,
            "fuzzy_upper": upper,
            "rule_lower": rule_l,
            "rule_upper": rule_u,
            "logits": logits,
            "prob": prob,
            **semantic,
        }
