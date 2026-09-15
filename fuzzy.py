from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn

RULES = [
    ("R1", [("wavelet_energy","high"), ("vit_context","irregular_cluster")], "likely_malignant"),
    ("R2", [("eff_texture","smooth"), ("wavelet_kurtosis","low")], "likely_benign"),
    ("R3", [("wavelet_entropy","moderate"), ("vit_context","diffuse_pattern")], "uncertain"),
    ("R4", [("wavelet_skewness","positive"), ("wavelet_homogeneity","moderate")], "moderate"),
    ("R5", [("vit_context","clustered_calcification"), ("eff_density","high")], "likely_malignant"),
    ("R6", [("wavelet_entropy","high"), ("vit_context","linear_pattern")], "likely_malignant"),
    ("R7", [("eff_texture","coarse"), ("wavelet_energy","low")], "likely_benign"),
    ("R8", [("wavelet_contrast","moderate"), ("vit_context","spotty_pattern")], "uncertain"),
    ("R9", [("eff_density","low"), ("wavelet_homogeneity","high")], "likely_benign"),
    ("R10",[("vit_context","complex_cluster"), ("wavelet_skewness","negative")], "moderate"),
    ("R11",[("wavelet_contrast","high"), ("eff_texture","fine")], "likely_malignant"),
    ("R12",[("vit_context","scattered_calcification"), ("wavelet_kurtosis","moderate")], "uncertain"),
]

CONSEQUENT_VALUE = {
    "likely_benign": 0.10,
    "uncertain": 0.45,
    "moderate": 0.60,
    "likely_malignant": 0.90,
}

class GaussianIT2Membership(nn.Module):
    """
    Interval type-2 Gaussian membership with uncertain sigma.
    """
    def __init__(self, mu=0.5, sigma_low=0.10, sigma_high=0.20):
        super().__init__()
        self.mu = float(mu)
        self.sigma_low = float(sigma_low)
        self.sigma_high = float(sigma_high)

    def forward(self, x):
        lo = torch.exp(-0.5 * ((x-self.mu) / self.sigma_low)**2)
        hi = torch.exp(-0.5 * ((x-self.mu) / self.sigma_high)**2)
        lower = torch.minimum(lo, hi)
        upper = torch.maximum(lo, hi)
        return lower, upper

class LinguisticAdapter(nn.Module):
    """
    Converts the fused 256-D embedding into soft antecedent scores in [0,1].

    IMPORTANT:
    The manuscript names the linguistic antecedents but does not specify the
    exact deterministic mapping from CNN/ViT embeddings to each linguistic
    variable. This small learned adapter makes the published architecture
    executable while keeping the 12 published rules unchanged.
    """
    def __init__(self, in_dim=256):
        super().__init__()
        self.names = [
            "wavelet_energy","wavelet_kurtosis","wavelet_entropy",
            "wavelet_skewness","wavelet_homogeneity","wavelet_contrast",
            "vit_irregular","vit_diffuse","vit_clustered","vit_linear",
            "vit_spotty","vit_complex","vit_scattered",
            "eff_texture","eff_density"
        ]
        self.proj = nn.Linear(in_dim, len(self.names))

    def forward(self, fused):
        vals = torch.sigmoid(self.proj(fused))
        return {name: vals[:,i] for i, name in enumerate(self.names)}

def _linguistic_degree(name, term, vals):
    x = vals[name]

    # Simple monotone/moderate membership surrogates in [0,1].
    # The manuscript does not publish the exact centers/sigmas.
    if term in ("high","positive","coarse","fine","smooth"):
        return x
    if term in ("low","negative"):
        return 1.0 - x
    if term == "moderate":
        return 1.0 - torch.abs(2*x - 1.0)

    # context terms are already separate soft scores
    return x

class IntervalType2FuzzyHead(nn.Module):
    def __init__(self, fused_dim=256):
        super().__init__()
        self.adapter = LinguisticAdapter(fused_dim)

    def forward(self, fused):
        raw = self.adapter(fused)

        vals = {
            "wavelet_energy": raw["wavelet_energy"],
            "wavelet_kurtosis": raw["wavelet_kurtosis"],
            "wavelet_entropy": raw["wavelet_entropy"],
            "wavelet_skewness": raw["wavelet_skewness"],
            "wavelet_homogeneity": raw["wavelet_homogeneity"],
            "wavelet_contrast": raw["wavelet_contrast"],
            "eff_texture": raw["eff_texture"],
            "eff_density": raw["eff_density"],
            "vit_context_irregular_cluster": raw["vit_irregular"],
            "vit_context_diffuse_pattern": raw["vit_diffuse"],
            "vit_context_clustered_calcification": raw["vit_clustered"],
            "vit_context_linear_pattern": raw["vit_linear"],
            "vit_context_spotty_pattern": raw["vit_spotty"],
            "vit_context_complex_cluster": raw["vit_complex"],
            "vit_context_scattered_calcification": raw["vit_scattered"],
        }

        numer = torch.zeros(fused.shape[0], device=fused.device)
        denom = torch.zeros_like(numer)

        for _, ants, cons in RULES:
            firing = torch.ones_like(numer)
            for var, term in ants:
                if var == "vit_context":
                    key = f"vit_context_{term}"
                    d = vals[key]
                else:
                    d = _linguistic_degree(var, term, vals)
                firing = firing * d.clamp(1e-6, 1.0)

            y = CONSEQUENT_VALUE[cons]
            numer = numer + firing * y
            denom = denom + firing

        score = numer / (denom + 1e-8)
        return score.clamp(0,1)
