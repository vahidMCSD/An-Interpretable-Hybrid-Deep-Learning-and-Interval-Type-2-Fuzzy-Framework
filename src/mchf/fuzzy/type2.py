from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np
import torch
from torch import nn

from .rules import RULES, VIT_CONTEXT_NAMES


class RuleFeatureNormalizer:
    """Fit on training data only; transform fuzzy rule variables to [0,1]."""
    CONTINUOUS = (
        "wavelet_energy", "wavelet_entropy", "wavelet_contrast",
        "wavelet_homogeneity", "wavelet_kurtosis", "wavelet_skewness",
    )

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.params: dict[str, tuple[float, float]] = {}

    def fit(self, rows: list[dict[str, float]]):
        for key in self.CONTINUOUS:
            vals = np.asarray([r[key] for r in rows], dtype=float)
            if key == "wavelet_skewness":
                center = float(np.nanmedian(vals)); scale = float(np.nanstd(vals)) or 1.0
                self.params[key] = (center, scale)
            else:
                lo, hi = np.nanquantile(vals, [0.01, 0.99])
                if not np.isfinite(lo): lo = 0.0
                if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
                self.params[key] = (float(lo), float(hi))
        return self

    def transform_value(self, key: str, value):
        if key not in self.params:
            return value
        a, b = self.params[key]
        if key == "wavelet_skewness":
            z = (value - a) / max(b, self.eps)
            return 0.5 * (torch.tanh(z) + 1.0) if torch.is_tensor(value) else float(0.5 * (np.tanh(z) + 1.0))
        out = (value - a) / max(b - a, self.eps)
        return torch.clamp(out, 0.0, 1.0) if torch.is_tensor(out) else float(np.clip(out, 0.0, 1.0))

    def state_dict(self): return dict(self.params)
    def load_state_dict(self, state):
        self.params = {k: tuple(v) for k, v in state.items()}; return self


class SemanticRuleAdapter(nn.Module):
    """Explicit implementation choice for a manuscript-under-specified mapping.

    Supplementary Table S1 names EfficientNet texture/density and ViT semantic
    contexts, but the manuscript does not provide the numerical mapping from
    embeddings to those linguistic variables. This compact supervised adapter
    exposes the assumption instead of hiding it.
    """
    def __init__(self, eff_dim: int = 128, vit_dim: int = 256):
        super().__init__()
        self.eff_heads = nn.Linear(eff_dim, 2)
        self.vit_head = nn.Linear(vit_dim, len(VIT_CONTEXT_NAMES))

    def forward(self, eff: torch.Tensor, vit: torch.Tensor) -> dict[str, torch.Tensor]:
        eff_scores = torch.sigmoid(self.eff_heads(eff))
        vit_probs = torch.softmax(self.vit_head(vit), dim=-1)
        out = {"eff_texture": eff_scores[:, 0], "eff_density": eff_scores[:, 1]}
        out.update({f"vit_{name}": vit_probs[:, i] for i, name in enumerate(VIT_CONTEXT_NAMES)})
        return out


@dataclass
class Type2Config:
    sigma_uncertainty_fraction: float = 0.20
    epsilon: float = 1e-8
    consequent_values: Mapping[str, float] | None = None
    max_km_iterations: int = 50
    km_tolerance: float = 1e-4
    membership_shift_fraction: float = 0.0
    type_reduction: str = "km"  # km or midpoint

    def __post_init__(self):
        if self.consequent_values is None:
            # Consequent centroids are not numerically specified in the paper.
            self.consequent_values = {
                "likely_benign": 0.10, "uncertain": 0.40,
                "moderate": 0.60, "likely_malignant": 0.90,
            }
        if self.type_reduction not in {"km", "midpoint"}:
            raise ValueError("type_reduction must be 'km' or 'midpoint'")


def _km_endpoint(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, left: bool,
                 max_iter: int = 50, tol: float = 1e-4, eps: float = 1e-12) -> float:
    """Karnik-Mendel centroid endpoint for singleton consequents.

    y, lower and upper are one-dimensional and y is sorted internally. For the
    left endpoint, upper firing strengths are used below the switch point and
    lower strengths above it; the reverse is used for the right endpoint.
    """
    order = np.argsort(y)
    y = np.asarray(y, float)[order]
    lo = np.asarray(lower, float)[order]
    hi = np.asarray(upper, float)[order]
    w = 0.5 * (lo + hi)
    den = max(w.sum(), eps)
    estimate = float(np.dot(y, w) / den)
    for _ in range(max_iter):
        k = int(np.searchsorted(y, estimate, side="right") - 1)
        k = max(-1, min(k, len(y)-1))
        if left:
            weights = np.where(np.arange(len(y)) <= k, hi, lo)
        else:
            weights = np.where(np.arange(len(y)) <= k, lo, hi)
        den = max(weights.sum(), eps)
        new = float(np.dot(y, weights) / den)
        if abs(new - estimate) <= tol:
            estimate = new; break
        estimate = new
    return estimate


def karnik_mendel_type_reduce(y, lower, upper, max_iter=50, tol=1e-4):
    """Exact KM type-reduction. Supports [R] or [N,R] arrays."""
    y = np.asarray(y, dtype=float)
    L = np.asarray(lower, dtype=float); U = np.asarray(upper, dtype=float)
    if L.ndim == 1:
        return (_km_endpoint(y, L, U, True, max_iter, tol),
                _km_endpoint(y, L, U, False, max_iter, tol))
    left=[]; right=[]
    for l,u in zip(L,U):
        left.append(_km_endpoint(y,l,u,True,max_iter,tol))
        right.append(_km_endpoint(y,l,u,False,max_iter,tol))
    return np.asarray(left), np.asarray(right)


class IntervalType2Fuzzy(nn.Module):
    """Interval type-2 inference using the 12 rules in Supplementary Table S1.

    Min t-norm is used for rule firing. Exact KM type-reduction is available and
    is the default for inference. A differentiable midpoint approximation is
    also exposed for training / GPU throughput, matching the manuscript's claim
    that a vectorized approximation was validated against iterative KM.
    """
    def __init__(self, cfg: Type2Config | None = None):
        super().__init__(); self.cfg = cfg or Type2Config()

    def _term_center_sigma(self, term: str) -> tuple[float, float]:
        mapping = {
            "low": (0.20,0.22), "moderate":(0.50,0.22), "high":(0.80,0.22),
            "negative":(0.25,0.20), "positive":(0.75,0.20),
            "smooth":(0.20,0.22), "fine":(0.50,0.22), "coarse":(0.80,0.22),
        }
        if term not in mapping: raise KeyError(term)
        center,sigma=mapping[term]
        center=float(np.clip(center*(1.0+self.cfg.membership_shift_fraction),0.0,1.0))
        return center,sigma

    def _gaussian_interval(self, x: torch.Tensor, term: str):
        center,sigma=self._term_center_sigma(term); u=self.cfg.sigma_uncertainty_fraction
        sl=max(sigma*(1-u),self.cfg.epsilon); su=max(sigma*(1+u),sl+self.cfg.epsilon)
        lower=torch.exp(-0.5*((x-center)/sl)**2); upper=torch.exp(-0.5*((x-center)/su)**2)
        return torch.minimum(lower,upper),torch.maximum(lower,upper)

    def _membership(self, feature: str, term: str, values: dict[str, torch.Tensor]):
        x=values[feature]
        if term=="present":
            u=self.cfg.sigma_uncertainty_fraction
            return torch.clamp(x*(1-u),0,1),torch.clamp(x*(1+u),0,1)
        return self._gaussian_interval(x,term)

    def firing_intervals(self, values: dict[str, torch.Tensor]):
        lowers=[]; uppers=[]; ys=[]
        for rule in RULES:
            lm=[]; um=[]
            for ant in rule.antecedents:
                l,u=self._membership(ant.feature,ant.term,values); lm.append(l); um.append(u)
            lowers.append(torch.stack(lm,-1).amin(-1)); uppers.append(torch.stack(um,-1).amin(-1))
            ys.append(float(self.cfg.consequent_values[rule.consequent]))
        return torch.stack(lowers,-1),torch.stack(uppers,-1),torch.tensor(ys,dtype=lowers[0].dtype,device=lowers[0].device)

    def midpoint_type_reduce(self,L,U,y):
        eps=self.cfg.epsilon
        ol=(L*y).sum(-1)/(L.sum(-1)+eps); ou=(U*y).sum(-1)/(U.sum(-1)+eps)
        lo=torch.minimum(ol,ou); hi=torch.maximum(ol,ou)
        no_fire=(L.sum(-1)+U.sum(-1)) < 10*eps
        score=0.5*(lo+hi); score=torch.where(no_fire,torch.full_like(score,0.5),score)
        return score,lo,hi

    def km_type_reduce(self,L,U,y):
        # Exact KM is non-smooth at switch changes. Preserve a straight-through
        # gradient from the differentiable midpoint score for trainability.
        mid,_,_=self.midpoint_type_reduce(L,U,y)
        left,right=karnik_mendel_type_reduce(y.detach().cpu().numpy(),L.detach().cpu().numpy(),U.detach().cpu().numpy(),
                                             self.cfg.max_km_iterations,self.cfg.km_tolerance)
        left_t=torch.as_tensor(left,dtype=L.dtype,device=L.device); right_t=torch.as_tensor(right,dtype=L.dtype,device=L.device)
        score_exact=0.5*(left_t+right_t)
        score=mid + (score_exact-mid).detach()
        return score,left_t,right_t

    def forward(self, values: dict[str, torch.Tensor]):
        L,U,y=self.firing_intervals(values)
        if self.cfg.type_reduction=="km": score,lo,hi=self.km_type_reduce(L,U,y)
        else: score,lo,hi=self.midpoint_type_reduce(L,U,y)
        return score,lo,hi,L,U
