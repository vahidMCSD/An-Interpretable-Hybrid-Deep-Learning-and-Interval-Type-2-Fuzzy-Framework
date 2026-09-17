from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Antecedent:
    feature: str
    term: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    antecedents: tuple[Antecedent, ...]
    consequent: str


RULES: tuple[Rule, ...] = (
    Rule("R1", (Antecedent("wavelet_energy", "high"), Antecedent("vit_irregular", "present")), "likely_malignant"),
    Rule("R2", (Antecedent("eff_texture", "smooth"), Antecedent("wavelet_kurtosis", "low")), "likely_benign"),
    Rule("R3", (Antecedent("wavelet_entropy", "moderate"), Antecedent("vit_diffuse", "present")), "uncertain"),
    Rule("R4", (Antecedent("wavelet_skewness", "positive"), Antecedent("wavelet_homogeneity", "moderate")), "moderate"),
    Rule("R5", (Antecedent("vit_clustered", "present"), Antecedent("eff_density", "high")), "likely_malignant"),
    Rule("R6", (Antecedent("wavelet_entropy", "high"), Antecedent("vit_linear", "present")), "likely_malignant"),
    Rule("R7", (Antecedent("eff_texture", "coarse"), Antecedent("wavelet_energy", "low")), "likely_benign"),
    Rule("R8", (Antecedent("wavelet_contrast", "moderate"), Antecedent("vit_spotty", "present")), "uncertain"),
    Rule("R9", (Antecedent("eff_density", "low"), Antecedent("wavelet_homogeneity", "high")), "likely_benign"),
    Rule("R10", (Antecedent("vit_complex", "present"), Antecedent("wavelet_skewness", "negative")), "moderate"),
    Rule("R11", (Antecedent("wavelet_contrast", "high"), Antecedent("eff_texture", "fine")), "likely_malignant"),
    Rule("R12", (Antecedent("vit_scattered", "present"), Antecedent("wavelet_kurtosis", "moderate")), "uncertain"),
)

VIT_CONTEXT_NAMES = ("irregular", "diffuse", "clustered", "linear", "spotty", "complex", "scattered")
