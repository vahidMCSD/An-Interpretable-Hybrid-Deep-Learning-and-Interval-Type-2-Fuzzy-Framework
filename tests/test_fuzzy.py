import torch
from mchf.fuzzy.type2 import IntervalType2Fuzzy

def test_fuzzy_score_range():
    b=8
    values={
      'wavelet_energy':torch.rand(b),'wavelet_entropy':torch.rand(b),'wavelet_contrast':torch.rand(b),
      'wavelet_homogeneity':torch.rand(b),'wavelet_skewness':torch.rand(b),'wavelet_kurtosis':torch.rand(b),
      'eff_texture':torch.rand(b),'eff_density':torch.rand(b),
      'vit_irregular':torch.rand(b),'vit_diffuse':torch.rand(b),'vit_clustered':torch.rand(b),'vit_linear':torch.rand(b),
      'vit_spotty':torch.rand(b),'vit_complex':torch.rand(b),'vit_scattered':torch.rand(b),
    }
    score,lo,hi,L,U=IntervalType2Fuzzy()(values)
    assert score.shape==(b,); assert torch.all((score>=0)&(score<=1)); assert L.shape==(b,12); assert U.shape==(b,12)
