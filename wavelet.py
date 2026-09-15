from typing import List
import numpy as np
import pywt
from scipy.stats import skew, kurtosis

def _stats(a: np.ndarray) -> List[float]:
    x = np.asarray(a, dtype=np.float64).ravel()
    eps = 1e-12
    energy = float(np.mean(x**2))
    entropy = float(-np.sum((np.abs(x)/(np.sum(np.abs(x))+eps)) *
                            np.log(np.abs(x)/(np.sum(np.abs(x))+eps) + eps)))
    mean = float(np.mean(x))
    std = float(np.std(x))
    sk = float(skew(x, bias=False)) if x.size > 2 else 0.0
    ku = float(kurtosis(x, fisher=True, bias=False)) if x.size > 3 else 0.0
    return [mean, std, energy, entropy, sk, ku]

def extract_wavelet_features(gray_224: np.ndarray, wavelet="db4", levels=3) -> np.ndarray:
    """
    Three-level DWT using db4.
    The paper states 36 raw wavelet features before PCA.
    To produce exactly 36, we use the six detail sub-bands
    {LH,HL} from three levels x six statistics each.

    NOTE:
    The manuscript states the final raw dimension (36) but does not
    enumerate the exact 36 formulas/sub-bands. This implementation is a
    transparent reproducible realization consistent with the reported
    dimension, not a claim that these were the authors' hidden exact formulas.
    """
    x = gray_224.astype(np.float32) / 255.0
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=levels)

    feats = []
    # coeffs[1:] -> [(cH3,cV3,cD3), (cH2,cV2,cD2), (cH1,cV1,cD1)]
    # Select cH and cV at each level: 6 bands * 6 stats = 36 features.
    for cH, cV, cD in coeffs[1:]:
        feats.extend(_stats(cH))
        feats.extend(_stats(cV))
    return np.asarray(feats, dtype=np.float32)
