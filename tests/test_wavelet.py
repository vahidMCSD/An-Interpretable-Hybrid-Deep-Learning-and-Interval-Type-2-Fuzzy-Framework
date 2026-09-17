import numpy as np
import pytest

def test_wavelet_36_features():
    pytest.importorskip('pywt')
    pytest.importorskip('skimage')
    from mchf.features.wavelet import wavelet_raw_features
    rng=np.random.default_rng(1); img=rng.random((224,224),dtype=np.float32)
    x,stats=wavelet_raw_features(img,'db4',3)
    assert x.shape==(36,)
    for k in ['wavelet_energy','wavelet_entropy','wavelet_contrast','wavelet_homogeneity','wavelet_skewness','wavelet_kurtosis']:
        assert k in stats and np.isfinite(stats[k])
