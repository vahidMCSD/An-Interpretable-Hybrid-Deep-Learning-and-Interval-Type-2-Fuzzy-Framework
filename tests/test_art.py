import numpy as np
from mchf.art.fuzzy_art import FuzzyART, FuzzyARTConfig

def test_art_fit_predict():
    rng=np.random.default_rng(2); X=rng.random((40,10)); y=(X[:,0]>.5).astype(int)
    m=FuzzyART(FuzzyARTConfig(vigilance=.7,beta=.5)).fit(X,y)
    p=m.predict(X); s=m.decision_score(X)
    assert p.shape==(40,); assert s.shape==(40,); assert len(m.weights)>0
