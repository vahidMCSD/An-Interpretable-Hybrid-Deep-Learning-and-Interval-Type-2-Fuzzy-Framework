import numpy as np
from mchf.fuzzy.type2 import karnik_mendel_type_reduce

def test_km_interval_order():
    y=np.array([0.1,0.4,0.6,0.9]);lo=np.array([.2,.4,.3,.1]);hi=np.array([.5,.7,.6,.4]);l,r=karnik_mendel_type_reduce(y,lo,hi);assert 0<=l<=r<=1

def test_km_collapses_when_weights_crisp():
    y=np.array([0.1,0.9]);w=np.array([.25,.75]);l,r=karnik_mendel_type_reduce(y,w,w);expected=float(np.dot(y,w)/w.sum());assert abs(l-expected)<1e-8 and abs(r-expected)<1e-8
