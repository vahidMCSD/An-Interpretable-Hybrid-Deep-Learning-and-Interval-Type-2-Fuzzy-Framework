from mchf.analysis import mcnemar_exact

def test_mcnemar():
    y=[0,0,1,1];a=[0,0,1,0];b=[0,1,1,1];r=mcnemar_exact(y,a,b);assert 0<=r['pvalue']<=1
