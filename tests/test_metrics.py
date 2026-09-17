import pandas as pd
from mchf.metrics import classification_metrics, dataset_specific_table

def test_metrics_and_table():
    y=[0,1,0,1,0,1,0,1]; s=[.1,.8,.2,.9,.15,.85,.25,.95]
    m=classification_metrics(y,s,.5); assert m['accuracy']==1
    df=pd.DataFrame({'dataset':['DDSM']*4+['INbreast']*4,'y_true':y,'score':s})
    t=dataset_specific_table(df,threshold=.5,n_boot=20)
    assert len(t)==3
