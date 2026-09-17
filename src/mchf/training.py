from __future__ import annotations
import numpy as np,torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from mchf.features.wavelet import WaveletPCA
from mchf.fuzzy.type2 import RuleFeatureNormalizer,Type2Config
from mchf.hybrid import FeatureLevelHybrid
from mchf.art.fuzzy_art import FuzzyART,FuzzyARTConfig
from mchf.metrics import youden_threshold

def rule_rows(arr,keys): return [{str(k):float(v) for k,v in zip(keys,row)} for row in arr]
def norm_rules(n,arr,keys,device):
    return {str(k):n.transform_value(str(k),torch.tensor(arr[:,j],dtype=torch.float32,device=device)) for j,k in enumerate(keys)}
def T(x,d): return torch.tensor(x,dtype=torch.float32,device=d)

def train_feature_model(data,tr,va,te,device='cpu',seed=12345,pca_variance=.95,lr=1e-4,batch_size=16,weight_decay=1e-4,epochs=50,patience=5,dropout=.1,fuzzy_uncertainty=.2,membership_shift=0,rho=.85,beta=.5,art_alpha=1e-3,type_reduction='km'):
    device=torch.device(device);y=data['label'].astype(int);keys=[str(x) for x in data['wave_rule_keys']];wpca=WaveletPCA(variance=pca_variance);Wtr=wpca.fit_transform(data['wave_raw'][tr]);Wva=wpca.transform(data['wave_raw'][va]);Wte=wpca.transform(data['wave_raw'][te]);Etr,Eva,Ete=data['eff'][tr],data['eff'][va],data['eff'][te];Vtr,Vva,Vte=data['vit'][tr],data['vit'][va],data['vit'][te];scaler=MinMaxScaler().fit(np.c_[Wtr,Etr,Vtr]);Ftr=scaler.transform(np.c_[Wtr,Etr,Vtr]).astype('float32');Fva=scaler.transform(np.c_[Wva,Eva,Vva]).astype('float32');Fte=scaler.transform(np.c_[Wte,Ete,Vte]).astype('float32');rn=RuleFeatureNormalizer().fit(rule_rows(data['wave_rule'][tr],keys));cfg=Type2Config(sigma_uncertainty_fraction=fuzzy_uncertainty,membership_shift_fraction=membership_shift,type_reduction=type_reduction);m=FeatureLevelHybrid(Wtr.shape[1],hidden_dim=256,heads=4,dropout=dropout,fuzzy_cfg=cfg).to(device);opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=weight_decay);pos=max((y[tr]==1).sum(),1);neg=max((y[tr]==0).sum(),1);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,dtype=torch.float32,device=device))
    blocks=[]
    for idx,W,E,V,F in [(tr,Wtr,Etr,Vtr,Ftr),(va,Wva,Eva,Vva,Fva),(te,Wte,Ete,Vte,Fte)]: blocks.append(dict(w=T(W,device),e=T(E,device),v=T(V,device),f=T(F,device),wr=norm_rules(rn,data['wave_rule'][idx],keys,device)))
    best=None;best_auc=-1;wait=0;rng=np.random.default_rng(seed)
    for ep in range(epochs):
        m.train();order=rng.permutation(len(tr))
        for s in range(0,len(tr),batch_size):
            bi=order[s:s+batch_size];out=m(blocks[0]['w'][bi],blocks[0]['e'][bi],blocks[0]['v'][bi],{k:v[bi] for k,v in blocks[0]['wr'].items()},fusion_scaled=blocks[0]['f'][bi]);loss=lossfn(out['logits'],T(y[tr][bi],device));opt.zero_grad();loss.backward();opt.step()
        m.eval()
        with torch.no_grad():ov=m(blocks[1]['w'],blocks[1]['e'],blocks[1]['v'],blocks[1]['wr'],fusion_scaled=blocks[1]['f']);pv=ov['prob'].cpu().numpy()
        auc=roc_auc_score(y[va],pv) if len(np.unique(y[va]))>1 else .5
        if auc>best_auc+1e-5:best_auc=auc;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};wait=0
        else:
            wait+=1
            if wait>=patience:break
    if best:m.load_state_dict(best)
    m.eval()
    with torch.no_grad():otr=m(blocks[0]['w'],blocks[0]['e'],blocks[0]['v'],blocks[0]['wr'],fusion_scaled=blocks[0]['f']);ova=m(blocks[1]['w'],blocks[1]['e'],blocks[1]['v'],blocks[1]['wr'],fusion_scaled=blocks[1]['f']);ote=m(blocks[2]['w'],blocks[2]['e'],blocks[2]['v'],blocks[2]['wr'],fusion_scaled=blocks[2]['f'])
    th=youden_threshold(y[va],ova['prob'].cpu().numpy());As=MinMaxScaler().fit(np.c_[otr['fused'].cpu().numpy(),otr['fuzzy_score'].cpu().numpy()]);Atr=np.clip(As.transform(np.c_[otr['fused'].cpu().numpy(),otr['fuzzy_score'].cpu().numpy()]),0,1);Ate=np.clip(As.transform(np.c_[ote['fused'].cpu().numpy(),ote['fuzzy_score'].cpu().numpy()]),0,1);art=FuzzyART(FuzzyARTConfig(vigilance=rho,beta=beta,alpha=art_alpha)).fit(Atr,y[tr]);return {'model':m,'outputs':ote,'prob':ote['prob'].cpu().numpy(),'threshold':th,'art_pred':art.predict(Ate),'art_score':art.decision_score(Ate),'n_art_categories':len(art.weights),'pca_dim':Wtr.shape[1],'val_auc':best_auc,'y_test':y[te], 'wavelet_pca':wpca, 'fusion_scaler':scaler, 'rule_normalizer':rn, 'art':art, 'art_scaler':As}
