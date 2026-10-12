import json, sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
sys.path.insert(0,"evaluation")
from common import load_isot, dedup, scrub
from run_eval import models
SEED=42
dd=dedup(load_isot()); y=dd.label.values; E2=np.load("evaluation/kaggle_outputs/emb_P2_scrubbed.npy")
idx=np.arange(len(dd)); tr,te=train_test_split(idx,test_size=.2,stratify=y,random_state=SEED)
cut=pd.Timestamp("2017-07-01"); ok=dd.date_p.notna().values; trm=ok&(dd.date_p<cut).values; tem=ok&(dd.date_p>=cut).values
def ece_pos(y,p,b=10):
    e=0.0
    for lo,hi in zip(np.linspace(0,1,b+1)[:-1],np.linspace(0,1,b+1)[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): e+=m.mean()*abs(p[m].mean()-y[m].mean())
    return float(e)
rf=lambda: RandomForestClassifier(200,n_jobs=-1,random_state=SEED)
x=dd.raw.map(scrub).values
res={}
res["rf_mpnet_holdout"]=ece_pos(y[te],rf().fit(E2[tr],y[tr]).predict_proba(E2[te])[:,1])
res["rf_mpnet_temporal"]=ece_pos(y[tem],rf().fit(E2[trm],y[trm]).predict_proba(E2[tem])[:,1])
res["rf_mpnet_temporal_isotonic"]=ece_pos(y[tem],CalibratedClassifierCV(rf(),method="isotonic",cv=3).fit(E2[trm],y[trm]).predict_proba(E2[tem])[:,1])
l=models()["TF-IDF + LR"](); l.fit(x[tr],y[tr]); res["lr_holdout"]=ece_pos(y[te],l.predict_proba(x[te])[:,1])
l=models()["TF-IDF + LR"](); l.fit(x[trm],y[trm]); res["lr_temporal"]=ece_pos(y[tem],l.predict_proba(x[tem])[:,1])
d=json.load(open("evaluation/results/results_calib_lodo.json"))
for k,v in res.items(): d[k]["ece_pos"]=v; print(k,round(v,4))
json.dump(d,open("evaluation/results/results_calib_lodo.json","w"),indent=1)
