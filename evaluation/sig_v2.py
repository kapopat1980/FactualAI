import json, sys, numpy as np
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
sys.path.insert(0,"evaluation")
from common import load_isot, dedup
SEED=42
dd=dedup(load_isot()); y=dd.label.values; E2=np.load("evaluation/kaggle_outputs/emb_P2_scrubbed.npy")
idx=np.arange(len(dd)); tr,te=train_test_split(idx,test_size=.2,stratify=y,random_state=SEED)
s2=RandomForestClassifier(200,n_jobs=-1,random_state=SEED).fit(E2[tr],y[tr]).predict_proba(E2[te])[:,1]
pos={j:i for i,j in enumerate(te)}
te_s=train_test_split(te,train_size=1000,stratify=y[te],random_state=SEED)[0]; yy=y[te_s]; h=500
rfx=np.array([s2[pos[j]] for j in te_s])
def mc(y,a,b):
    b_=int(((a==y)&(b!=y)).sum()); c_=int(((a!=y)&(b==y)).sum()); st=(abs(b_-c_)-1)**2/(b_+c_) if b_+c_ else 0
    return dict(b=b_,c=c_,chi2=float(st),p=float(chi2.sf(st,1)))
def boot_diff(y,a,b,B=2000):
    rng=np.random.default_rng(SEED); n=len(y); d=[]
    for _ in range(B):
        i=rng.integers(0,n,n); d.append((b[i]==y[i]).mean()-(a[i]==y[i]).mean())
    return [float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
OUT={}
for name,f,w in [("1b","evaluation/kaggle_outputs/llm_cache_llama3.2_1b.json",None),("3b","evaluation/kaggle_outputs/llm_cache_llama3.2_3b.json",0.9),("8b","evaluation/kaggle_outputs/llm_cache_llama3.1_8b.json",0.8)]:
    c=json.load(open(f)); fs=np.array([c[str(int(j))][1] for j in te_s])
    prf=(rfx[h:]>=.5).astype(int)
    for lab,ww in [("w0.6",0.6)]+([("wsel",w)] if w else []):
        pe=((ww*rfx[h:]+(1-ww)*fs[h:])>=.5).astype(int)
        OUT[f"{name}_{lab}"]=dict(rf_acc=float((prf==yy[h:]).mean()),ens_acc=float((pe==yy[h:]).mean()),mcnemar=mc(yy[h:],prf,pe),diff_ci=boot_diff(yy[h:],prf,pe))
        print(name,lab,OUT[f"{name}_{lab}"])
json.dump(OUT,open("evaluation/results/results_sig_v2.json","w"),indent=1)
