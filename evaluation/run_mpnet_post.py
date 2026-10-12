"""Post-hoc analysis of the Kaggle run: alignment check, mpnet stress test, CI, McNemar,
temporal split with mpnet embeddings, and LLM / ensemble diagnostics from llm_cache.json."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score, confusion_matrix
sys.path.insert(0, "evaluation")
from common import load_isot, dedup, scrub
from run_eval import models

SEED = 42
R = "evaluation/kaggle_outputs"
OUT = {}


def met(y, p, s=None):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    d = dict(acc=float(accuracy_score(y, p)), prec=float(pr), rec=float(rc), f1=float(f), macro_f1=float(f1_score(y, p, average="macro")))
    if s is not None:
        d["auc"] = float(roc_auc_score(y, s))
    return d


def mcnemar(y, p1, p2):
    b = int(((p1 == y) & (p2 != y)).sum()); c = int(((p1 != y) & (p2 == y)).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return dict(b=b, c=c, chi2=float(stat), p=float(chi2.sf(stat, 1)))


def boot_ci(y, p, B=2000):
    rng = np.random.default_rng(SEED); n = len(y)
    a = [(p[i] == y[i]).mean() for i in (rng.integers(0, n, n) for _ in range(B))]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


dd = dedup(load_isot()); y = dd.label.values
E1 = np.load(f"{R}/emb_P1_raw.npy"); E2 = np.load(f"{R}/emb_P2_scrubbed.npy")
assert len(E1) == len(dd) == len(E2)
idx = np.arange(len(dd))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
rf = lambda: RandomForestClassifier(200, n_jobs=-1, random_state=SEED)

# Alignment check + scrubbed holdout (should reproduce Kaggle's rf_mpnet_holdout_P2)
m2 = rf().fit(E2[tr], y[tr]); s2 = m2.predict_proba(E2[te])[:, 1]; p2 = (s2 >= .5).astype(int)
OUT["holdout_P2_recomputed"] = met(y[te], p2, s2)
print("holdout P2 recomputed", OUT["holdout_P2_recomputed"]["acc"], flush=True)

# Stress test: raw-trained RF on raw and on scrubbed test embeddings
m1 = rf().fit(E1[tr], y[tr])
OUT["stress"] = {"raw_train_raw_test": met(y[te], m1.predict(E1[te])),
                 "raw_train_scrubbed_test": met(y[te], m1.predict(E2[te])),
                 "scrubbed_train_scrubbed_test": OUT["holdout_P2_recomputed"]}
OUT["ci_P2"] = boot_ci(y[te], p2)
print("stress", {k: v["acc"] for k, v in OUT["stress"].items()}, flush=True)

# McNemar vs TF-IDF + LR on the same scrubbed split
x = dd.raw.map(scrub).values
lr = models()["TF-IDF + LR"](); lr.fit(x[tr], y[tr]); plr = lr.predict(x[te])
OUT["mcnemar_LR_vs_RFmpnet"] = mcnemar(y[te], plr, p2)
print("mcnemar", OUT["mcnemar_LR_vs_RFmpnet"], flush=True)

# Temporal split with mpnet (scrubbed) embeddings
cut = pd.Timestamp("2017-07-01"); ok = dd.date_p.notna().values
trm = ok & (dd.date_p < cut).values; tem = ok & (dd.date_p >= cut).values
mt = rf().fit(E2[trm], y[trm]); st = mt.predict_proba(E2[tem])[:, 1]
OUT["temporal_mpnet"] = met(y[tem], (st >= .5).astype(int), st)
print("temporal", OUT["temporal_mpnet"], flush=True)

# LLM diagnostics from the cached answers (keys are row indices into dd)
cache = json.load(open(f"{R}/llm_cache.json"))
keys = np.array([int(k) for k in cache]); labs = np.array([cache[k][0] for k in cache])
fs = np.array([cache[k][1] for k in cache])
yl = y[keys]
OUT["llm_label_counts"] = {l: int((labs == l).sum()) for l in np.unique(labs)}
OUT["llm_confusion"] = confusion_matrix(yl, (fs >= .5).astype(int)).tolist()
OUT["llm_sample_fake_share"] = float(yl.mean())
pos = {j: i for i, j in enumerate(te)}
assert all(k in pos for k in keys), "LLM sample not inside held-out split"
# Ensemble vs RF-only on the same test half (same ordering as the Kaggle script)
te_s = train_test_split(te, train_size=1000, stratify=y[te], random_state=SEED)[0]
fsd = {int(k): v[1] for k, v in cache.items()}
llm_fs = np.array([fsd[int(j)] for j in te_s]); rfx = np.array([s2[pos[j]] for j in te_s]); yy = y[te_s]
h = len(yy) // 2
p_rf = (rfx[h:] >= .5).astype(int); p_en = ((0.6 * rfx[h:] + 0.4 * llm_fs[h:]) >= .5).astype(int)
OUT["ens_test_half_recomputed"] = {"rf": met(yy[h:], p_rf), "ens06": met(yy[h:], p_en)}
OUT["mcnemar_RF_vs_ens06"] = mcnemar(yy[h:], p_rf, p_en)
print("ensemble", OUT["ens_test_half_recomputed"], OUT["mcnemar_RF_vs_ens06"], flush=True)
json.dump(OUT, open("evaluation/results/results_mpnet_post.json", "w"), indent=1)
print("saved")
