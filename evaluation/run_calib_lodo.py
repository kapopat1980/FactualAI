"""Calibration / selective prediction (RF-mpnet, TF-IDF+LR) and multi-source leave-one-dataset-out."""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
sys.path.insert(0, "evaluation")
from common import load_isot, dedup, scrub, load_fnn
from run_eval import models

SEED = 42; OUT = {}
dd = dedup(load_isot()); y = dd.label.values
E2 = np.load("evaluation/kaggle_outputs/emb_P2_scrubbed.npy")
idx = np.arange(len(dd)); tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
cut = pd.Timestamp("2017-07-01"); ok = dd.date_p.notna().values
trm = ok & (dd.date_p < cut).values; tem = ok & (dd.date_p >= cut).values


def ece(y, p, bins=10):
    """Expected calibration error of the confidence in the predicted label."""
    conf = np.where(p >= .5, p, 1 - p); pred = (p >= .5).astype(int); e = 0.0
    edges = np.linspace(.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > .5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() >= 20:
            out.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return out


def coverage_curve(y, p):
    conf = np.where(p >= .5, p, 1 - p); order = np.argsort(-conf); correct = ((p >= .5).astype(int) == y)[order]
    res = []
    for c in [1.0, .9, .8, .7, .6, .5]:
        k = int(round(c * len(y))); res.append([c, float(correct[:k].mean()), float(np.sort(conf)[::-1][k - 1])])
    return res


def summarize(name, y, p):
    pred = (p >= .5).astype(int)
    OUT[name] = dict(acc=float(accuracy_score(y, pred)), f1=float(f1_score(y, pred)), auc=float(roc_auc_score(y, p)),
                     brier=float(brier_score_loss(y, p)), ece=ece(y, p), reliability=reliability(y, p), coverage=coverage_curve(y, p))
    print(name, {k: round(v, 4) for k, v in OUT[name].items() if isinstance(v, float)}, flush=True)


rf = lambda: RandomForestClassifier(200, n_jobs=-1, random_state=SEED)
# RF-mpnet, in-distribution (scrubbed holdout) and temporal
m = rf().fit(E2[tr], y[tr]); summarize("rf_mpnet_holdout", y[te], m.predict_proba(E2[te])[:, 1])
mt = rf().fit(E2[trm], y[trm]); p_t = mt.predict_proba(E2[tem])[:, 1]; summarize("rf_mpnet_temporal", y[tem], p_t)
# Isotonic recalibration fitted on training-period data only (does not see the shifted test period)
cal = CalibratedClassifierCV(rf(), method="isotonic", cv=3).fit(E2[trm], y[trm])
summarize("rf_mpnet_temporal_isotonic", y[tem], cal.predict_proba(E2[tem])[:, 1])
# TF-IDF + LR (probabilities) for comparison
x = dd.raw.map(scrub).values
lr = models()["TF-IDF + LR"](); lr.fit(x[tr], y[tr]); summarize("lr_holdout", y[te], lr.predict_proba(x[te])[:, 1])
lrt = models()["TF-IDF + LR"](); lrt.fit(x[trm], y[trm]); summarize("lr_temporal", y[tem], lrt.predict_proba(x[tem])[:, 1])

# Multi-source leave-one-dataset-out on scrubbed titles (lexical models)
src = {"ISOT": (dd.title.map(scrub).values, y)}
for p in ["politifact", "gossipcop"]:
    v = load_fnn(p); src[p] = (v.title.map(scrub).values, v.label.values)
OUT["lodo"] = {}
for target in src:
    others = [s for s in src if s != target]
    Xtr = np.concatenate([src[s][0] for s in others]); Ytr = np.concatenate([src[s][1] for s in others])
    for mname in ["TF-IDF + LR"]:
        mm = models()[mname](); mm.fit(Xtr, Ytr); pr = mm.predict(src[target][0]); sc = mm.decision_function(src[target][0])
        OUT["lodo"][f"{'+'.join(others)} -> {target}"] = dict(acc=float(accuracy_score(src[target][1], pr)), macro_f1=float(f1_score(src[target][1], pr, average="macro")),
                                                             auc=float(roc_auc_score(src[target][1], sc)))
    # single-source references for the same target
    for s in others:
        mm = models()["TF-IDF + LR"](); mm.fit(*src[s]); pr = mm.predict(src[target][0]); sc = mm.decision_function(src[target][0])
        OUT["lodo"][f"{s} -> {target}"] = dict(acc=float(accuracy_score(src[target][1], pr)), macro_f1=float(f1_score(src[target][1], pr, average="macro")),
                                               auc=float(roc_auc_score(src[target][1], sc)))
for k, v in OUT["lodo"].items():
    print(f"{k:35s} acc={v['acc']:.3f} macroF1={v['macro_f1']:.3f} auc={v['auc']:.3f}")
json.dump(OUT, open("evaluation/results/results_calib_lodo.json", "w"), indent=1)
