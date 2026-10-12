"""Leakage-aware re-evaluation of the FactualAI text classifier on ISOT.

Protocols
  P0 Original : title+text, basic cleaning, random stratified 5-fold CV, no de-duplication
  P1 Dedup    : as P0 after removing exact duplicates
  P2 Scrubbed : P1 + removal of publisher/format artifacts (Reuters dateline/tag, URLs,
                handles, image credits, source brand names, casing)
  P3 Temporal : P2 data, train < 2017-07-01, test >= 2017-07-01
  P4 Cross-dataset : train on ISOT titles (scrubbed), test on FakeNewsNet PolitiFact / GossipCop titles
Also: artifact-only classifier, and raw-trained models tested on scrubbed text.
"""
import json, sys, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
sys.path.insert(0, "evaluation")
from common import *

SEED = 42
OUT = {}


def models():
    return {
        "TF-IDF + LR": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True), LogisticRegression(C=4, max_iter=2000)),
        "TF-IDF + LinearSVM": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True), LinearSVC(C=0.5)),
        "TF-IDF + CNB": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000), ComplementNB()),
        "RF on TF-IDF-768 (repo fallback)": lambda: make_pipeline(TfidfVectorizer(max_features=768), RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)),
        "RF on LSA-768": lambda: make_pipeline(TfidfVectorizer(min_df=2, max_features=100000, sublinear_tf=True), TruncatedSVD(768, random_state=SEED), RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)),
    }


def score(y, pred, s=None):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    d = dict(acc=accuracy_score(y, pred), prec=p, rec=r, f1=f, macro_f1=f1_score(y, pred, average="macro"))
    if s is not None:
        d["auc"] = roc_auc_score(y, s)
    return d


def get_score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def cv(X, y, names, k=5):
    skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
    res = {}
    for n in names:
        fold = []
        t0 = time.time()
        for tr, te in skf.split(X, y):
            m = models()[n](); m.fit(X[tr], y[tr])
            pr = m.predict(X[te]); fold.append(score(y[te], pr, get_score(m, X[te])))
        res[n] = {kk: (float(np.mean([f[kk] for f in fold])), float(np.std([f[kk] for f in fold]))) for kk in fold[0]}
        print(f"  {n:34s} acc={res[n]['acc'][0]:.4f}±{res[n]['acc'][1]:.4f} f1={res[n]['f1'][0]:.4f} auc={res[n]['auc'][0]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    df = load_isot()
    df0 = df.copy(); df0["x"] = df0.raw.map(basic_clean)
    dd = dedup(df); dd["x0"] = dd.raw.map(basic_clean); dd["x"] = dd.raw.map(scrub)
    dd["t"] = dd.title.map(scrub)
    OUT["n"] = dict(original=len(df0), dedup=len(dd), dedup_fake=int(dd.label.sum()), dedup_real=int((1 - dd.label).sum()))
    print(OUT["n"])
    names = list(models())
    y0 = df0.label.values; y = dd.label.values

    print("P0 original"); OUT["P0"] = cv(df0.x.values, y0, names)
    print("P1 dedup"); OUT["P1"] = cv(dd.x0.values, y, names)
    print("P2 scrubbed"); OUT["P2"] = cv(dd.x.values, y, names)
    print("P2-title scrubbed"); OUT["P2_title"] = cv(dd.t.values, y, ["TF-IDF + LR", "TF-IDF + LinearSVM", "RF on LSA-768"])
    json.dump(OUT, open("evaluation/results/results_cv.json", "w"), indent=1)


if __name__ == "__main__":
    main()
