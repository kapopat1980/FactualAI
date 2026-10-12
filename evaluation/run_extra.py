"""Shortcut, stress, temporal, cross-dataset and significance experiments."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
sys.path.insert(0, "evaluation")
from common import *
from run_eval import models, score, get_score, SEED

OUT = {}
df = load_isot(); dd = dedup(df)
dd["x0"] = dd.raw.map(basic_clean); dd["x"] = dd.raw.map(scrub); dd["t"] = dd.title.map(scrub)
y = dd.label.values

# 1. Artifact-only classifier: 10 binary markers, no content words
markers = {
    "reuters": r"\(Reuters\)", "dateline": r"^[A-Z][A-Za-z .,/'-]{1,60}\(Reuters\) -", "featured": r"Featured image",
    "wire": r"21st Century Wire", "url": r"https?://|www\.", "pictw": r"pic\.twitter\.com", "handle": r"@\w+",
    "getty": r"Getty", "credit": r"(?i)image via|photo by",
}
A = np.column_stack([dd.text.str.contains(p, regex=True).values for p in markers.values()] +
                    [dd.title.str.contains(r"\b[A-Z]{4,}\b").values]).astype(float)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
pa = cross_val_predict(LogisticRegression(max_iter=1000), A, y, cv=skf)
OUT["artifact_only"] = score(y, pa)
Ar = A[:, :1]
OUT["reuters_tag_only"] = score(y, cross_val_predict(LogisticRegression(), Ar, y, cv=skf))
print("artifact-only", OUT["artifact_only"], "reuters-only", OUT["reuters_tag_only"], flush=True)

# 2. Single held-out split (dedup) for stress test + significance
idx = np.arange(len(dd))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
OUT["split"] = dict(train=len(tr), test=len(te))
preds = {}
OUT["stress"] = {}
for n in ["TF-IDF + LR", "RF on TF-IDF-768 (repo fallback)", "RF on LSA-768"]:
    m = models()[n](); m.fit(dd.x0.values[tr], y[tr])
    a = score(y[te], m.predict(dd.x0.values[te]))
    b = score(y[te], m.predict(dd.x.values[te]))
    m2 = models()[n](); m2.fit(dd.x.values[tr], y[tr])
    p2 = m2.predict(dd.x.values[te]); preds[n] = p2
    c = score(y[te], p2, get_score(m2, dd.x.values[te]))
    OUT["stress"][n] = {"raw_train_raw_test": a, "raw_train_scrubbed_test": b, "scrubbed_train_scrubbed_test": c}
    print(n, a["acc"], b["acc"], c["acc"], flush=True)


def mcnemar(y, p1, p2):
    b = int(((p1 == y) & (p2 != y)).sum()); c = int(((p1 != y) & (p2 == y)).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return dict(b=b, c=c, chi2=stat, p=float(chi2.sf(stat, 1)))


def boot_ci(y, p, B=2000):
    rng = np.random.default_rng(SEED); n = len(y); acc = []
    for _ in range(B):
        i = rng.integers(0, n, n); acc.append((p[i] == y[i]).mean())
    return [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


yt = y[te]
OUT["sig"] = {
    "LR_vs_RF-LSA": mcnemar(yt, preds["TF-IDF + LR"], preds["RF on LSA-768"]),
    "LR_vs_RF-TFIDF768": mcnemar(yt, preds["TF-IDF + LR"], preds["RF on TF-IDF-768 (repo fallback)"]),
    "ci": {n: boot_ci(yt, p) for n, p in preds.items()},
}
print(OUT["sig"], flush=True)

# 3. Temporal split on scrubbed data
cut = pd.Timestamp("2017-07-01")
ok = dd.date_p.notna()
trm = ok & (dd.date_p < cut); tem = ok & (dd.date_p >= cut)
OUT["temporal_counts"] = dict(train_fake=int(dd.label[trm].sum()), train_real=int((1 - dd.label[trm]).sum()),
                              test_fake=int(dd.label[tem].sum()), test_real=int((1 - dd.label[tem]).sum()))
OUT["temporal"] = {}
for n in ["TF-IDF + LR", "RF on LSA-768"]:
    m = models()[n](); m.fit(dd.x[trm].values, y[trm])
    OUT["temporal"][n] = score(y[tem], m.predict(dd.x[tem].values), get_score(m, dd.x[tem].values))
print("temporal", OUT["temporal_counts"], OUT["temporal"], flush=True)

# 4. Cross-dataset: ISOT titles -> FakeNewsNet titles
OUT["cross"] = {}
fnn = {p: load_fnn(p) for p in ["politifact", "gossipcop"]}
OUT["fnn_counts"] = {p: dict(fake=int(v.label.sum()), real=int((1 - v.label).sum())) for p, v in fnn.items()}
for n in ["TF-IDF + LR", "RF on LSA-768"]:
    m = models()[n](); m.fit(dd.t.values, y)
    for p, v in fnn.items():
        xt = v.title.map(scrub).values
        OUT["cross"][f"{n} -> {p}"] = score(v.label.values, m.predict(xt), get_score(m, xt))
    # in-domain reference: 5-fold CV inside PolitiFact
pf = fnn["politifact"]; xp = pf.title.map(scrub).values; yp = pf.label.values
pp = cross_val_predict(models()["TF-IDF + LR"](), xp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
OUT["cross"]["in-domain PolitiFact 5-fold (TF-IDF + LR)"] = score(yp, pp)
print("cross", OUT["cross"], flush=True)

# 5. Top LR features under raw vs scrubbed (interpretability of shortcut)
for tag, col in [("raw", "x0"), ("scrubbed", "x")]:
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    X = v.fit_transform(dd[col].values); lr = LogisticRegression(C=4, max_iter=2000).fit(X, y)
    f = np.array(v.get_feature_names_out()); o = np.argsort(lr.coef_[0])
    OUT[f"top_{tag}"] = dict(fake=f[o[-15:]][::-1].tolist(), real=f[o[:15]].tolist())
print(OUT["top_raw"], OUT["top_scrubbed"])
json.dump(OUT, open("evaluation/results/results_extra.json", "w"), indent=1, default=float)
