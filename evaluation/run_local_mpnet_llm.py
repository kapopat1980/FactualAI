#!/usr/bin/env python3
"""
FactualAI - local experiments that need models not available in the cloud workspace.
Fills the highlighted [RUN] cells in Tables 4 and 5 of the revised manuscript.

What it does (all on the de-duplicated ISOT corpus, same split/seed as the manuscript):
  A. RF on all-mpnet-base-v2 embeddings, Protocol P1 (raw) and P2 (artifact-scrubbed),
     5-fold stratified CV, plus the same single 80/20 held-out split.
  B. LLaMA 3.2 1B (Ollama) zero-shot on a stratified sample of the held-out split
     (default 1,000 articles; use --llm-n 0 for the full split).
     Reports accuracy/P/R/F1, rate of unparseable or "Unknown" outputs, median latency.
  C. Ensemble RF(mpnet) + LLM on the same sample: weights w in {0,0.1,...,1.0},
     selecting w on a validation slice and reporting test results for the chosen w
     and for the original fixed 0.6/0.4.
  D. Translation round-trip robustness (optional, --translate): 200 scrubbed test
     articles translated EN->{hi,mr,gu,ta,bn}->EN with deep-translator, then classified.

Setup (one time):
    pip install pandas scikit-learn sentence-transformers requests scipy deep-translator
    ollama pull llama3.2:1b   &&   ollama serve        (in another terminal)
Put Fake.csv and True.csv in ./data and copy common.py next to this script.

Run:
    python run_local_mpnet_llm.py --data data            # A + B + C   (~1-3 h on CPU)
    python run_local_mpnet_llm.py --data data --translate   # also D
Send results_local.json back; every number in the paper's [RUN] cells comes from it.
"""
import argparse, json, re, time
import numpy as np, pandas as pd, requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from common import load_isot, dedup, basic_clean, scrub

SEED = 42
PROMPT = """You are a fact-checking assistant. Read the following news article excerpt and decide whether it is more likely to be real or fake news.

Respond with ONLY a JSON object of the form:
{{"label": "Fake" | "Real", "confidence": <float between 0.5 and 1.0>, "justification": "<one or two sentence plain-language explanation>"}}

Article excerpt:
\"\"\"{article}\"\"\"
"""


def met(y, p, s=None):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    d = dict(acc=float(accuracy_score(y, p)), prec=float(pr), rec=float(rc), f1=float(f))
    if s is not None:
        d["auc"] = float(roc_auc_score(y, s))
    return d


def llm(text, url, model):
    t0 = time.time()
    try:
        r = requests.post(url, json={"model": model, "prompt": PROMPT.format(article=text[:1500]),
                                     "stream": False, "format": "json", "options": {"temperature": 0, "seed": SEED}}, timeout=60)
        o = json.loads(r.json()["response"])
        lab = str(o.get("label", "Unknown")).strip().capitalize()
        conf = float(o.get("confidence", 0.5))
        conf = min(max(conf, 0.5), 1.0)
    except Exception:
        lab, conf = "Unparsed", 0.5
    fs = conf if lab == "Fake" else (1 - conf if lab == "Real" else 0.5)
    return lab, fs, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--llm-n", type=int, default=1000)
    ap.add_argument("--ollama", default="http://localhost:11434/api/generate")
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2,000 articles, 40 LLM calls, writes results_quicktest.json")
    a = ap.parse_args()
    # Pre-flight: make sure Ollama is running and the model is pulled
    base = a.ollama.split("/api/")[0]
    try:
        tags = [m["name"] for m in requests.get(base + "/api/tags", timeout=10).json().get("models", [])]
    except Exception:
        raise SystemExit("ERROR: cannot reach Ollama at " + base + ". Start the Ollama app (or run 'ollama serve') and try again.")
    if not any(t.startswith(a.model) for t in tags):
        raise SystemExit(f"ERROR: model '{a.model}' not found in Ollama. Run:  ollama pull {a.model}")
    print("Ollama OK, model found:", a.model, flush=True)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-mpnet-base-v2")
    R = {}
    dd = dedup(load_isot(a.data))
    if a.quick:
        dd = dd.sample(2000, random_state=SEED).reset_index(drop=True); a.llm_n = 40
        print("QUICK TEST MODE - numbers are NOT for the paper", flush=True)
    y = dd.label.values
    raw_text = dd.raw.values  # LLM sees raw text, as in the deployed system
    E = {}
    for tag, fn in [("P1_raw", basic_clean), ("P2_scrubbed", scrub)]:
        cache = f"emb_{tag}{'_quick' if a.quick else ''}.npy"
        try:
            E[tag] = np.load(cache)
            assert E[tag].shape[0] == len(dd)
            print("loaded saved embeddings", cache, flush=True)
        except Exception:
            print("encoding", tag, f"({len(dd):,} articles)", flush=True)
            E[tag] = enc.encode(dd.raw.map(fn).tolist(), batch_size=64, show_progress_bar=True)
            np.save(cache, E[tag])
        folds = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(E[tag], y):
            rf = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(E[tag][tr], y[tr])
            folds.append(met(y[te], rf.predict(E[tag][te]), rf.predict_proba(E[tag][te])[:, 1]))
        R[f"rf_mpnet_cv_{tag}"] = {k: [float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds]))] for k in folds[0]}
        print(tag, R[f"rf_mpnet_cv_{tag}"], flush=True)

    idx = np.arange(len(dd))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    rf = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(E["P2_scrubbed"][tr], y[tr])
    rf_s = rf.predict_proba(E["P2_scrubbed"][te])[:, 1]
    R["rf_mpnet_holdout_P2"] = met(y[te], (rf_s >= .5).astype(int), rf_s)

    # LLM on a stratified sample of the held-out split (scrubbed text so the LLM cannot read "(Reuters)")
    te_s = te if a.llm_n == 0 else train_test_split(te, train_size=a.llm_n, stratify=y[te], random_state=SEED)[0]
    pos = {j: i for i, j in enumerate(te)}
    labs, llm_fs, lat = [], [], []
    lcache = "llm_cache_quick.json" if a.quick else "llm_cache.json"
    try:
        done = json.load(open(lcache))
    except Exception:
        done = {}
    for k, j in enumerate(te_s):
        key = str(int(j))
        if key in done:
            l, s, t = done[key]
        else:
            txt = re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", raw_text[j])
            l, s, t = llm(txt, a.ollama, a.model)
            done[key] = [l, s, t]
            if k % 10 == 0:
                json.dump(done, open(lcache, "w"))
        labs.append(l); llm_fs.append(s); lat.append(t)
        if k % 50 == 0:
            print(f"LLM {k}/{len(te_s)}", flush=True)
    json.dump(done, open(lcache, "w"))
    llm_fs = np.array(llm_fs); yy = y[te_s]; rfx = np.array([rf_s[pos[j]] for j in te_s])
    R["llm"] = met(yy, (llm_fs >= .5).astype(int), llm_fs)
    R["llm"].update(n=int(len(te_s)), unknown_rate=float(np.mean([l not in ("Fake", "Real") for l in labs])),
                    median_latency_s=float(np.median(lat)), p95_latency_s=float(np.percentile(lat, 95)))
    R["rf_on_llm_sample"] = met(yy, (rfx >= .5).astype(int), rfx)
    # Ensemble: choose w on first half, report on second half
    h = len(yy) // 2; grid = np.round(np.arange(0, 1.01, .1), 1)
    val = {float(w): float(accuracy_score(yy[:h], ((w * rfx[:h] + (1 - w) * llm_fs[:h]) >= .5))) for w in grid}
    wb = max(val, key=val.get)
    R["ensemble_weight_val_acc"] = val; R["ensemble_selected_w_rf"] = wb
    for name, w in [("ensemble_selected", wb), ("ensemble_fixed_0.6", 0.6)]:
        s = w * rfx[h:] + (1 - w) * llm_fs[h:]
        R[name] = met(yy[h:], (s >= .5).astype(int), s)
    R["rf_only_test_half"] = met(yy[h:], (rfx[h:] >= .5).astype(int), rfx[h:])
    R["llm_only_test_half"] = met(yy[h:], (llm_fs[h:] >= .5).astype(int), llm_fs[h:])
    print(json.dumps({k: R[k] for k in R if "ensemble" in k or k == "llm"}, indent=1))

    if a.translate:
        from deep_translator import GoogleTranslator
        sub = train_test_split(te, train_size=20 if a.quick else 200, stratify=y[te], random_state=SEED)[0]
        base = [scrub(dd.raw.values[j])[:1500] for j in sub]
        R["translation"] = {"en": met(y[sub], rf.predict(enc.encode(base)))}
        for lg in ["hi", "mr", "gu", "ta", "bn"]:
            back = []
            for t in base:
                try:
                    f = GoogleTranslator(source="en", target=lg).translate(t[:1400])
                    back.append(scrub(GoogleTranslator(source=lg, target="en").translate(f)))
                except Exception:
                    back.append(t)
            R["translation"][lg] = met(y[sub], rf.predict(enc.encode(back)))
            print(lg, R["translation"][lg], flush=True)
    out = "results_quicktest.json" if a.quick else "results_local.json"
    json.dump(R, open(out, "w"), indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()
