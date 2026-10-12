#!/usr/bin/env python3
"""FactualAI evaluation, part 2 (Kaggle GPU).

S1  Production RF (all-mpnet-base-v2) on titles: P2-titles 5-fold CV, cross-dataset P4
    (ISOT -> PolitiFact / GossipCop) and leave-one-dataset-out with multi-source training.
S2  Larger local LLMs (default llama3.2:3b, llama3.1:8b) on the same 1,000-article sample as part 1,
    with validation-selected ensemble weight.
S3  Multilingual robustness with an open MT model (NLLB-200 distilled 600M):
    (a) round trip EN -> {hi, mr, gu, ta, bn} -> EN, then the English production classifier;
    (b) no translation: RF on a multilingual encoder (paraphrase-multilingual-mpnet-base-v2) trained on
        English, tested directly on the native-script text.
Results are written to results_v2.json after every section, so a failure in one section keeps the others.
"""
import argparse, json, os, re, time, traceback
import numpy as np, pandas as pd, requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
from common import load_isot, dedup, scrub, basic_clean

SEED = 42
OUTF = "results_v2.json"
R = json.load(open(OUTF)) if os.path.exists(OUTF) else {}
PROMPT = """You are a fact-checking assistant. Read the following news article excerpt and decide whether it is more likely to be real or fake news.

Respond with ONLY a JSON object of the form:
{{"label": "Fake" | "Real", "confidence": <float between 0.5 and 1.0>, "justification": "<one or two sentence plain-language explanation>"}}

Article excerpt:
\"\"\"{article}\"\"\"
"""
LANGS = {"hi": "hin_Deva", "mr": "mar_Deva", "gu": "guj_Gujr", "ta": "tam_Taml", "bn": "ben_Beng"}


def save():
    json.dump(R, open(OUTF, "w"), indent=1)


def met(y, p, s=None):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    d = dict(acc=float(accuracy_score(y, p)), prec=float(pr), rec=float(rc), f1=float(f), macro_f1=float(f1_score(y, p, average="macro")), n=int(len(y)))
    if s is not None and len(set(y)) > 1:
        d["auc"] = float(roc_auc_score(y, s))
    return d


def rf():
    return RandomForestClassifier(200, n_jobs=-1, random_state=SEED)


def load_fnn(prefix, d):
    f = pd.read_csv(f"{d}/{prefix}_fake.csv"); f["label"] = 1
    r = pd.read_csv(f"{d}/{prefix}_real.csv"); r["label"] = 0
    x = pd.concat([f, r], ignore_index=True).dropna(subset=["title"])
    return x.loc[~x.title.str.strip().str.lower().duplicated()].reset_index(drop=True)


def section(name):
    def deco(fn):
        def run(*a, **k):
            if R.get(name, {}).get("_done"):
                print(f"[{name}] already done, skipping", flush=True); return
            t0 = time.time(); print(f"\n===== {name} =====", flush=True)
            try:
                fn(*a, **k); R.setdefault(name, {})["_done"] = True
            except Exception as e:
                R.setdefault(name, {})["_error"] = repr(e); traceback.print_exc()
            R[name]["_minutes"] = round((time.time() - t0) / 60, 1); save()
        return run
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--fnn", required=True)
    ap.add_argument("--llms", default="llama3.2:3b,llama3.1:8b")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--mt-n", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-mpnet-base-v2")
    dd = dedup(load_isot(a.data)); y = dd.label.values
    if a.quick:
        dd = dd.sample(3000, random_state=SEED).reset_index(drop=True); y = dd.label.values; a.mt_n = 20
    idx = np.arange(len(dd)); tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    te_s = train_test_split(te, train_size=min(1000, len(te) // 2), stratify=y[te], random_state=SEED)[0]
    if a.quick:
        te_s = te_s[:30]; a.llms = a.llms.split(",")[0]
    fnn = {p: load_fnn(p, a.fnn) for p in ["politifact", "gossipcop"]}
    if a.quick:
        fnn = {p: v.sample(min(600, len(v)), random_state=SEED).reset_index(drop=True) for p, v in fnn.items()}

    def embed(texts, cache):
        if os.path.exists(cache):
            e = np.load(cache)
            if len(e) == len(texts):
                return e
        e = np.asarray(enc.encode(list(texts), batch_size=128, show_progress_bar=True)); np.save(cache, e); return e

    q = "_quick" if a.quick else ""

    @section("S1_titles")
    def s1():
        S = R.setdefault("S1_titles", {})
        T = {"ISOT": (embed(dd.title.map(scrub), f"embT_isot{q}.npy"), y)}
        for p, v in fnn.items():
            T[p] = (embed(v.title.map(scrub), f"embT_{p}{q}.npy"), v.label.values)
        X, Y = T["ISOT"]; folds = []
        for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, Y):
            m = rf().fit(X[a_], Y[a_]); s = m.predict_proba(X[b_])[:, 1]; folds.append(met(Y[b_], (s >= .5).astype(int), s))
        S["isot_titles_cv"] = {k: [float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds]))] for k in folds[0]}
        print("P2 titles CV", S["isot_titles_cv"]["acc"], flush=True)
        for target in T:
            others = [s for s in T if s != target]
            for combo in [[o] for o in others] + [others]:
                Xtr = np.concatenate([T[c][0] for c in combo]); Ytr = np.concatenate([T[c][1] for c in combo])
                m = rf().fit(Xtr, Ytr); s = m.predict_proba(T[target][0])[:, 1]
                S[f"{'+'.join(combo)} -> {target}"] = met(T[target][1], (s >= .5).astype(int), s)
                print("+".join(combo), "->", target, round(S[f"{'+'.join(combo)} -> {target}"]["acc"], 4), flush=True)
            save()
    s1()

    # Full-text production classifier on scrubbed text (needed by S2 and S3)
    E2 = embed(dd.raw.map(scrub), f"embF_P2{q}.npy")
    clf = rf().fit(E2[tr], y[tr]); rf_s = clf.predict_proba(E2[te])[:, 1]; pos = {j: i for i, j in enumerate(te)}

    @section("S2_llms")
    def s2():
        S = R.setdefault("S2_llms", {})
        base = a.ollama.rstrip("/")
        for model in [m for m in a.llms.split(",") if m]:
            if S.get(model, {}).get("acc") is not None:
                continue
            print("pulling", model, flush=True); os.system(f"ollama pull {model}")
            cache_f = f"llm_cache_{model.replace(':', '_')}{q}.json"
            done = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
            labs, fs, lat = [], [], []
            for k, j in enumerate(te_s):
                key = str(int(j))
                if key not in done:
                    txt = re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", dd.raw.values[j]); t0 = time.time()
                    try:
                        r = requests.post(base + "/api/generate", json={"model": model, "prompt": PROMPT.format(article=txt[:1500]), "stream": False,
                                                                       "format": "json", "options": {"temperature": 0, "seed": SEED}}, timeout=120)
                        o = json.loads(r.json()["response"]); lab = str(o.get("label", "Unknown")).strip().capitalize()
                        conf = min(max(float(o.get("confidence", 0.5)), 0.5), 1.0)
                    except Exception:
                        lab, conf = "Unparsed", 0.5
                    done[key] = [lab, conf if lab == "Fake" else (1 - conf if lab == "Real" else 0.5), time.time() - t0]
                    if k % 25 == 0:
                        json.dump(done, open(cache_f, "w")); print(f"{model} {k}/{len(te_s)}", flush=True)
                l, s, t = done[key]; labs.append(l); fs.append(s); lat.append(t)
            json.dump(done, open(cache_f, "w"))
            fs = np.array(fs); yy = y[te_s]; rfx = np.array([rf_s[pos[j]] for j in te_s]); h = len(yy) // 2
            d = met(yy, (fs >= .5).astype(int), fs)
            d.update(label_counts={l: labs.count(l) for l in set(labs)}, median_latency_s=float(np.median(lat)), p95_latency_s=float(np.percentile(lat, 95)))
            grid = np.round(np.arange(0, 1.01, .1), 1)
            val = {float(w): float(accuracy_score(yy[:h], (w * rfx[:h] + (1 - w) * fs[:h]) >= .5)) for w in grid}
            wb = max(val, key=lambda w: (val[w], -w))  # ties -> smaller RF weight, i.e. favour using the LLM
            d["val_acc_by_w"] = val; d["selected_w_rf"] = wb
            for name, w in [("ens_selected", wb), ("ens_fixed_0.6", 0.6), ("rf_only", 1.0), ("llm_only", 0.0)]:
                s = w * rfx[h:] + (1 - w) * fs[h:]; d[name + "_test_half"] = met(yy[h:], (s >= .5).astype(int), s)
            S[model] = d; save()
            print(model, "acc", round(d["acc"], 4), "auc", round(d.get("auc", float("nan")), 4), "w*", wb, d["label_counts"], flush=True)
    s2()

    @section("S3_multilingual")
    def s3():
        S = R.setdefault("S3_multilingual", {})
        os.system("pkill -x ollama >/dev/null 2>&1")  # free GPU memory for NLLB
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        name = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name); mt = AutoModelForSeq2SeqLM.from_pretrained(name).to(dev).eval()
        if dev == "cuda":
            mt = mt.half()

        def translate(sents, src, tgt, bs=24):
            out = []
            tok.src_lang = src
            for i in range(0, len(sents), bs):
                b = tok(sents[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=200).to(dev)
                with torch.no_grad():
                    g = mt.generate(**b, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), max_new_tokens=256, num_beams=2)
                out += tok.batch_decode(g, skip_special_tokens=True)
            return out

        sub = train_test_split(te, train_size=a.mt_n, stratify=y[te], random_state=SEED)[0]
        # English excerpt: first sentences up to ~700 characters, publisher dateline removed
        def excerpt(t):
            t = re.sub(r"^.{0,120}?\(Reuters\)\s*-\s*", "", t)
            ss = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
            o = []
            for s in ss:
                if sum(len(x) for x in o) + len(s) > 700 and o:
                    break
                o.append(s[:400])
            return o or [t[:400]]
        ex = [excerpt(dd.raw.values[j]) for j in sub]; ys = y[sub]
        flat = [s for e in ex for s in e]; lens = [len(e) for e in ex]
        def regroup(fl):
            out, i = [], 0
            for n in lens:
                out.append(" ".join(fl[i:i + n])); i += n
            return out
        en_text = regroup(flat)
        p_en = clf.predict_proba(enc.encode([scrub(t) for t in en_text]))[:, 1]
        S["en_excerpt"] = met(ys, (p_en >= .5).astype(int), p_en)
        # multilingual encoder, trained on English full-text training articles
        menc = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        cacheM = f"embM_train{q}.npy"
        if os.path.exists(cacheM) and len(np.load(cacheM)) == len(tr):
            EM = np.load(cacheM)
        else:
            EM = np.asarray(menc.encode([scrub(t) for t in dd.raw.values[tr]], batch_size=128, show_progress_bar=True)); np.save(cacheM, EM)
        mclf = rf().fit(EM, y[tr])
        pm_en = mclf.predict_proba(menc.encode(en_text))[:, 1]
        S["multiling_encoder_en_excerpt"] = met(ys, (pm_en >= .5).astype(int), pm_en)
        en_emb = enc.encode(en_text)
        samples = {}
        for lg, code in LANGS.items():
            fwd = translate(flat, "eng_Latn", code); back = translate(fwd, code, "eng_Latn")
            native, back_t = regroup(fwd), regroup(back)
            pb = clf.predict_proba(enc.encode([scrub(t) for t in back_t]))[:, 1]
            pn = mclf.predict_proba(menc.encode(native))[:, 1]
            be = enc.encode(back_t)
            cos = float(np.mean(np.sum(en_emb * be, 1) / (np.linalg.norm(en_emb, axis=1) * np.linalg.norm(be, axis=1))))
            S[lg] = {"roundtrip_then_english_rf": met(ys, (pb >= .5).astype(int), pb),
                     "native_multilingual_rf": met(ys, (pn >= .5).astype(int), pn),
                     "roundtrip_semantic_similarity": cos,
                     "pred_agreement_with_english": float(np.mean((pb >= .5) == (p_en >= .5)))}
            samples[lg] = {"native": native[0][:300], "back": back_t[0][:300]}
            print(lg, {k: (round(v["acc"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in S[lg].items()}, flush=True)
            save()
        S["example"] = {"english": en_text[0][:300], **samples}
    s3()
    print("\nSaved", OUTF)


if __name__ == "__main__":
    main()
