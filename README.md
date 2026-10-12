# FactualAI

**Multilingual and multimodal fake news detection, with a leakage-aware evaluation.**

Companion code for:

> Rane, C., Patil, L. K., & Popat, K. A. *FactualAI: A Multilingual and Multimodal Fake News
> Detection System with a Leakage-Aware Evaluation of Its Text Classifier.*

FactualAI accepts news content as typed text (100+ languages via machine translation), images
(OCR), audio (speech recognition) and video URLs, and classifies the resulting text with a
weighted ensemble of a Random Forest over sentence embeddings and a locally hosted LLM (served
through Ollama) that also returns a plain-language explanation.

This repository contains (1) the system and (2) every script, notebook and result file behind
the numbers reported in the paper.

## What the evaluation found (short version)

| Finding | Number |
|---|---|
| Accuracy on ISOT from 10 publisher/format markers alone, no article content | 99.7% |
| ISOT articles that are exact duplicates | 12.9% |
| Sparse-feature Random Forest, raw → raw vs. raw → scrubbed test text | 99.6% → 45.9% |
| Production classifier (RF on all-mpnet-base-v2), de-duplicated + scrubbed, 5-fold CV | 92.3% |
| Production classifier under temporal shift (train < Jul 2017, test ≥ Jul 2017) | 66.1% |
| Best lexical model, ISOT titles → FakeNewsNet PolitiFact titles | 62.7% (AUC 0.73) |
| LLaMA 3.2 1B / 3.2 3B / 3.1 8B alone, zero-shot | 45.9% / 71.1% / 87.4% |
| Ensemble (w = 0.6): RF alone → with 1B → with 8B | 91.4% → 88.0% → 94.4% |
| Round trip EN → {hi, mr, gu, ta, bn} → EN, accuracy change | −2.0 to −5.5 points |

In-corpus accuracy on ISOT mainly reflects source identification. See the paper for details.

## Repository structure

```
src/factualai/          the system
  preprocessing.py      language detection, translation, cleaning (Sect. 4.1)
  embeddings.py         sentence embeddings (all-mpnet-base-v2; TF-IDF fallback for CPU/CI)
  classifier.py         Random Forest (Sect. 4.3)
  llm_reasoner.py       LLM client via Ollama (Sect. 4.4)
  ensemble.py           weighted ensemble, fallback and abstention (Sect. 4.4-4.5)
  ocr.py, asr.py, youtube.py, rss.py   multimodal inputs (Sect. 5)
  app.py                Flask REST API (7 endpoints)
scripts/train.py        train / evaluate the classifier
tests/                  pytest suite (26 tests)
evaluation/             all experiments in the paper -> see evaluation/README.md
data/                   dataset instructions (data are not redistributed)
```

## Installation

```bash
git clone https://github.com/kapopat1980/FactualAI.git
cd FactualAI
pip install -r requirements.txt
```

The core install uses a lightweight TF-IDF embedder so the API and tests run without PyTorch.
For the production configuration and the multimodal inputs install the optional extras:

```bash
pip install sentence-transformers          # all-mpnet-base-v2 embeddings (production)
pip install opencv-python pytesseract      # OCR (needs the tesseract-ocr system package)
pip install faster-whisper                 # speech recognition
pip install yt-dlp pydub                   # video URLs (needs ffmpeg)
```

LLM component: install [Ollama](https://ollama.com) and pull a model, e.g.

```bash
ollama pull llama3.2:1b      # deployed default
ollama pull llama3.1:8b      # recommended by the evaluation (Table 7)
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `FACTUALAI_LLM_MODEL` | `llama3.2:1b` | Ollama model used by the LLM reasoner |
| `FACTUALAI_RF_WEIGHT` | `0.6` | Ensemble weight w of the Random Forest (LLM gets 1 − w). Use `1.0` with the 1B model (the LLM then only explains); about `0.6`–`0.8` with `llama3.1:8b` |
| `FACTUALAI_OCR_LANG` | `eng` | Tesseract languages, e.g. `eng+hin+mar` (install the language data first) |

If the LLM times out, returns unparseable output or answers "Unknown", the ensemble falls back
to the Random Forest score and the result is flagged (`llm_abstained`).

## Running the API

```python
from factualai.app import create_app
from factualai.classifier import RandomForestNewsClassifier
from factualai.embeddings import SentenceTransformerEmbedder
from factualai.llm_reasoner import LLMReasoner

classifier = RandomForestNewsClassifier(SentenceTransformerEmbedder())
classifier.load("models/rf_classifier.pkl")
app = create_app(classifier=classifier, llm_reasoner=LLMReasoner())
app.run(port=5000)
```

## Training

Put the ISOT files (`Fake.csv`, `True.csv`) in `data/raw/` (see `data/README.md`), then

```bash
python scripts/train.py --data-dir data/raw --embedder sentence-transformer
```

Note: `scripts/train.py` reproduces the *original* random 80/20 split (protocol P0), which is
affected by duplicate leakage. For the leakage-aware protocols use the scripts in `evaluation/`.

## Reproducing the paper

See [`evaluation/README.md`](evaluation/README.md). In short: CPU scripts reproduce the audit,
baselines, stress test, temporal and cross-dataset results; two Kaggle notebooks (free GPU)
reproduce the embedding, LLM and multilingual results. All result files are included in
`evaluation/results/`.

## Testing

```bash
python -m pytest tests/ -v
```

26 tests covering the ensemble logic (including the timeout fallback and the "Unknown"
abstention path), preprocessing, the classifier and the Flask API (model and LLM mocked).

## Changes in v1.1.0

See [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT, see `LICENSE`.

## Citation

Please cite the paper and the archived software release (see `CITATION.cff` and the Zenodo
record linked from the paper's Code availability statement).
