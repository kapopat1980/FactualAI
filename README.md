# FactualAI

An AI-Powered Multilingual and Multimodal Fake News Detection System.

Companion code for:

> Rane, C., Patil, L. K., & Popat, K. A. *FactualAI: An AI-Powered
> Multilingual and Multimodal Fake News Detection System.*

FactualAI accepts news content as typed text (100+ languages via automatic
translation), images (OCR), audio (ASR), and YouTube video URLs, and
classifies it using a weighted ensemble of a Random Forest classifier
(sentence-embedding features) and a locally hosted LLaMA 3.2 model served
through Ollama, producing a confidence-scored verdict with a plain-language
explanation.

## Repository structure

```
src/factualai/
  preprocessing.py   Section 4.2 — text cleaning, language detection, translation
  embeddings.py       Section 4.3 — sentence embeddings (SentenceTransformer / TF-IDF fallback)
  classifier.py        Section 4.4 — Random Forest classifier over embeddings
  llm_reasoner.py       Section 4.5 — LLaMA 3.2 client via Ollama (30s timeout)
  ensemble.py            Section 4.6 — weighted ensemble decision + fallback logic
  ocr.py                  Section 5.2 — image input (OpenCV + Tesseract)
  asr.py                   Section 5.3 — audio input (Faster-Whisper)
  youtube.py                Section 5.4 — YouTube audio extraction + transcription
  rss.py                     Section 5.5 — live RSS feed classification
  app.py                      Section 3 — Flask REST API (7 endpoints, Table 1)
scripts/train.py     Training/evaluation script (Section 8.1 methodology)
tests/                Pytest suite (24 tests, see "Testing" below)
data/                 Dataset instructions (raw data not redistributed)
```

## Installation

```bash
git clone https://github.com/<your-username>/FactualAI.git
cd FactualAI
pip install -r requirements.txt
```

The core install above is intentionally lightweight and uses a TF-IDF
fallback embedder so the classifier, ensemble logic, and API are fully
testable without a multi-gigabyte PyTorch download. To reproduce the exact
pipeline and accuracy figures reported in the paper (all-mpnet-base-v2
embeddings), install the optional extras listed at the bottom of
`requirements.txt`:

```bash
pip install sentence-transformers          # embeddings.py production backend
pip install opencv-python pytesseract      # ocr.py (needs system tesseract-ocr)
pip install faster-whisper                 # asr.py
pip install yt-dlp pydub                   # youtube.py (needs system ffmpeg)
```

The LLM reasoning component additionally requires [Ollama](https://ollama.com)
running locally with the `llama3.2:1b` model pulled:

```bash
ollama pull llama3.2:1b
ollama serve
```

## Reproducing the reported results

1. Obtain the ISOT Fake News Dataset (`Fake.csv`, `True.csv`) — see
   `data/README.md` for sources and citation.
2. Place both files in `data/raw/`.
3. Run:
   ```bash
   python scripts/train.py --data-dir data/raw --embedder sentence-transformer
   ```
   This performs the same 80/20 stratified split described in Section 8.1 and
   prints accuracy/precision/recall/F1 for comparison against Table 2.

Omit `--embedder sentence-transformer` to use the lightweight TF-IDF fallback
instead (faster, no large model download, but not the configuration used for
the paper's reported 95% accuracy).

## Running the API server

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

## Testing

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

24 tests covering: the weighted-ensemble decision logic (including the
LLM-timeout fallback path and the LLM label→fake-score conversion, matching
the worked example in Figure 3 of the paper), the text preprocessing
pipeline, the Random Forest classifier, and the Flask API endpoints (with
the model and LLM mocked out). All tests run against the lightweight TF-IDF
embedder and require no GPU, external network access, or Ollama instance.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite the paper (see the manuscript's reference
list) and, if applicable, this repository's archived release (see the
project's Zenodo record, linked from the paper's Data/Code Availability
statement).
