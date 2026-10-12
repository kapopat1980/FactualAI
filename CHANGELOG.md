# Changelog

## v1.1.0

Accompanies the revised manuscript (leakage-aware evaluation).

### System
- `ensemble.py`: an LLM answer other than "Fake"/"Real" (e.g. "Unknown") is now treated as an
  abstention: the ensemble falls back to the Random Forest score and sets `llm_abstained=True`
  (previously the answer was mapped to a neutral 0.5, which pulled the score towards 0.5).
  Labels are case-normalised.
- `ensemble.py`: the Random Forest weight is configurable with `FACTUALAI_RF_WEIGHT`
  (default 0.6, as deployed).
- `llm_reasoner.py`: the Ollama model is configurable with `FACTUALAI_LLM_MODEL`
  (default `llama3.2:1b`; `llama3.1:8b` performed best in the evaluation).
- `ocr.py`: Tesseract languages configurable with `FACTUALAI_OCR_LANG` (default `eng`).
- `rss.py`: removed the discontinued Reuters RSS feed (four outlets remain).
- Tests: 26 tests, including the abstention fallback.

### Evaluation (new `evaluation/` folder)
- ISOT artifact audit, de-duplication and artifact scrubbing.
- Protocols P0–P4 (original, de-duplicated, scrubbed, temporal, cross-dataset) with five
  baselines, stress test, McNemar tests and bootstrap confidence intervals.
- Production classifier and LLMs (1B, 3B, 8B) evaluated on Kaggle GPU notebooks.
- Calibration, selective prediction, multi-source training and multilingual (NLLB-200) analyses.
- All result files included.

## v1.0.0
Initial release.
