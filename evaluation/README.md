# Evaluation: how to reproduce every number in the paper

Run all commands from the **repository root**. Data go in:

```
data/raw/Fake.csv, data/raw/True.csv          ISOT (see data/README.md)
data/fakenewsnet/politifact_fake.csv ...       FakeNewsNet title files (4 CSVs)
```

All outputs are already included in `evaluation/results/`, so you can inspect them without
re-running anything.

## Part A – CPU (any machine, ~3–4 h in total)

| Script | Paper | Output |
|---|---|---|
| `python evaluation/artifacts.py` | Table 3, Fig. 2 | `results/artifacts.csv` |
| `python evaluation/run_eval.py` | Table 4 (P0–P2, 5-fold CV) | `results/results_cv.json` |
| `python evaluation/run_extra.py` | Tables 3, 5, 6 (marker-only, stress test, McNemar, temporal, cross-dataset) | `results/results_extra.json` |

## Part B – GPU (Kaggle notebooks, free)

Open each notebook on Kaggle with the ISOT dataset
(`clmentbisaillon/fake-and-real-news-dataset`) attached, Accelerator = GPU, Internet = On,
then *Save Version → Save & Run All*.

| Notebook | Paper | Output (download from the Output tab) |
|---|---|---|
| `notebooks/FactualAI_Kaggle.ipynb` (runs `run_local_mpnet_llm.py`) | Table 4 (production classifier), Table 7 (1B LLM) | `results_local.json`, `emb_P1_raw.npy`, `emb_P2_scrubbed.npy`, `llm_cache.json` |
| `notebooks/FactualAI_Kaggle_part2.ipynb` (runs `run_v2.py`) | Tables 6–9 (titles, cross-dataset, 3B/8B LLMs, multilingual) | `results_v2.json`, `llm_cache_llama3.2_3b.json`, `llm_cache_llama3.1_8b.json` |

The LLM answer caches are included in `kaggle_outputs/`. The embedding files (`emb_*.npy`,
about 120 MB each) are not included; regenerate them with the first notebook and put them in
`evaluation/kaggle_outputs/`.

## Part C – analyses that use the Kaggle outputs (CPU)

| Script | Paper | Output |
|---|---|---|
| `python evaluation/run_mpnet_post.py` | Table 5 (production row), Table 6 (temporal), Table 7 notes | `results/results_mpnet_post.json` |
| `python evaluation/run_calib_lodo.py` then `python evaluation/calib_fix.py` | Sect. 7.6–7.7, Fig. 4, Table 8 | `results/results_calib_lodo.json` |
| `python evaluation/sig_v2.py` | Table 7 (McNemar, CIs) | `results/results_sig_v2.json` |

## Figures

`python evaluation/figures/fig_artifacts.py`, `fig_protocols.py`, `fig_calib.py`, `fig_arch.py`
write the PNGs in `evaluation/figures/`.

## Notes

- Random seed 42 everywhere. Random Forest results can differ by a few tenths of a point
  between scikit-learn versions (e.g. 92.2% vs 92.5% on the scrubbed split).
- The `--translate` option of `run_local_mpnet_llm.py` (Google Translate round trip) failed in
  our Kaggle run and is not reported; the multilingual results use NLLB-200 (`run_v2.py`, S3).
- `common.py` holds data loading, de-duplication and the artifact scrubber used by all scripts.
