# Dataset

FactualAI is trained and evaluated on the **ISOT Fake News Dataset**
(44,898 articles: 23,481 labeled fake, 21,417 labeled real), as described in
Section 4.1 of the paper.

The raw data is **not redistributed in this repository** (it is a large,
separately licensed third-party dataset). To reproduce training:

1. Download `Fake.csv` and `True.csv`, e.g. from:
   - Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   - Original ISOT source (University of Victoria):
     https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
2. Place both files in `data/raw/`.
3. Run `python scripts/train.py --data-dir data/raw`.

## Citation

If you use this dataset, please cite it per the ISOT project's request:

> Ahmed H, Traore I, Saad S. "Detecting opinion spams and fake news using
> text classification", Journal of Security and Privacy, Volume 1, Issue 1,
> Wiley, 2018.
>
> Ahmed H, Traore I, Saad S. (2017) "Detection of Online Fake News Using
> N-Gram Analysis and Machine Learning Techniques." In: Traore I., Woungang
> I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in
> Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer
> Science, vol 10618. Springer, Cham.

## Sample data

`data/sample.csv` contains a small set of hand-written illustrative examples
(not drawn from ISOT) used only by the test suite and for smoke-testing the
pipeline without downloading the full dataset.
