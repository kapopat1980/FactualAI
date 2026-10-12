# Data

No data are redistributed in this repository. Place the files as follows:

```
data/raw/Fake.csv
data/raw/True.csv
data/fakenewsnet/politifact_fake.csv
data/fakenewsnet/politifact_real.csv
data/fakenewsnet/gossipcop_fake.csv
data/fakenewsnet/gossipcop_real.csv
```

## ISOT Fake and Real News dataset

44,898 articles (23,481 fake, 21,417 real). Download `Fake.csv` and `True.csv` from

- Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- University of Victoria ISOT Lab: https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

Please cite:

> Ahmed H, Traore I, Saad S (2017) Detection of online fake news using n-gram analysis and
> machine learning techniques. ISDDC 2017, LNCS 10618, Springer, pp 127–138.
>
> Ahmed H, Traore I, Saad S (2018) Detecting opinion spams and fake news using text
> classification. Security and Privacy 1(1):e9.

Note: the two classes come from different publishers. The real articles carry Reuters
datelines and tags and 12.9% of rows are exact duplicates. Use the de-duplication and scrubbing
in `evaluation/common.py` before evaluating any model (see the paper, Section 6).

## FakeNewsNet (titles)

Download the four CSV files from the `dataset/` folder of
https://github.com/KaiDMML/FakeNewsNet. Please cite:

> Shu K, Mahudeswaran D, Wang S, Lee D, Liu H (2020) FakeNewsNet: A data repository with news
> content, social context, and spatiotemporal information for studying fake news on social
> media. Big Data 8(3):171–188.

## Sample data

`data/sample.csv` contains a few hand-written illustrative examples (not from ISOT) used only by
the test suite.
