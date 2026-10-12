"""Shared data loading and artifact-scrubbing for the FactualAI re-evaluation."""
import re
import numpy as np
import pandas as pd

URL = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+|\b\S+\.(com|org|net)/\S*)", re.I)
HANDLE = re.compile(r"@\w+")
DATELINE = re.compile(r"^.{0,120}?\(Reuters\)\s*-\s*")
REUTERS = re.compile(r"\breuters\b", re.I)
CREDIT = re.compile(r"(featured image|photo|image)\s+(via|by|credit)[^.\n]*", re.I)
SOURCE_BRANDS = re.compile(r"21st century wire|21wire|getty images?|youtube|twitter|facebook", re.I)
PUNCT = re.compile(r"[^\w\s]", re.U)
WS = re.compile(r"\s+")


def basic_clean(s: str) -> str:
    """Original FactualAI preprocessing (Section 4.2): strip punctuation, lowercase."""
    return WS.sub(" ", PUNCT.sub(" ", s)).strip().lower()


def scrub(s: str) -> str:
    """Remove source/format artifacts that identify the publisher rather than the claim."""
    s = DATELINE.sub("", s)
    s = URL.sub(" ", s)
    s = HANDLE.sub(" ", s)
    s = CREDIT.sub(" ", s)
    s = SOURCE_BRANDS.sub(" ", s)
    s = REUTERS.sub(" ", s)
    return basic_clean(s)  # lowercasing also removes the ALL-CAPS title cue


def load_isot(d="data/raw"):
    f = pd.read_csv(f"{d}/Fake.csv"); f["label"] = 1
    t = pd.read_csv(f"{d}/True.csv"); t["label"] = 0
    df = pd.concat([f, t], ignore_index=True)
    df["title"] = df.title.astype(str); df["text"] = df.text.astype(str)
    df["date_p"] = pd.to_datetime(df.date.str.strip(), errors="coerce", format="mixed")
    df["raw"] = df.title + ". " + df.text
    return df


def dedup(df):
    key = df.raw.str.strip().str.lower()
    return df.loc[~key.duplicated()].reset_index(drop=True)


def load_fnn(prefix, d="data/fakenewsnet"):
    f = pd.read_csv(f"{d}/{prefix}_fake.csv"); f["label"] = 1
    r = pd.read_csv(f"{d}/{prefix}_real.csv"); r["label"] = 0
    x = pd.concat([f, r], ignore_index=True).dropna(subset=["title"])
    x = x.loc[~x.title.str.strip().str.lower().duplicated()].reset_index(drop=True)
    return x
