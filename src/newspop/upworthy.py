"""Upworthy Research Archive helpers: loading, headline features (Pattern lexicon via TextBlob, the same
lexicon behind the Mashable sentiment features) and test-level aggregates."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from textblob import TextBlob

from .data import ROOT

UP = ROOT / "data" / "external" / "upworthy"


def load(split: str) -> pd.DataFrame:
    d = pd.read_csv(UP / f"{split}.csv", low_memory=False)
    d = d.dropna(subset=["headline"])
    d = d[(d.impressions > 0) & (d.clicks >= 0)].copy()
    d["t"] = pd.to_datetime(d.created_at, format="mixed")
    d["test"] = d.clickability_test_id
    d["lede_clean"] = d.lede.fillna("").map(lambda s: re.sub(r"<[^>]+>", " ", s).lower().strip())
    d["logit_ctr"] = np.log((d.clicks + 0.5) / (d.impressions - d.clicks + 0.5))
    return d.reset_index(drop=True)


_POL: dict[str, float] = {}


def _pol(w: str) -> float:
    if w not in _POL:
        _POL[w] = TextBlob(w).sentiment.polarity
    return _POL[w]


PRON = {"first_person": r"\b(i|me|my|mine|i'm|i've|i'd)\b", "second_person": r"\b(you|your|yours|you're|you'll)\b",
        "third_person": r"\b(he|she|him|her|his|hers|they|them|their)\b", "this_these": r"\b(this|these)\b"}
FEATURES = ["subjectivity", "polarity", "share_positive_words", "share_negative_words", "n_words", "avg_word_length",
            "question", "exclamation", "number", "quote", "all_caps_word", *PRON]


def headline_features(h: pd.Series) -> pd.DataFrame:
    """Features of each headline (computed once per unique headline)."""
    u = pd.Series(h.astype(str).unique())
    F = _features(u)
    F.index = u.to_numpy()
    return F.loc[h.astype(str).to_numpy()].set_index(h.index)


def _features(h: pd.Series) -> pd.DataFrame:
    out = []
    for s in h:
        b = TextBlob(s)
        words = re.findall(r"[a-z']+", s.lower())
        pol = [_pol(w) for w in words] if words else []
        n = max(len(words), 1)
        row = dict(subjectivity=b.sentiment.subjectivity, polarity=b.sentiment.polarity,
                   share_positive_words=sum(p > 0 for p in pol) / n, share_negative_words=sum(p < 0 for p in pol) / n,
                   n_words=len(words), avg_word_length=np.mean([len(w) for w in words]) if words else 0.0,
                   question=float("?" in s), exclamation=float("!" in s), number=float(bool(re.search(r"\d", s))),
                   quote=float(bool(re.search(r"[\"“”]", s))),
                   all_caps_word=float(bool(re.search(r"\b[A-Z]{3,}\b", s))))
        low = s.lower()
        for k, rx in PRON.items():
            row[k] = float(bool(re.search(rx, low)))
        out.append(row)
    return pd.DataFrame(out, index=h.index)
