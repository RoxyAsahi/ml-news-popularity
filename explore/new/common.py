"""Shared loader for the new-direction probes.

The url column of OnlineNewsPopularity.csv is shifted against the feature/shares rows from 2013-06-06 on
(see data/external/realigned.csv, built by the keyword-rebuild thread). Here every row keeps its own features
and shares, gets its true url/title from the repaired table, and its publication day from `timedelta`
(2015-01-08 minus timedelta days), never from the url in the file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HGB_PARAMS = dict(learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, max_iter=500)


def load_fixed() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "OnlineNewsPopularity.csv")
    raw.columns = raw.columns.str.strip()
    fix = pd.read_csv(ROOT / "data" / "external" / "realigned.csv")
    assert len(raw) == len(fix)
    df = raw.drop(columns=["url"]).copy()
    df["file_row"] = np.arange(len(df))
    df["url"] = fix.true_url.to_numpy()
    df["title"] = fix.true_title.to_numpy()
    df["date"] = pd.to_datetime("2015-01-08") - pd.to_timedelta(df.timedelta, unit="D")
    df["art_id"] = df.url.str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2}/[^/]+)")[0].str.lower()
    df = df[df.n_unique_tokens <= 1]
    return df.sort_values(["date", "file_row"]).reset_index(drop=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    drop = ["url", "title", "date", "art_id", "file_row", "shares", "timedelta"]
    return df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes("number")


def channel(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("data_channel_is_")]
    names = np.array([c.replace("data_channel_is_", "") for c in cols] + ["other"])
    M = df[cols].to_numpy()
    return names[np.where(M.sum(1) > 0, M.argmax(1), len(cols))]
