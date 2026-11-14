"""Shared loader for the new-direction probes.

The url column of OnlineNewsPopularity.csv is shifted against the feature/shares rows from 2013-06-06 on
(see data/external/realigned.csv, built by the keyword-rebuild thread). Here every row keeps its own features
and shares, gets its true url/title from the repaired table, and its publication day from `timedelta`
(2015-01-08 minus timedelta days), never from the url in the file.
"""
from __future__ import annotations

from pathlib import Path

import os

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


NETS = ["twitter", "facebook", "linked_in", "google_plus", "pinterest", "stumble_upon"]
STATES = ["new", "rising", "hot", "page"]


def load_obs(df: pd.DataFrame, path=None, lists_only: bool = False) -> pd.DataFrame:
    """Archive sightings joined to the rows of load_fixed() (column `row`), sorted by article and capture time."""
    O = pd.read_parquet(path or ROOT / "data" / "interim" / "wb_obs.parquet")
    if lists_only:
        O = O[O.src != "article"]
    src = os.environ.get("WB_SRC", "home")  # homepage captures by default; "all" adds stories.json and channel pages
    if src != "all":
        O = O[O.src.isin(src.split(","))]
    O = O[(O.age_h > 0) & O.total.notna()].copy()
    thin = int(os.environ.get("WB_THIN", "0"))  # keep at most this many captures per day (evenly spaced), 0 = all
    if thin:
        cap = pd.Series(np.sort(O.ts.unique()))
        day = cap.dt.normalize(); pos = cap.groupby(day).cumcount(); cnt = cap.groupby(day).transform("size")
        keep = np.zeros(len(cap), bool)
        for d, g in cap.groupby(day):
            k = len(g)
            keep[g.index[np.unique(np.round(np.linspace(0, k - 1, min(thin, k))).astype(int))]] = True
        O = O[O.ts.isin(cap[keep])].copy()
    row_of = pd.Series(np.arange(len(df)), index=df.art_id.to_numpy())
    row_of = row_of[row_of.index.notna() & ~row_of.index.duplicated()]
    O["row"] = O.art_id.map(row_of)
    return O.dropna(subset=["row"]).astype({"row": int}).sort_values(["row", "ts"]).reset_index(drop=True)


def early_state(O: pd.DataFrame, h: float) -> pd.DataFrame:
    """One row per article seen at age <= h: its LAST such sighting plus what was seen before it."""
    o = O[O.age_h <= h]
    g = o.groupby("row")
    last = g.tail(1).set_index("row")
    E = pd.DataFrame(index=last.index)
    E["age"] = last.age_h
    E["l_total"] = np.log1p(last.total)
    for k in NETS:
        E["l_" + k] = np.log1p(last[k].fillna(0))
    E["fb_share"] = last.facebook.fillna(0) / (last.total + 1)
    E["l_rate0"] = E.l_total - np.log(E.age.clip(lower=0.05))  # log average shares per hour so far
    for c in STATES:
        E["col_" + c] = (last.col == c).astype(float)
    E["rank"] = last["rank"]
    E["n_sight"] = g.size()
    E["ever_rising"] = g.col.apply(lambda v: float((v == "rising").any()))
    E["ever_hot"] = g.col.apply(lambda v: float((v == "hot").any()))
    for k in ["vel_last", "vel_slope", "vel_max", "vel_min", "vel_first", "vel_n"]:
        E[k] = last[k]
    return E


COUNT_COLS = ["age", "l_total"] + ["l_" + k for k in NETS] + ["fb_share", "l_rate0"]
STATE_COLS = ["col_" + c for c in STATES] + ["rank", "n_sight", "ever_rising", "ever_hot"]
VEL_COLS = ["vel_last", "vel_slope", "vel_max", "vel_min", "vel_first", "vel_n"]
