"""Loading, cleaning and feature groups for the Online News Popularity data.

Dataset: Fernandes, Vinagre & Cortez (2015), UCI ML Repository id 332.
Place the CSV at data/raw/OnlineNewsPopularity.csv (see README).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "OnlineNewsPopularity.csv"

REALIGN_CSV = ROOT / "data" / "external" / "url_realignment.csv"
ACQUIRED = pd.Timestamp("2015-01-08")  # timedelta counts days back from this date (agrees with weekday_is_* on every row)

TARGET = "shares"
NON_PREDICTIVE = ["url", "url_in_file", "timedelta"]


def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """Read the CSV and strip the leading spaces the original column names carry."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Unzip the course dataset (or download it from UCI) "
            "into data/raw/ - see README.md."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def add_publish_date(df: pd.DataFrame, from_url: bool = False) -> pd.DataFrame:
    """Publication date of the row's content: ACQUIRED minus timedelta days.

    from_url=True parses the date in the URL instead (http://mashable.com/YYYY/MM/DD/slug/). In the file as
    distributed that is the date of another article for most rows after June 2013 (see repair_urls).
    """
    out = df.copy()
    if from_url:
        out["publish_date"] = pd.to_datetime(
            out["url"].str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2})/")[0], format="%Y/%m/%d"
        )
    else:
        out["publish_date"] = ACQUIRED - pd.to_timedelta(out["timedelta"], unit="D")
    return out


def repair_urls(df: pd.DataFrame, path: str | Path = REALIGN_CSV) -> pd.DataFrame:
    """Give each row the URL its features and shares belong to.

    In OnlineNewsPopularity.csv the url column is shifted against all other columns from 6 June 2013 on: row i's
    content belongs to url[i + k], with k growing from 1 to 152 through the file (scripts/a1_realign_urls.py
    estimates k and writes data/external/url_realignment.csv). `url` becomes the repaired URL (NaN for the last
    152 rows, whose URL is not in the file) and `url_in_file` keeps the column as distributed.
    df must be in the original file order.
    """
    k = pd.read_csv(path).offset.to_numpy()
    if len(k) != len(df):
        raise ValueError("url_realignment.csv does not match the CSV; rerun scripts/a1_realign_urls.py")
    given = df["url"].to_numpy()
    j = np.arange(len(df)) + np.maximum(k, 0)
    out = df.copy()
    out["url_in_file"] = given
    out["url"] = np.where((k >= 0) & (j < len(df)), given[np.minimum(j, len(df) - 1)], None)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, documented cleaning. Every rule here must be mentioned in the report.

    - drop the single row with n_unique_tokens > 1 (a rate cannot exceed 1; data error)
    - flag articles with empty content (n_tokens_content == 0)
    - flag kw_min_min == -1, which encodes 'missing' rather than a real share count
    """
    out = df[df["n_unique_tokens"] <= 1].copy()
    out["empty_content"] = (out["n_tokens_content"] == 0).astype(int)
    out["kw_min_min_missing"] = (out["kw_min_min"] == -1).astype(int)
    return out


def load(path: str | Path = RAW_CSV, repaired: bool = True) -> pd.DataFrame:
    """Load, repair the url column, clean, add publish_date and sort chronologically (oldest first).

    repaired=False returns the file as distributed (shifted url column, dates parsed from that URL), which is what
    every analysis before the repair used; it is kept for the diagnosis in scripts/a2_alignment_validation.py.
    """
    raw = load_raw(path).reset_index(drop=True)
    if not repaired:
        df = clean(add_publish_date(raw, from_url=True))
        return df.sort_values(["publish_date", "url"]).reset_index(drop=True)
    raw = repair_urls(raw)
    raw["_file_row"] = np.arange(len(raw))
    df = clean(add_publish_date(raw))
    return df.sort_values(["publish_date", "_file_row"]).drop(columns="_file_row").reset_index(drop=True)


# ---------------------------------------------------------------- feature groups
def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Named groups of predictive columns, used for ablations (RQ2) and clustering (RQ4)."""
    cols = [c for c in df.columns if c not in NON_PREDICTIVE + [TARGET, "publish_date"]]
    keyword = [c for c in cols if c.startswith("kw_")]
    self_ref = [c for c in cols if c.startswith("self_reference_")]
    channel = [c for c in cols if c.startswith("data_channel_is_")]
    weekday = [c for c in cols if c.startswith("weekday_is_") or c == "is_weekend"]
    lda = [c for c in cols if c.startswith("LDA_")]
    sentiment = [
        c for c in cols
        if "polarity" in c or "subjectivity" in c or "positive" in c or "negative" in c
    ]
    used = set(keyword + self_ref + channel + weekday + lda + sentiment)
    content = [c for c in cols if c not in used]
    return {
        "content": content,
        "keyword": keyword,
        "self_reference": self_ref,
        "channel": channel,
        "weekday": weekday,
        "lda": lda,
        "sentiment": sentiment,
    }


def feature_set(df: pd.DataFrame, name: str = "all") -> list[str]:
    """Feature sets for the share-derived feature analysis (RQ2).

    all       : every predictive column
    no_kw     : drop keyword-share statistics (kw_*)
    no_kw_ref : also drop self_reference_* (shares of linked Mashable articles)
    no_clock  : drop only the keyword columns that act as a clock (CLOCK_COLUMNS)
    no_clock_strict : also drop CLOCK_COLUMNS_EXTRA
    """
    g = feature_groups(df)
    every = [c for group in g.values() for c in group]
    if name == "all":
        return every
    if name == "no_kw":
        return [c for c in every if c not in g["keyword"]]
    if name == "no_kw_ref":
        return [c for c in every if c not in g["keyword"] + g["self_reference"]]
    if name == "no_clock":
        return [c for c in every if c not in CLOCK_COLUMNS]
    if name == "no_clock_strict":
        return [c for c in every if c not in CLOCK_COLUMNS + CLOCK_COLUMNS_EXTRA]
    raise ValueError(f"unknown feature set {name!r}")


# Keyword statistics are point-in-time (computed from articles published earlier), but some
# of them mostly measure *how large the archive was* at publication time: a built-in clock.
# |Spearman with publication date| > 0.8 for kw_min_min / its -1 flag, 0.74 for kw_max_max.
CLOCK_COLUMNS = ["kw_max_max", "kw_min_min", "kw_min_min_missing"]
# next tier (|rho| 0.45-0.52); dropped only in the strict variant
CLOCK_COLUMNS_EXTRA = ["kw_avg_max", "kw_avg_min"]


def make_label(shares: pd.Series, threshold: float) -> np.ndarray:
    """Binary popularity label: 1 if shares >= threshold."""
    return (shares >= threshold).astype(int).to_numpy()
