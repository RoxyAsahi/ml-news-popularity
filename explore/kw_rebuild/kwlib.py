"""Shared loading for explore/kw_rebuild: the dataset under two row-to-URL alignments.

  given   the CSV as distributed: url, slug, headline and publication date all come from the url column
          (this is what every analysis up to v19 used; identical to newspop.data.load()).
  true    the url column re-aligned by k4_realign.py: each row gets the URL / headline its features and shares
          belong to, and the publication date comes from timedelta (2015-01-08 minus timedelta days).
          Rows whose true URL lies beyond the end of the file (the last 152) have no URL or headline.
Both frames carry file_row (0-based row of the CSV) so results can be mapped between alignments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import add_publish_date, clean, load_raw  # noqa: E402
from newspop.titles import aligned_titles  # noqa: E402

INT = ROOT / "data" / "interim" / "kw_rebuild"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
ACQUIRED = pd.Timestamp("2015-01-08")
FIRST_SHIFTED = pd.Timestamp("2013-06-06")  # content date of the first shifted row (k4_realign.py)


def frame(align: str) -> pd.DataFrame:
    """Cleaned dataset, oldest first, with columns url, publish_date, title (NaN if unknown), file_row."""
    raw = add_publish_date(load_raw()).reset_index(drop=True)
    raw["file_row"] = np.arange(len(raw))
    if align == "given":
        df = clean(raw).sort_values(["publish_date", "url"]).reset_index(drop=True)
        df["title"] = aligned_titles(df).to_numpy()
        return df
    if align != "true":
        raise ValueError(align)
    R = pd.read_csv(INT / "realigned.csv")
    raw["url_as_given"] = raw["url"]
    raw["url"] = R.true_url.to_numpy()
    raw["title"] = R.true_title.to_numpy()
    raw["publish_date"] = (ACQUIRED - pd.to_timedelta(raw.timedelta, unit="D")).dt.normalize()
    return clean(raw).sort_values(["publish_date", "file_row"]).reset_index(drop=True)


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    drop = ["url", "url_as_given", "shares", "publish_date", "timedelta", "file_row", "title"]
    return df.drop(columns=[c for c in drop if c in df.columns]).select_dtypes("number")
