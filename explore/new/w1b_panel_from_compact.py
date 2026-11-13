"""Build data/interim/wb_obs.parquet and wb_meta.parquet from the compact CSVs (scripts/wayback/wb_compact.py)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(ROOT / "data" / "wayback"))
a = ap.parse_args()
O = pd.read_csv(Path(a.src) / "wb_obs.csv.gz", dtype={"ts": str})
M = pd.read_csv(Path(a.src) / "wb_meta.csv.gz")
M["post_date"] = pd.to_datetime(M.post_date, utc=True, errors="coerce").dt.tz_localize(None)
O["ts"] = pd.to_datetime(O.ts, format="%Y%m%d%H%M%S")
O = O.merge(M[["art_id", "post_date"]], on="art_id", how="left")
O["age_h"] = (O.ts - O.post_date).dt.total_seconds() / 3600
O = O.drop(columns=["post_date"]).drop_duplicates(["art_id", "ts", "src", "col"]).sort_values(["art_id", "ts"])
out = ROOT / "data" / "interim"; out.mkdir(parents=True, exist_ok=True)
O.to_parquet(out / "wb_obs.parquet", index=False); M.to_parquet(out / "wb_meta.parquet", index=False)
print(f"sightings {len(O):,}; articles {O.art_id.nunique():,}; captures {O.ts.nunique():,}; "
      f"span {O.ts.min()} .. {O.ts.max()}")
print(O.groupby("src").size().to_string())
