"""External check of the url repair: share counts seen on Mashable's homepage shortly after publication.

Internet Archive captures of mashable.com (data/wayback/, collected by scripts/wayback/wb_fetch.py; 2,152 captures,
January 2013 to February 2014 so far) list articles with the share count shown at capture time. A count seen hours
after publication cannot exceed the count recorded in January 2015, and should predict it. Each sighting is joined
to a dataset row through the article's URL, once with the repaired URL of the row and once with the URL written in
the row (file as distributed), and compared with that row's shares:
  sightings            sightings (age > 0) of articles that match a row
  early_above_final    share of sightings whose count exceeds the row's shares (strictly, and by more than 5% + 100)
  articles             articles with a sighting 6 to 48 hours after publication (the last such sighting is used)
  corr_log, spearman   correlation of that early count with the row's shares (log1p; rank)
by period of the row's content date (before / from 6 June 2013, the first shifted row).
Output: results/tables/a4_archive_check.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import load  # noqa: E402

W = ROOT / "data" / "wayback"
O = pd.read_csv(W / "wb_obs.csv.gz", dtype={"ts": str}); M = pd.read_csv(W / "wb_meta.csv.gz")
M["post_date"] = pd.to_datetime(M.post_date, utc=True, errors="coerce").dt.tz_localize(None)
O["ts"] = pd.to_datetime(O.ts, format="%Y%m%d%H%M%S")
O = O.merge(M[["art_id", "post_date"]], on="art_id", how="left").drop_duplicates(["art_id", "ts", "src", "col"])
O["age_h"] = (O.ts - O.post_date).dt.total_seconds() / 3600
O = O[(O.age_h > 0) & O.total.notna()].sort_values(["art_id", "ts"])
print(f"captures {O.ts.nunique():,}, sightings {len(O):,}, {O.ts.min():%Y-%m-%d} to {O.ts.max():%Y-%m-%d}")

df = load()
aid = lambda u: u.str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2}/[^/]+)")[0].str.lower()
rows = []
for name, col in [("repaired", "url"), ("as_distributed", "url_in_file")]:
    mp = pd.DataFrame({"art_id": aid(df[col].fillna("")), "shares": df.shares, "date": pd.to_datetime(df.publish_date)})
    mp = mp.dropna(subset=["art_id"]).drop_duplicates("art_id").set_index("art_id")
    j = O.join(mp, on="art_id", how="inner")
    for per, mk in [("before 2013-06-06", j.date < "2013-06-06"), ("from 2013-06-06", j.date >= "2013-06-06")]:
        o = j[mk]; e = o[(o.age_h >= 6) & (o.age_h <= 48)].groupby("art_id").tail(1)
        rows.append(dict(mapping=name, period=per, sightings=len(o), early_above_final=float((o.total > o.shares).mean()),
                         early_above_final_5pct_100=float((o.total > 1.05 * o.shares + 100).mean()), articles=len(e),
                         corr_log=float(np.corrcoef(np.log1p(e.total), np.log1p(e.shares))[0, 1]),
                         spearman=float(spearmanr(e.total, e.shares)[0])))
R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / "tables" / "a4_archive_check.csv", index=False)
print(R.round(4).to_string(index=False))
