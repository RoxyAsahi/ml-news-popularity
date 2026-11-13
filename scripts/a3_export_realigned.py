"""Write the repair as one flat table for the side analyses in explore/kw_rebuild/ and explore/editorial_dml/.

One row per row of OnlineNewsPopularity.csv (file order): file_row, url_in_file, offset, true_url, true_title,
content_date (2015-01-08 minus timedelta). Built from data/external/url_realignment.csv (scripts/a1_realign_urls.py)
and the sitemap headlines. Output: data/external/realigned.csv and data/interim/kw_rebuild/realigned.csv (not
committed; both side analyses read this file).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import ACQUIRED, REALIGN_CSV, load_raw, repair_urls  # noqa: E402
from newspop.titles import aligned_titles  # noqa: E402

raw = load_raw().reset_index(drop=True)
R = repair_urls(raw)
t = aligned_titles(R.assign(url=R.url.fillna("")))
out = pd.DataFrame({"file_row": np.arange(len(raw)), "url_in_file": R.url_in_file, "offset": pd.read_csv(REALIGN_CSV).offset,
                    "true_url": R.url, "true_title": t.where(R.url.notna()),
                    "content_date": (ACQUIRED - pd.to_timedelta(raw.timedelta, unit="D")).dt.date})
for p in [ROOT / "data" / "external" / "realigned.csv", ROOT / "data" / "interim" / "kw_rebuild" / "realigned.csv"]:
    p.parent.mkdir(parents=True, exist_ok=True); out.to_csv(p, index=False)
print(len(out), "rows;", int(out.true_url.notna().sum()), "with a URL;", int(out.true_title.notna().sum()), "with a headline")
