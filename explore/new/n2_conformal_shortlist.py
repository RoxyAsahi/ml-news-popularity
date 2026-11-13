"""Probe B2: shortlists of likely blockbusters with a false-discovery guarantee (conformal selection).

Jin & Candes (JMLR 2023): for a test article j with score V_j, the conformal p-value against "shares <= c" is
  p_j = (1 + #{calibration i : shares_i <= c and V_i >= V_j}) / (n_cal + 1);
Benjamini-Hochberg at level q on the month's p-values gives a shortlist whose expected share of
non-blockbusters is at most q (exchangeability between calibration and test articles assumed).
Past-only: scores come from n1_tail_value.py (model fit on months < m); calibration = the scored articles of the
months before m (at least 3 months), test = month m. c = 95th or 90th percentile of shares in the months before m.
Reported per score and level q: shortlist size per month, realised false-discovery proportion, power (share of true
blockbusters shortlisted), and the same for a naive "top-k by score" list of equal size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_fixed  # noqa: E402

df = load_fixed()
S = pd.read_parquet(ROOT / "data" / "interim" / "n1_scores_fixed.parquet")
s = df.shares.to_numpy().astype(float)
month = df.date.dt.to_period("M").astype(str).to_numpy()
ms = sorted(m for m in set(month) if not np.isnan(S.loc[month == m, "log_mean"]).any())
rows = []
for pct in (0.95, 0.90):
    for score in ["clf_1400", "log_mean", "lognorm", "q90", "clf_top5"]:
        v = S[score].to_numpy()
        for q in (0.2, 0.3, 0.5):
            R = D = T = 0
            fdp = []
            for m in ms[3:]:
                cal = np.isin(month, [x for x in ms if x < m]); te = month == m
                c = np.quantile(s[month < m], pct)
                null = cal & (s <= c)
                vn = np.sort(v[null])
                p = (1 + (len(vn) - np.searchsorted(vn, v[te], side="left"))) / (cal.sum() + 1)
                o = np.argsort(p); k = np.arange(1, te.sum() + 1)
                ok = np.flatnonzero(p[o] <= q * k / te.sum())
                r = ok.max() + 1 if len(ok) else 0
                sel = o[:r]; hit = s[te][sel] > c
                R += r; D += hit.sum(); T += (s[te] > c).sum()
                if r:
                    fdp.append(1 - hit.mean())
            rows.append(dict(pct=pct, score=score, q=q, shortlisted=R, per_month=R / len(ms[3:]), fdp=1 - D / max(R, 1),
                             power=D / T, months_nonempty=len(fdp)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
out.to_csv(ROOT / "results" / "tables" / "n2_conformal_shortlist.csv", index=False)
