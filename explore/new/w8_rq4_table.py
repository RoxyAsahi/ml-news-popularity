"""Direction A, the one table for the report: what the first hours add to the paper's best pre-publication model.

Harmonised with the main report: evaluation months 2013-07 .. 2014-12, all dataset articles; the "before
publication" row is the report's own best model, `ens` (mean of the 58-feature HistGB and the headline-embedding
ridge, AUC 0.746; per-article past-only scores from scripts/v1_past_only_called.py on main, copied to
data/external/from_main/v1_scores.csv together with the cached headline vectors).

A look "within h hours" = the last homepage sighting at age <= h (nothing if the article was not sighted; the model
then sees missing values, the sighted flag and the number of captures taken that day and the next).
Early model, past-only monthly rolling origin (fit on all earlier months from 2013-01):
  HistGB on the 58 features + 32 PCA components of the headline vectors (PCA fit on the training months) +
  the report's ens score (out-of-sample by construction; missing before 2013-07) + the archive state at h.
Columns
  all          AUC (>= 1400) over all articles, unseen as missing            [day-bootstrap interval]
  all_thin4    the same with at most 4 homepage captures per day
  not_yet      AUC among articles sighted by h and not yet past 1,400 at the look
  not_yet_sh   the same for Szabo-Huberman (rank by the count at the look)
  not_yet_pre  the same for the report's ens score
plus the share sighted, the share already past 1,400 and the median actual age of the look.
Output: results/tables/w8_rq4_table.csv, data/interim/w8_scores.parquet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COUNT_COLS, HGB_PARAMS, ROOT, STATE_COLS, VEL_COLS, early_state, features, load_fixed,  # noqa: E402
                    load_obs)

FIRST, B = "2013-07", 500
HS = [1.0, 3.0, 6.0, 12.0, 24.0]
df = load_fixed()
n = len(df)
X = features(df).to_numpy()
s = df.shares.to_numpy().astype(float); y = np.log1p(s); lab = s >= 1400
month = df.date.dt.to_period("M").astype(str).to_numpy()
dn = (df.date - df.date.min()).dt.days.to_numpy()
t95 = np.quantile(s, .95)

# the report's pre-publication scores, mapped from its row index (file order without the dropped row) to ours
V1 = pd.read_csv(ROOT / "data" / "external" / "from_main" / "v1_scores.csv")
fr = np.sort(df.file_row.to_numpy())
ens = pd.Series(V1.ens.to_numpy(), index=fr[V1.row.to_numpy()]).reindex(df.file_row.to_numpy()).to_numpy()
assert np.allclose(pd.Series(V1.y.to_numpy(), index=fr[V1.row.to_numpy()]).reindex(df.file_row.to_numpy()).dropna(),
                   y[~np.isnan(ens)])
z = np.load(ROOT / "data" / "external" / "from_main" / "mashable_title_bge.npz", allow_pickle=False)
emb = dict(zip(z["titles"].tolist(), z["E"]))
E = np.stack([emb[t] for t in df.title.fillna("")])
ev = (month >= FIRST) & ~np.isnan(ens)
print(f"evaluation articles {ev.sum():,}; ens AUC {roc_auc_score(lab[ev], ens[ev]):.4f}", flush=True)
cols = COUNT_COLS + STATE_COLS + VEL_COLS
rng = np.random.default_rng(0)
days = np.unique(dn[ev]); gi = [np.flatnonzero(dn[ev] == d) for d in days]
boot = rng.integers(0, len(days), size=(B, len(days)))
pcs = {}  # PCA components per test month (fit on the training months), shared by all horizons
for m in sorted(set(month[ev])):
    pcs[m] = PCA(32, random_state=0).fit(E[month < m]).transform(E)

sc = {}
rows = [dict(look="before publication", n=int(ev.sum()), all=roc_auc_score(lab[ev], ens[ev]),
             all_top5=roc_auc_score(s[ev] >= t95, ens[ev]), all_spearman=spearmanr(ens[ev], y[ev])[0])]
bs = [roc_auc_score(lab[ev][sub], ens[ev][sub]) for sub in (np.concatenate([gi[j] for j in b]) for b in boot)]
rows[0]["all_lo"], rows[0]["all_hi"] = np.quantile(bs, [.025, .975])
for thin in (0, 4):
    os.environ["WB_THIN"] = str(thin)
    O = load_obs(df, lists_only=True)
    cap_day = pd.Series(pd.to_datetime(O.ts.unique())).dt.normalize().value_counts()
    ncap = (df.date.map(cap_day).fillna(0) + (df.date + pd.Timedelta(days=1)).map(cap_day).fillna(0)).to_numpy()
    for h in HS:
        Es = early_state(O, h).reindex(np.arange(n))
        seen = Es.age.notna().to_numpy()
        A0 = np.c_[X, ens, Es[cols].to_numpy(dtype=float), seen.astype(float), ncap]
        z_ = np.full(n, np.nan)
        for m in sorted(set(month[ev])):
            tr, te = month < m, ev & (month == m)
            A = np.c_[A0, pcs[m]]
            ok = ~np.isnan(A[tr]).all(0)
            z_[te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(A[tr][:, ok], y[tr]).predict(A[te][:, ok])
        sc[(thin, h)] = z_
        zz = z_[ev]
        if thin:
            r = next(r for r in rows if r.get("h") == h)
            r["all_thin4"] = roc_auc_score(lab[ev], zz); r["sighted_thin4"] = float(seen[ev].mean())
            print(f"thin4 h={h}: {r['all_thin4']:.4f}", flush=True)
            continue
        cur = np.expm1(Es.l_total.to_numpy())
        ny = ev & seen & (cur < 1400)
        d = np.array([[roc_auc_score(lab[ev][sub], zz[sub]), roc_auc_score(lab[ev][sub], ens[ev][sub])]
                      for sub in (np.concatenate([gi[j] for j in b]) for b in boot)])
        r = dict(look=f"within {h:g} h", h=h, n=int(ev.sum()), all=roc_auc_score(lab[ev], zz),
                 all_lo=np.quantile(d[:, 0], .025), all_hi=np.quantile(d[:, 0], .975),
                 gain=roc_auc_score(lab[ev], zz) - roc_auc_score(lab[ev], ens[ev]),
                 gain_lo=np.quantile(d[:, 0] - d[:, 1], .025), gain_hi=np.quantile(d[:, 0] - d[:, 1], .975),
                 all_top5=roc_auc_score(s[ev] >= t95, zz), all_spearman=spearmanr(zz, y[ev])[0],
                 sighted=float(seen[ev].mean()), already_1400=float((cur[ev & seen] >= 1400).mean()),
                 age_median=float(np.nanmedian(Es.age.to_numpy()[ev & seen])), n_not_yet=int(ny.sum()),
                 not_yet=roc_auc_score(lab[ny], z_[ny]), not_yet_sh=roc_auc_score(lab[ny], cur[ny]),
                 not_yet_pre=roc_auc_score(lab[ny], ens[ny]),
                 unseen_model=roc_auc_score(lab[ev & ~seen], z_[ev & ~seen]), unseen_pre=roc_auc_score(lab[ev & ~seen], ens[ev & ~seen]))
        for hf in ("2013H2", "2014H1", "2014H2"):
            mk = ev & np.array([(m[:4] + ("H1" if m[5:] <= "06" else "H2")) == hf for m in month])
            r["all_" + hf] = roc_auc_score(lab[mk], z_[mk]); r["pre_" + hf] = roc_auc_score(lab[mk], ens[mk])
        rows.append(r)
        print(pd.DataFrame([r]).round(3).T.to_string(header=False), flush=True)
R = pd.DataFrame(rows); R.to_csv(ROOT / "results" / "tables" / "w8_rq4_table.csv", index=False)
pd.DataFrame({"file_row": df.file_row.to_numpy()[ev], "ens": ens[ev],
              **{f"h{h:g}" + ("_thin4" if t else ""): v[ev] for (t, h), v in sc.items()}}).to_parquet(
    ROOT / "data" / "interim" / "w8_scores.parquet", index=False)
pd.set_option("display.width", 220)
print(R[["look", "n", "all", "all_lo", "all_hi", "all_thin4", "not_yet", "not_yet_sh", "not_yet_pre", "sighted",
         "already_1400", "age_median"]].round(3).to_string(index=False))
