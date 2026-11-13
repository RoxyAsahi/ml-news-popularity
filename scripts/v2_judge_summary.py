"""v19 RQ3: can a zero-shot judge (or a supervised text model) call popularity from the headline?

Reads the judge outputs saved by explore/jev/*.py (copied to explore/jev/results/) and the past-only scores of
scripts/v1_past_only_called.py, and reports with 95% bootstrap intervals (1,000 draws over pairs, or days):
  Mashable, same-day same-section pairs (4,000) and same-day pairs (4,000), real headlines:
     judge accuracy (order-debiased), order consistency, accuracy on its most confident 25% / 10%,
     past-only HistGB on the same pairs (pairs in the scored months), and the two combined (10-fold logistic)
  Mashable, judge given headline + 6 dataset features (2,000 same-day same-section pairs)
  Mashable, whole-day lineup: Spearman of the judge's 'shared most' probability with log shares within the day,
     and how often its top pick lands in the day's top 10%
  Upworthy, significantly different headline pairs of one randomized test (2,000), API judge
  Open-weights JEV-9B (int8, CPU): 400 Upworthy pairs and 400 Mashable section pairs, if available
Output: results/tables/v2_judge_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.data import load  # noqa: E402

J = ROOT / "explore" / "jev" / "results"
B = 1000
rng = np.random.default_rng(0)
df = load().reset_index(drop=True)
y = np.log1p(df.shares.to_numpy())
day = pd.to_datetime(df.publish_date).dt.normalize().to_numpy()
V1 = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv")
hgb = np.full(len(df), np.nan); hgb[V1.row] = V1.hgb
rows = []


def boot_mean(x, clusters=None):
    x = np.asarray(x, float)
    if clusters is None:
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    else:
        u, inv = np.unique(clusters, return_inverse=True)
        s, c = np.bincount(inv, x), np.bincount(inv)
        bs = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
            bs.append((w * s).sum() / (w * c).sum())
    return x.mean(), np.quantile(bs, .025), np.quantile(bs, .975)


def add(setting, scorer, metric, x, n, clusters=None):
    est, lo, hi = boot_mean(x, clusters)
    rows.append(dict(setting=setting, scorer=scorer, metric=metric, est=est, lo=lo, hi=hi, n=n))


def debias(p0, p1):
    """Probability that the first-listed option (order 0) wins, averaged over both orders."""
    return (np.asarray(p0) + 1 - np.asarray(p1)) / 2


def selective(setting, scorer, p, correct):
    conf = np.abs(p - .5); o = np.argsort(-conf, kind="stable")
    for q in (.25, .10):
        k = o[: int(q * len(p))]
        add(setting, scorer, f"acc_top{int(q * 100)}pct_conf", correct[k], len(k))


# --- Mashable pairs, real headlines ---------------------------------------------------------------
for mode in ["section", "day"]:
    P = pd.read_csv(J / f"api_mash_titles_{mode}.csv")
    t = y[P.a] > y[P.b]
    p = debias(P.p0_title, P.p1_title)
    setting = f"mashable_same_{mode}_pairs"
    add(setting, "judge_api", "acc", (p > .5) == t, len(P), day[P.a])
    add(setting, "judge_api", "order_consistency", (P.p0_title > .5) == (P.p1_title < .5), len(P))
    selective(setting, "judge_api", p, (p > .5) == t)
    pk = debias(P.p0_slugt, P.p1_slugt)
    add(setting, "judge_api_urlkeywords", "acc", (pk > .5) == t, len(P), day[P.a])
    sc = ~np.isnan(hgb[P.a]) & ~np.isnan(hgb[P.b])
    dh = hgb[P.a] - hgb[P.b]
    add(setting, "hgb_past_only", "acc", ((dh > 0) == t)[sc], int(sc.sum()), day[P.a][sc])
    add(setting, "judge_api_same_subset", "acc", ((p > .5) == t)[sc], int(sc.sum()), day[P.a][sc])
    L = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    Xc = np.c_[dh, L][sc]
    pc = cross_val_predict(LogisticRegression(), Xc, t[sc], cv=10, method="predict_proba")[:, 1]
    add(setting, "hgb_plus_judge", "acc", (pc > .5) == t[sc], int(sc.sum()), day[P.a][sc])

# --- judge given headline + dataset features -------------------------------------------------------
F = pd.read_csv(J / "api_feat_pairs.csv")
t = y[F.a] > y[F.b]
add("mashable_same_section_pairs_featprompt", "judge_api_title_plus_features", "acc",
    (debias(F.p0_feat, F.p1_feat) > .5) == t, len(F), day[F.a])
kw = df.kw_avg_avg.to_numpy()
add("mashable_same_section_pairs_featprompt", "kw_avg_avg", "acc", (kw[F.a] > kw[F.b]) == t, len(F), day[F.a])

# --- whole-day lineup -------------------------------------------------------------------------------
Lu = pd.read_csv(J / "api_lineup.csv")
Lu["y"], Lu["day"] = y[Lu.row], day[Lu.row]
rho, top, base = [], [], []
for d, g in Lu.groupby("day"):
    if len(g) < 10:
        continue
    rho.append(np.corrcoef(rankdata(g.p_top), rankdata(g.y))[0, 1])
    thr = np.quantile(g.y, .9)
    top.append(float(g.y.to_numpy()[np.argmax(g.p_top.to_numpy())] >= thr)); base.append(np.mean(g.y >= thr))
add("mashable_day_lineup", "judge_api", "rho_day", rho, len(rho))
add("mashable_day_lineup", "judge_api", "top10", top, len(top))
add("mashable_day_lineup", "random", "top10", base, len(base))

# --- Upworthy, randomized headline pairs -------------------------------------------------------------
U = pd.read_csv(J / "api_upw_pairs.csv")
pu = debias(U.p0, U.p1); tu = U.zz.to_numpy() > 0
add("upworthy_sig_pairs", "judge_api", "acc", (pu > .5) == tu, len(U))
add("upworthy_sig_pairs", "judge_api", "order_consistency", (U.p0 > .5) == (U.p1 < .5), len(U))
selective("upworthy_sig_pairs", "judge_api", pu, (pu > .5) == tu)

# --- open-weights JEV-9B --------------------------------------------------------------------------
for fname, tag in [("local_runs.csv", ""), ("local_runs_late.csv", "_scored_months")]:
    lr = J / fname
    if not lr.exists():
        continue
    D = pd.read_csv(lr).pivot_table(index=["set", "k"], columns="order", values="p_first").dropna().reset_index()
    for s, g in D.groupby("set"):
        p = debias(g[0], g[1]); kk = g.k.to_numpy()
        if s == "upw":
            truth = (U.zz.to_numpy()[kk] > 0); api = pu[kk]; setting = "upworthy_sig_pairs"
        else:
            P = pd.read_csv(J / "api_mash_titles_section.csv")
            truth = y[P.a.to_numpy()[kk]] > y[P.b.to_numpy()[kk]]
            api = debias(P.p0_title, P.p1_title)[kk]; setting = "mashable_same_section_pairs" + tag
        add(setting, "judge_open_9b", "acc", (p > .5) == truth, len(g))
        add(setting, "judge_open_9b", "order_consistency", (g[0] > .5) == (g[1] < .5), len(g))
        add(setting, "judge_api_same_subset_9b", "acc", (api > .5) == truth, len(g))
        rows.append(dict(setting=setting, scorer="open_9b_vs_api", metric="agreement",
                         est=np.mean((p > .5) == (api > .5)), lo=np.nan, hi=np.nan, n=len(g)))

# --- second wording ---------------------------------------------------------------------------------
W = J / "api_prompt2.csv"
if W.exists():
    W = pd.read_csv(W)
    for s, g in W.groupby("set"):
        p2 = debias(g.p0, g.p1); kk = g.k.to_numpy()
        if s == "upw":
            truth, p1, setting = U.zz.to_numpy()[kk] > 0, pu[kk], "upworthy_sig_pairs"
        else:
            P = pd.read_csv(J / "api_mash_titles_section.csv")
            truth = y[P.a.to_numpy()[kk]] > y[P.b.to_numpy()[kk]]
            p1, setting = debias(P.p0_title, P.p1_title)[kk], "mashable_same_section_pairs"
        add(setting, "judge_api_wording2", "acc", (p2 > .5) == truth, len(g))
        add(setting, "judge_api_wording1_same_subset", "acc", (p1 > .5) == truth, len(g))
        rows.append(dict(setting=setting, scorer="wording1_vs_wording2", metric="agreement",
                         est=np.mean((p1 > .5) == (p2 > .5)), lo=np.nan, hi=np.nan, n=len(g)))

# --- judge accuracy by quarter (Mashable section pairs) -------------------------------------------
P = pd.read_csv(J / "api_mash_titles_section.csv"); p = debias(P.p0_title, P.p1_title)
quarter = pd.to_datetime(df.publish_date).dt.to_period("Q").astype(str).to_numpy()[P.a]
for qq in sorted(set(quarter)):
    sel = quarter == qq
    add("mashable_same_section_pairs_" + qq, "judge_api", "acc", ((p > .5) == (y[P.a] > y[P.b]))[sel], int(sel.sum()))

R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / "v2_judge_summary.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
