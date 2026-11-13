"""Can a zero-shot judge call popularity from the headline? (repaired data)

Reads the judge outputs saved by explore/jev/*.py (copied to explore/jev/results/) and the past-only scores of
scripts/v1_past_only_called.py, and reports with 95% bootstrap intervals (1,000 draws over days, or over pairs):
  Mashable, same-day same-section pairs (4,000) and same-day pairs (4,000), real headlines, rows of the repaired data:
     judge accuracy (order-debiased), order consistency, accuracy on its most confident 25% / 10%, the judge on the
     URL keywords, the past-only models on the same pairs (scored months), and judge + model (10-fold logistic)
  the same by period (before the first shifted row, 2013 H2, 2014 H1, 2014 H2)
  Mashable, judge given headline + 6 dataset features (2,000 same-day same-section pairs)
  Mashable, whole-day lineup: Spearman of the judge's 'shared most' probability with log shares within the day,
     and how often its top pick lands in the day's top 10%
  Mashable, single-article question (expected share band vs true shares), early and late period
  Mashable, paraphrase tests: 250 pairs from January-May 2013 and 250 pairs from 2014, every headline reworded by an LLM
  Upworthy, significantly different headline pairs of one randomized test (2,000), API judge; the same for pairs
     without and with the test's winning (published) headline
  Open-weights JEV-9B (int8, CPU): 400 Upworthy pairs and 400 Mashable section pairs, if available
  second question wording (1,000 pairs each)
Output: results/tables/v2_judge_summary.csv
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
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
date = pd.to_datetime(df.publish_date).dt.normalize()
day = date.to_numpy()
V1 = pd.read_csv(ROOT / "data" / "interim" / "v1_scores.csv")
SC = {}
for c in ["hgb", "hgb_bge", "ens", "bge", "tfidf"]:
    SC[c] = np.full(len(df), np.nan); SC[c][V1.row] = np.where(V1.has_title, V1[c], np.nan)
row_of_url = pd.Series(np.arange(len(df)), index=df.url)[lambda s: s.index.notna()]
rows = []


def boot_mean(x, clusters=None, key=""):
    """Mean with a 95% bootstrap interval; the draws are seeded by the row's name, so adding rows changes nothing."""
    x = np.asarray(x, float); rng = np.random.default_rng(zlib.crc32(key.encode()))
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
    est, lo, hi = boot_mean(x, clusters, f"{setting}|{scorer}|{metric}")
    rows.append(dict(setting=setting, scorer=scorer, metric=metric, est=est, lo=lo, hi=hi, n=n))


def debias(p0, p1):
    """Probability that the first-listed option (order 0) wins, averaged over both orders."""
    return (np.asarray(p0) + 1 - np.asarray(p1)) / 2


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def selective(setting, scorer, p, correct):
    conf = np.abs(p - .5); o = np.argsort(-conf, kind="stable")
    for q in (.25, .10):
        k = o[: int(q * len(p))]
        add(setting, scorer, f"acc_top{int(q * 100)}pct_conf", correct[k], len(k))


PERIODS = [("before_2013-06-06", "2013-01-01", "2013-06-06"), ("2013-06-06_to_2013-12", "2013-06-06", "2014-01-01"),
           ("2014_H1", "2014-01-01", "2014-07-01"), ("2014_H2", "2014-07-01", "2015-01-01")]

# --- Mashable pairs, real headlines ---------------------------------------------------------------
for mode in ["section", "day"]:
    P = pd.read_csv(J / f"rep_mash_titles_{mode}.csv")
    a, b = P.a.to_numpy(), P.b.to_numpy()
    assert (df.url.to_numpy()[a] == P.url_a.to_numpy()).all() and (df.url.to_numpy()[b] == P.url_b.to_numpy()).all()
    t = y[a] > y[b]
    p = debias(P.p0_title, P.p1_title)
    setting = f"mashable_same_{mode}_pairs"
    add(setting, "judge_api", "acc", (p > .5) == t, len(P), day[a])
    add(setting, "judge_api", "order_consistency", (P.p0_title > .5) == (P.p1_title < .5), len(P))
    selective(setting, "judge_api", p, (p > .5) == t)
    if "p0_slugt" in P:
        pk = debias(P.p0_slugt, P.p1_slugt)
        add(setting, "judge_api_urlkeywords", "acc", (pk > .5) == t, len(P), day[a])
    for nm, lo, hi in PERIODS:
        sel = (day[a] >= np.datetime64(lo)) & (day[a] < np.datetime64(hi))
        add(f"{setting}_{nm}", "judge_api", "acc", ((p > .5) == t)[sel], int(sel.sum()), day[a][sel])
    sc = ~np.isnan(SC["hgb"][a]) & ~np.isnan(SC["hgb"][b])
    add(setting, "judge_api_same_subset", "acc", ((p > .5) == t)[sc], int(sc.sum()), day[a][sc])
    D = {c: SC[c][a] - SC[c][b] for c in SC}
    for c in SC:
        add(setting, f"{c}_past_only", "acc", ((D[c] > 0) == t)[sc], int(sc.sum()), day[a][sc])
    for c in ["hgb", "bge", "hgb_bge", "ens"]:
        hit = {}
        for nm, Xc in [(f"{c}_cv", np.c_[D[c]][sc]), (f"{c}_plus_judge", np.c_[D[c], logit(p)][sc])]:
            pc = cross_val_predict(LogisticRegression(), Xc, t[sc], cv=10, method="predict_proba")[:, 1]
            hit[nm] = ((pc > .5) == t[sc]).astype(float)
            add(setting, nm, "acc", hit[nm], int(sc.sum()), day[a][sc])
        add(setting, f"{c}_plus_judge_minus_{c}", "acc_diff", hit[f"{c}_plus_judge"] - hit[f"{c}_cv"], int(sc.sum()), day[a][sc])

# --- judge given headline + dataset features -------------------------------------------------------
F = pd.read_csv(J / "rep_feat_pairs.csv")
t = y[F.a] > y[F.b]
S = "mashable_same_section_pairs_featprompt"
add(S, "judge_api_title_plus_features", "acc", (debias(F.p0_feat, F.p1_feat) > .5) == t, len(F), day[F.a])
add(S, "judge_api_title_only_same_pairs", "acc", (debias(F.p0_title, F.p1_title) > .5) == t, len(F), day[F.a])
kw = df.kw_avg_avg.to_numpy()
add(S, "kw_avg_avg", "acc", (kw[F.a] > kw[F.b]) == t, len(F), day[F.a])

# --- whole-day lineup -------------------------------------------------------------------------------
Lu = pd.read_csv(J / "rep_lineup.csv")
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

# --- single-article question: expected share band vs true shares ------------------------------------
for fname, tag in [("rep_recall_probe.csv", "with_url_date_section"), ("rep_recall_probe_titleonly.csv", "headline_only")]:
    if (J / fname).exists():
        Rp = pd.read_csv(J / fname); Rp["band"] = Rp[["pA", "pB", "pC", "pD"]].to_numpy() @ np.arange(4)
        for per, g in Rp.groupby("period"):
            r = spearmanr(g.band, y[g.row])[0]
            bs = [spearmanr(g.band.to_numpy()[i], y[g.row.to_numpy()[i]])[0] for i in (rng.integers(0, len(g), len(g)) for _ in range(B))]
            rows.append(dict(setting=f"mashable_single_article_{tag}_{per}", scorer="judge_api", metric="spearman_band_vs_shares",
                             est=r, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n=len(g)))

# --- paraphrase tests: 250 pairs from January-May 2013 and 250 pairs from 2014, every headline reworded by an LLM ----
pooled = []
for fname, tag in [("api_paraphrase.csv", "2013_Jan-May"), ("api_paraphrase2014.csv", "2014")]:
    if not (J / fname).exists():
        continue
    Q = pd.read_csv(J / fname); a, b = row_of_url[Q.url_a].to_numpy(), row_of_url[Q.url_b].to_numpy()
    t = y[a] > y[b]; po, pp = debias(Q.p0_orig, Q.p1_orig), debias(Q.p0_para, Q.p1_para)
    H = pd.DataFrame(dict(orig=((po > .5) == t).astype(float), para=((pp > .5) == t).astype(float), same=((po > .5) == (pp > .5)).astype(float)))
    pooled.append(H)
    for H_, tg in [(H, tag)] + ([(pd.concat(pooled), "pooled")] if len(pooled) == 2 else []):
        add(f"mashable_paraphrase_{tg}", "judge_api_original", "acc", H_.orig, len(H_))
        add(f"mashable_paraphrase_{tg}", "judge_api_reworded", "acc", H_.para, len(H_))
        add(f"mashable_paraphrase_{tg}", "reworded_minus_original", "acc_diff", H_.para - H_.orig, len(H_))
        add(f"mashable_paraphrase_{tg}", "original_vs_reworded", "same_decision", H_.same, len(H_))

# --- Upworthy, randomized headline pairs -------------------------------------------------------------
U = pd.read_csv(J / "api_upw_pairs.csv")
pu = debias(U.p0, U.p1); tu = U.zz.to_numpy() > 0
add("upworthy_sig_pairs", "judge_api", "acc", (pu > .5) == tu, len(U))
add("upworthy_sig_pairs", "judge_api", "order_consistency", (U.p0 > .5) == (U.p1 < .5), len(U))
selective("upworthy_sig_pairs", "judge_api", pu, (pu > .5) == tu)
if (J / "api_upw_winner_split.csv").exists():  # pairs with / without the headline that won its test (and was published)
    for nm, g in pd.read_csv(J / "api_upw_winner_split.csv").groupby("set"):
        add(f"upworthy_sig_pairs_{nm}", "judge_api", "acc", (debias(g.p0, g.p1) > .5) == (g.zz.to_numpy() > 0), len(g))

# --- open-weights JEV-9B --------------------------------------------------------------------------
M = pd.read_csv(J / "rep_mash_titles_section.csv"); pm = debias(M.p0_title, M.p1_title); tm = y[M.a] > y[M.b]
for fname, keep in [("local_runs.csv", "upw"), ("rep_local_runs.csv", "mash")]:
    if not (J / fname).exists():
        continue
    D = pd.read_csv(J / fname).pivot_table(index=["set", "k"], columns="order", values="p_first").dropna().reset_index()
    g = D[D.set == keep]
    if len(g) == 0:
        continue
    p = debias(g[0], g[1]); kk = g.k.to_numpy()
    truth, api, setting = (tu[kk], pu[kk], "upworthy_sig_pairs") if keep == "upw" else (tm[kk], pm[kk], "mashable_same_section_pairs")
    add(setting, "judge_open_9b", "acc", (p > .5) == truth, len(g))
    add(setting, "judge_open_9b", "order_consistency", (g[0] > .5) == (g[1] < .5), len(g))
    add(setting, "judge_api_same_subset_9b", "acc", (api > .5) == truth, len(g))
    rows.append(dict(setting=setting, scorer="open_9b_vs_api", metric="agreement",
                     est=np.mean((p > .5) == (api > .5)), lo=np.nan, hi=np.nan, n=len(g)))

# --- second wording ---------------------------------------------------------------------------------
W = J / "rep_prompt2.csv"
if W.exists():
    W = pd.read_csv(W)
    for s, g in W.groupby("set"):
        p2 = debias(g.p0, g.p1); kk = g.k.to_numpy()
        truth, p1, setting = (tu[kk], pu[kk], "upworthy_sig_pairs") if s == "upw" else (tm[kk], pm[kk], "mashable_same_section_pairs")
        add(setting, "judge_api_wording2", "acc", (p2 > .5) == truth, len(g))
        add(setting, "judge_api_wording1_same_subset", "acc", (p1 > .5) == truth, len(g))
        rows.append(dict(setting=setting, scorer="wording1_vs_wording2", metric="agreement",
                         est=np.mean((p1 > .5) == (p2 > .5)), lo=np.nan, hi=np.nan, n=len(g)))

R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / "v2_judge_summary.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
