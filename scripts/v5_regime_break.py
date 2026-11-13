"""v19: when does the headline carry signal? A break in mid-2013.

The main past-only evaluation starts in 2013-07. Here the rolling origin starts in 2013-03 (train on all earlier
months, at least January and February 2013) and accuracy on same-day same-section pairs is reported per month for
  tfidf, bge       ridge on the real headline (TF-IDF / bge embedding)
  hgb_history      HistGB on the 13 history features
  hgb_all          HistGB on all dataset features
  hgb_all_bge      HistGB on all features + 32 PCA components of the headline embedding
and, per quarter, for the zero-shot judge on its 4,000 judged pairs.
Transfer check for the bge ridge: train on one period, test on another (does the early headline signal carry
forward, and does a later model find it backwards?).
95% intervals: bootstrap over days (1,000 draws) for pooled windows.
The break is dated from the daily within-day correlation of a headline model trained on January-March 2013; the
same is done with the URL keywords (dataset only). Also: judge before/after the break, and the paraphrase test.
Output: results/tables/v5_regime_monthly.csv, v5_regime_windows.csv, v5_regime_transfer.csv, v5_regime_daily.csv,
        v5_regime_break.csv,
        results/figures/v19_fig_regime.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import load  # noqa: E402
from newspop.stories import channel  # noqa: E402
from newspop.titles import aligned_titles, bge_embeddings  # noqa: E402

T = ROOT / "results" / "tables"; J = ROOT / "explore" / "jev" / "results"
B = 1000; rng = np.random.default_rng(0)
df = load().reset_index(drop=True); n = len(df)
y = np.log1p(df.shares.to_numpy())
X = df.drop(columns=[c for c in ["url", "shares", "publish_date", "timedelta"] if c in df.columns]).select_dtypes("number")
hist = [c for c in X.columns if c.startswith(("kw_", "self_reference_"))]
title = aligned_titles(df).fillna(""); ok = (title != "").to_numpy(); E = bge_embeddings(title)
day = pd.to_datetime(df.publish_date).dt.normalize(); dn = (day - day.min()).dt.days.to_numpy()
month = day.dt.to_period("M").astype(str).to_numpy(); chan = channel(df)
MODELS = ["tfidf", "bge", "hgb_history", "hgb_all", "hgb_all_bge"]
FIRST = "2013-03"


def hgb():
    return HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS)


def bge_ridge(tr):
    sc = StandardScaler().fit(E[tr]); m = Ridge(alpha=1000).fit(sc.transform(E[tr]), y[tr])
    return lambda te: m.predict(sc.transform(E[te]))


S = {k: np.full(n, np.nan) for k in MODELS}
for m in sorted(set(month)):
    if m < FIRST:
        continue
    tr, te = np.flatnonzero((month < m) & ok), np.flatnonzero((month == m) & ok)
    v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(title[tr])
    S["tfidf"][te] = Ridge(alpha=10).fit(v.transform(title[tr]), y[tr]).predict(v.transform(title[te]))
    S["bge"][te] = bge_ridge(tr)(te)
    S["hgb_history"][te] = hgb().fit(X.loc[tr, hist], y[tr]).predict(X.loc[te, hist])
    S["hgb_all"][te] = hgb().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    pca = PCA(32, random_state=0).fit(E[tr]); XE = np.c_[X.to_numpy(), pca.transform(E)]
    S["hgb_all_bge"][te] = hgb().fit(XE[tr], y[tr]).predict(XE[te])
    print(m, flush=True)


def day_counts(s, idx):
    """Concordant and total same-day same-section pairs per day (dict day -> (c, t))."""
    out = {}
    for (d, _), ix in pd.DataFrame({"i": idx, "d": dn[idx], "c": chan[idx]}).groupby(["d", "c"]).i:
        ix = ix.to_numpy()
        if len(ix) < 2:
            continue
        a, b = np.triu_indices(len(ix), 1)
        dy = np.sign(y[ix[a]] - y[ix[b]]); ds = np.sign(s[ix[a]] - s[ix[b]]); k = dy != 0
        c, t = out.get(d, (0.0, 0.0))
        out[d] = (c + np.sum(dy[k] == ds[k]) + 0.5 * np.sum(ds[k] == 0), t + k.sum())
    return out


def acc_ci(counts):
    ds = sorted(counts); c = np.array([counts[d][0] for d in ds]); t = np.array([counts[d][1] for d in ds])
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ds), len(ds)), minlength=len(ds)); bs.append((w * c).sum() / (w * t).sum())
    return c.sum() / t.sum(), np.quantile(bs, .025), np.quantile(bs, .975), int(t.sum())


rows, wrows = [], []
WINDOWS = {"2013-03..05": ("2013-03", "2013-05"), "2013-06": ("2013-06", "2013-06"),
           "2013-07..12": ("2013-07", "2013-12"), "2014": ("2014-01", "2014-12")}
for k in MODELS:
    for m in sorted(set(month)):
        if m < FIRST:
            continue
        est, lo, hi, npairs = acc_ci(day_counts(S[k], np.flatnonzero((month == m) & ok)))
        rows.append(dict(scorer=k, month=m, acc=est, lo=lo, hi=hi, pairs=npairs))
    for w, (a, b) in WINDOWS.items():
        est, lo, hi, npairs = acc_ci(day_counts(S[k], np.flatnonzero((month >= a) & (month <= b) & ok)))
        wrows.append(dict(scorer=k, window=w, acc=est, lo=lo, hi=hi, pairs=npairs))
M = pd.DataFrame(rows); M.to_csv(T / "v5_regime_monthly.csv", index=False)

# zero-shot judge on its judged pairs, by window and quarter
P = pd.read_csv(J / "api_mash_titles_section.csv"); a, b = P.a.to_numpy(), P.b.to_numpy()
pj = (P.p0_title + 1 - P.p1_title).to_numpy() / 2; cj = ((pj > .5) == (y[a] > y[b])).astype(float)


def pair_ci(sel):
    u, inv = np.unique(dn[a][sel], return_inverse=True); c = np.bincount(inv, cj[sel]); t = np.bincount(inv)
    bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)); bs.append((w * c).sum() / (w * t).sum())
    return cj[sel].mean(), np.quantile(bs, .025), np.quantile(bs, .975), int(sel.sum())


for w, (lo_m, hi_m) in {"2013-01..02": ("2013-01", "2013-02"), **WINDOWS}.items():
    est, lo, hi, npairs = pair_ci((month[a] >= lo_m) & (month[a] <= hi_m))
    wrows.append(dict(scorer="judge_api", window=w, acc=est, lo=lo, hi=hi, pairs=npairs))
W = pd.DataFrame(wrows); W.to_csv(T / "v5_regime_windows.csv", index=False)
quarter = day.dt.to_period("Q").astype(str).to_numpy()
JQ = pd.DataFrame([dict(quarter=q, **dict(zip(["acc", "lo", "hi", "pairs"], pair_ci(quarter[a] == q)))) for q in sorted(set(quarter))])

# transfer of the headline (bge) model across periods
PER = {"2013 Jan-Mar": ("2013-01", "2013-03"), "2013 Apr-May": ("2013-04", "2013-05"), "2013 Aug-Dec": ("2013-08", "2013-12"),
       "2014 Jan-Jun": ("2014-01", "2014-06"), "2014 Jul-Dec": ("2014-07", "2014-12")}
ix = {k: np.flatnonzero((month >= lo_m) & (month <= hi_m) & ok) for k, (lo_m, hi_m) in PER.items()}
trows = []
for tr_name, te_name in [("2013 Jan-Mar", "2013 Apr-May"), ("2013 Jan-Mar", "2013 Aug-Dec"), ("2013 Jan-Mar", "2014 Jul-Dec"),
                         ("2013 Aug-Dec", "2013 Apr-May"), ("2014 Jan-Jun", "2013 Apr-May"), ("2014 Jan-Jun", "2014 Jul-Dec"),
                         ("2013 Aug-Dec", "2014 Jan-Jun")]:
    s = np.zeros(n); s[ix[te_name]] = bge_ridge(ix[tr_name])(ix[te_name])
    est, lo, hi, npairs = acc_ci(day_counts(s, ix[te_name]))
    trows.append(dict(train=tr_name, test=te_name, acc=est, lo=lo, hi=hi, pairs=npairs))
TR = pd.DataFrame(trows); TR.to_csv(T / "v5_regime_transfer.csv", index=False)

# --- exact date: within-day correlation of a headline model trained on Jan-Mar 2013 with log shares, per day ------
BREAK = pd.Timestamp("2013-06-07")
tr0 = np.flatnonzero((month <= "2013-03") & ok)
s0 = np.full(n, np.nan); s0[ok] = bge_ridge(tr0)(np.flatnonzero(ok))
slug = df.url.str.extract(r"/\d{4}/\d{2}/\d{2}/([^/]+)/?\s*$")[0].str.replace(r"-\d+$", "", regex=True).str.replace("-", " ").fillna("")
vs = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(slug[month <= "2013-03"])
s_slug = Ridge(alpha=10).fit(vs.transform(slug[month <= "2013-03"]), y[month <= "2013-03"]).predict(vs.transform(slug))
daily = []
for dd in pd.date_range("2013-04-01", "2013-08-31"):
    ixd = np.flatnonzero((day == dd).to_numpy() & ok)
    if len(ixd) > 5:
        daily.append(dict(day=dd, n=len(ixd), corr_headline=np.corrcoef(s0[ixd], y[ixd])[0, 1], corr_slug=np.corrcoef(s_slug[ixd], y[ixd])[0, 1]))
Dd = pd.DataFrame(daily); Dd.to_csv(T / "v5_regime_daily.csv", index=False)
cs = Dd.corr_headline.to_numpy()
best = min(range(20, len(cs) - 20), key=lambda i: ((cs[:i] - cs[:i].mean()) ** 2).sum() + ((cs[i:] - cs[i:].mean()) ** 2).sum())
print(f"change point of the daily headline correlation (least squares, one break): {Dd.day[best].date()}; "
      f"mean before {cs[:best].mean():.3f}, after {cs[best:].mean():.3f}")
bef, aft = Dd[Dd.day < BREAK], Dd[Dd.day >= BREAK]
print(f"dataset-only URL keywords, mean daily correlation: before {bef.corr_slug.mean():.3f}, after {aft.corr_slug.mean():.3f}")
brows = [dict(what="change_point", value=str(Dd.day[best].date())),
         dict(what="daily_corr_headline_before", value=cs[:best].mean()), dict(what="daily_corr_headline_after", value=cs[best:].mean()),
         dict(what="daily_corr_urlkeywords_before", value=bef.corr_slug.mean()), dict(what="daily_corr_urlkeywords_after", value=aft.corr_slug.mean())]
# judge before / after the break date (its 4,000 judged pairs), and on the day lineups
for nm, sel in [("judge_pairs_before", day.to_numpy()[a] < BREAK), ("judge_pairs_after", day.to_numpy()[a] >= BREAK)]:
    est, lo, hi, npairs = pair_ci(sel); brows.append(dict(what=nm, value=est, lo=lo, hi=hi, n=npairs))
Lu = pd.read_csv(J / "api_lineup.csv"); Lu["y"], Lu["day"] = y[Lu.row], day.to_numpy()[Lu.row]
from scipy.stats import rankdata  # noqa: E402
for nm, sub in [("before", Lu[Lu.day < BREAK]), ("after", Lu[Lu.day >= BREAK])]:
    rho, top = [], []
    for _, g in sub.groupby("day"):
        if len(g) >= 10:
            rho.append(np.corrcoef(rankdata(g.p_top), rankdata(g.y))[0, 1])
            top.append(float(g.y.to_numpy()[np.argmax(g.p_top.to_numpy())] >= np.quantile(g.y, .9)))
    rho, top = np.array(rho), np.array(top)
    for mn, x in [("rho_day", rho), ("top10", top)]:
        bsx = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
        brows.append(dict(what=f"judge_lineup_{mn}_{nm}", value=x.mean(), lo=np.quantile(bsx, .025), hi=np.quantile(bsx, .975), n=len(x)))
# paraphrase test (explore/jev/api_paraphrase.py)
PPh = pd.read_csv(J / "api_paraphrase.csv"); tt = y[PPh.a] > y[PPh.b]
po, pp = (PPh.p0_orig + 1 - PPh.p1_orig) / 2, (PPh.p0_para + 1 - PPh.p1_para) / 2
for nm, c in [("paraphrase_original_acc", ((po > .5) == tt)), ("paraphrase_reworded_acc", ((pp > .5) == tt)),
              ("paraphrase_same_decision", (po > .5) == (pp > .5))]:
    c = np.asarray(c, float); bsx = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(B)]
    brows.append(dict(what=nm, value=c.mean(), lo=np.quantile(bsx, .025), hi=np.quantile(bsx, .975), n=len(c)))
BR = pd.DataFrame(brows); BR.to_csv(T / "v5_regime_break.csv", index=False)

pd.set_option("display.width", 200)
print(M.pivot(index="month", columns="scorer", values="acc").round(3).to_string())
print(BR.to_string(index=False)); print(W.round(3).to_string(index=False)); print(JQ.round(3).to_string(index=False)); print(TR.round(3).to_string(index=False))

# figure
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED})
fig, ax = plt.subplots(figsize=(3.4, 2.6))
months = sorted(set(M.month)); xi = {m: i for i, m in enumerate(months)}
for k, lab, col, ls in [("bge", "headline embedding (ridge)", ORANGE, "-"), ("hgb_history", "history features (HistGB)", AQUA, "-"),
                        ("hgb_all", "all features (HistGB)", BLUE, "-")]:
    g = M[M.scorer == k]; x = [xi[m] for m in g.month]
    ax.plot(x, 100 * g.acc, color=col, ls=ls, lw=1.5, label=lab)
    ax.fill_between(x, 100 * g.lo, 100 * g.hi, color=col, alpha=0.13, lw=0)
qx = {q: np.mean([xi[m] for m in months if str(pd.Period(m, "M").asfreq("Q")) == q] or [np.nan]) for q in JQ.quarter}
jq = JQ[JQ.quarter >= "2013Q2"]
ax.errorbar([qx[q] for q in jq.quarter], 100 * jq.acc, yerr=[100 * (jq.acc - jq.lo), 100 * (jq.hi - jq.acc)], fmt="s", ms=3.5,
            color=INK, lw=0.9, label="JEV judge, headline (quarters)")
ax.axhline(50, color=MUTED, lw=0.8, ls="--")
ax.axvline(xi["2013-06"] - 0.5 + 6 / 30, color=INK, lw=0.8, ls=":"); ax.text(xi["2013-06"] - 0.1, 41.2, "7 June 2013", fontsize=6, color=INK)
ticks = [m for m in months if m.endswith(("-03", "-07", "-11"))]
ax.set_xticks([xi[m] for m in ticks], [pd.Period(m, "M").strftime("%b '%y") for m in ticks], fontsize=6.5)
ax.set_ylabel("% correct, same day & section pairs"); ax.set_ylim(40, 74)
ax.legend(frameon=False, fontsize=6, loc="upper right", handlelength=1.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "v19_fig_regime.png", dpi=300); print("saved")
