"""Does the repaired url column behave like the part of the file that was never shifted?

Everything is computed twice: on the file as distributed (`load(repaired=False)`: each row keeps the URL written in
it, dates parsed from that URL, which is what every analysis before the repair used) and on the repaired data
(`load()`). Three checks that the alignment (scripts/a1_realign_urls.py) never used:
  1. outcome: a headline model (ridge on bge embeddings) trained on January-March 2013 only, where the file is
     aligned, scores later articles; accuracy on same-day same-section pairs by period
  2. a feature: the section (data_channel_is_*) predicted from the headline by a classifier trained on
     January-March 2013; accuracy by period
  3. rolling origin from March 2013 (train on all earlier months): headline models month by month, with the
     feature models on the repaired data for scale
and the zero-shot judge on the pairs judged before the repair (as distributed), by period; and the correlation of\nlog shares between same-story articles (URL-slug twins), an analysis that joins URLs to shares.
95% intervals: bootstrap over days (1,000 draws).
Output: results/tables/a2_validation_periods.csv, a2_validation_monthly.csv, results/figures/fig_alignment.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newspop.anatomy import HGB_PARAMS  # noqa: E402
from newspop.data import REALIGN_CSV, load, load_raw  # noqa: E402
from newspop.stories import channel, twin_pairs  # noqa: E402
from newspop.titles import aligned_titles, bge_embeddings  # noqa: E402

T = ROOT / "results" / "tables"; J = ROOT / "explore" / "jev" / "results"
B = 1000; rng = np.random.default_rng(0)
PERIODS = [("2013-04-01..06-05", "2013-04-01", "2013-06-06"), ("2013-06-06..12-31", "2013-06-06", "2014-01-01"),
           ("2014 H1", "2014-01-01", "2014-07-01"), ("2014 H2", "2014-07-01", "2015-01-01")]


class Data:
    def __init__(self, repaired: bool):
        self.df = df = load(repaired=repaired).reset_index(drop=True)
        self.n = len(df); self.y = np.log1p(df.shares.to_numpy())
        title = aligned_titles(df).fillna(""); self.ok = (title != "").to_numpy(); self.E = bge_embeddings(title)
        self.date = pd.to_datetime(df.publish_date).dt.normalize()
        self.dn = (self.date - pd.Timestamp("2013-01-01")).dt.days.to_numpy()
        self.month = self.date.dt.to_period("M").astype(str).to_numpy(); self.chan = np.asarray(channel(df))

    def between(self, lo, hi):
        return ((self.date >= lo) & (self.date < hi)).to_numpy()

    def day_counts(self, s, idx):
        """Concordant and total same-day same-section pairs per day."""
        out = {}
        for (d, _), ix in pd.DataFrame({"i": idx, "d": self.dn[idx], "c": self.chan[idx]}).groupby(["d", "c"]).i:
            ix = ix.to_numpy()
            if len(ix) < 2:
                continue
            a, b = np.triu_indices(len(ix), 1)
            dy = np.sign(self.y[ix[a]] - self.y[ix[b]]); ds = np.sign(s[ix[a]] - s[ix[b]]); k = dy != 0
            c, t = out.get(d, (0.0, 0.0))
            out[d] = (c + np.sum(dy[k] == ds[k]) + 0.5 * np.sum(ds[k] == 0), t + k.sum())
        return out

    def bge_ridge(self, tr):
        sc = StandardScaler().fit(self.E[tr]); m = Ridge(alpha=1000).fit(sc.transform(self.E[tr]), self.y[tr])
        return lambda te: m.predict(sc.transform(self.E[te]))


def ci(c, t):
    c, t = np.asarray(c, float), np.asarray(t, float); bs = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(c), len(c)), minlength=len(c)); bs.append((w * c).sum() / (w * t).sum())
    return c.sum() / t.sum(), np.quantile(bs, .025), np.quantile(bs, .975), int(t.sum())


def acc_ci(counts):
    ds = sorted(counts)
    return ci([counts[d][0] for d in ds], [counts[d][1] for d in ds])


V = {"as_distributed": Data(False), "repaired": Data(True)}
prow, mrow = [], []
for vname, D in V.items():
    # 1. outcome check and 2. section check, both trained on January-March 2013
    tr = np.flatnonzero((D.month <= "2013-03") & D.ok)
    s = np.full(D.n, np.nan); s[D.ok] = D.bge_ridge(tr)(np.flatnonzero(D.ok))
    sc = StandardScaler().fit(D.E[tr]); clf = LogisticRegression(C=0.01, max_iter=500).fit(sc.transform(D.E[tr]), D.chan[tr])
    hit = np.zeros(D.n); hit[D.ok] = clf.predict(sc.transform(D.E[D.ok])) == D.chan[D.ok]
    for pname, lo, hi in PERIODS:
        idx = np.flatnonzero(D.between(lo, hi) & D.ok)
        est, l, h, npairs = acc_ci(D.day_counts(s, idx))
        prow.append(dict(data=vname, check="headline_model_JanMar_pair_acc", period=pname, est=est, lo=l, hi=h, n=npairs))
        g = pd.DataFrame({"d": D.dn[idx], "h": hit[idx]}).groupby("d").h.agg(["sum", "count"])
        est, l, h, nn = ci(g["sum"], g["count"])
        prow.append(dict(data=vname, check="section_from_headline_JanMar_acc", period=pname, est=est, lo=l, hi=h, n=nn))
    idx = np.flatnonzero((D.month >= "2013-07") & D.ok)
    est, l, h, npairs = acc_ci(D.day_counts(s, idx))
    prow.append(dict(data=vname, check="headline_model_JanMar_pair_acc", period="2013-07..2014-12", est=est, lo=l, hi=h, n=npairs))
    # 3. rolling origin from March 2013
    X = D.df.drop(columns=[c for c in ["url", "url_in_file", "shares", "publish_date", "timedelta"] if c in D.df.columns]).select_dtypes("number")
    models = ["bge"] + (["hgb_all", "hgb_all_bge"] if vname == "repaired" else [])
    S = {k: np.full(D.n, np.nan) for k in models}
    for m in sorted(set(D.month)):
        if m < "2013-03":
            continue
        tr, te = np.flatnonzero((D.month < m) & D.ok), np.flatnonzero((D.month == m) & D.ok)
        S["bge"][te] = D.bge_ridge(tr)(te)
        if vname == "repaired":
            S["hgb_all"][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(X.iloc[tr], D.y[tr]).predict(X.iloc[te])
            XE = np.c_[X.to_numpy(), PCA(32, random_state=0).fit(D.E[tr]).transform(D.E)]
            S["hgb_all_bge"][te] = HistGradientBoostingRegressor(random_state=0, **HGB_PARAMS).fit(XE[tr], D.y[tr]).predict(XE[te])
        print(vname, m, flush=True)
    for k in models:
        for m in sorted(set(D.month)):
            if m >= "2013-03":
                est, l, h, npairs = acc_ci(D.day_counts(S[k], np.flatnonzero((D.month == m) & D.ok)))
                mrow.append(dict(data=vname, scorer=k, month=m, acc=est, lo=l, hi=h, pairs=npairs))
        for pname, lo, hi in [("2013-03-01..06-05", "2013-03-01", "2013-06-06")] + PERIODS[1:] + [("2013-07..2014-12", "2013-07-01", "2015-01-01")]:
            est, l, h, npairs = acc_ci(D.day_counts(S[k], np.flatnonzero(D.between(lo, hi) & D.ok)))
            prow.append(dict(data=vname, check=f"rolling_{k}_pair_acc", period=pname, est=est, lo=l, hi=h, n=npairs))
    # what a random split shows: 5-fold cross-validated headline ridge over all articles, AUC for popular
    idx = np.flatnonzero(D.ok); oof = np.zeros(len(idx))
    for tr_, te_ in KFold(5, shuffle=True, random_state=0).split(idx):
        oof[te_] = D.bge_ridge(idx[tr_])(idx[te_])
    prow.append(dict(data=vname, check="random_5fold_headline_auc", period="all", est=roc_auc_score(D.df.shares.to_numpy()[idx] >= 1400, oof), lo=np.nan, hi=np.nan, n=len(idx)))
    # spread of the log-share difference over all same-day same-section pairs
    ss = nn = 0.0
    for _, v in pd.DataFrame({"y": D.y, "d": D.dn, "c": D.chan}).groupby(["d", "c"]).y:
        v = v.to_numpy()
        if len(v) > 1:
            a, b = np.triu_indices(len(v), 1); ss += np.sum((v[a] - v[b]) ** 2); nn += len(a)
    prow.append(dict(data=vname, check="sd_log_share_difference_same_day_section", period="all", est=np.sqrt(ss / nn), lo=np.nan, hi=np.nan, n=int(nn)))

# offset profile: give each row the headline found d rows away from its estimated URL (d = 0 is the repair) and
# score it with the January-March headline model; only the estimated offset should carry signal. Also: the
# repaired rows without those on which both title fingerprints disagree.
raw = load_raw().reset_index(drop=True); A = pd.read_csv(REALIGN_CSV); N = len(raw)
Ef = bge_embeddings(aligned_titles(raw).fillna("")); has_f = aligned_titles(raw).notna().to_numpy()
D = V["repaired"]; pos = pd.Series(np.arange(N), index=raw.url); frow = pos[D.df.url_in_file].to_numpy(); kk = A.offset.to_numpy()[frow]
tr = np.flatnonzero((D.month <= "2013-03") & D.ok); sc0 = StandardScaler().fit(D.E[tr]); m0 = Ridge(alpha=1000).fit(sc0.transform(D.E[tr]), D.y[tr])
late = (D.month >= "2013-07") & (kk > 0)
for d in (-3, -2, -1, 0, 1, 2, 3):
    j = frow + kk + d; okd = late & (j >= 0) & (j < N); okd[okd] &= has_f[j[okd]]
    s = np.full(D.n, np.nan); s[okd] = m0.predict(sc0.transform(Ef[j[okd]]))
    est, l, h, npairs = acc_ci(D.day_counts(s, np.flatnonzero(okd)))
    prow.append(dict(data="repaired", check=f"offset_profile_d={d:+d}_pair_acc", period="2013-07..2014-12", est=est, lo=l, hi=h, n=npairs))
both_bad = (A.title_length_agrees.to_numpy()[frow] == 0) & (A.title_sentiment_agrees.to_numpy()[frow] == 0)
s = np.full(D.n, np.nan); s[D.ok] = m0.predict(sc0.transform(D.E[D.ok]))
for nm, keep in [("all_rows", D.ok), ("without_double_disagreement", D.ok & ~both_bad)]:
    est, l, h, npairs = acc_ci(D.day_counts(s, np.flatnonzero(keep & (D.month >= "2013-07"))))
    prow.append(dict(data="repaired", check=f"headline_model_JanMar_pair_acc_{nm}", period="2013-07..2014-12", est=est, lo=l, hi=h, n=npairs))
print("rows on which both fingerprints disagree:", int((both_bad & D.ok).sum()), f"({(both_bad & D.ok).sum() / D.ok.sum():.4f})")

# zero-shot judge on the pairs judged before the repair (rows of the file as distributed)
D = V["as_distributed"]; P = pd.read_csv(J / "as_distributed" / "api_mash_titles_section.csv"); a, b = P.a.to_numpy(), P.b.to_numpy()
cj = (((P.p0_title + 1 - P.p1_title).to_numpy() / 2 > .5) == (D.y[a] > D.y[b])).astype(float)
for pname, lo, hi in [("2013-01-01..06-05", "2013-01-01", "2013-06-06")] + PERIODS[1:]:
    sel = D.between(lo, hi)[a]; g = pd.DataFrame({"d": D.dn[a][sel], "c": cj[sel]}).groupby("d").c.agg(["sum", "count"])
    est, l, h, nn = ci(g["sum"], g["count"])
    prow.append(dict(data="as_distributed", check="judge_api_pair_acc", period=pname, est=est, lo=l, hi=h, n=nn))
# what the shift did to an analysis that joins URLs to shares: same-story pairs (URL-slug cosine >= 0.5, at most
# 7 days apart) and the correlation of their log shares
for vname, D in V.items():
    d2 = D.df[D.df.url.notna()].reset_index(drop=True); y2 = np.log1p(d2.shares.to_numpy()); P2 = twin_pairs(d2, min_sim=0.5)
    i, j = P2.i.to_numpy(), P2.j.to_numpy(); r = np.corrcoef(y2[i], y2[j])[0, 1]
    bs = [np.corrcoef(y2[i[k]], y2[j[k]])[0, 1] for k in (rng.integers(0, len(i), len(i)) for _ in range(B))]
    prow.append(dict(data=vname, check="same_story_pairs_corr_log_shares", period="all", est=r, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), n=len(i)))
PR = pd.DataFrame(prow); PR.to_csv(T / "a2_validation_periods.csv", index=False)
M = pd.DataFrame(mrow); M.to_csv(T / "a2_validation_monthly.csv", index=False)
pd.set_option("display.width", 220)
print(PR.round(4).to_string(index=False))
print(M.assign(k=M.data + ":" + M.scorer).pivot(index="month", columns="k", values="acc").round(3).to_string())

# figure: offset through the file, fingerprint agreement, and the headline model before / after the repair
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED})
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw=dict(width_ratios=[1, 1.25]))
O = pd.read_csv(T / "a1_offset_by_month.csv"); x0 = np.arange(len(O))
ax0.step(x0, O["max"], where="mid", color=INK, lw=1.2)
ax0.set_yscale("symlog", linthresh=1); ax0.set_yticks([0, 1, 2, 10, 50, 150]); ax0.set_yticklabels(["0", "1", "2", "10", "50", "150"])
ax0.set_ylabel("rows between a row's content and its URL")
ax0.set_xticks(x0[::3]); ax0.set_xticklabels([m[2:] for m in O.month[::3]], rotation=45, ha="right")
ax0.set_title("(a) offset of the url column", fontsize=7.5, loc="left")
months = sorted(set(M.month)); xi = {m: i for i, m in enumerate(months)}
for data, k, lab, col, ls in [("as_distributed", "bge", "headline, file as distributed", MUTED, "--"), ("repaired", "bge", "headline, repaired", ORANGE, "-"),
                              ("repaired", "hgb_all", "58 features, repaired", BLUE, "-"), ("repaired", "hgb_all_bge", "features + headline, repaired", AQUA, "-")]:
    g = M[(M.data == data) & (M.scorer == k)]
    ax1.plot([xi[m] for m in g.month], g.acc, ls, color=col, lw=1.2, marker="o", ms=2, label=lab)
ax1.axhline(.5, color=MUTED, lw=.6)
ax1.axvline(xi["2013-06"] - .3, color=INK, lw=.6, ls=":"); ax1.text(xi["2013-06"] - .55, .505, "first shifted row\n6 June 2013", fontsize=6, va="bottom", ha="right")
ax1.set_xticks(range(0, len(months), 3)); ax1.set_xticklabels([m[2:] for m in months[::3]], rotation=45, ha="right")
ax1.set_ylabel("accuracy, same-day same-section pairs"); ax1.set_ylim(.47, .66); ax1.legend(frameon=False, fontsize=6, ncol=2, loc="center", bbox_to_anchor=(0.62, 0.40))
ax1.set_title("(b) rolling origin: train on all earlier months", fontsize=7.5, loc="left")
for a_ in (ax0, ax1):
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "fig_alignment.png", dpi=300)
print("saved")
