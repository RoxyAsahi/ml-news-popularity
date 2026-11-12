"""E7 (RQ4) - Do articles that cluster together in topic + sentiment space differ in popularity?

1. KMeans (k = 2..10, silhouette) and Gaussian mixtures (BIC) on the 5 LDA topic weights
   plus 16 sentiment features, standardised. No share information is used.
2. Per-cluster popularity: share of articles with >= 1400 shares, median shares,
   chi-square test and Cramer's V.
3. Profile of each cluster: dominant LDA topic, channel mix, mean polarity.
4. Isolation Forest on the same space: are the 2% most unusual articles more often popular?

Run from the repo root:  python scripts/e7_clustering.py
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from newspop.data import ROOT, feature_groups, load, make_label

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
SEED = 0


def cramers_v(table):
    chi2, p, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    return chi2, p, np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def main():
    df = load()
    df = df[df.empty_content == 0].reset_index(drop=True)   # no text -> no topic/sentiment
    y = make_label(df.shares, 1400)
    g = feature_groups(df)
    cols = g["lda"] + g["sentiment"]
    Z = StandardScaler().fit_transform(df[cols])
    rng = np.random.RandomState(SEED)
    sample = rng.choice(len(Z), 8000, replace=False)

    # ---- model selection
    sel = []
    for k in range(2, 11):
        km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        gm = GaussianMixture(k, covariance_type="full", random_state=SEED).fit(Z)
        sel.append({"k": k, "silhouette": silhouette_score(Z[sample], km.labels_[sample]),
                    "inertia": km.inertia_, "gmm_bic": gm.bic(Z)})
        print(sel[-1])
    sel = pd.DataFrame(sel)
    sel.to_csv(TAB / "e7_model_selection.csv", index=False)
    k_best = int(sel.loc[sel.silhouette.idxmax(), "k"])
    # silhouette is low (~0.1) for every k, so also report a small solution for comparison
    k_rich = 5
    print("best k by silhouette:", k_best)

    results = []
    for k in sorted({k_best, k_rich}):
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
        tab = pd.crosstab(lab, y)
        chi2, p, v = cramers_v(tab)
        prof = df.assign(cluster=lab, popular=y).groupby("cluster").agg(
            n=("shares", "size"), popular_rate=("popular", "mean"), median_shares=("shares", "median"),
            **{f"mean_{c}": (c, "mean") for c in g["lda"] + ["global_sentiment_polarity", "global_subjectivity"]},
            **{f"share_{c.replace('data_channel_is_', '')}": (c, "mean") for c in g["channel"]})
        prof["dominant_topic"] = prof[[f"mean_{c}" for c in g["lda"]]].idxmax(axis=1).str.replace("mean_", "")
        prof.to_csv(TAB / f"e7_profile_k{k}.csv")
        print(f"\nk={k}: chi2={chi2:.1f}, p={p:.1e}, Cramer's V={v:.3f}")
        print(prof[["n", "popular_rate", "median_shares", "dominant_topic", "mean_global_sentiment_polarity",
                    "share_entertainment", "share_world", "share_tech", "share_bus"]].round(3).to_string())
        results.append({"k": k, "chi2": chi2, "p": p, "cramers_v": v})
        if k == k_best:                      # the figure shows the silhouette-chosen k
            lab_rich, prof_rich = lab, prof

    # same test for the editorial channel, as a reference point
    chan = df[g["channel"]].idxmax(axis=1).where(df[g["channel"]].sum(axis=1) > 0, "none")
    chi2, p, v = cramers_v(pd.crosstab(chan, y))
    results.append({"k": "channel (reference)", "chi2": chi2, "p": p, "cramers_v": v})
    pd.DataFrame(results).to_csv(TAB / "e7_association.csv", index=False)
    print("\nchannel reference: Cramer's V =", round(v, 3))

    # ---- anomalies
    iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=SEED).fit(Z)
    out = iso.predict(Z) == -1
    an = pd.DataFrame({"group": ["outlier", "inlier"],
                       "n": [out.sum(), (~out).sum()],
                       "popular_rate": [y[out].mean(), y[~out].mean()],
                       "median_shares": [df.shares[out].median(), df.shares[~out].median()],
                       "top5pct_rate": [(df.shares[out] >= df.shares.quantile(.95)).mean(),
                                        (df.shares[~out] >= df.shares.quantile(.95)).mean()]})
    chi2, p, v = cramers_v(pd.crosstab(out, y))
    an["chi2_p"] = p
    an.to_csv(TAB / "e7_isolation_forest.csv", index=False)
    print(an.round(3).to_string())

    # ---- figure: PCA map + per-cluster popularity
    pcs = PCA(2, random_state=SEED).fit(Z)
    P = pcs.transform(Z[sample])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    order = prof_rich.sort_values("popular_rate").index
    share_cols = [c for c in prof_rich.columns if c.startswith("share_")]

    def name(c, i):
        top = prof_rich.loc[c, share_cols].astype(float).idxmax()
        chan = top.replace("share_", "") if prof_rich.loc[c, top] >= 0.5 else "mixed"
        return f"C{i+1}: {chan}, {prof_rich.loc[c, 'dominant_topic']}"
    names = {c: name(c, i) for i, c in enumerate(order)}
    for i, c in enumerate(order):
        m = lab_rich[sample] == c
        a1.scatter(P[m, 0], P[m, 1], s=2, alpha=0.4, color=f"C{i}", label=names[c])
    a1.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%})")
    a1.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
    a1.legend(fontsize=6, markerscale=4, loc="center left", bbox_to_anchor=(1.0, 0.5)); a1.set_title(f"KMeans, k = {k_best} (8,000-article sample)", fontsize=9)
    r = prof_rich.loc[order]
    a2.barh([names[c] for c in order], r["popular_rate"], color=[f"C{i}" for i in range(len(order))])
    a2.axvline(y.mean(), color="0.4", ls=":", lw=1)
    for i, v in enumerate(r["popular_rate"]):
        a2.text(v + 0.005, i, f"{v:.0%} (n={r['n'].iloc[i]:,})", va="center", fontsize=7)
    a2.set_xlim(0.3, 0.75); a2.set_xlabel("share of articles with >= 1400 shares")
    fig.tight_layout(); fig.savefig(FIG / "e7_clusters.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(sel.k, sel.silhouette, marker="o", color="C0")
    ax2 = ax.twinx(); ax2.plot(sel.k, sel.gmm_bic, marker="s", color="0.5")
    ax.set_xlabel("k"); ax.set_ylabel("silhouette (KMeans)", color="C0"); ax2.set_ylabel("BIC (GMM)", color="0.4")
    fig.tight_layout(); fig.savefig(FIG / "e7_model_selection.png", dpi=200)


if __name__ == "__main__":
    main()
