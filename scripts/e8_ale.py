"""E8 - Direction of feature effects: accumulated local effects (ALE).

Partial dependence averages predictions over feature combinations that never occur when
features are correlated (e.g. the keyword statistics). ALE (Apley & Zhu, 2020) avoids this by
averaging *local* prediction differences inside quantile bins and accumulating them.
Own implementation, first-order ALE on the predicted probability of "popular" (>= 1,400 shares)
from HistGB trained on the training block, evaluated on the same block.

Run from the repo root:  python scripts/e8_ale.py   (~1 min)
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from newspop.data import ROOT, feature_set, load, make_label
from newspop.models import fit, get_model
from newspop.splits import temporal_split

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FEATURES = {"kw_avg_avg": "keyword shares (kw_avg_avg)",
            "self_reference_avg_sharess": "linked-article shares",
            "num_hrefs": "number of links",
            "n_tokens_content": "words in body"}
N_BINS = 20


def ale(model, X, j, n_bins=N_BINS):
    """First-order ALE of column j; returns bin edges (quantiles) and centred effect."""
    x = X[:, j]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    local = np.zeros(len(edges) - 1)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    lo, hi = X.copy(), X.copy()
    lo[:, j], hi[:, j] = edges[idx], edges[idx + 1]
    d = model.predict_proba(hi)[:, 1] - model.predict_proba(lo)[:, 1]
    local = np.bincount(idx, weights=d, minlength=len(edges) - 1) / np.maximum(counts, 1)
    acc = np.concatenate([[0], np.cumsum(local)])
    # centre so the count-weighted mean effect is zero
    mid = (acc[:-1] + acc[1:]) / 2
    acc -= np.sum(mid * counts) / counts.sum()
    return edges, acc


def main():
    df = load()
    n = len(df)
    tr, _, _ = temporal_split(n, 0.7, 0.1)
    cols = feature_set(df, "all")
    X = df[cols].to_numpy(dtype=float)
    y = make_label(df.shares, 1400)
    model = fit(get_model("hgb", 0), X[tr], y[tr])
    rows = []
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    for ax, (f, name) in zip(axes, FEATURES.items()):
        edges, eff = ale(model, X[tr], cols.index(f))
        rows += [{"feature": f, "x": e, "ale": a} for e, a in zip(edges, eff)]
        # x axis = percentile of the training distribution, labelled with real values
        pct = np.array([(X[tr, cols.index(f)] <= e).mean() * 100 for e in edges])
        ax.plot(pct, eff, marker="o", ms=2, color="C0")
        ax.axhline(0, color="0.6", lw=0.6)
        ticks = [10, 50, 90]
        vals = np.quantile(X[tr, cols.index(f)], np.array(ticks) / 100)
        ax.set_xticks(ticks, [f"p{t}\n{v:,.0f}" for t, v in zip(ticks, vals)], fontsize=7.5)
        ax.set_xlim(0, 101)
        ax.set_title(name, fontsize=8)
        print(f, "range of effect:", round(eff.max() - eff.min(), 3))
    axes[0].set_ylabel("ALE on P(popular)")
    fig.tight_layout(); fig.savefig(FIG / "e8_ale.png", dpi=300)
    pd.DataFrame(rows).to_csv(TAB / "e8_ale.csv", index=False)


if __name__ == "__main__":
    main()
