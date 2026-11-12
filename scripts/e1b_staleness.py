"""E1b (RQ1) - Does AUC fall because the model gets stale, or because later articles are harder?

E1 found that a model trained up to 25 Jun 2014 ranks Jul-Sep 2014 articles better than
Oct-Dec 2014 articles (decay about 0.03). That gap mixes two effects:

* staleness: the model has not seen the most recent articles;
* difficulty: the late articles are harder to rank for any model.

Here we fix the evaluation window (Oct-Dec 2014) and move the training cutoff forward
(26 Jun, 1 Aug, 1 Sep, 1 Oct 2014), training on every article published before it.
If staleness drove the decay, a fresh model would recover most of the 0.03.

* staleness = AUC(fresh model, Oct-Dec) - AUC(old model, Oct-Dec): paired bootstrap on the
  same test articles;
* difficulty = AUC(old model, Jul-Sep) - AUC(fresh model, Oct-Dec): independent bootstrap
  (disjoint article sets).

Run from the repo root:  python scripts/e1b_staleness.py   (~2 min)
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from newspop.data import ROOT, feature_set, load, make_label
from newspop.evaluate import bootstrap_ci, paired_bootstrap
from newspop.models import fit, get_model

warnings.filterwarnings("ignore")
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
THRESHOLD = 1400
CUTOFFS = ["2014-06-26", "2014-08-01", "2014-09-01", "2014-10-01"]
EARLY = ("2014-07-01", "2014-10-01")     # Jul-Sep 2014
LATE = ("2014-10-01", "2015-01-01")      # Oct-Dec 2014
MODELS = {"hgb": "HistGB", "logreg": "logistic regression"}
N_BOOT = 1000


def window(dates, lo, hi):
    return np.where((dates >= pd.Timestamp(lo)) & (dates < pd.Timestamp(hi)))[0]


def main():
    df = load()
    y = make_label(df.shares, THRESHOLD)
    X = df[feature_set(df, "all")].to_numpy()
    dates = df.publish_date
    early, late = window(dates, *EARLY), window(dates, *LATE)

    rows, pairs, scores = [], [], {}
    for m in MODELS:
        for c in CUTOFFS:
            tr = np.where(dates < pd.Timestamp(c))[0]
            mod = fit(get_model(m, 0), X[tr], y[tr])
            scores[m, c] = mod.predict_proba(X)[:, 1]
            est, lo, hi = bootstrap_ci(y[late], scores[m, c][late], n_boot=N_BOOT)
            rows.append({"model": m, "cutoff": c, "n_train": len(tr), "eval": "Oct-Dec 2014",
                         "auc": est, "lo": lo, "hi": hi})
            print(m, c, len(tr), round(est, 3))
        old, fresh = scores[m, CUTOFFS[0]], scores[m, CUTOFFS[-1]]
        st = paired_bootstrap(y[late], fresh[late], old[late], n_boot=N_BOOT)
        # difficulty: old model on Jul-Sep vs fresh model on Oct-Dec (disjoint sets)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(N_BOOT):
            ie, il = rng.choice(early, len(early)), rng.choice(late, len(late))
            diffs.append(roc_auc_score(y[ie], old[ie]) - roc_auc_score(y[il], fresh[il]))
        total = roc_auc_score(y[early], old[early]) - roc_auc_score(y[late], old[late])
        diff = roc_auc_score(y[early], old[early]) - roc_auc_score(y[late], fresh[late])
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        pairs.append({"model": m, "total_decay": total,
                      "staleness": st["diff"], "staleness_lo": st["lo"], "staleness_hi": st["hi"],
                      "difficulty": diff, "difficulty_lo": lo, "difficulty_hi": hi})

    res, dec = pd.DataFrame(rows), pd.DataFrame(pairs)
    res.to_csv(TAB / "e1b_cutoff_auc.csv", index=False)
    dec.to_csv(TAB / "e1b_decomposition.csv", index=False)
    print(dec.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m, col in zip(MODELS, ["C0", "0.45"]):
        d = res[res.model == m]
        x = pd.to_datetime(d.cutoff)
        ax.errorbar(x, d.auc, yerr=[d.auc - d.lo, d.hi - d.auc], marker="o", ms=3,
                    capsize=2, color=col, label=MODELS[m])
    ax.set_ylabel("ROC-AUC on Oct-Dec 2014")
    ax.set_xlabel("training cutoff (2014)")
    ax.set_xticks(pd.to_datetime(CUTOFFS), ["26 Jun", "1 Aug", "1 Sep", "1 Oct"])
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "e1b_staleness.png", dpi=300)


if __name__ == "__main__":
    main()
