"""U6: the exchange rate between an observational archive and randomized headline tests (revision of u3).

Changes against u3, after review:
  * Two archives. "random": one random headline per test (an idealized archive: randomized, simultaneous
    traffic, no selection). "published": the headline that won the test (first_place), which is what a
    newsroom archive actually holds - selected by the outcome, with winner's curse.
  * Archive size: within each training fold, random subsets of 1,000 ... all training tests (3 draws each), so
    the exchange rate k*(archive size) is measured with one fixed pool of experiments.
  * Training variability enters the intervals: each bootstrap replicate resamples held-out articles and, for
    every budget k and archive, picks one of the 3 training draws at random.
  * A fixed-penalty prior (no cross-validation, alpha = 1,000) next to the cross-validated one, to check that the
    small-budget gains of the prior are not a cross-validation artefact.
  * Click-through lift of the picked headline (in % of the cell mean) for every configuration.
Learner and features as in u3 (ridge, GloVe-100 mean + 15 features); 5-fold CV grouped by article.

Usage: python scripts/u6_exchange_rate.py [confirmatory|exploratory|holdout]
Output: results/tables/u6_runs_<split>.csv (accuracy and lift per fold and configuration),
        results/tables/u6_pairs_<split>.npz (per-pair correctness, for u7)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

from newspop import upw_rank as ur
from newspop.data import ROOT

split = sys.argv[1] if len(sys.argv) > 1 else "confirmatory"
S = ur.setup(split)
Z = S.Z
ALPHAS = np.logspace(-1, 5, 13)
KS = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
SIZES = [1000, 2000, 4500, 9000] if split == "confirmatory" else [1000, 2000]
DRAWS = 3
rng = np.random.default_rng(1)
first = S.d.first_place.fillna(False).astype(bool).to_numpy()
print(f"{split}: {len(S.tests)} tests; first_place in {np.bincount(S.tix[first], minlength=len(S.tests)).astype(bool).mean():.3f}",
      flush=True)

SIG = np.where(np.abs(S.zz) > 1.96)[0]
CORR, rows = {}, []


def published(test_ids, r):
    """The winning headline of each test (first_place); a random one where the archive records none."""
    out = []
    for k in test_ids:
        w = S.pk[k][first[S.pk[k]]]
        out.append(w[0] if len(w) else S.pk[k][r.integers(0, len(S.pk[k]))])
    return np.array(out)


def record(cfg, fold, score_rows, pred_rows, te):
    pred = np.full(len(S.d), np.nan)
    pred[score_rows] = pred_rows
    m = np.isin(S.tix[S.ev_a[SIG]], te)
    dp = pred[S.ev_a[SIG[m]]] - pred[S.ev_b[SIG[m]]]
    v = CORR.setdefault(cfg, np.full(len(SIG), np.nan))
    v[m] = np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(S.zz[SIG[m]])).astype(float))
    rows.append(dict(fold=fold, method=cfg[0], archive=cfg[1], size=cfg[2], draw=cfg[3], k=cfg[4],
                     **ur.evaluate(S, score_rows, pred_rows, te)))


for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(len(S.tests)), groups=S.article)):
    score_rows = np.where(np.isin(S.tix, te))[0]
    Zte = Z[S.hid[score_rows]]
    full = {}
    for arch in ["random", "published"]:
        for size in SIZES + [-1]:
            for draw in range(DRAWS if size != -1 or arch == "random" else 1):
                sub = tr if size == -1 else rng.choice(tr, size, replace=False)
                rows_a = ur.one_per_test(S, sub, rng) if arch == "random" else published(sub, rng)
                O = RidgeCV(alphas=ALPHAS).fit(Z[S.hid[rows_a]], ur.week_demeaned(S, rows_a))
                record(("O", arch, size, draw, 0), fold, score_rows, O.predict(Zte), te)
                if size == -1:
                    Oraw = RidgeCV(alphas=ALPHAS).fit(Z[S.hid[rows_a]], S.y[rows_a])
                    record(("O-raw", arch, size, draw, 0), fold, score_rows, Oraw.predict(Zte), te)
                    if draw == 0:
                        full[arch] = O.coef_
    exp_tests = np.unique(S.tix[ur.within_rows(S, tr)[0]])
    for k in KS + [len(exp_tests)]:
        if k > len(exp_tests):
            continue
        kk = k if k < len(exp_tests) else -1
        for rep in range(DRAWS if kk != -1 else 1):
            sub = rng.choice(exp_tests, k, replace=False)
            er, cf = ur.within_rows(S, sub)
            Xe = ur.demean(Z[S.hid[er]], cf)
            ye = S.y[er] - pd.Series(S.y[er]).groupby(cf).transform("mean").to_numpy()
            cv = list(GroupKFold(min(5, k)).split(Xe, ye, groups=S.tix[er]))
            E = RidgeCV(alphas=ALPHAS, fit_intercept=False, cv=cv).fit(Xe, ye)
            record(("E", "-", 0, rep, kk), fold, score_rows, Zte @ E.coef_, te)
            for arch, bO in full.items():
                D = RidgeCV(alphas=ALPHAS, fit_intercept=False, cv=cv).fit(Xe, ye - Xe @ bO)
                record(("PRIOR", arch, -1, rep, kk), fold, score_rows, Zte @ (bO + D.coef_), te)
                Df = Ridge(alpha=1000.0, fit_intercept=False).fit(Xe, ye - Xe @ bO)
                record(("PRIOR-fixed", arch, -1, rep, kk), fold, score_rows, Zte @ (bO + Df.coef_), te)
        print(fold, kk, flush=True)

keys = sorted(CORR, key=str)
np.savez_compressed(ROOT / "results" / "tables" / f"u6_pairs_{split}.npz",
                    configs=np.array(["|".join(map(str, c)) for c in keys]), correct=np.array([CORR[c] for c in keys]),
                    article=S.article[S.tix[S.ev_a[SIG]]])
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / f"u6_runs_{split}.csv", index=False)
print(R.groupby(["method", "archive", "size", "k"])[["within_acc_sig", "lift_pct"]].mean().round(4).to_string())
