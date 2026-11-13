"""U8: which randomized tests should a newsroom with an archive run next? (archive-guided experiment selection)

Setting as in u6 (confirmatory folds, ridge on GloVe-100 mean + 15 features). Within each training fold the pool is
every training test with a cell of >= 2 headlines; a strategy picks k of them (nested, k = 25 ... 800), and the
archive-as-prior learner (PRIOR-fixed, alpha = 1,000, and PRIOR with cross-validated penalty) and the
experiments-only learner E are trained on the picked tests. Strategies, all label-free (they never see the
outcome of a test before picking it):
  random      uniform (3 draws)
  margin-     smallest archive-predicted spread between the headlines of a test (uncertainty sampling)
  margin+     largest archive-predicted spread (tests where the archive claims the biggest differences)
  committee   largest disagreement among 8 archive models fit on bootstrap resamples of the archive
              (query by committee: the archive's own estimation uncertainty)
  design      greedy Bayesian experimental design: each step adds the test that most reduces the summed posterior
              variance of within-cell contrasts over the whole pool, under the prior of PRIOR-fixed
              (precision alpha; transductive/V-optimal design, rank-r Woodbury updates)
Usage: python scripts/u8_active_experiments.py [confirmatory|exploratory|holdout]
Output: results/tables/u8_active_<split>.csv (per fold), results/tables/u8_pairs_<split>.npz (per-pair correctness)
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
ALPHA_FIXED = 1000.0
KS = [25, 50, 100, 200, 400, 800]
rng = np.random.default_rng(2)
SIG = np.where(np.abs(S.zz) > 1.96)[0]
CORR, rows = {}, []


def record(cfg, fold, score_rows, pred_rows, te):
    pred = np.full(len(S.d), np.nan)
    pred[score_rows] = pred_rows
    m = np.isin(S.tix[S.ev_a[SIG]], te)
    dp = pred[S.ev_a[SIG[m]]] - pred[S.ev_b[SIG[m]]]
    v = CORR.setdefault(cfg, np.full(len(SIG), np.nan))
    v[m] = np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(S.zz[SIG[m]])).astype(float))
    rows.append(dict(fold=fold, learner=cfg[0], strategy=cfg[1], draw=cfg[2], k=cfg[3],
                     **ur.evaluate(S, score_rows, pred_rows, te)))


def per_test(values, row_test, tests):
    """Mean of a per-row quantity within each test, aligned with `tests`."""
    s = pd.Series(values).groupby(row_test).mean()
    return s.reindex(tests).to_numpy()


def greedy_design(X, row_test, tests, n, alpha):
    """Greedy V-optimal design: pick tests whose contrast rows most reduce sum_j x_j' A^-1 x_j over all pool rows,
    A = alpha I + X_S' X_S. Gain of a test approximated by the sum of its rows' single-row gains."""
    d = X.shape[1]
    Ai = np.eye(d) / alpha
    C = X.T @ X                                   # target: all pool contrasts
    P = X @ Ai                                     # N x d
    Q = P @ C                                      # N x d
    pos = {t: np.where(row_test == t)[0] for t in tests}
    tidx = pd.factorize(row_test)[0]
    order_t = pd.factorize(row_test)[1]
    chosen, avail = [], np.ones(len(order_t), bool)
    for _ in range(n):
        num = np.einsum("ij,ij->i", Q, P)
        den = 1 + np.einsum("ij,ij->i", P, X)
        g = np.bincount(tidx, weights=num / den, minlength=len(order_t))
        g[~avail] = -np.inf
        j = int(np.argmax(g))
        avail[j] = False
        chosen.append(order_t[j])
        r = pos[order_t[j]]
        Xt = X[r]
        Pt = P[r]                                  # Xt Ai
        M = np.linalg.inv(np.eye(len(r)) + Pt @ Xt.T)
        PXt = P @ Xt.T                             # N x r
        Ai = Ai - Pt.T @ M @ Pt
        P = P - PXt @ M @ Pt
        Q = Q - PXt @ M @ (Pt @ C)
    return np.array(chosen)


for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(len(S.tests)), groups=S.article)):
    score_rows = np.where(np.isin(S.tix, te))[0]
    Zte = Z[S.hid[score_rows]]
    rows_a = ur.one_per_test(S, tr, rng)
    Xa, ya = Z[S.hid[rows_a]], ur.week_demeaned(S, rows_a)
    O = RidgeCV(alphas=ALPHAS).fit(Xa, ya)
    bO = O.coef_
    record(("O", "-", 0, 0), fold, score_rows, O.predict(Zte), te)
    committee = []
    for _ in range(8):
        bi = rng.integers(0, len(rows_a), len(rows_a))
        committee.append(Ridge(alpha=O.alpha_).fit(Xa[bi], ya[bi]).coef_)
    committee = np.array(committee)

    exp_tests = np.unique(S.tix[ur.within_rows(S, tr)[0]])
    er, cf = ur.within_rows(S, exp_tests)
    Xall = ur.demean(Z[S.hid[er]], cf)
    yall = S.y[er] - pd.Series(S.y[er]).groupby(cf).transform("mean").to_numpy()
    rt = S.tix[er]
    spread = per_test(np.abs(Xall @ bO), rt, exp_tests)
    disagree = per_test((Xall @ committee.T).std(1), rt, exp_tests)
    nmax = max(KS)
    seqs = {("random", r): rng.permutation(exp_tests)[:nmax] for r in range(3)}
    seqs[("margin-", 0)] = exp_tests[np.argsort(spread + 1e-12 * rng.random(len(spread)))][:nmax]
    seqs[("margin+", 0)] = exp_tests[np.argsort(-spread + 1e-12 * rng.random(len(spread)))][:nmax]
    seqs[("committee", 0)] = exp_tests[np.argsort(-disagree + 1e-12 * rng.random(len(spread)))][:nmax]
    seqs[("design", 0)] = greedy_design(Xall, rt, exp_tests, nmax, ALPHA_FIXED)
    for (strategy, draw), seq in seqs.items():
        for k in KS:
            sub = seq[:k]
            m = np.isin(rt, sub)
            Xe, ye, g = Xall[m], yall[m], rt[m]
            cv = list(GroupKFold(min(5, k)).split(Xe, ye, groups=g))
            E = RidgeCV(alphas=ALPHAS, fit_intercept=False, cv=cv).fit(Xe, ye)
            record(("E", strategy, draw, k), fold, score_rows, Zte @ E.coef_, te)
            D = RidgeCV(alphas=ALPHAS, fit_intercept=False, cv=cv).fit(Xe, ye - Xe @ bO)
            record(("PRIOR", strategy, draw, k), fold, score_rows, Zte @ (bO + D.coef_), te)
            Df = Ridge(alpha=ALPHA_FIXED, fit_intercept=False).fit(Xe, ye - Xe @ bO)
            record(("PRIOR-fixed", strategy, draw, k), fold, score_rows, Zte @ (bO + Df.coef_), te)
    print(fold, "done", flush=True)

keys = sorted(CORR, key=str)
np.savez_compressed(ROOT / "results" / "tables" / f"u8_pairs_{split}.npz",
                    configs=np.array(["|".join(map(str, c)) for c in keys]), correct=np.array([CORR[c] for c in keys]),
                    article=S.article[S.tix[S.ev_a[SIG]]])
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / f"u8_active_{split}.csv", index=False)
print(R.groupby(["learner", "strategy", "k"])[["within_acc_sig", "lift_pct"]].mean().unstack(0).round(4).to_string())
