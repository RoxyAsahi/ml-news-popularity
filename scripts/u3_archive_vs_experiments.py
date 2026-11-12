"""U3: how many randomized experiments is an observational archive worth?

A newsroom that wants a headline-ranking model has two kinds of data: a large observational archive (each
article with the headline it ran with and its click-through) and a small, expensive set of randomized
headline tests. We learn within-test headline effects (GloVe-100 mean + 15 interpretable features, ridge) from
  E(k)      k randomized tests only (within-cell demeaned contrasts);
  O-raw     the archive only, raw logit CTR (standard popularity-model training);
  O         the archive only: one package per training test, logit CTR demeaned within week (OBS-FE in u2);
  POOL(k)   both stacked in one regression;
  PRIOR(k)  experimental grounding: the archive model as a prior, a ridge correction learnt on the k tests
            (beta = beta_O + delta, delta shrunk to 0), in the spirit of Kallus et al. (2018).
Held-out articles (5-fold CV grouped by article); k = number of randomized training tests (-1: all of the fold), 3 random subsets.
Metric: pairwise accuracy on significantly different within-test pairs, and the CTR lift of the picked headline.
The value of the archive in experiments is the k at which E(k) reaches O.

Usage: python scripts/u3_archive_vs_experiments.py [exploratory|confirmatory|holdout]
Output: results/tables/u3_fusion_<split>.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from newspop import upw_rank as ur
from newspop.data import ROOT

split = sys.argv[1] if len(sys.argv) > 1 else "exploratory"
S = ur.setup(split)
Z = S.Z
ALPHAS = np.logspace(-1, 5, 13)
KS = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
REPS = 3
rng = np.random.default_rng(0)
print(f"{split}: {len(S.tests)} tests, {S.article.max() + 1} articles", flush=True)

rows = []
SIG = np.where(np.abs(S.zz) > 1.96)[0]          # significant within-test pairs, scored once (in their fold)
CORR = {}                                         # config -> correctness per significant pair


def record(name, k, rep, score_rows, pred_rows, te):
    pred = np.full(len(S.d), np.nan)
    pred[score_rows] = pred_rows
    m = np.isin(S.tix[S.ev_a[SIG]], te)
    dp = pred[S.ev_a[SIG[m]]] - pred[S.ev_b[SIG[m]]]
    v = CORR.setdefault((name, k, rep), np.full(len(SIG), np.nan))
    v[m] = np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(S.zz[SIG[m]])).astype(float))


for fold, (tr, te) in enumerate(GroupKFold(5).split(np.zeros(len(S.tests)), groups=S.article)):
    obs = ur.one_per_test(S, tr, rng)
    O = RidgeCV(alphas=ALPHAS).fit(Z[S.hid[obs]], ur.week_demeaned(S, obs))
    score_rows = np.where(np.isin(S.tix, te))[0]
    Zte = Z[S.hid[score_rows]]
    rows.append(dict(fold=fold, method="O", k=0, rep=0, **ur.evaluate(S, score_rows, O.predict(Zte), te)))
    record("O", 0, 0, score_rows, O.predict(Zte), te)
    O_raw = RidgeCV(alphas=ALPHAS).fit(Z[S.hid[obs]], S.y[obs])
    rows.append(dict(fold=fold, method="O-raw", k=0, rep=0, **ur.evaluate(S, score_rows, O_raw.predict(Zte), te)))
    record("O-raw", 0, 0, score_rows, O_raw.predict(Zte), te)
    exp_tests = np.unique(S.tix[ur.within_rows(S, tr)[0]])
    Xo, yo = Z[S.hid[obs]], ur.week_demeaned(S, obs)
    for k in KS + [len(exp_tests)]:
        if k > len(exp_tests):
            continue
        for rep in range(REPS if k < len(exp_tests) else 1):
            sub = rng.choice(exp_tests, k, replace=False)
            er, cf = ur.within_rows(S, sub)
            Xe = ur.demean(Z[S.hid[er]], cf)
            ye = S.y[er] - pd.Series(S.y[er]).groupby(cf).transform("mean").to_numpy()
            # penalties chosen by CV over whole tests (rows of one test are dependent after demeaning)
            cv = list(GroupKFold(min(5, k)).split(Xe, ye, groups=S.tix[er]))
            E = RidgeCV(alphas=ALPHAS, fit_intercept=False, cv=cv).fit(Xe, ye)
            P = RidgeCV(alphas=ALPHAS).fit(np.r_[Xo, Xe], np.r_[yo, ye])
            D = RidgeCV(alphas=ALPHAS, fit_intercept=False, cv=cv).fit(Xe, ye - Xe @ O.coef_)
            for name, pred in [("E", Zte @ E.coef_), ("POOL", P.predict(Zte)), ("PRIOR", Zte @ (O.coef_ + D.coef_))]:
                kk = k if k < len(exp_tests) else -1
                rows.append(dict(fold=fold, method=name, k=kk, rep=rep, **ur.evaluate(S, score_rows, pred, te)))
                record(name, kk, rep, score_rows, pred, te)
            print(fold, k, rep, {r["method"]: round(r["within_acc_sig"], 3) for r in rows[-3:]}, flush=True)

keys = sorted(CORR, key=str)
np.savez_compressed(ROOT / "results" / "tables" / f"u3_pairs_{split}.npz",
                    configs=np.array(["|".join(map(str, c)) for c in keys]), correct=np.array([CORR[c] for c in keys]),
                    article=S.article[S.tix[S.ev_a[SIG]]])
R = pd.DataFrame(rows)
R.to_csv(ROOT / "results" / "tables" / f"u3_fusion_{split}.csv", index=False)
print(R.groupby(["method", "k"])[["within_acc_sig", "within_acc", "lift_pct"]].mean().round(4).to_string())
