# E1 — Temporal drift (RQ1)

Script: `scripts/e1_drift.py` (~2 min). Threshold 1400 shares, all 58 features, no tuning.
Tables in `results/tables/e1_*.csv`, figures `results/figures/e1_*.png`.

## Findings

1. **Ranking quality survives a chronological split.** Test ROC-AUC, random split (mean of 5 seeds) vs chronological (train on oldest 80%, test on newest 20%):

   | Model | Random | Chronological |
   | --- | --- | --- |
   | Logistic regression | 0.707 ± 0.002 | 0.717 |
   | kNN (k = 50) | 0.697 ± 0.005 | 0.693 |
   | Random forest | 0.734 ± 0.003 | 0.736 |
   | HistGradientBoosting | 0.735 ± 0.002 | 0.737 |

   Rolling-origin (5 expanding windows): HistGB 0.730 ± 0.019, RF 0.733 ± 0.015, LogReg 0.715 ± 0.021, kNN 0.698 ± 0.010. The fold-to-fold spread (≈ 0.02) is ten times the seed-to-seed spread of a random split (≈ 0.002), so **random-split error bars understate the real uncertainty**.

2. **Threshold-dependent metrics do degrade.** PR-AUC drops from 0.756 to 0.719 (HistGB) and F1 from 0.699 to 0.674, mostly because the share of popular articles falls from 53% (whole data) to 48% in the newest 20%. Month by month after training, the models predict 49–53% positives while the true rate is 47–49% (Sep–Dec 2014): a calibration drift, not a ranking failure.

3. **Mild decay with distance from training data.** Monthly ROC-AUC of HistGB goes from 0.75–0.77 in the first four months after the training cutoff (Jun–Sep 2014) to 0.71–0.74 in Oct–Dec 2014.

4. **Old and new articles are easy to tell apart, largely because some keyword columns act as a
   clock.** A classifier separates the oldest 70% from the newest 30% with ROC-AUC 0.958 on all
   features and 0.804 without `kw_*`/`self_reference_*`. E3 shows the keyword statistics are
   point-in-time and that three of them mostly encode archive size (constant inside the test
   block). The remaining separability (0.80 from content features) is real covariate drift, but
   point 1 shows it does not hurt ranking.

## Figures

- `e1_drift_monthly.png` — top: monthly share of popular articles (44%–68%); bottom: per-month ROC-AUC after the training cutoff.
- `e1_split_comparison.png` — random vs chronological ROC-AUC per model.

## Caveats

- 2014-06 is a partial month at the cutoff (303 articles).
- No hyper-parameter tuning yet; E1 compares split types, not models.
