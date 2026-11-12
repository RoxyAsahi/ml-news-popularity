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

3. **AUC is about 0.03 lower in Oct–Dec than in Jul–Sep 2014 — but this is mostly not staleness (see E1b below).** Training
   on the oldest 70%, AUC on the first three full months after the cutoff (Jul–Sep 2014) vs the
   last three (Oct–Dec 2014): HistGB 0.754 → 0.722, decay 0.032 [0.014, 0.049]; logistic
   regression 0.736 → 0.707, decay 0.030 [0.012, 0.048] (independent bootstrap, 95% CI). Monthly
   AUCs with intervals are in `e1_drift_monthly.png`; single months are too noisy (±0.02–0.03) to
   read on their own.

   **Correcting for the label shift does not help.** Rescaling probabilities from the training
   positive rate (54.8%) to the most recent known one (validation block, 54.0%) leaves HistGB
   accuracy at 0.667 and Brier at 0.209 (logistic regression 0.658 → 0.660). The shift happens
   *inside* the test period (47.9% positive), so no rate known at training time can anticipate it.

4. **Old and new articles are easy to tell apart; the keyword clock columns explain only a small
   part of it** (0.958 → 0.940 without them, see E3). A classifier separates the oldest 70% from the newest 30% with ROC-AUC 0.958 on all
   features and 0.804 without `kw_*`/`self_reference_*`. E3 shows the keyword statistics are
   point-in-time and that three of them mostly encode archive size (constant inside the test
   block). The remaining separability (0.80 from content features) is real covariate drift, but
   point 1 shows it does not hurt ranking.

## Figures

- `e1_drift_monthly.png` — top: monthly share of popular articles (44%–68%); bottom: per-month ROC-AUC after the training cutoff with 95% bootstrap intervals.
- Tables: `e1_early_vs_late.csv`, `e1_label_shift.csv`, `e1_monthly_eval.csv` (with CIs).
- `e1_split_comparison.png` — random vs chronological ROC-AUC per model.

## Caveats

- 2014-06 is a partial month at the cutoff (303 articles).
- No hyper-parameter tuning yet; E1 compares split types, not models.

## E1b — stale model or harder articles? (added after review round 2)

Script: `scripts/e1b_staleness.py` (~1 min); tables `e1b_cutoff_auc.csv`, `e1b_decomposition.csv`,
figure `e1b_staleness.png` (and panel (b) of the report's Fig. 2).

Fix the evaluation window at Oct–Dec 2014 and move the training cutoff forward, training on every
earlier article:

| Cutoff | HistGB AUC on Oct–Dec | LogReg AUC on Oct–Dec |
| --- | --- | --- |
| 26 Jun 2014 | 0.722 | 0.707 |
| 1 Aug 2014 | 0.723 | 0.707 |
| 1 Sep 2014 | 0.726 | 0.708 |
| 1 Oct 2014 | 0.729 | 0.710 |

- **Staleness** (fresh − old model on the same Oct–Dec articles, paired bootstrap): HistGB
  0.006 [0.002, 0.011], LogReg 0.003 [0.001, 0.005]. Detectable but small.
- **Difficulty** (old model on Jul–Sep − fresh model on Oct–Dec, independent bootstrap): HistGB
  0.024 [0.007, 0.042], LogReg 0.027 [0.010, 0.045].
- So most of the 0.03 gap is that late-2014 articles are harder to rank, not that the model is
  outdated. This also explains why half-life weighting (E2) does not help.
- Note: the Jul–Sep window includes the validation block (not used for training here), so
  call it "after the training cutoff", not "within the test period".

### Review round 3 additions
- E1b intervals now use a weekly block bootstrap: staleness 0.006 [0.003, 0.010], difficulty
  0.024 [0.005, 0.045] (HistGB); total gap 0.030 (cutoff 26 Jun) is the number used in the report.
- Fixed-size sliding windows (15,857 most recent articles, same five test blocks): HistGB AUC
  0.704, 0.718, 0.743, 0.744, 0.732 (sd 0.017); LogReg sd 0.024. So period-to-period variation is
  not an artefact of growing training size. Table `e1b_sliding_window.csv`.
- Age vs shares: all articles were collected on the same day, so age and publication date are
  one-to-one; the flat age curve cannot by itself show saturation. Report relies on Szabo &
  Huberman (2010) for that.
