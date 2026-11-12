# E3 — Share-derived features: point-in-time, but carrying a clock (RQ2)

Scripts: `scripts/e3b_point_in_time.py` (~3 min; point-in-time check, clock columns,
adversarial validation, predictive AUC with bootstrap CIs and paired fold differences) and
`scripts/e3_leakage.py` (group ablation + grouped permutation importance).
Tables `results/tables/e3b_*.csv`, `e3_*.csv`; figures `e3b_staircase.png`, `e3_ablation.png`,
`e3_grouped_permutation.png`.

> **Revised 2026-11-12 after independent review.** The first version called this a "leakage
> audit". The data show the keyword statistics are computed point-in-time, so that framing was
> wrong. Do not use the word "leakage" for these features in the report.

## Findings

1. **The keyword statistics are point-in-time.** For 99.87% of the 39,564 articles with an earlier
   publication day, `kw_max_max` equals the largest share count of any article published on an
   earlier day; it exceeds it for exactly 1 article (a same-day ordering effect). The staircase
   figure shows `kw_max_max` jumping the day *after* each new record article appears. The keyword
   features behave like a historical target encoding: legitimate at prediction time.

2. **Some keyword columns are a clock.** `kw_min_min` (ρ with date = −0.88), its −1 flag
   (+0.86) and `kw_max_max` (+0.74) mostly measure how large the archive was when the article
   appeared: `kw_min_min = −1` in 8.5% of 2013 articles vs 99.9% of 2014 articles. Inside the
   chronological test block these three columns are constant (1, 1 and 3 distinct values).

3. **Removing the clock columns costs nothing.** Test ROC-AUC (95% bootstrap CI) and the paired
   drop versus all features:

   | Feature set | HistGB test AUC | Drop vs all (test, paired 95% CI) | Drop vs all (5 rolling folds, 95% CI) |
   | --- | --- | --- | --- |
   | all (60) | 0.737 [0.727, 0.749] | — | — |
   | no clock columns (57) | 0.735 [0.724, 0.746] | 0.002 [−0.001, 0.005] | 0.001 [−0.001, 0.002] |
   | no clock, strict (55) | 0.735 [0.724, 0.746] | 0.002 [−0.002, 0.005] | 0.003 [0.000, 0.006] |
   | no `kw_*` (50) | 0.721 [0.710, 0.732] | 0.016 [0.010, 0.021] | 0.011 [0.005, 0.017] |
   | no `kw_*`, no `self_reference_*` (47) | 0.700 [0.689, 0.712] | 0.036 [0.030, 0.043] | 0.027 [0.021, 0.032] |

   Logistic regression shows the same pattern (no clock: −0.002; no `kw_*`: −0.017; neither group: −0.022).
   The predictive value of the keyword group comes from the non-clock statistics (`kw_avg_avg`
   etc.), i.e. from the historical popularity of an article's keywords — a real, usable signal.

4. **The clock explains only part of the old-vs-new separability.** Adversarial validation
   (telling the oldest 70% from the newest 30%): all features 0.958, without clock columns 0.940,
   strict 0.910, without `kw_*` 0.810, without `kw_*` and `self_reference_*` 0.804. The remaining
   keyword averages still drift with the growing archive; content features alone still separate
   periods at 0.80, so there is genuine covariate drift too, but it does not hurt ranking (E1).

5. **Importance** (unchanged from the first version): `keyword` is HistGB's most important group
   (validation AUC drop 0.052 when permuted), then channel (0.034) and weekday (0.029).

## Limitation for the report

The shares of earlier articles used to build these statistics were measured when the dataset
was collected (Jan 2015), not when the later article was published. For an article published
in 2013 the statistics therefore use share counts that were not yet final at that time. Because
shares saturate within about a week (E0), the effect should be small; mention it as a limitation.
