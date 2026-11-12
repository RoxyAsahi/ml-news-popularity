# Main results table (review item R2)

Script: `scripts/main_results.py` (~6 min). Hyper-parameters chosen on the validation block
(train = oldest 70%, validation = next 10%), refit on train + validation, test block (newest 20%,
7,929 articles, 47.9% popular) used once. Threshold 1,400 shares. 95% percentile bootstrap
intervals (1,000 resamples); model gaps are paired bootstraps on the same test articles.
Table: `results/tables/main_results.csv`.

| Features | Model | Chosen setting | Test ROC-AUC [95% CI] | Test PR-AUC [95% CI] | Gap to best [95% CI] |
| --- | --- | --- | --- | --- | --- |
| all (60) | majority | — | 0.500 | 0.479 | — |
| all | L1 logistic regression | C = 0.1 | 0.717 [0.705, 0.729] | 0.696 [0.680, 0.712] | 0.022 [0.016, 0.028] |
| all | kNN | k = 500 | 0.705 [0.694, 0.717] | 0.682 [0.666, 0.698] | 0.034 [0.027, 0.041] |
| all | random forest | leaf 5, sqrt | 0.736 [0.725, 0.747] | 0.715 [0.699, 0.730] | 0.003 [−0.001, 0.007] |
| all | **HistGB** | lr 0.03, 15 leaves, L2 1, 500 iter | **0.739 [0.728, 0.750]** | 0.721 [0.705, 0.736] | best |
| no share-derived (47) | L1 logistic regression | C = 0.01 | 0.693 [0.681, 0.704] | 0.670 [0.653, 0.685] | 0.015 [0.009, 0.021] |
| no share-derived | kNN | k = 500 | 0.688 [0.676, 0.700] | 0.664 [0.648, 0.680] | 0.020 [0.014, 0.026] |
| no share-derived | **random forest** | leaf 20, 0.3 | **0.708 [0.696, 0.719]** | 0.691 [0.675, 0.706] | best |
| no share-derived | HistGB | lr 0.03, 15 leaves, L2 1, 500 iter | 0.705 [0.693, 0.716] | 0.687 [0.671, 0.702] | 0.003 [−0.002, 0.007] |

## Reading

- **Random forest and HistGB are indistinguishable** (gap 0.003, interval contains 0, in both
  feature settings). Say "no detectable difference", not "HistGB is best".
- Both tree ensembles beat L1 logistic regression by about 0.02 and kNN by about 0.03; those
  intervals exclude 0.
- Removing the share-derived features costs 0.034 [0.027, 0.042] (HistGB), 0.028 [0.021, 0.036]
  (RF), 0.024 [0.019, 0.029] (LogReg), 0.017 [0.013, 0.021] (kNN).
- Tuning moved HistGB from 0.737 (defaults) to 0.739 and kNN from 0.693 (k = 50) to 0.705
  (k = 500); differences of this size between tuned and untuned runs elsewhere in the notes are
  within the ±0.011 interval and should not be interpreted.
- Every model's validation AUC is 0.01–0.02 above its test AUC: the validation block (25 Jun – 28 Aug 2014)
  is slightly easier than the test block (28 Aug – 27 Dec 2014), consistent with E1.

## Update after review round 3 (2026-11-12)

- Reference ("best") model is now chosen by **validation** AUC, not test AUC.
- kNN grid extended to k = 2,000 (still picks 500, now inside the grid); HistGB grid has 12
  configurations with the number of boosting rounds chosen on the validation block (temporal
  early stopping).
- Intervals are now **weekly block bootstrap** (whole publication weeks resampled); about 1.4×
  wider than article-level (HistGB [0.721, 0.752] vs [0.726, 0.748]).
- New numbers: all features — HistGB 0.737 (ref.), RF 0.736 (gap 0.001 [−0.003, 0.005]), L1 LogReg
  0.717 (gap 0.020 [0.014, 0.025]), kNN 0.705 (gap 0.032 [0.027, 0.037]). Without share-derived
  features — RF 0.708 (ref.), HistGB 0.703 (gap 0.005 [0.001, 0.009], now detectable), LogReg 0.693,
  kNN 0.688. Share-derived drop: HistGB 0.034 [0.027, 0.040], RF 0.028 [0.022, 0.034].
- The table above is superseded by `results/tables/main_results.csv`.
