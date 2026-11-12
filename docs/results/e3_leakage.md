# E3 — Leakage audit of share-derived features (RQ2)

Script: `scripts/e3_leakage.py` (~2 min). Threshold 1400 shares, chronological 70/10/20 split
plus 5-fold rolling-origin. Tables `results/tables/e3_*.csv`, figures `results/figures/e3_*.png`.

`kw_*` (10 columns incl. the `kw_min_min = -1` flag) summarise the shares of *other* articles
with the same keywords; `self_reference_*` (3 columns) the shares of linked Mashable articles.
If these statistics were computed at data-collection time (Jan 2015) they may include shares
that were not known when the article was published.

## Findings

1. **Ablation** (rolling-origin ROC-AUC, mean ± sd over 5 folds):

   | Feature set | # features | LogReg | HistGB |
   | --- | --- | --- | --- |
   | all | 60 | 0.715 ± 0.019 | 0.730 ± 0.017 |
   | no `self_reference_*` | 57 | 0.711 ± 0.019 | 0.721 ± 0.020 |
   | no `kw_*` | 50 | 0.700 ± 0.018 | 0.719 ± 0.016 |
   | no `kw_*`, no `self_reference_*` | 47 | 0.694 ± 0.019 | 0.703 ± 0.017 |
   | only `kw_*` + `self_reference_*` | 13 | 0.646 ± 0.011 | 0.680 ± 0.014 |

   Removing both share-derived groups costs 0.021 (LogReg) to 0.027 (HistGB). The 47 content,
   channel, weekday, topic and sentiment features alone still reach 0.70, so **most of the
   predictive signal does not depend on share-derived features**, but the best model's edge over
   logistic regression largely does (0.015 gap with them, 0.009 without).

2. **Grouped permutation importance** (validation block, HistGB): `keyword` is the most important
   group (AUC drop 0.052 ± 0.003), then channel (0.034), weekday (0.029), self_reference (0.023),
   LDA topics (0.021), content (0.021), sentiment (0.007). For logistic regression the content group
   ranks first (0.052) and keyword third (0.038) — the boosting model leans more on share-derived features.

3. **The keyword statistics encode the publication date.** Spearman correlation with the publication
   date: `kw_min_min` −0.88, `kw_min_min = -1` flag +0.86, `kw_max_max` +0.74, `kw_avg_max` +0.52,
   `kw_avg_min` −0.45. No non-keyword feature exceeds |0.24|. Together with E1 (old vs new articles
   separable with AUC 0.96, 0.80 without these groups), this suggests the keyword statistics were
   computed over a window that depends on the article's date. That is a sign of time dependence,
   **not proof of leakage**: the original paper should be checked for how they were computed before
   we claim leakage in the report.

## Figures

- `e3_ablation.png` — rolling-origin AUC per feature set (HistGB).
- `e3_grouped_permutation.png` — AUC drop per permuted feature group (HistGB).

## For the report

Phrase it as "features derived from other articles' share counts" and report results with and
without them; recommend the no-share-derived setting as the conservative estimate (≈ 0.70 AUC).
