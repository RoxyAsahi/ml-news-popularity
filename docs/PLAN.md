# Research plan

_Last updated 2026-11-12. Draft for discussion at the first group meeting._

## Storyline

How reliable is the "popular" label, and how much of what a model learns from these 58 pre-computed features generalises to future articles? This lets every required part of the assignment (classification, clustering, training-set size, regularisation, feature importance) sit in one argument rather than a list of unrelated experiments.

## Pilot results (`scripts/01_pilot.py`, untuned)

Data: 39,643 Mashable articles after dropping one corrupt row, published 2013-01-07 to 2014-12-27.

- `shares` is heavy-tailed: median 1,400, mean 3,395, 99th percentile 31,658, max 843,300.
- Shares do not grow with article age: Spearman ρ(timedelta, shares) = 0.03; median shares per age bin stays between 1,300 and 1,600. The youngest article is 8 days old, so shares have saturated. **No age correction of the label is needed**; this is itself a finding.
- Monthly share of "popular" (≥ 1400) articles ranges from 44% (Jul 2013) to 68% (Mar 2013).

ROC-AUC, threshold 1400, 80/20 split:

| Split | Features | LogReg | kNN (k=50) | HistGB |
| --- | --- | --- | --- | --- |
| random | all | 0.709 | 0.704 | 0.737 |
| random | no `kw_*`, no `self_reference_*` | 0.686 | 0.687 | 0.713 |
| temporal | all | 0.717 | 0.693 | 0.737 |
| temporal | no `kw_*` | 0.700 | 0.683 | 0.721 |
| temporal | no `kw_*`, no `self_reference_*` | 0.696 | 0.679 | 0.700 |

Note: random and chronological splits give the same AUC for boosting, but removing the share-derived features hurts more under the chronological split (−0.037 vs −0.024).

- Threshold at 50/75/90/95th percentile: ROC-AUC stays 0.73–0.74, PR-AUC falls 0.68 → 0.10.
- Learning curve: 3% of training data (~950 articles) already gives ROC-AUC ≈ 0.70; logistic regression plateaus after 10%, boosting keeps improving.
- Half-life recency weights (30/90/180/365 days) gave 0.720–0.735 vs 0.737 unweighted: no gain in the pilot.
- Top permutation importances: `kw_avg_avg`, `self_reference_min_shares`, `is_weekend`, tech and entertainment channels.

Data issues to report: 1,181 articles with 0 content words; 58% of rows have `kw_min_min = -1` (missing marker); one row with `n_unique_tokens = 701`.

## Research questions and experiments

| ID | RQ | What | Output |
| --- | --- | --- | --- |
| E0 | — | EDA: shares distribution, shares vs age, channel/weekday, monthly base rate | Fig. 1 |
| E1 | RQ1 | Random vs chronological split; monthly positive rate and per-month AUC | drift plot |
| E2 | RQ1 | Half-life sample weighting (none, 30, 90, 180, 365 days) under rolling-origin evaluation | AUC vs half-life |
| E3 | RQ2 | Feature sets all / no `kw_*` / no `kw_*`+`self_reference_*`; permutation importance + SHAP or PDP | importance comparison |
| E4 | RQ3 | Threshold at 50/75/90/95th percentile; ROC-AUC vs PR-AUC; log-shares regression as reference | threshold plot |
| E5 | RQ3 | Learning curves 1–100% of training data, 5 seeds per point | learning curves |
| E6 | RQ3 | Logistic regression C from 1e-4 to 1e2, L1 and L2; L1 path | validation curve, L1 path |
| E7 | RQ4 | KMeans / GMM on LDA + sentiment features (k by silhouette/BIC); chi-square test of popularity rate per cluster; optional Isolation Forest | PCA scatter, per-cluster bars |

E1–E3 carry the novelty; E4–E7 cover what the assignment lists. If time runs short, drop the Isolation Forest part of E7 and the rolling-origin part of E2.

## Evaluation protocol

- Sort by publication date (parsed from the URL). Main split: oldest 70% train, next 10% validation (tuning), newest 20% test, used once. Plus 5-fold rolling-origin (expanding window), report mean ± sd.
- Threshold quantiles from training data only.
- Metrics: ROC-AUC, PR-AUC, F1, balanced accuracy; majority-class baseline.
- All preprocessing inside sklearn Pipelines. Fixed seeds. Results as CSV, figures from scripts.
- Models: majority baseline, L1/L2 logistic regression, kNN, random forest, HistGradientBoosting.

## Timeline

| Dates | Milestone |
| --- | --- |
| 12–16 Nov | Repo set up, first meeting: agree RQs and owners, everyone runs the setup |
| 17–23 Nov | Shared protocol code merged; first results for E0, E1, E4 |
| 24–30 Nov | E2, E3, E5, E6, E7 results (A3 also due 29 Nov) |
| 1–3 Dec | Freeze experiments, fix report outline and figures; bring results to the 3 Dec tutorial |
| 4–10 Dec | Full draft before the 10 Dec tutorial, peer review inside the group |
| 11–13 Dec | Revise, cut to 8 pages, rerun everything from a clean environment, zip and submit |

## Workstreams (owners to be agreed)

| Workstream | Scope | Owner |
| --- | --- | --- |
| Shared code & reproducibility | data, splits, evaluation, run-all script, E0 | |
| Time & drift | E1, E2 | |
| Leakage & interpretability | E3 | |
| Threshold, data size, regularisation | E4, E5, E6 | |
| Clustering & anomalies | E7 | |

## References (verify each before citing)

- Fernandes, Vinagre & Cortez (2015). A proactive intelligent decision support system for predicting the popularity of online news. EPIA. — dataset, baseline (RF, AUC 0.73).
- Yang et al. (2014). Active learning for networked data. WSDM. — motivation for learning curves.
- Yang, Tang & Zhang (2014). Streaming active learning. CIKM. — drift / retweet prediction.
- Szabo & Huberman (2010). Predicting the popularity of online content. CACM. — early saturation of popularity.
- Kaufman et al. (2012). Leakage in data mining. KDD / TKDD. — leakage audit.
- Gama et al. (2014). A survey on concept drift adaptation. ACM Computing Surveys.
- Saito & Rehmsmeier (2015). The precision-recall plot is more informative than the ROC plot on imbalanced datasets. PLOS ONE.
- Pedregosa et al. (2011). Scikit-learn. JMLR.
