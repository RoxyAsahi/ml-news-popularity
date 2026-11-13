# explore/kw_rebuild: rebuilding the keyword memory, and a shifted `url` column

Exploration of 2026-11-13 (not part of `run_all.sh`, nothing in `report/` touched). The task was to rebuild the
dataset's `kw_*` columns as a stream from keys we can see (URL-slug words, real headline words) using only earlier
articles. On the way it turned out that the `url` column of the course CSV is misaligned, which changes what
every headline-based analysis in this repository measured.

## 1. The `url` column is shifted against all other columns from 2013-06-06 on

In the original row order of `OnlineNewsPopularity.csv`, the features and `shares` of row *i* belong to the URL of
row *i + k*. `k` is 0 for the first 8,009 rows, 1 from row 8,009 (content date 2013-06-06), 2 from 2013-11-25, and
grows quickly from April 2014 to 152 at the end of the file (`results/tables/k4_offset_steps.csv`, 144 steps).
About 80% of the rows therefore carry the URL of another article.

Evidence (`k4_realign.py`, `k6_check_alignment.py`):

| Check | rows before 2013-06 | later rows, url as distributed | later rows, url re-aligned |
| --- | --- | --- | --- |
| `n_tokens_title` = alphanumeric runs in the HTML-escaped real headline + 1 | 86% | 13–14% (chance 14%) | 80–91% |
| `title_sentiment_polarity`, `title_subjectivity` = TextBlob on the real headline | 94% | 20–22% (chance 23%) | 90–94% |
| date in the URL = 2015-01-08 minus `timedelta` (which agrees with `weekday_is_*` on 100% of rows) | 100% | 98% falling to 9% in 2014 Q4 | 100% |
| section predicted from slug words, classifier fit on Jan–Apr 2013 (not used by the alignment) | 54% held out | 16–19% (majority class 23%) | 44–51% |

The alignment is a Viterbi path over a non-decreasing offset with the date as a hard constraint and the title
token count and sentiment as fingerprints. 39,492 rows get their URL back; the true URLs of the last 152 rows lie
beyond the end of the file, and 152 URLs in the file belong to no row.

What is affected: everything derived from `url` after 2013-06-06, i.e. the slug, the real headline, the
publication date taken from the URL (68% of 2014 H2 rows are 1 to 5 days off), same-story twins, and the headlines
shown to the JEV judge. What is not: all row-wise columns, including `kw_*`, `self_reference_*` and `shares`.
The true publication date is `2015-01-08 - timedelta`.

`kwlib.frame("given")` reproduces the data as every analysis up to v19 used it; `kwlib.frame("true")` is the
re-aligned frame. The full table is `data/interim/kw_rebuild/realigned.csv` (copy in the project folder under
`kw_rebuild/`).

## 2. Headline words do carry popularity; the "break" in June 2013 was the shift

Past-only headline-word memory (`tit_all_all_mean`: mean empirical-Bayes lift of the headline's words, learned only
from articles of earlier days), mean Spearman with log shares inside a day and section, 95% intervals over days
(`k3_headline_signal.py`, figure `results/figures/k3_headline_signal.png`):

| | 2013-03-01..06-05 | 2013-06-06..12-31 | 2014 |
| --- | --- | --- | --- |
| headline-word memory, url as distributed | 0.239 [0.206, 0.272] | 0.005 [-0.018, 0.029] | 0.000 [-0.015, 0.014] |
| headline-word memory, url re-aligned | 0.238 [0.205, 0.271] | 0.292 [0.273, 0.312] | 0.260 [0.245, 0.275] |
| `kw_avg_avg` (dataset) | 0.142 [0.111, 0.174] | 0.204 [0.182, 0.227] | 0.192 [0.177, 0.208] |

Split-half reliability of the lifts of frequent headline words inside a day and section is 0.63 before the shift
and 0.00 / -0.01 / -0.04 afterwards with the url as distributed; re-aligned it is 0.58 / 0.58 / 0.42.

Which memory works (re-aligned, 2013-07..2014-12, `results/tables/k3_horizons.csv`): all history with shrinkage
0.270; raw-share averages as in the original recipe 0.220; windows of 90 / 30 / 7 days 0.234 / 0.191 / 0.130;
decay with half-life 90 / 30 / 7 days 0.262 / 0.235 / 0.182. Word popularity is long-lived, not a short-term
trend, and shrinkage matters. A memory fed only by articles at least 30 days old still reaches 0.251, so the
signal does not rest on freshly published articles whose January-2015 share counts would not have been known.
(The "momentum" columns correlate negatively only because the short-window lift is shrunk towards zero.)

## 3. Rebuilt memory versus the dataset's `kw_*` (v19 protocol)

Monthly rolling origin, HistGB with the v19 parameters, evaluation 2013-07..2014-12, 1,000 bootstrap draws of
days (`k2_evaluate.py`; `results/tables/k2_metrics_true.csv`, `k2_diffs_true.csv`; the same with the url as
distributed in `k2_*_given.csv`, whose `orig_all` row reproduces v19's `hgb` row exactly).

| Features (re-aligned) | AUC | same-day same-section pairs | top pick in day's top 10% |
| --- | --- | --- | --- |
| dataset without `kw_*` | 0.711 [0.704, 0.718] | 59.0% | 31.9% |
| dataset (with `kw_*`), = v19 model | 0.724 [0.718, 0.730] | 60.6% | 34.3% |
| without `kw_*` + rebuilt word memory | 0.738 [0.732, 0.743] | 62.1% | 39.3% |
| the same, memory fed only by articles >= 30 days old | 0.730 | 61.4% | 37.6% |
| dataset + rebuilt memory | 0.742 [0.736, 0.748] | 62.5% | 36.7% |
| dataset + rebuilt memory + cached JEV lineup score | 0.758 [0.753, 0.763] | 64.1% | 42.6% |
| rebuilt memory alone | 0.710 | 60.7% | 30.5% |
| `kw_*` alone | 0.664 | 58.3% | 30.5% |
| ridge on headline TF-IDF (v19 text baseline) | 0.684 | 60.0% | 33.6% |
| random | | | 11.1% |

Paired differences against the v19 model: rebuilt memory instead of `kw_*` +0.0135 AUC [+0.0095, +0.0176] and
+1.6 points on section pairs [+1.2, +2.0]; added to it +0.0176 [+0.0140, +0.0213]; with the judge score +0.0336
[+0.0296, +0.0379] and +8.3 points on the top pick [+3.7, +13.2]. Section memory alone adds nothing (the model
already has the section flags). With the url as distributed the same word features lower the AUC (0.704 vs 0.711
without `kw_*`), and the TF-IDF ridge sits at 0.510, which is what v19 reported.

## 4. The cached JEV answers, scored against the right shares

No new API calls (`k5_rescore_judge.py`, `results/tables/k5_judge_rescored.csv`). Rows from 2013-06-06 on:

| | as scored in v19 | re-scored |
| --- | --- | --- |
| 4,000 "same-day same-section" pairs, accuracy | 48.9% [47.2, 50.5] | 59.4% [57.8, 61.0] |
| its most confident 25% | | 67.7% [64.4, 71.0] |
| 4,000 same-day pairs, accuracy | 51.4% [49.6, 53.1] | 58.0% [56.3, 59.7] |
| whole-day lineup, top pick in the day's top 10% (random 11.2%) | 11.6% [9.1, 14.4] | 23.3% [20.0, 27.0] |
| whole-day lineup, within-day Spearman | -0.003 | 0.160 [0.148, 0.173] |

On the same re-scored section pairs the past-only word memory gets 62.8% and `kw_avg_avg` 59.8%. The early-2013
versus later gap that v19b read as memorisation disappears (59.5% before the shift, 59.4% after).

## Limits

* Shares are the January-2015 crawl values for every article; "past-only" restricts which articles feed the
  memory, not when their shares were counted. The 30-day embargo bounds how much that matters.
* The keys are headline and slug words. Mashable's editor tags, which the original `kw_*` use, are not in the data.
* Title fingerprints agree on 80–91% of re-aligned rows, the same rate as in the untouched first months (headlines
  were edited after 2015, tokenisation is reverse-engineered). Around the 144 offset steps a few rows may be off by
  one. Rows without a recovered headline (176) are left out of the metrics, as v19 did with its 24.
* After re-scoring, the judged "same-section" pairs are same-day pairs whose true sections differ in most cases
  (546 of 3,070 share a true section; accuracy there 56.8% [52.5, 60.9]). The lineup prompt also showed section
  labels taken from the shifted rows. The memorisation probe of v19b has to be redone on aligned data before the
  judge score is used as a feature.
* The empirical-Bayes strength is estimated once on the months before 2013-07.

## Scripts (run in this order from the repo root)

| Script | What it does |
| --- | --- |
| `k4_realign.py` | measures the shift and writes the row-to-URL table |
| `k6_check_alignment.py` | independent check with sections and videos |
| `k1_build_features.py true` / `given` | past-only keyword memory (windows, decay, shrinkage, embargo lags 1–30 days) |
| `k3_headline_signal.py` | headline signal by period under both alignments, horizons, figure |
| `k5_rescore_judge.py` | re-scores the cached JEV answers |
| `k2_evaluate.py true` / `given` | rolling evaluation of feature sets (about 25 minutes each on 4 cores, resumable) |

Needs `pyarrow` and `textblob` on top of `requirements.txt`.
