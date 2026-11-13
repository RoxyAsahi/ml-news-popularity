# explore/kw_rebuild: rebuilding the keyword memory, and a shifted `url` column

Exploration of 2026-11-13 (not part of `run_all.sh`, nothing in `report/` touched). The task was to rebuild the
dataset's `kw_*` columns as a stream from keys we can see (URL-slug words, real headline words) using only earlier
articles. On the way it turned out that the `url` column of the course CSV is misaligned, which changes what
every headline-based analysis in this repository measured. Revised after the review of the teacher-view thread
(outcome-based alignment check, wording of "past-only", status of the judge numbers, robustness).

## 1. The `url` column is shifted against all other columns from 2013-06-06 on

In the original row order of `OnlineNewsPopularity.csv`, the features and `shares` of row *i* belong to the URL of
row *i + k*. `k` is 0 for the first 8,009 rows, 1 from row 8,009 (content date 2013-06-06), 2 from 2013-11-25, and
grows quickly from April 2014 to 152 at the end of the file (`results/tables/k4_offset_steps.csv`, 143 steps).
About 80% of the rows therefore carry the URL of another article. To our knowledge this is undocumented (the
review thread searched the UCI page, Kaggle, OpenML, GitHub and papers using the data and found no mention).

**Outcome-based check** (`k7_offset_profile.py`, proposed by the review thread). Shares played no part in the
alignment, so they can test it. A TF-IDF ridge on headlines is fit on file rows 0–4,599 and scored on later rows
with the headline taken at offset *k + d*; the entries are within-day correlations of its prediction with log
shares:

| Rows | n | url as distributed | d = -1 | **d = 0** | d = +1 |
| --- | --- | --- | --- | --- | --- |
| rows 4,600–8,008 (aligned, held out) | 3,409 | 0.263 | 0.004 | **0.263** | -0.005 |
| 2013-06-06..2013-12-31 | 10,188 | 0.005 | 0.017 | **0.231** | 0.029 |
| 2014 H1 | 9,837 | -0.010 | 0.021 | **0.204** | 0.008 |
| 2014 H2 | 11,458 | 0.007 | 0.016 | **0.151** | -0.002 |
| within 5 rows of an offset step | 1,474 | 0.009 | -0.006 | **0.152** | -0.005 |

d = ±2, ±3 are also within ±0.03 (±0.06 near the steps). The signal sits exactly at the estimated offset.

**Fingerprints used by the alignment** (`k4_realign.py`) and an independent section check (`k6_check_alignment.py`):

| Check | rows before 2013-06 | later rows, url as distributed | later rows, url re-aligned |
| --- | --- | --- | --- |
| `n_tokens_title` = alphanumeric runs in the HTML-escaped real headline + 1 | 86% | 13–14% (chance 14%) | 80–91% |
| `title_sentiment_polarity`, `title_subjectivity` = TextBlob on the real headline | 94% | 20–22% (chance 23%) | 90–94% |
| date in the URL = 2015-01-08 minus `timedelta` (which agrees with `weekday_is_*` on 100% of rows) | 100% | 98% falling to 9% in 2014 Q4 | 100% |
| section predicted from slug words, classifier fit on Jan–Apr 2013 (not used by the alignment) | 54% held out | 16–19% (majority class 23%) | 44–51% |

The alignment is a Viterbi path over a non-decreasing offset with the date as a hard constraint and the title
token count and sentiment as fingerprints. 39,492 rows get their URL back; the true URLs of the last 152 rows lie
beyond the end of the file, and 152 URLs in the file belong to no row. On 1,338 rows (3.4%; 2.8% in the untouched
first months) neither fingerprint agrees; the longest run of such rows is 3, so there is no misaligned stretch.

What is affected: everything derived from `url` after 2013-06-06, i.e. the slug, the real headline, the
publication date taken from the URL (68% of 2014 H2 rows are 1 to 5 days off), same-story twins, and the headlines
shown to the JEV judge. What is not: all row-wise columns, including `kw_*`, `self_reference_*` and `shares`.
The true publication date is `2015-01-08 - timedelta`.

`kwlib.frame("given")` reproduces the data as every analysis up to v19 used it; `kwlib.frame("true")` is the
re-aligned frame. The full table is `data/interim/kw_rebuild/realigned.csv` (copy in the project folder under
`kw_rebuild/`).

## 2. Headline words do carry popularity; the "break" in June 2013 was the shift

**Protocol: past articles, retrospective labels.** The memory of an article uses only articles published on
earlier days, but their shares are the January-2015 counts, as for every row of the dataset. For an article
published yesterday that count was not known today. The dataset's own `kw_*` columns have exactly this property,
so comparisons with them are like for like. The stricter number is the **30-day embargo**: only articles at least
30 days old feed the memory, whose share counts were essentially final by then.

Headline-word memory = mean empirical-Bayes lift of the headline's words. Mean Spearman with log shares inside a
day and section, 95% intervals over days (`k3_headline_signal.py`, figure `results/figures/k3_headline_signal.png`):

| | 2013-03-01..06-05 | 2013-06-06..12-31 | 2014 |
| --- | --- | --- | --- |
| headline-word memory, url as distributed | 0.239 [0.206, 0.272] | 0.005 [-0.018, 0.029] | 0.000 [-0.015, 0.014] |
| headline-word memory, url re-aligned | 0.238 [0.205, 0.271] | 0.292 [0.273, 0.312] | 0.260 [0.245, 0.275] |
| `kw_avg_avg` (dataset) | 0.142 [0.111, 0.174] | 0.204 [0.182, 0.227] | 0.192 [0.177, 0.208] |

On the evaluation window 2013-07..2014-12: **0.251 [0.239, 0.263] with the 30-day embargo, 0.270 [0.258, 0.282]
with a 1-day lag**, against 0.196 [0.183, 0.208] for `kw_avg_avg`.

Split-half reliability of the lifts of frequent headline words inside a day and section is 0.63 before the shift
and 0.00 / -0.01 / -0.04 afterwards with the url as distributed; re-aligned it is 0.58 / 0.58 / 0.42.

Which memory works (`results/tables/k3_horizons.csv`). The variant is chosen on 2013-03..06, before the
evaluation window, and the ranking is the same there:

| Headline-word memory variant (1-day lag) | 2013-03..06 (choice) | 2013-07..2014-12 (evaluation) |
| --- | --- | --- |
| all history, shrinkage | 0.248 | 0.270 |
| decay, half-life 90 / 30 / 7 days | 0.247 / 0.242 / 0.196 | 0.262 / 0.235 / 0.182 |
| window 90 / 30 / 7 days | 0.241 / 0.205 / 0.147 | 0.234 / 0.191 / 0.130 |
| all history, raw-share averages (original recipe, no shrinkage) | 0.183 | 0.220 |
| `kw_avg_avg` | 0.154 | 0.196 |

Word popularity is long-lived, not a short-term trend, and shrinkage matters. Every variant except the 7-day ones
is at or above `kw_avg_avg`. (The "momentum" columns correlate negatively only because the short-window lift is
shrunk towards zero.)

## 3. A reproducible substitute for `kw_*` (v19 protocol)

`kw_*` are built from Mashable's editor tags, which are not in the data. The memory here uses headline and slug
words, so it is not the same feature rebuilt but a substitute anyone can recompute, and it does better.

Monthly rolling origin, HistGB with the v19 parameters, evaluation 2013-07..2014-12, 1,000 bootstrap draws of
days (`k2_evaluate.py`; `results/tables/k2_metrics_true.csv`, `k2_diffs_true.csv`; the same with the url as
distributed in `k2_*_given.csv`, whose `orig_all` row reproduces v19's `hgb` row exactly). The model sets below
receive all memory families at once, so no variant is selected on the evaluation window.

| Features (re-aligned) | AUC | same-day same-section pairs | top pick in day's top 10% |
| --- | --- | --- | --- |
| dataset without `kw_*` | 0.711 [0.704, 0.718] | 59.0% | 31.9% |
| dataset (with `kw_*`), = v19 model | 0.724 [0.718, 0.730] | 60.6% | 34.3% |
| without `kw_*` + word memory, 30-day embargo | 0.730 | 61.4% | 37.6% |
| without `kw_*` + word memory, 1-day lag | 0.738 [0.732, 0.743] | 62.1% | 39.3% |
| without `kw_*` + raw-share mimic only (weakest sensible variant) | 0.729 | 61.2% | 36.7% |
| dataset + word memory, 30-day embargo | 0.736 | 62.0% | 36.7% |
| dataset + word memory, 1-day lag | 0.742 [0.736, 0.748] | 62.5% | 36.7% |
| word memory alone | 0.710 | 60.7% | 30.5% |
| `kw_*` alone | 0.664 | 58.3% | 30.5% |
| ridge on headline TF-IDF (v19 text baseline) | 0.684 | 60.0% | 33.6% |
| random | | | 11.1% |

Paired differences against the v19 model (AUC; section pairs in points):

| | day bootstrap | week-block bootstrap | doubtful rows dropped |
| --- | --- | --- | --- |
| word memory (30-day embargo) instead of `kw_*` | +0.0058 [+0.0021, +0.0096]; +0.9 [+0.5, +1.3] | +0.0058 [+0.0020, +0.0098] | +0.0054 [+0.0018, +0.0094] |
| word memory (1-day lag) instead of `kw_*` | +0.0135 [+0.0095, +0.0176]; +1.6 [+1.2, +2.0] | +0.0135 [+0.0093, +0.0176] | +0.0132 [+0.0093, +0.0174] |
| raw-share mimic instead of `kw_*` | +0.0050 [+0.0014, +0.0086] | +0.0050 [+0.0014, +0.0087] | +0.0044 [+0.0009, +0.0080] |
| word memory (30-day embargo) added | +0.0113 [+0.0080, +0.0145] | +0.0113 [+0.0081, +0.0146] | +0.0111 [+0.0078, +0.0144] |
| word memory (1-day lag) added | +0.0176 [+0.0140, +0.0213] | +0.0176 [+0.0139, +0.0217] | +0.0174 [+0.0139, +0.0211] |

"Doubtful rows dropped" leaves the 1,338 rows where neither fingerprint agrees out of the evaluation
(`k2_robust_*_true.csv`). The top-pick differences are positive but their intervals mostly include zero (545
days). Section memory alone adds nothing (the model already has the section flags). With the url as distributed
the same word features lower the AUC (0.704 vs 0.711 without `kw_*`), and the TF-IDF ridge sits at 0.510, which is
what v19 reported.

## 4. The cached JEV answers: indicative only, not a measurement

`k5_rescore_judge.py` scores the cached answers of v19 against the shares of the rows the headlines really belong
to (no new API calls). On rows from 2013-06-06 on, the accuracy on the 4,000 judged pairs moves from 48.9%
[47.2, 50.5] to 59.4% [57.8, 61.0], and the early-2013 versus later gap that v19b read as memorisation disappears
(59.5% before the shift, 59.4% after). This shows the judge has signal on Mashable headlines. It should not be
quoted as a result, because the test was drawn on shifted rows:

* the "same-section" pairs are same-day pairs whose true sections mostly differ (546 of 3,070 share a true
  section; accuracy there 56.8% [52.5, 60.9]);
* the prompts showed section labels taken from the shifted rows (the day lineups themselves are true days, since a
  URL carries its own date);
* the memorisation probe of v19b compared 2013 H1 with later rows that were misaligned, so it has to be redone, and
  the paraphrase test has to be extended beyond early 2013.

For the same reason the feature sets with the judge's lineup score (`orig_all+judge`, `orig_all+full+judge`: AUC
0.743 and 0.758 in `k2_metrics_true.csv`) are **contamination untested** and stay out of the table above. Pairs
and lineups need to be redrawn on aligned data first. For orientation only: on the same re-scored pairs the
past-only word memory gets 62.8%, the judge 59.4% and `kw_avg_avg` 59.8%.

## Limits

* Retrospective labels (see section 2). The 30-day embargo is the number to quote when strictness matters.
* The keys are headline and slug words, not Mashable's editor tags.
* Title fingerprints agree on 80–91% of re-aligned rows, the same rate as in the untouched first months (headlines
  were edited after 2015, tokenisation is reverse-engineered). Rows without a recovered headline (176) are left
  out of the metrics, as v19 did with its 24.
* The empirical-Bayes strength is estimated once on the months before 2013-07.
* Slug-based results elsewhere in the repository (twins, split attention, v19 headline analyses, publication days
  taken from the URL) were computed on the shifted column and are not corrected here.

## Scripts (run in this order from the repo root)

| Script | What it does |
| --- | --- |
| `k4_realign.py` | measures the shift and writes the row-to-URL table |
| `k6_check_alignment.py`, `k7_offset_profile.py` | checks with sections / videos, and with shares at neighbouring offsets |
| `k1_build_features.py true` / `given` | keyword memory (windows, decay, shrinkage, embargo lags 1–30 days) |
| `k3_headline_signal.py` | headline signal by period under both alignments, variants, figure |
| `k5_rescore_judge.py` | re-scores the cached JEV answers (indicative) |
| `k2_evaluate.py true` / `given` | rolling evaluation of feature sets with robustness runs (about 25 minutes of fits each on 4 cores, resumable) |

Needs `pyarrow` and `textblob` on top of `requirements.txt`.
