# Editorial levers: sample, estimator and scripts (for citing)

**Sample.** 38,302 Mashable articles from `OnlineNewsPopularity.csv` (39,644 rows), in file order. Dropped: 1,181
rows with unparsed text (`n_tokens_content = 0`), 1 corrupt row (`n_unique_tokens > 1`), the last 152 rows whose
true url is outside the file, 24 without a sitemap headline. Publication date = 2015-01-08 minus `timedelta`
(agrees with the dataset's weekday flags on every row); range 2013-01-07 to 2014-12-27, 720 days, 103 weeks.
Urls and real headlines come from `realigned.csv` (file_row -> true_url, true_title). The day's volume counts
all 39,644 dataset rows by content date and section. Outcome: log(1 + shares).

**Estimator.** Partially linear double machine learning (Robinson 1988; Chernozhukov et al. 2018): log shares and
each lever are predicted from the confounders W with LightGBM (400 trees, learning rate 0.03, 15 leaves), out of
fold, and the residual of log shares is regressed on the residual(s) of the lever(s). Cross-fitting uses 5 folds
made of whole calendar weeks, so articles from the same day or week never sit on both sides of a split; this is
repeated 5 times with different week assignments and the median estimate is reported (Chernozhukov et al. 2018,
eq. 3.14). Standard errors are clustered by week. W = publication date, day of week, section, the five LDA
topics, the keyword-history columns `kw_*`, the number of keywords, and 64 principal components of the real
headline's bge-base-en-v1.5 vector (88 columns). Levers enter as log(1 + count); "adjusted" numbers have all
levers in the final regression together. The weekend term uses W without the day of week.

**Checks.** Robustness values (Cinelli & Hazlett 2020); negative-control outcome = log shares of the older Mashable
articles the article links to (32,323 articles with at least one such link; not valid for the internal-link
lever); placebo lever = same-section volume 7 days later; random folds and extra-trees nuisances; same-story
twin differences (2,720 pairs); for crowding: within-week identification, neighbouring-day volumes, composition
of content-predicted shares, log-linearity and overlap.

**Scripts** (`explore/editorial_dml/`, branch `claude/project-thread-3zjyz6`, PR #43): `common.py` (sample, W,
cross-fitting), `d1_naive.py` (raw, SHAP, model what-if), `d2_dml_ladder.py` (adjusted estimates),
`d5_sensitivity.py` (robustness values, negative control, placebo, twins), `d7_weekend_crowding.py` (weekend and
crowding checks), `d8_cite_tables.py` (the two `cite_*.csv` files).

**Files to cite.** `cite_levers.csv` (per lever: raw, model what-if, adjusted jointly and alone, negative control,
robustness value; % change in shares per doubling of 1 + count, or per SD for wording) and
`cite_weekend_crowding.csv` (weekend premium raw / adjusted / predicted from Monday-Friday slopes / at fixed
volume, the volume slopes, placebo, and every check with its interval).

**Wording agreed with the review.** Article levers: "adjusted associations of 2-4% per doubling; a negative
control indicates these are mostly or wholly residual confounding by story or format". Weekend: "the premium is
accounted for by crowding or exposure on the day", not "is". Headline sentiment: "no association beyond what
the headline's content carries". Lever values are read off the page as crawled in January 2015.

**Accepted wording (teacher-view review, round 2).** "Within the same week and weekday, a day with twice as many
articles carries 8-12% fewer shares per article; observable composition explains at most a quarter of this.
Extrapolated to weekend volumes, this slope is large enough to account for the whole weekend premium (+34%
[23, 46] predicted, +29% observed), though weekdays almost never reach weekend volumes, and the data cannot
separate crowding or exposure from unobserved same-day selection." Article levers: one clause, "adjusted
associations of 2-4% that fail a negative control". Scope in the report: a short paragraph after RQ2 (about five
sentences, at most a three-row table).
