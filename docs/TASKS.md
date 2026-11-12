# Task queue

Claude works through this list top to bottom on a schedule (every 3 hours until the deadline).
Each run: take the first unchecked task, do it on a branch, run the checks below, open a PR,
self-review the diff, merge, tick the box here with the PR number, and report in the project thread.
Roxy can reorder, add or delete lines at any time; the next run follows whatever is here.

**Checks before every merge:** changed scripts run end to end; numbers quoted in docs/report
match `results/tables/`; no data files committed; report stays ≤ 8 pages once it compiles.

**Stop and ask Roxy instead of doing it:** anything that changes the research questions or the
storyline, submitting on Canvas, inviting people, deleting results.

Deadline: Sun 13 Dec 2026, 23:59 CET.

## Queue

- [x] Repo skeleton, EDA, pilot (main)
- [x] E1 temporal drift (#1)
- [x] E3 leakage audit (#2)
- [x] E4–E6 threshold, learning curves, regularisation (#3)
- [x] E7 clustering and anomalies (#4)
- [x] Report skeleton + run_all.sh (#5)
- [x] Task queue + deterministic L1 logistic regression (#6)

Review notes (2026-11-12): https://claude.ai/artifact/UnYjVnCpepjSA6yGzj5XMr — items R1–R8 below follow its priority order.

- [x] **R1 E3 rewrite (#8):** `kw_*` are point-in-time features that carry a clock, not leakage (kw_max_max = max shares of earlier articles for 99.7% of rows; kw_min_min = -1 in 8.5% of 2013 vs 99.9% of 2014). New point-in-time check script + staircase figure (kw_max_max vs running max over time). Drop pure clock columns (kw_max_max, kw_min_min, kw_min_min_missing; if needed kw_max_min, kw_min_max) and redo adversarial validation and predictive AUC. Rename RQ2 accordingly. Limitation: earlier articles' shares measured at collection time (Jan 2015).
- [x] **R2 Main results table (#9):** tune HistGB, RF, LogReg, kNN on the validation block; touch test once; bootstrap 95% CIs for everything. Add bootstrap + paired-comparison helpers to `newspop/evaluate.py`. Reviewer's HistGB test AUC 0.733 [0.720, 0.743] → 0.01-level differences are "no detectable difference".
- [x] **R3 E1 (#10):** error bars on monthly AUC (else no "decay" claim); rewrite "strong covariate shift" per the clock finding; add label-shift correction (shift probabilities to validation positive rate) as a one-line negative result (accuracy 0.667→0.661, Brier worse).
- [x] **R4 E3 ablation (#11):** report paired differences across rolling folds with intervals, not two means with separate sds.
- [x] **R5 E4 (#12):** decision threshold currently set with `np.quantile` on test scores (transductive peeking) → set it on the validation block. Paired bootstrap CI for regressor vs classifier at the 95th percentile (~300 positives).
- [x] **R6 E7 (#13):** negative-sentiment effect is confounded by channel → CMH test or logistic regression with channel. Isolation Forest: profile the outliers; contamination 1% and 5% too.
- [x] **R7 E2 (#14):** half-life weighting under rolling-origin folds, paired differences, written as a negative result tied to "ranking is stable".
- [x] **R8 Misc (#15):** state the threshold convention once (overall median 1,400 vs training-block median 1,500); drop jupyter from requirements.txt; compile LaTeX early.
- [x] Get LaTeX compiling (#15): `report/build.sh`; in a fresh container first run `apt-get update && apt-get install -y --no-install-recommends texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra`
- [ ] Report: Data section + Table 1 (feature groups) + Fig. 1 (shares histogram / shares vs age)
- [ ] Report: Methodology section
- [ ] Report: Results RQ1 + RQ2 (combined figure: drift + ablation)
- [ ] Report: Results RQ3 (one combined figure: threshold, learning curve, regularisation)
- [ ] Report: Results RQ4 (cluster figure)
- [ ] Report: Introduction, Related Work, Discussion, Conclusion, Abstract
- [ ] Verify every reference against its source; fill the two Yang et al. 2014 titles
- [ ] Check page count ≤ 8; cut or merge figures if needed
- [ ] Clean-environment reproduction: fresh venv, `pip install`, `./run_all.sh`, diff results
- [ ] Build the submission zip (report PDF + code, no data) and hand it to Roxy for upload
