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
- [x] Task queue + deterministic L1 logistic regression (this PR)
- [ ] **Revise E3 + E1 after review (2026-11-12):** `kw_*` are point-in-time features (kw_max_max ≤ max shares of earlier articles for 99.99% of rows, equal for 99.7%; kw_min_min = -1 in 8.5% of 2013 vs 99.9% of 2014 rows). Reframe RQ2 from "leakage audit" to "are these features point-in-time, and do they carry a clock?"; add a staircase figure of kw_max_max vs running max of earlier shares over time; soften E1's "strong covariate shift" (kw_max_max / kw_min_min are near-constant inside the chronological test block). Remaining caveat for limitations: earlier articles' shares were measured at collection time (Jan 2015). Wait for the full review notes before finalising.
- [ ] Work through the remaining points of the independent review (CI, tuning, E2, E4 threshold, E7 confounding)
- [ ] E2 proper: half-life weighting under rolling-origin folds, HistGB and LogReg, 5 half-lives
- [ ] Get LaTeX compiling (tectonic or TeX Live in the container; fall back to documenting Overleaf)
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
