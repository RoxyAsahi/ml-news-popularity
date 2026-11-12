# Task queue

Claude works through this list top to bottom on a schedule (every 3 hours until the deadline).
Each run: take the first unchecked task, do it on a branch, run the checks below, open a PR,
self-review the diff, merge, tick the box here with the PR number, and report in the project thread.
Roxy can reorder, add or delete lines at any time; the next run follows whatever is here.

**Checks before every merge:** changed scripts run end to end; numbers quoted in docs/report
match `results/tables/`; no data files committed; report stays ≤ 8 pages once it compiles.

**Stop and ask Roxy instead of doing it:** anything that changes the research questions or the
storyline, inviting people, deleting results.

**Do not raise submission matters with Roxy** (Canvas upload, solo-submission permission, author line):
Roxy said on 2026-11-12 to leave them alone and focus on improving the project. Keep the author line as is.

**When the queue is empty:** add the next most valuable improvement (check the latest review in
docs/ or the project's review artifacts first) as a new item, then do it.

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
- [x] Report: Data section + Table 1 (feature groups) + Fig. 1 (shares histogram / shares vs age) (#16)
- [x] Report: Methodology section (#17)
- [x] Report: Results RQ1 + RQ2 (combined figure: drift + ablation)
- [x] Report: Results RQ3 (one combined figure: threshold, learning curve, regularisation)
- [x] Report: Results RQ4 (cluster figure)
- [x] Report: Table 2 (main results, tuned on validation)
- [x] Report: final float placement / layout pass (resolved by #20: figures follow reference order, 7 pages)
- [x] Report: Introduction, Related Work, Discussion, Conclusion, Abstract (first full draft, 6 pages)
- [x] Fill the two Yang et al. 2014 titles + DOIs (checked via web search)
- [x] Review round 2: E1b staleness vs difficulty, ALE figure (E8), weekly precision@top-5% (E4b), text corrections, all references with DOIs/pages (Fernandes claim confirmed from abstract + .names file)
- [x] Check page count ≤ 8 (7 pages after review round 2)
- [x] Clean-environment reproduction (2026-11-12): fresh clone + venv (numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1), `./run_all.sh` in 26 min, all 44 tables match main (max abs diff 1.2e-10)
- [x] Build the submission zip: `./make_submission.sh` -> `dist/group3_news_popularity.zip` (report.pdf + code/, no data)
- [x] Review round 3 (adversarial grading, items 2–8): age not identifiable, kw values from collection date (upper bound), weekly block bootstrap, sliding-window variance, validation-chosen reference model, wider kNN grid, HistGB temporal early stopping, Breslow–Day + per-channel effects, all Isolation Forest settings, method citations, VCPToolBox footnote, figure fonts, abstract rewrite; 8 pages
- [x] Review round 4 fixes (kw alone 0.016, 'likely optimistic', per-fold block SE 0.008 vs period sd 0.017, staleness as upper bound, RQ1 wording, Related Work compressed, larger Fig 3/4 fonts)
- [~] ~~Prose tightening, seed robustness, calibration~~ superseded by the v2 storyline below (2026-11-12).

## v2 storyline (decided 2026-11-12)

Plan: https://claude.ai/artifact/H5xxBoLJgx5Qad7ADPgKrc ("新闻热度项目 v2"). Working title:
*Same Story, Different Fate: Context, Content and Chance in Online News Popularity.*
Feasibility code: branch `claude/project-thread-byh7ye`, folder `explore/` (f1–f9). Formal versions
live in `src/newspop/` + `scripts/` and are **strictly past-only**: every score used for an RQ comes
from a model trained on articles published before the scored article (rolling origin), never
from cross-fitting. Report rewrite happens on main only once the RQ results exist.
Independent review after each milestone (end of each week).

### Week 1 (13–19 Nov): infrastructure + RQ1 "what is AUC 0.74 made of?"

- [x] (#25) `newspop.stories`: slug parsing (strip trailing ids), TF-IDF slug vectors (1–2 grams, sublinear), twin pairs (cos ≥ 0.5, ≤ 7 days apart), story = connected components of the twin graph; channel and day helpers; tests on a few known pairs; script writes `results/tables/stories_pairs.csv` summary (counts by threshold, same-channel share)
- [x] (#25, 180 pairs, series filter added after review 3) Twin precision: sample 200 pairs stratified by similarity band, annotate "same story / same topic / unrelated" from slug + date + channel (`data/annotations/twins_200.csv`, committed; annotation protocol and annotator disclosed in docs); precision per band with Wilson CI; choose the threshold used downstream
- [x] (#25) RQ1 hierarchical pairwise AUC, strictly past-only: rolling-origin out-of-sample scores (HistGB + LogReg), pairwise AUC over all / same week / same day / same channel / same channel-day / twins / same-channel twins; weekly cluster bootstrap CIs; figure (ladder plot)
- [x] (#25, done as fixed-effect variance shares + twin correlation vs matched neighbours + first-vs-later) RQ1 nested variance decomposition of log shares: random effects week ⊃ channel×week ⊃ story ⊃ residual (statsmodels MixedLM or REML via variance components); share of variance at each level with bootstrap CIs; how much of each level the model's out-of-sample score captures
- [ ] Milestone review 1 (independent reviewer thread) → apply fixes

### Week 2 (20–26 Nov): RQ2 content within stories + RQ3 attention competition
- [ ] RQ2: twin-difference regression vs cross-sectional regression of log shares on content features (length, subjectivity, polarity, images/videos, links); paired CIs; which effects survive topic control
- [ ] RQ3: same-day volume effect (week + weekday + channel FE, day-clustered CIs), robustness to content controls; semantic-memory time decay as negative result (merges old E2)
- [ ] Milestone review 2

### Week 3 (27 Nov–3 Dec): RQ4 honest uncertainty + packaging
- [ ] RQ4: CQR + adaptive conformal (ACI) monthly coverage under drift; selective prediction risk–coverage (merges old E1/E4)
- [ ] Figures, `run_all.sh` update, clean-environment reproduction
- [ ] Milestone review 3

### Week 4 (4–10 Dec): report rewrite on main (old E3→RQ1 context, E1/E4→RQ4, E2→RQ3, E5/E6 one sentence + appendix, E7 dropped from report, E7b/E8 replaced by RQ2)
### 11–13 Dec: review only, no new experiments
