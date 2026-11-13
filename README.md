# Online News Popularity — ML Final Group Project

UvA *Machine Learning* (6012B0809Y), Final Group Project, Group 3.
Deadline: **Sunday 13 December 2026, 23:59 CET** (Canvas). Deliverable: one zip with a ≤ 8-page PDF report and all code.

## The `url` column of the benchmark is shifted

From row 8,009 on (articles published from 6 June 2013), the `url` column of `OnlineNewsPopularity.csv` is shifted
against all other columns: the features and shares of a row belong to the URL written 1 row further down at first,
152 rows further down by December 2014. 31,635 rows (79.8%) carry the URL of another article. The file on the UCI
repository is byte-identical to the course copy (MD5 `6e6f1cfc9e03cfa1bc42dbc5c24075e7`), so this is the benchmark
itself. Features and shares stayed together, so feature-only analyses are not affected; anything that joins the
URL, the date in it, its slug or scraped text to a row is.

* `data/external/url_realignment.csv`: for every row of the file, the offset to the row that holds its real URL
  (empty for the last 152 rows, which have none), plus two per-row agreement flags. Built by
  `scripts/a1_realign_urls.py`; validated on held-out signals by `scripts/a2_alignment_validation.py`.
* `newspop.data.load()` returns the **repaired** data by default: `url` is the recovered URL, `url_in_file` the one
  written in the row, `publish_date` comes from `timedelta` (2015-01-08 minus `timedelta` days).
  `load(repaired=False)` gives the file as distributed (URL as written, date parsed from it).

## Research questions (report v20: *What Can Be Called Before Publication? A Misaligned Benchmark, Its Repair, and the Signal in the Headline*)

All Mashable results use the repaired data, a strictly past-only protocol (monthly rolling origin) and
decision-level metrics (day ranking, the day's top pick, same-day same-section pairs).

| RQ | Question | Scripts |
| --- | --- | --- |
| RQ1 | Is the `url` column aligned with features and shares, and can a repair be verified without ground truth? | `a1_realign_urls.py`, `a2_alignment_validation.py` |
| RQ2 | How much can be called before publication, and from which signals (history, body, headline)? | `v1_past_only_called.py`, `v3_signal_groups.py`, `v4_top_pick_value.py`, `v5_learners.py`, `v6_headline_terms.py` |
| RQ3 | Can a zero-shot decision model (JEV) call popularity from the headline alone, and does it add to supervised models? | `v2_judge_summary.py`, `explore/jev/` |
| RQ4 | Does the same judge carry over to choosing between headlines of one article (Upworthy randomized tests)? | `v2_judge_summary.py`, `explore/jev/` |

Real headlines: `data/external/mashable_sitemap_titles.csv` (40,161 sitemap entries;
`scripts/fetch_mashable_titles.py`, `src/newspop/titles.py`).

**Earlier reports are superseded.** `report/v18/` and `report/v19/` were written before the shift was found. Their
Mashable results that used the URL, its date, its slug or the headline (same-story twins in v18; "the headline
carries no signal after mid-2013" in v19) are artefacts of the shifted column and should not be used. The scripts
behind them (`e*`, `s*`, `explore/f*`) are kept for the record only.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `import newspop` work everywhere
```

### Data (not committed)

Put `OnlineNewsPopularity.csv` in `data/raw/`. Either:

- unzip `ML Group Project Datasets.zip` from Canvas → `Online News Popularity/OnlineNewsPopularity.zip` → `OnlineNewsPopularity.csv`, or
- download it from the [UCI repository](https://archive.ics.uci.edu/dataset/332/online+news+popularity).

The file has 39,644 rows and 61 columns (the course PDF's 39,797 is the pre-cleaning count).

## Running

```bash
./run_all.sh report           # everything in the current report (about 40 minutes on 4 cores)
./run_all.sh                  # the report, then the experiments of earlier reports (downloads Upworthy and GloVe; several hours)
./report/build.sh             # compile the report (pdflatex + bibtex), prints the page count
./make_submission.sh          # dist/group3_news_popularity.zip: report.pdf + code/ (tracked files, no data)
# or one at a time:
python scripts/00_eda.py      # E0: figures + summary tables
python scripts/01_pilot.py    # pilot numbers used in the plan (~1 min)
```

Outputs go to `results/figures/` and `results/tables/`. Every figure in the report must be produced by a script in `scripts/`.

## Repository layout

```
src/newspop/       shared code — import this, don't copy-paste it
  data.py          loading, cleaning, url repair, publish_date, feature groups / sets
  titles.py        real headlines (sitemap) matched to rows, cached sentence embeddings
  splits.py        chronological split, rolling-origin folds, half-life weights
  models.py        model zoo (all preprocessing inside Pipelines)
  evaluate.py      ROC-AUC, PR-AUC, F1, balanced accuracy
scripts/           one script per experiment (00_eda.py, 01_pilot.py, e1_*.py, …)
notebooks/         scratch exploration only; nothing in the report comes from here
results/           generated figures and tables
report/            LaTeX source of the report (main.tex, references.bib; Overleaf-compatible)
docs/PLAN.md       research plan, protocol, timeline
```

## Working rules

1. **Evaluation protocol** (current report): for every month from July 2013 on, train on all earlier months and score that month; hyper-parameters are chosen on March-June 2013. (Earlier reports used a 70/10/20 chronological split, see `docs/PLAN.md`.)
   **Threshold convention:** the main experiments use 1,400 shares (median of the full data, as in Fernandes et al. 2015, kept for comparability); the threshold sweep (E4) uses quantiles of the training block only (its median is 1,500). Decision cutoffs are always chosen on the validation block.
2. Fixed seeds; results saved as CSV; figures generated by scripts.
3. Work on a branch per experiment (`e3-leakage`, …) and open a pull request; one other member reviews before merge.
4. **Academic integrity:** write your own code. The course checks code similarity across groups and against public solutions. Ideas taken from elsewhere are cited in the report, never pasted.

## Acknowledgements

Dataset: K. Fernandes, P. Vinagre, P. Cortez (2015), *A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News*, EPIA 2015. The half-life weighting idea was inspired by the TimeDecay mechanism in [VCPToolBox](https://github.com/lioensky/VCPToolBox); the implementation here is independent.
