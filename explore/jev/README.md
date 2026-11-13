# Exploration: can a decision model (Jev) judge news popularity?

Scripts that produced the judge outputs (paths point at the exploration machine; they need a TypeSafe key or the
open model, so they are not part of run_all.sh; their outputs in `results/` are).
Outputs on the repaired data are `results/rep_*.csv` and `results/api_paraphrase*.csv`; outputs made before the
url column was repaired are in `results/as_distributed/` and are used only to show what the shift did.

* `jevapi.py` - cached client for TypeSafe's System One API (reads the key from `$TYPESAFE_API_KEY`; no key in the repo).
* `api_mash_titles.py` - Jev on 4,000 same-day(-section) Mashable pairs, real sitemap headlines vs URL keywords.
* `api_upw_pairs.py`, `api_upw_random.py`, `api_controls.py` - Jev on Upworthy headline A/B pairs (significant / random), slug-form control.
* `title_eval.py`, `pair_eval.py`, `mash_base.py` - evaluation and out-of-fold HGB / TF-IDF baselines.
* `transfer_law.py` - predicts Jev's Mashable accuracy from Upworthy's headline-effect size and Jev's correlation with it.
* `jev.py`, `convert.py` - local int8 CPU run of the open JEV-9B student (pilot only).
* `results/` - outputs used in the summary.
* `local_runs.py`, `local_runs_rep.py` - open JEV-9B (int8, CPU) on 400 Upworthy and 400 Mashable pairs.
* `api_prompt2.py` - second question wording on 1,000 pairs of each kind.
* `api_recall_probe.py`, `api_recall_probe_titleonly.py` - single-article share recall (memorization probe).
* `api_lineup.py`, `api_feat_pairs.py`, `api_contam.py` - whole-day lineups, headline + features, Upworthy non-winner check.
* `results/rep_curve_confirmatory_{roberta,glove}.csv` - supervised Upworthy ridge on frozen RoBERTa/GloVe (exploration).

* `api_paraphrase.py`, `api_paraphrase2014.py` - rewording tests: 250 pairs from January-May 2013 and 250 pairs
  from 2014 with every headline reworded. The rewordings were written by an AI assistant (Claude), not by people:
  `results/paraphrases_2013H1.txt`, `results/paraphrases_2014.txt`; sources in `results/paraphrase_sources.txt`,
  `results/paraphrase_sources_2014.txt`.

`scripts/v2_judge_summary.py` and `scripts/fig_judge.py` read the cached outputs in `results/`; rerunning the API
scripts needs a TypeSafe key in `$TYPESAFE_API_KEY`.
