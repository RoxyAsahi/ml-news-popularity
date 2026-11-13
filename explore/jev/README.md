# Exploration: can a decision model (Jev) judge news popularity?

Exploratory scripts (paths point at the exploration machine; not part of run_all.sh yet).

* `jevapi.py` - cached client for TypeSafe's System One API (reads the key from `$TYPESAFE_API_KEY`; no key in the repo).
* `api_mash_titles.py` - Jev on 4,000 same-day(-section) Mashable pairs, real sitemap headlines vs URL keywords.
* `api_upw_pairs.py`, `api_upw_random.py`, `api_controls.py` - Jev on Upworthy headline A/B pairs (significant / random), slug-form control.
* `title_eval.py`, `pair_eval.py`, `mash_base.py` - evaluation and out-of-fold HGB / TF-IDF baselines.
* `transfer_law.py` - predicts Jev's Mashable accuracy from Upworthy's headline-effect size and Jev's correlation with it.
* `jev.py`, `convert.py` - local int8 CPU run of the open JEV-9B student (pilot only).
* `results/` - outputs used in the summary.
* `local_runs.py`, `local_runs_late.py` - open JEV-9B (int8, CPU) on 400 Upworthy and 400 + 400 Mashable pairs.
* `api_prompt2.py` - second question wording on 1,000 pairs of each kind.
* `api_recall_probe.py`, `api_recall_probe_titleonly.py` - single-article share recall (memorization probe).
* `api_lineup.py`, `api_feat_pairs.py`, `api_contam.py` - whole-day lineups, headline + features, Upworthy non-winner check.
* `results/rep_curve_confirmatory_{roberta,glove}.csv` - supervised Upworthy ridge on frozen RoBERTa/GloVe (exploration).

The v19 scripts (`scripts/v2_judge_summary.py`, `scripts/v4_by_quarter_and_value.py`, `scripts/fig_v19.py`) read the
cached outputs in `results/`; rerunning the API scripts needs a TypeSafe key in `$TYPESAFE_API_KEY`.
* `api_paraphrase.py` - paraphrase test: 250 early-2013 pairs with every headline reworded
  (`results/paraphrases_2013H1.txt`, written by an AI assistant; sources in `results/paraphrase_sources.txt`).
