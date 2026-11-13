# Exploration: can a decision model (Jev) judge news popularity?

Exploratory scripts (paths point at the exploration machine; not part of run_all.sh yet).

* `jevapi.py` - cached client for TypeSafe's System One API (reads the key from `$TYPESAFE_API_KEY`; no key in the repo).
* `api_mash_titles.py` - Jev on 4,000 same-day(-section) Mashable pairs, real sitemap headlines vs URL keywords.
* `api_upw_pairs.py`, `api_upw_random.py`, `api_controls.py` - Jev on Upworthy headline A/B pairs (significant / random), slug-form control.
* `title_eval.py`, `pair_eval.py`, `mash_base.py` - evaluation and out-of-fold HGB / TF-IDF baselines.
* `transfer_law.py` - predicts Jev's Mashable accuracy from Upworthy's headline-effect size and Jev's correlation with it.
* `jev.py`, `convert.py` - local int8 CPU run of the open JEV-9B student (pilot only).
* `results/` - outputs used in the summary.
