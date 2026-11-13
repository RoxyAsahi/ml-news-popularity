#!/usr/bin/env bash
# Reproduce every table and figure. Run from the repo root after setup (see README).
#   ./run_all.sh report   only the current report (v20), about 40 minutes on 4 cores
#   ./run_all.sh          the report, then the experiments of the earlier reports (v1-v18), several hours
set -euo pipefail
# ---------------------------------------------------------------------------------------------------------
# Report v20 ("What Can Be Called Before Publication? A Misaligned Benchmark, Its Repair, and the Signal in the
# Headline"). Everything in the report comes from this block; run it alone with  ./run_all.sh report
# Needs data/raw/OnlineNewsPopularity.csv and data/external/mashable_sitemap_titles.csv (committed;
# scripts/fetch_mashable_titles.py rebuilds it); downloads bge-base-en-v1.5 from Hugging Face once (embeddings are
# cached in data/interim/). Judge (JEV) outputs and the Upworthy reference curves are read from
# explore/jev/results/, so no API key is needed; explore/jev/ holds the scripts that produced them.
# About 40 minutes on 4 cores.
report() {
  for s in scripts/a1_realign_urls.py scripts/a2_alignment_validation.py scripts/v1_past_only_called.py \
           scripts/v2_judge_summary.py scripts/v3_signal_groups.py scripts/v4_top_pick_value.py \
           scripts/v5_learners.py scripts/v6_headline_terms.py scripts/v7_word_memory.py \
           scripts/fig_called.py scripts/fig_judge.py scripts/a3_export_realigned.py; do
    echo "== $s"; python "$s"
  done
}
# The day-volume and lever estimates quoted in Sec. 5.2 (double ML) come from explore/editorial_dml/ (its own
# run_all.sh; extra packages lightgbm, econml, onnxruntime; about an hour; outputs in explore/editorial_dml/out/,
# cite_*.csv). It reads data/external/realigned.csv, which a3_export_realigned.py writes. Run with
#   ./run_all.sh levers
levers() { (cd explore/editorial_dml && ./run_all.sh); }
if [ "${1:-}" = "levers" ]; then levers; exit 0; fi
if [ "${1:-}" = "report" ]; then report; exit 0; fi
report
# ---------------------------------------------------------------------------------------------------------
# Earlier reports (v1-v18), kept for the record; nothing in the current report comes from here. The Mashable
# scripts (00_eda, e*, s*, main_results) were written before the url column was found to be shifted: they now run
# on the repaired loader, so anything that used the URL, its date or its slug no longer reproduces the numbers in
# report/v18 (those numbers were affected by the shift). The Upworthy part (u*) is unaffected; it downloads the
# archive from OSF and GloVe vectors from GitHub and takes several hours on 4 cores.
./scripts/u0_download_upworthy.sh
for split in exploratory holdout confirmatory; do
  python scripts/u1_design_ladder.py $split          # version fixed on the exploratory split (superseded by u1b)
  python scripts/u1b_design_ladder_strict.py $split  # effect estimates vs out-of-split randomized truth
  python scripts/u2_interventional_ranking.py $split # observational vs decision metric
  python scripts/u3_archive_vs_experiments.py $split # archive vs randomized tests (pooling)
  python scripts/u6_exchange_rate.py $split          # idealized/published archives, sizes, priors
done
python scripts/u7_exchange_curve.py                  # exchange rate with training-draw intervals
for split in confirmatory exploratory holdout; do
  python scripts/u8_active_experiments.py $split     # archive-guided test selection
done
python scripts/u10_capacity.py                       # learner capacity vs archive bias
python scripts/u9_exchange_law.py                    # exchange-rate law, k* robustness, selection summary
python scripts/u4_archive_worth.py
python scripts/u5_leaderboard.py
for s in scripts/00_eda.py scripts/01_pilot.py scripts/e1_drift.py scripts/e3_leakage.py \
         scripts/e4_threshold.py scripts/e5_learning_curve.py scripts/e6_regularisation.py \
         scripts/e7_clustering.py scripts/e3b_point_in_time.py scripts/main_results.py scripts/e7b_confounding.py scripts/e2_half_life.py scripts/e1b_staleness.py scripts/e4b_precision_at_k.py scripts/e8_ale.py scripts/fig_report_data.py scripts/fig_report_results.py \
         scripts/s1_stories.py scripts/s1b_twin_precision.py scripts/s1c_annotation_agreement.py scripts/s2_rq1_anatomy.py scripts/s3_rq1_story_similarity.py \
         scripts/s4_rq2_twin_effects.py scripts/s5_rq3_competition.py scripts/s6_rq3_memory.py scripts/s7_rq4_uncertainty.py scripts/s8_rashomon.py scripts/s9_precision_power.py scripts/s10_split_attention.py scripts/s11_chance.py scripts/s12_slug_model.py scripts/s13_sensitivity.py scripts/s14_context_training.py scripts/fig_v2.py; do
  if [ -f "$s" ]; then echo "== $s"; python "$s"; fi
done
