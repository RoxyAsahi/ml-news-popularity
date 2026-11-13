#!/usr/bin/env bash
# Reproduce every table and figure. Run from the repo root after setup (see README).
# The Upworthy part (u*) downloads the archive from OSF and GloVe vectors from GitHub; it takes several hours
# on 4 cores (u1b and u2 on the confirmatory split dominate).
set -euo pipefail
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
