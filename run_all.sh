#!/usr/bin/env bash
# Reproduce every table and figure (~26 min). Run from the repo root after setup (see README).
set -euo pipefail
for s in scripts/00_eda.py scripts/01_pilot.py scripts/e1_drift.py scripts/e3_leakage.py \
         scripts/e4_threshold.py scripts/e5_learning_curve.py scripts/e6_regularisation.py \
         scripts/e7_clustering.py scripts/e3b_point_in_time.py scripts/main_results.py scripts/e7b_confounding.py scripts/e2_half_life.py scripts/e1b_staleness.py scripts/e4b_precision_at_k.py scripts/e8_ale.py scripts/fig_report_data.py scripts/fig_report_results.py \
         scripts/s1_stories.py scripts/s1b_twin_precision.py scripts/s2_rq1_anatomy.py scripts/s3_rq1_story_similarity.py \
         scripts/s4_rq2_twin_effects.py scripts/s5_rq3_competition.py scripts/s6_rq3_memory.py scripts/s7_rq4_uncertainty.py scripts/s8_rashomon.py scripts/s9_precision_power.py scripts/fig_v2.py; do
  if [ -f "$s" ]; then echo "== $s"; python "$s"; fi
done
