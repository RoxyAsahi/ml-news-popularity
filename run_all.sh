#!/usr/bin/env bash
# Reproduce every table and figure (~10 min). Run from the repo root after setup (see README).
set -euo pipefail
for s in scripts/00_eda.py scripts/01_pilot.py scripts/e1_drift.py scripts/e3_leakage.py \
         scripts/e4_threshold.py scripts/e5_learning_curve.py scripts/e6_regularisation.py \
         scripts/e7_clustering.py scripts/e3b_point_in_time.py scripts/main_results.py scripts/e7b_confounding.py scripts/e2_half_life.py; do
  if [ -f "$s" ]; then echo "== $s"; python "$s"; fi
done
