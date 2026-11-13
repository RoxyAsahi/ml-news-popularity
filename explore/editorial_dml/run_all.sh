#!/usr/bin/env bash
# Runs the whole editorial-lever analysis. Every step caches its out-of-fold predictions in
# data/interim/dml/, so the script can be killed and restarted at any point.
# Needs: data/raw/OnlineNewsPopularity.csv, data/external/realigned.csv (file_row -> true url / headline),
#        pip install lightgbm econml onnxruntime tokenizers huggingface_hub   (on top of requirements.txt)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
for s in d0_title_embeddings d1_naive d2_dml_ladder d3_dose_response d4_heterogeneity d5_sensitivity d6_figures; do
  [ -f "$s.py" ] || continue
  echo "== $s"; HF_HUB_DISABLE_XET=1 $PY -W ignore "$s.py" > "out/$s.log" 2>&1 || { echo "FAILED: $s (see out/$s.log)"; exit 1; }
done
