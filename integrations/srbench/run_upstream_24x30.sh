#!/usr/bin/env bash
set -euo pipefail

: "${SRBENCH_ROOT:?Set SRBENCH_ROOT to the prepared pinned checkout}"
: "${SRBENCH_IMAGES:?Set SRBENCH_IMAGES to the Docker image directory}"
cd "$SRBENCH_ROOT"

COMMON=(
  -results results_alphasymbolic_2025/
  -images "$SRBENCH_IMAGES"
  -n_trials 30
  -job_time_limit 8:00
  -fit_time_limit 3600
  -m 10000
  -max_samples 40000
  --scale_x
  --scale_y
  --local
  --ecotracker
  -ml alphasymbolic
)

# The estimator has an internal frozen 60 s cap, while the external 3600 s cap
# and infrastructure match the comparator protocol.
python experiment/analyze.py datasets/blackbox/ -script optimize_model "${COMMON[@]}"
python experiment/analyze.py datasets/firstprinciples/ -script optimize_model "${COMMON[@]}"

# Run SRBench's own exact symbolic verifier without replacing its aggregation.
python experiment/analyze.py datasets/blackbox/ -script assess_symbolic_model "${COMMON[@]}"
python experiment/analyze.py datasets/firstprinciples/ -script assess_symbolic_model "${COMMON[@]}"

python postprocessing/scripts/collate_experiments_results.py \
  ./results_alphasymbolic_2025/ ./results/alphasymbolic-2025/
