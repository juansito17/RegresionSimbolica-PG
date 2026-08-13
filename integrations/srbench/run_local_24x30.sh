#!/usr/bin/env bash
set -euo pipefail

# Independent reproduction harness.  This is intentionally not labelled as an
# upstream SRBench result.
python3 AlphaSymbolic/scripts/freeze_adaptive_config.py benchmarks/adaptive_config.json
python3 -m AlphaSymbolic.scripts.benchmark_srbench \
  --profile official \
  --output benchmarks/srbench_adaptive_24x30.jsonl \
  --resume
