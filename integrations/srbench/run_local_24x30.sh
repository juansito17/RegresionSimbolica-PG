#!/usr/bin/env bash
set -euo pipefail

# Independent reproduction harness.  This is intentionally not labelled as an
# upstream SRBench result.
python3 -m warpsymbolic.cli.freeze_adaptive_config benchmarks/adaptive_config.json
python3 -m warpsymbolic.cli.benchmark_srbench \
  --profile official \
  --output benchmarks/srbench_adaptive_24x30.jsonl \
  --resume
