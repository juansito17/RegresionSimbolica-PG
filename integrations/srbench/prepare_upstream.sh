#!/usr/bin/env bash
set -euo pipefail

: "${SRBENCH_ROOT:?Set SRBENCH_ROOT to a clean cavalab/srbench checkout}"
WARP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_COMMIT="dc3f6daa93bf10955df8775256a6f8644f38fd93"
ACTUAL_COMMIT="$(git -C "$SRBENCH_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Expected SRBench $EXPECTED_COMMIT, found $ACTUAL_COMMIT" >&2
  exit 2
fi

mkdir -p "$SRBENCH_ROOT/algorithms/alphasymbolic"
mkdir -p "$SRBENCH_ROOT/experiment/methods/alphasymbolic"
cp "$WARP_ROOT/integrations/srbench/algorithms/alphasymbolic/"* \
  "$SRBENCH_ROOT/algorithms/alphasymbolic/"
cp "$WARP_ROOT/integrations/srbench/experiment/methods/alphasymbolic/"*.py \
  "$SRBENCH_ROOT/experiment/methods/alphasymbolic/"

echo "Prepared WarpSymbolic in pinned SRBench checkout: $SRBENCH_ROOT"
echo "Build with: cd '$SRBENCH_ROOT' && bash scripts/make_docker_compose_file.sh && docker compose build alphasymbolic"
