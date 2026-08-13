#!/usr/bin/env bash
set -euo pipefail

# Pin WARPSYMBOLIC_REF to the reviewed release/commit in an official run.
# The old variables remain accepted for one transition cycle.
repo="${WARPSYMBOLIC_REPO:-${ALPHASYMBOLIC_REPO:-https://github.com/juansito17/Algoritmo-Genetico---Formulas.git}}"
ref="${WARPSYMBOLIC_REF:-${ALPHASYMBOLIC_REF:-main}}"

python -m pip install \
  "git+${repo}@${ref}"

if [[ "${ALPHASYMBOLIC_SKIP_CUDA_BUILD:-0}" != "1" ]]; then
  cuda_dir="$(
    python - <<'PY'
from pathlib import Path
import warpsymbolic.gpu

print(Path(warpsymbolic.gpu.__file__).resolve().parent / "cuda")
PY
  )"
  # torch is supplied by requirements.txt; disabling build isolation lets the
  # extension setup import torch.utils.cpp_extension and target the runner GPU.
  python -m pip install --no-build-isolation "${cuda_dir}"
  python - <<'PY'
from warpsymbolic.gpu.cuda_loader import load_rpn_cuda_native

load_rpn_cuda_native()
PY
fi
