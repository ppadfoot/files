#!/usr/bin/env bash
set -euo pipefail

ROOT="${GPT2_NANO_REPO:-$(pwd)}"
cd "$ROOT"

NB_DIR="notebooks/gauge_covariant_theory"
OUT_DIR="analysis_outputs/gauge_covariant_theory/executed_notebooks"
mkdir -p "$OUT_DIR"

for nb in "$NB_DIR"/[0-1][0-9]_*.ipynb; do
  [ -f "$nb" ] || continue
  name="$(basename "$nb")"
  echo "[execute] $name"
  jupyter nbconvert \
    --to notebook \
    --execute "$nb" \
    --output "$name" \
    --output-dir "$OUT_DIR" \
    --ExecutePreprocessor.timeout="${GAUGE_NOTEBOOK_TIMEOUT:-3600}" \
    --ExecutePreprocessor.kernel_name=python3
 done

echo "Executed notebooks are in $OUT_DIR"
