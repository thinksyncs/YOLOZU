#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Where to put the extracted BOP dataset.
# Note: On many RunPod images, `/` (and therefore `/tmp`) is a small overlay FS.
# Prefer `/workspace` for large downloads.
OUT_DIR="${OUT_DIR:-/workspace/bop}"

python3 tools/download_bop_dataset.py \
  --dataset tless \
  --archives tless_base.zip,tless_train_primesense.zip \
  --out "${OUT_DIR}"

echo "[bop] extracted under: ${OUT_DIR}"
