#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v yolozu >/dev/null 2>&1; then
  YOLOZU_BIN=(yolozu)
else
  YOLOZU_BIN=(python3 -m yolozu.cli)
fi

DATASET="data/smoke"
PREDICTIONS="data/smoke/predictions/predictions_dummy.json"
REPORT="reports/smoke_coco_eval_dry_run.json"

if ! "${YOLOZU_BIN[@]}" doctor --output -; then
  echo "doctor reported environment issues; continuing smoke checks"
fi
"${YOLOZU_BIN[@]}" validate dataset "$DATASET"
"${YOLOZU_BIN[@]}" validate predictions "$PREDICTIONS" --strict
"${YOLOZU_BIN[@]}" eval-coco \
  --dataset "$DATASET" \
  --split val \
  --predictions "$PREDICTIONS" \
  --dry-run \
  --output "$REPORT"

echo "smoke OK: $REPORT"