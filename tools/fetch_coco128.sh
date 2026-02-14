#!/usr/bin/env bash
set -euo pipefail

# Fetches the tiny COCO subset (YOLO-format) used by the unit tests.
# Source: Official COCO hosting (images.cocodataset.org).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/data/coco128"

if [[ -d "$OUT_DIR/images/train2017" && -d "$OUT_DIR/labels/train2017" ]]; then
  echo "coco128 already present at: $OUT_DIR"
  exit 0
fi

INSECURE_FLAG=""
if [[ "${YOLOZU_INSECURE_SSL:-}" == "1" || "${CI:-}" == "true" ]]; then
  INSECURE_FLAG="--insecure"
fi

python3 "$REPO_ROOT/tools/fetch_coco128_official.py" --out "$OUT_DIR" $INSECURE_FLAG
