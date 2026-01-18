#!/usr/bin/env bash
set -euo pipefail

# Fetches the tiny COCO subset (YOLO-format) used by the unit tests.
# Source: Ultralytics YOLOv5 release assets.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
ZIP_PATH="${TMPDIR:-/tmp}/coco128.zip"

mkdir -p "$DATA_DIR"

if [[ -d "$DATA_DIR/coco128/images/train2017" && -d "$DATA_DIR/coco128/labels/train2017" ]]; then
  echo "coco128 already present at: $DATA_DIR/coco128"
  exit 0
fi

if command -v curl >/dev/null 2>&1; then
  curl -L -o "$ZIP_PATH" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ZIP_PATH" "$URL"
else
  echo "Error: need curl or wget" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip not found (install it with apt)" >&2
  exit 1
fi

unzip -q -o "$ZIP_PATH" -d "$DATA_DIR"

# Basic sanity checks for expected layout.
[[ -d "$DATA_DIR/coco128/images/train2017" ]] || { echo "Missing images/train2017 after unzip" >&2; exit 1; }
[[ -d "$DATA_DIR/coco128/labels/train2017" ]] || { echo "Missing labels/train2017 after unzip" >&2; exit 1; }

echo "Installed coco128 into: $DATA_DIR/coco128"
