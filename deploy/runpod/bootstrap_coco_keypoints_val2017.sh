#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COCO_ROOT="${COCO_ROOT:-/workspace/coco2017}"

mkdir -p "${COCO_ROOT}/annotations"

# RunPod sometimes has TLS issues with images.cocodataset.org; prefer http.
VAL_URL="${VAL_URL:-http://images.cocodataset.org/zips/val2017.zip}"
ANN_URL="${ANN_URL:-http://images.cocodataset.org/annotations/annotations_trainval2017.zip}"

if [[ ! -d "${COCO_ROOT}/val2017" ]]; then
  echo "[coco] downloading val2017..."
  wget -q -O "${COCO_ROOT}/val2017.zip" "${VAL_URL}"
  unzip -q "${COCO_ROOT}/val2017.zip" -d "${COCO_ROOT}"
fi

if [[ ! -f "${COCO_ROOT}/annotations/person_keypoints_val2017.json" ]]; then
  echo "[coco] downloading annotations..."
  wget -q -O "${COCO_ROOT}/annotations_trainval2017.zip" "${ANN_URL}"
  unzip -q "${COCO_ROOT}/annotations_trainval2017.zip" -d "${COCO_ROOT}"
fi

echo "${COCO_ROOT}"

