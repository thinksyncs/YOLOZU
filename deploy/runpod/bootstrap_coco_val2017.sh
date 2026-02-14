#!/usr/bin/env bash
set -euo pipefail

# Bootstrap COCO val2017 on RunPod and prepare YOLO-format labels.
#
# Why /tmp?
# - Some RunPod /workspace mounts enforce quotas that can fail mid-conversion.
# - /tmp is usually backed by the container's local overlay and is reliable.
#
# Outputs:
# - COCO raw:      /tmp/coco/{val2017,annotations}
# - YOLO dataset:  /tmp/coco-yolo (dataset.json + labels/val2017 + images/val2017)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${REPO_ROOT}"

PY="${PY:-./.venv-runpod/bin/python}"

if [[ ! -x "${PY}" ]]; then
  echo "[bootstrap] Missing ${PY}. Create your venv first (see deploy/runpod/README.md)."
  exit 1
fi

echo "[bootstrap] Installing OS deps (unzip)"
apt-get update -qq
apt-get install -y -qq unzip >/dev/null

echo "[bootstrap] Installing Python deps (pycocotools)"
"${PY}" -m pip install -q -U pip
"${PY}" -m pip install -q pycocotools

COCO_ROOT="/tmp/coco"
YOLO_ROOT="/tmp/coco-yolo"
mkdir -p "${COCO_ROOT}"

cd "${COCO_ROOT}"

# NOTE: HTTPS certificate verification can fail in some environments; use HTTP.
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

if [[ ! -d val2017 ]]; then
  echo "[bootstrap] Downloading val2017.zip"
  rm -f val2017.zip
  curl -L --retry 5 --retry-delay 2 -o val2017.zip "${VAL_URL}"
  unzip -q val2017.zip
fi

if [[ ! -d annotations ]]; then
  echo "[bootstrap] Downloading annotations_trainval2017.zip"
  rm -f annotations_trainval2017.zip
  curl -L --retry 5 --retry-delay 2 -o annotations_trainval2017.zip "${ANN_URL}"
  unzip -q annotations_trainval2017.zip
fi

cd "${REPO_ROOT}"
rm -rf "${YOLO_ROOT}"

echo "[bootstrap] Converting COCO -> YOLO labels at ${YOLO_ROOT}"
"${PY}" tools/prepare_coco_yolo.py --coco-root "${COCO_ROOT}" --split val2017 --out "${YOLO_ROOT}"

echo "[bootstrap] Done"
echo "${YOLO_ROOT}"

