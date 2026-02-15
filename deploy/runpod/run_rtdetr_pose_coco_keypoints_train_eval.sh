#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COCO_ROOT="${COCO_ROOT:-/workspace/coco2017}"
OUT_DATASET="${OUT_DATASET:-/workspace/coco2017_kp_yolozu_smoke}"
MAX_IMAGES="${MAX_IMAGES:-200}"

IMAGE_SIZE="${IMAGE_SIZE:-320}"
EPOCHS="${EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-500}"
DEVICE="${DEVICE:-cuda}"

CONFIG_PATH="${CONFIG_PATH:-rtdetr_pose/configs/coco_keypoints_smoke.json}"

echo "[coco] bootstrap..."
bash deploy/runpod/bootstrap_coco_keypoints_val2017.sh

echo "[coco] prepare YOLOZU dataset..."
python3 tools/prepare_coco_keypoints_yolozu.py \
  --coco-root "${COCO_ROOT}" \
  --out "${OUT_DATASET}" \
  --out-split val2017 \
  --max-images "${MAX_IMAGES}"

ts="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
RUN_DIR="${RUN_DIR:-/workspace/runs/rtdetr_pose_coco_kp_${ts}}"
mkdir -p "${RUN_DIR}"

echo "[train] run_dir=${RUN_DIR}"
python3 rtdetr_pose/tools/train_minimal.py \
  --config "${CONFIG_PATH}" \
  --dataset-root "${OUT_DATASET}" \
  --split val2017 \
  --real-images \
  --image-size "${IMAGE_SIZE}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --device "${DEVICE}" \
  --run-dir "${RUN_DIR}"

CKPT="${RUN_DIR}/checkpoint.pt"
PRED="${RUN_DIR}/predictions.json"

echo "[predict] checkpoint=${CKPT}"
python3 tools/export_predictions.py \
  --adapter rtdetr_pose \
  --dataset "${OUT_DATASET}" \
  --split val2017 \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}" \
  --image-size "${IMAGE_SIZE}" \
  --score-threshold 0.05 \
  --max-images "${MAX_IMAGES}" \
  --output "${PRED}" \
  --wrap

echo "[eval] OKS mAP..."
python3 tools/eval_keypoints.py \
  --dataset "${OUT_DATASET}" \
  --split val2017 \
  --predictions "${PRED}" \
  --oks \
  --output "${RUN_DIR}/keypoints_eval_oks.json"

echo "[eval] PCK..."
python3 tools/eval_keypoints.py \
  --dataset "${OUT_DATASET}" \
  --split val2017 \
  --predictions "${PRED}" \
  --output "${RUN_DIR}/keypoints_eval_pck.json"

echo "${RUN_DIR}"

