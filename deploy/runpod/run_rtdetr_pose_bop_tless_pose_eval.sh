#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 1) Download/extract BOP T-LESS train_primesense (real RGBD + GT pose).
OUT_DIR="${OUT_DIR:-/workspace/bop}"
bash deploy/runpod/bootstrap_bop_tless_train_primesense.sh

# 2) Convert to YOLOZU dataset (YOLO labels + per-image sidecar with K/R/t).
BOP_ROOT="${BOP_ROOT:-${OUT_DIR}/tless}"
BOP_SPLIT="${BOP_SPLIT:-train_primesense}"
DATASET_OUT="${DATASET_OUT:-/workspace/bop-yolozu-tless-train}"
OUT_SPLIT="${OUT_SPLIT:-train2017}"
MAX_IMAGES="${MAX_IMAGES:-200}"
VISIB_MIN="${VISIB_MIN:-0.2}"

rm -rf "${DATASET_OUT}"
python3 tools/prepare_bop_yolozu.py \
  --bop-root "${BOP_ROOT}" \
  --split "${BOP_SPLIT}" \
  --out "${DATASET_OUT}" \
  --out-split "${OUT_SPLIT}" \
  --bbox-source bbox_vis \
  --visib-fract-min "${VISIB_MIN}" \
  --max-images "${MAX_IMAGES}" \
  --link-images

# 3) Train → export predictions → COCOeval (detection mAP proxy) + pose eval.
CONFIG="${CONFIG:-configs/yolo26_rtdetr_pose/yolo26n.json}"
DEVICE="${DEVICE:-cuda}"
IMG_SIZE="${IMG_SIZE:-320}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.001}"
SCORE_THRESH="${SCORE_THRESH:-0.01}"
MAX_DETS="${MAX_DETS:-100}"
EPOCHS_CSV="${EPOCHS_CSV:-1,5,20}"
RUN_BASE="${RUN_BASE:-/workspace/runs/rtdetr_pose_bop_tless}"

mkdir -p "${RUN_BASE}"
IFS=',' read -r -a epochs_arr <<< "${EPOCHS_CSV}"
for ep in "${epochs_arr[@]}"; do
  ep="$(echo "${ep}" | xargs)"
  [[ -z "${ep}" ]] && continue
  stamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
  run_dir="${RUN_BASE}/ep${ep}_${stamp}"
  mkdir -p "${run_dir}"
  echo
  echo "[bop-tless] === epochs=${ep} run=${run_dir} ==="

  python3 rtdetr_pose/tools/train_minimal.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_OUT}" \
    --split "${OUT_SPLIT}" \
    --device "${DEVICE}" \
    --real-images \
    --image-size "${IMG_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --use-matcher \
    --epochs "${ep}" \
    --run-dir "${run_dir}" \
    --checkpoint-out "${run_dir}/checkpoint.pt" \
    --onnx-out "${run_dir}/model.onnx" \
    --log-every 200 >/dev/null

  python3 tools/export_predictions.py \
    --adapter rtdetr_pose \
    --dataset "${DATASET_OUT}" \
    --split "${OUT_SPLIT}" \
    --config "${CONFIG}" \
    --checkpoint "${run_dir}/checkpoint.pt" \
    --device "${DEVICE}" \
    --image-size "${IMG_SIZE}" \
    --score-threshold "${SCORE_THRESH}" \
    --max-detections "${MAX_DETS}" \
    --max-images "${MAX_IMAGES}" \
    --wrap \
    --output "${run_dir}/pred.json" >/dev/null

  python3 tools/eval_suite.py \
    --dataset "${DATASET_OUT}" \
    --split "${OUT_SPLIT}" \
    --predictions-glob "${run_dir}/pred.json" \
    --bbox-format cxcywh_norm \
    --max-images "${MAX_IMAGES}" \
    --output "${run_dir}/eval_suite.json" >/dev/null

  python3 tools/eval_pose.py \
    --dataset "${DATASET_OUT}" \
    --split "${OUT_SPLIT}" \
    --predictions "${run_dir}/pred.json" \
    --min-score "${SCORE_THRESH}" \
    --iou-threshold 0.5 \
    --max-images "${MAX_IMAGES}" \
    --output "${run_dir}/pose_eval.json" >/dev/null

  python3 - <<PY
import json
from pathlib import Path
run = Path("${run_dir}")
suite = json.loads((run / "eval_suite.json").read_text())
m = suite["results"][0].get("metrics", {})
pose = json.loads((run / "pose_eval.json").read_text()).get("metrics", {})
print("map50_95", m.get("map50_95"), "map50", m.get("map50"))
print("rot_deg_median", pose.get("rot_deg_median"), "depth_abs_median", pose.get("depth_abs_median"))
print("pose_success", pose.get("pose_success"), "rot_success", pose.get("rot_success"), "trans_success", pose.get("trans_success"))
PY
done

echo
echo "[bop-tless] Done. Runs under: ${RUN_BASE}"
