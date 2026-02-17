# YOLOZU (萬) — 日本語README

English README: [`README.md`](README.md)

YOLOZU は Apache-2.0 の **contract-first evaluation + tooling harness** です。  
推論バックエンド（PyTorch / ONNXRuntime / TensorRT / C++ / Rust など）は自由に選び、**同一の `predictions.json` 契約**に落として評価・比較できることを最重視します。

対象:
- リアルタイム単眼 RGB **検出**
- 単眼 **depth + 6DoF pose**（RT-DETRベースの最小学習スキャフォールド）
- **セマンティックセグ**（データ準備 + mIoU評価）
- **インスタンスセグ**（PNGマスク契約 + mask mAP評価）

推奨デプロイ（標準パス）: **PyTorch → ONNX → TensorRT**

---

## Quickstart（pip 利用者）

```bash
python3 -m pip install yolozu
yolozu doctor --output -
yolozu predict-images --backend dummy --input-dir /path/to/images
yolozu demo instance-seg
```

追加機能（必要なものだけ）:

```bash
python3 -m pip install 'yolozu[demo]'    # torch demos（CPU可）
python3 -m pip install 'yolozu[onnxrt]'  # ONNXRuntime CPU exporter
python3 -m pip install 'yolozu[coco]'    # pycocotools COCOeval
python3 -m pip install 'yolozu[train]'   # 学習スキャフォールド（torch+onnxrt等）
python3 -m pip install 'yolozu[full]'
```

ドキュメント入口: [`docs/README.md`](docs/README.md)

---

## 何が“売り”か（設計の中心）

- **Bring-your-own inference + 契約ファースト評価**  
  推論はどこで回してもよく、評価は `predictions.json` に統一して **公平に比較**できます。
- **Safe TTT（test-time training）**  
  Tent / MIM のプリセット・ガード・リセットポリシーを用意（`docs/ttt_protocol.md`）。
- **再現性/運用性（Run Contract）**  
  `yolozu train` の run contract で、成果物の置き場・run_meta・resume・export/parity を固定（`docs/run_contract.md`）。

---

## CLIの使い分け（pip vs ソースチェックアウト）

### pip: `yolozu`（インストール安全・CPU中心）
- `yolozu doctor`（環境診断）
- `yolozu validate dataset|predictions|instance-seg`（成果物検証）
- `yolozu eval-coco` / `yolozu eval-instance-seg`（評価）
- `yolozu onnxrt export`（ONNXRuntime推論→predictions出力、要 `yolozu[onnxrt]`）
- `yolozu onnxrt quantize`（ONNXRuntime dynamic quantize、要 `yolozu[onnxrt]`）
- `yolozu train`（RT-DETR pose 学習、要 `yolozu[train]`）
- `yolozu test`（シナリオスイート実行）

### repo: `python3 tools/yolozu.py`（研究/評価の“全部盛り”）
- `export --backend {torch,onnxrt,trt}`（統一引数 + キャッシュ + runメタ）
- TTA/TTT など、重いワークフローをまとめて扱うためのツール群が `tools/` にあります。

---

## 予測JSON（評価契約）

評価の中心は `predictions.json` です（スキーマ: [`schemas/predictions.schema.json`](schemas/predictions.schema.json) / 解説: [`docs/predictions_schema.md`](docs/predictions_schema.md)）。

- どのバックエンドでも **同じスキーマ**で出力
- 変換・評価・差分（parity）を統一

---

## Dataset 形式（YOLO + 任意メタデータ）

基本:
- 画像: `images/<split>/*.(jpg|png|...)`
- ラベル: `labels/<split>/*.txt`（`class cx cy w h` 正規化）

任意メタデータ（JSON）: `labels/<split>/<image>.json`
- Mask/Seg: `mask_path` / `mask` / `M`
- Depth: `depth_path` / `depth` / `D_obj`
- Pose: `R_gt` / `t_gt`（または `pose`）
- Intrinsics: `K_gt` / `intrinsics`

検証:
```bash
yolozu validate dataset /path/to/dataset --strict
```

### 互換（YOLOv8 / YOLO11 / YOLOX）

- Ultralytics YOLOv8 / YOLO11:
  - `images/train` + `labels/train`（および `images/val` + `labels/val`）ならそのままOK
  - Ultralytics の `data.yaml` も `--dataset` に渡せます（`path:` + `train:`/`val:` が `images/<split>` を指す想定）
- YOLOX:
  - COCO JSON（`instances_*.json`）が多いので、`tools/prepare_coco_yolo.py` で YOLO形式へ一度変換するのが最短です

---

## TTA / TTT（Test-Time Adaptation / Training）

- TTA: 予測の後処理で軽量に揺らす（`--tta`）
- TTT: **推論前**にモデルパラメータを少し更新（Tent / MIM、torch backendのみ）

TTTは repo 側のエクスポータで使うのが基本です（`docs/ttt_protocol.md`）:

```bash
python3 tools/yolozu.py export \
  --backend torch \
  --dataset /path/to/yolo-dataset \
  --checkpoint /path/to/checkpoint.pt \
  --device cuda \
  --ttt --ttt-preset safe --ttt-reset sample \
  --ttt-log-out reports/ttt_log_safe.json \
  --output reports/predictions_ttt_safe.json
```

注意:
- TTT は torch backend 限定です（ONNXRuntime/TensorRT は TTA か precomputed predictions を推奨）

---

## Training scaffold（RT-DETR pose）+ Run Contract（本番級の再現性）

実装: `rtdetr_pose/rtdetr_pose/train_minimal.py`（ラッパ: `rtdetr_pose/tools/train_minimal.py`）

### 最短（ソースチェックアウト）
```bash
python3 -m pip install -r requirements-test.txt
bash tools/fetch_coco128.sh
python3 rtdetr_pose/tools/train_minimal.py \
  --dataset-root data/coco128 \
  --config rtdetr_pose/configs/base.json \
  --max-steps 50 \
  --run-dir runs/train_minimal_smoke
```

### 反復運用（Run Contract 推奨）

```bash
yolozu train configs/examples/train_contract.yaml --run-id exp01

# 完全resume（model/optim/sched/AMP scaler/EMA/progress + RNG）
yolozu train configs/examples/train_contract.yaml --run-id exp01 --resume

# 配線スモーク（最初のoptimizer stepで止め、保存/export/parityまで通す）
yolozu train configs/examples/train_contract.yaml --run-id exp01 --dry-run
```

契約された成果物（固定パス）:
- `runs/<run_id>/checkpoints/{last,best}.pt`
- `runs/<run_id>/reports/{train_metrics.jsonl,val_metrics.jsonl,config_resolved.yaml,run_meta.json,onnx_parity.json}`
- `runs/<run_id>/exports/model.onnx`（+ meta）

実装済みの“壊れない”学習ループ要件:
- Resume（完全復帰）
- NaN/Inf guard（skip + LR decay + stop）
- Grad clip（推奨）
- AMP / EMA / DDP（torchrun）
- Validation cadence（epoch/step）+ early stop

拡張（任意）:
- フォトメトリックAug（`--hsv-*`, `--gray-prob`, `--gaussian-noise-*`, `--blur-*`）  
  ※実画像を使う場合は `--real-images` を併用（スキャフォールドはデフォルトで合成画像）。
- `torch.compile`（実験的）: `--torch-compile`（失敗時はデフォルトでfallback）

Run Contract仕様: [`docs/run_contract.md`](docs/run_contract.md)

---

## ONNX 量子化（低コスト）

ONNXRuntime の dynamic quantize で int8-ish の ONNX を生成できます（CPU向け、校正データ不要）:

```bash
yolozu onnxrt quantize --onnx model.onnx --output model_int8.onnx --weight-type qint8
```

---

## 対称性（Symmetry）チェック

- 設定: `configs/runtime/symmetry.json`（ローダ: `yolozu.config.load_symmetry_map`）
- 実装: `yolozu/symmetry.py`（`none`, `Cn`/`C2`/`C4`, `Cinf`）
- テンプレ検証: `yolozu/template_verification.py`

---

## 実行ファイル化（PyInstaller / PyArmor）

手順: [`deploy/pyinstaller/README.md`](deploy/pyinstaller/README.md)

---

## 開発者向け（ローカル検証）

```bash
.venv/bin/ruff check .
.venv/bin/python -m unittest
```

