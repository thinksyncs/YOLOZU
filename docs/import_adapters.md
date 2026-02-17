# Import adapters (canonical schema projection)

YOLOZU’s fastest path to “use other platform datasets/configs as-is” is:

1) Fix **one internal canonical schema** (YOLOZU meaning is stable)
2) Add platform-specific **Import adapters** (read-only projection into the canonical schema)
3) Keep evaluation apples-to-apples via the `predictions.json` contract

This doc describes the **canonical schema** and the current import entry points.

## Canonical schema

### SampleRecord (per image)

Canonical representation is a list of records:

```json
{
  "image": "/abs/or/rel/path.jpg",
  "labels": [
    { "class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2 }
  ],
  "image_hw": [480, 640]
}
```

Notes:
- bbox is **`cxcywh_norm`** (normalized to `[0,1]`).
- Optional fields are allowed and may be absent:
  - `mask` / `mask_path`, `depth` / `depth_path` / `D_obj`
  - `pose`, `intrinsics`, `meta`

Implementation reference: `yolozu/canonical.py`.

### TrainConfig (major keys only)

Canonical training config projection focuses on “same meaning” keys:
- `imgsz`, `batch`, `epochs/steps`, `optimizer`, `lr`, `weight_decay`, `seed`, `device`
- plus optional buckets: `preprocess`, `aug`, `loss`, `eval`, `export`

Implementation reference: `yolozu/canonical.py` (`TrainConfig`).

## Dataset import (read-only)

Import adapters **do not rewrite the original dataset**. They generate a small descriptor artifact that points to
the original dataset and explains how to interpret it.

### COCO instances JSON (Detectron2 / MMDetection / YOLOX style)

Create a wrapper directory:

```bash
yolozu import dataset \
  --from coco-instances \
  --instances /path/to/instances_val2017.json \
  --images-dir /path/to/images/val2017 \
  --split val2017 \
  --output data/coco_as_is_wrapper \
  --force
```

Then validate or evaluate by pointing at the wrapper:

```bash
yolozu validate dataset data/coco_as_is_wrapper --split val2017
```

## Config import (Mode A / static files)

### Ultralytics args.yaml (YOLOv8 / YOLO11)

```bash
yolozu import config \
  --from ultralytics \
  --args /path/to/runs/.../args.yaml \
  --output reports/train_config_import.json \
  --force
```

This produces a canonical `TrainConfig` JSON (major keys only).

## Goals / non-goals

The goal is compatibility for:
1) dataset reference (paths/splits/classes/label interpretation)
2) major hyperparameters meaning (`imgsz/batch/lr/epochs`)
3) resolved artifacts to make “it worked as-is” visible

Non-goal (initially): perfectly reproducing framework-specific training behavior (aug hooks, schedulers, etc.).

