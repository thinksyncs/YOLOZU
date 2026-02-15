# Predictions JSON schema (v1)

This document defines the stable **Predictions JSON** contract used by YOLOZU's
evaluation tools. It exists so you can run inference anywhere and still compare
results here.

## Versioning

- `schema_version`: integer, current version is `1`.
- Backward compatible additions are allowed (new optional fields).
- Breaking changes must bump `schema_version`.

## Allowed top-level shapes

You may submit predictions in one of the following shapes.

### Shape A: Array of entries

```json
[
  {
    "image": "path/or/name.jpg",
    "detections": [
      {
        "class_id": 0,
        "score": 0.9,
        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
      }
    ]
  }
]
```

### Shape B: Wrapped object

```json
{
  "schema_version": 1,
  "predictions": [
    {
      "image": "path/or/name.jpg",
      "detections": [
        {
          "class_id": 0,
          "score": 0.9,
          "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
        }
      ]
    }
  ],
  "meta": {
    "adapter": "rtdetr_pose",
    "images": 1
  }
}
```

### Shape C: Map (image -> detections)

```json
{
  "path/or/name.jpg": [
    {
      "class_id": 0,
      "score": 0.9,
      "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    }
  ]
}
```

## Image keys (how predictions join to the dataset)

Each entry must include an `image` string. In YOLOZU, `image` is primarily a
**join key** used to match predictions to dataset records.

Current evaluator behavior:

1) First try an **exact string match** to the dataset record `image` field.
2) If that fails, try matching by **basename** (e.g. `000000123.jpg`).

Recommendations:

- For portability across machines and OS path separators, prefer **basenames** as
  `image` values (and keep them unique within the evaluated split).
- Do not rely on the predictions file location to resolve `image` paths.
  Evaluators treat `image` as an identifier, not as “relative to predictions.json”.

## Detection fields

Required:

- `class_id`: int
- `score`: float
- `bbox`: `{cx, cy, w, h}`

Optional:

- `bbox_abs`: absolute bbox if present
- `mask`, `mask_path`, `depth`, `depth_path`, `rot6d`, `R`, `t_xyz`, `k_delta`, `k_prime`, ...
- `keypoints`: optional 2D keypoints per detection (YOLO pose-style)
  - Recommended: flat list `[x1,y1,v1,x2,y2,v2,...]` in normalized coords
    (`v`: visibility 0/1/2)
  - Also accepted by some tools: list of objects `[{x,y,v?}, ...]`

## BBox format

The canonical `bbox` dict uses keys `{cx, cy, w, h}`. How those numbers are
interpreted is controlled by the evaluating tool via `--bbox-format`.

Common formats:

- `cxcywh_norm` (canonical): normalized to `[0,1]` relative to the image size.
- `cxcywh_abs`: pixels.
- `xywh_abs`: pixels (top-left origin).
- `xyxy_abs`: pixels (top-left origin; bottom-right corner).

Tool expectations:

- `tools/eval_coco.py` and `tools/eval_suite.py` default to `--bbox-format cxcywh_norm`
  but allow overriding.
- The pinned YOLO26 protocol **requires** `cxcywh_norm`
  (see `docs/yolo26_eval_protocol.md`).
- `tools/export_predictions.py --adapter rtdetr_pose` emits `cxcywh_norm`.

## Masks (PNG) and path resolution

This repo supports segmentation workflows, but note that there are **two related
artifacts**:

1) This document: detection-style predictions (`{image, detections}`).
2) Instance segmentation predictions (`{image, instances}`), validated by
   `docs/schemas/instance_segmentation_predictions.schema.json`.

### `mask` vs `mask_path`

When a tool accepts both fields:

- `mask` is treated as the primary field.
- `mask_path` is treated as an alias for `mask` (kept for compatibility).

### PNG requirements (instance segmentation)

For `eval-instance-seg`, each predicted instance mask must:

- Be a 2D mask (PNG recommended).
- Use the standard image coordinate system: top-left origin, `y` down, `x` right.
- Have the **same `(H, W)` as the corresponding GT mask** (size mismatches are errors).
- Use “non-zero = foreground” semantics (binary mask).

### Relative path resolution

Mask paths are resolved as follows:

1) If the evaluation command provides a `--pred-root`, relative mask paths are
   resolved under it.
2) Otherwise, tools default to resolving relative paths under the predictions
   JSON directory.

For reproducible scripts, prefer passing an explicit absolute `--pred-root`.

## Units & coordinate frames (important)

YOLOZU does not automatically convert units. All geometry is computed in the
**units you provide**.

- `intrinsics` / `K` / `K_gt`:
  - Interpreted as `(fx, fy, cx, cy)` in **pixels**, in the **same image coordinate
    system** as the bbox values used for translation recovery.
  - If you resize/letterbox images before inference, provide intrinsics for the
    *post-transform* image.
- `bbox` + `offsets`:
  - Translation recovery uses bbox center in **pixels**.
  - If `--bbox-format cxcywh_norm` is used, bbox is first converted to pixels
    using the entry `image_size` / `image_hw`.
  - `offsets` (if present) are treated as **pixel offsets** `(du, dv)`.
- `log_z` / `z`:
  - Used as the depth/translation `z` component.
  - Whatever unit your dataset uses (e.g., meters or millimeters) becomes the unit
    of `t_xyz`.
  - Recommendation: store depth and translation in **meters** for interoperability.
- `k_delta` / `k_prime`:
  - `k_delta = [dfx, dfy, dcx, dcy]` is a **correction** applied to the provided
    intrinsics:
    - `fx' = fx * (1 + dfx)`
    - `fy' = fy * (1 + dfy)`
    - `cx' = cx + dcx`
    - `cy' = cy + dcy`
  - This is not a full intrinsics estimator; it adjusts a given baseline `K`.

### Intrinsics after resize/letterbox (rule of thumb)

If the original image is resized by `(s_x, s_y)` and padded by `(p_x, p_y)`
(pixels), then:

$$
fx' = fx\,s_x,\quad fy' = fy\,s_y,\quad cx' = cx\,s_x + p_x,\quad cy' = cy\,s_y + p_y
$$

## Schema files

- Detection predictions schema: `schemas/predictions.schema.json`
- Instance segmentation predictions schema:
  `docs/schemas/instance_segmentation_predictions.schema.json`
