# Predictions JSON schema (v1)

This document defines the stable **Predictions JSON** contract used by evaluation tools.

## Versioning
- `schema_version`: integer, current version is `1`.
- Backward compatible additions are allowed (new optional fields).
- Breaking changes must bump `schema_version`.

## Allowed top-level shapes
You may submit predictions in one of the following shapes:

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

## Detection fields
Required:
- `class_id`: int
- `score`: float
- `bbox`: {`cx`,`cy`,`w`,`h`}

Optional:
- `bbox_abs`: absolute bbox if present
- `mask`, `mask_path`, `depth`, `depth_path`, `rot6d`, `R`, `t_xyz`, `k_delta`, `k_prime`...

## BBox format
- Default in eval tools is `cxcywh_norm` (normalized to [0,1]).
- For `cxcywh_norm`, `cx,cy,w,h` are relative to the image width/height.
- Use `--bbox-format` to override at evaluation time.

## Units & coordinate frames (important)

YOLOZU does not automatically convert units. All geometry is computed in the **units you provide**.

- `intrinsics` / `K` / `K_gt`:
  - Interpreted as $(fx, fy, cx, cy)$ in **pixels**, in the **same image coordinate system** as the bbox values used for translation recovery.
  - If you resize/letterbox images before inference, you must provide intrinsics for the *post-transform* image.
- `bbox` + `offsets`:
  - Translation recovery uses bbox center in **pixels**.
  - If `--bbox-format cxcywh_norm` is used, bbox is first converted to pixels using the entry `image_size` / `image_hw`.
  - `offsets` (if present) are treated as **pixel offsets** $(du, dv)$.
- `log_z` / `z`:
  - Used as the depth/translation $z$ component.
  - Whatever unit your dataset uses (e.g., meters or millimeters) becomes the unit of `t_xyz`.
  - Recommendation: store depth and translation in **meters** for interoperability.
- `k_delta` / `k_prime`:
  - `k_delta = [dfx, dfy, dcx, dcy]` is a **correction** applied to the provided intrinsics:
    - $fx' = fx \cdot (1 + dfx)$
    - $fy' = fy \cdot (1 + dfy)$
    - $cx' = cx + dcx$
    - $cy' = cy + dcy$
  - This is not a full intrinsics estimator; it adjusts a given baseline K.

### Intrinsics after resize/letterbox (rule of thumb)

If the original image is resized by $(s_x, s_y)$ and padded by $(p_x, p_y)$ (pixels), then:

$$
fx' = fx\,s_x,\quad fy' = fy\,s_y,\quad cx' = cx\,s_x + p_x,\quad cy' = cy\,s_y + p_y
$$

## Schema file
A JSON Schema file is provided at:
- `schemas/predictions.schema.json`
