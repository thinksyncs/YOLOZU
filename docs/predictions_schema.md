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

## Schema file
A JSON Schema file is provided at:
- `schemas/predictions.schema.json`
