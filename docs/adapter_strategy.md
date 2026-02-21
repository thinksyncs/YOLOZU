# Adapter strategy (inference backends → `predictions.json`)

YOLOZU’s main workflow is:

1) Run inference with your preferred backend
2) Export results into the **canonical** `predictions.json`
3) Validate + evaluate consistently

This page describes the **recommended adapter path** and priorities.

## Priorities (the “one thick road”)

1. **Ultralytics YOLO (v8/v11)**
   - Most common field format.
   - Fast path to high-quality `predictions.json` for detection/seg.

2. **MMDetection**
   - Strong research/production baseline; common in internal stacks.

3. **Detectron2**
   - Widely used for instance segmentation; predictable outputs.

4. **OpenCV DNN (ONNX)**
   - Deployment-friendly baseline for CPU / edge scenarios.

## Contract: what adapters must produce

Adapters should emit the canonical schema:
- `predictions.json` compliant with: [predictions_schema.md](predictions_schema.md)

The point is not “perfectly mirroring a framework’s internal objects”, but producing:
- Stable IDs / image keys
- Boxes/masks/keypoints in agreed coordinates
- Confidence scores
- Category mapping

## Recommended workflow

- Start with: [External inference backends](external_inference.md)
- Validate first:

```bash
yolozu validate predictions --predictions predictions.json
```

- Then evaluate (repo example dataset):

```bash
yolozu eval-coco --dataset data/coco128 --predictions predictions.json
```

## Related docs

- Import adapters (schema-centric): [import_adapters.md](import_adapters.md)
- Real model interface: [real_model_interface.md](real_model_interface.md)
- Evaluation protocol: [yolo26_eval_protocol.md](yolo26_eval_protocol.md)
