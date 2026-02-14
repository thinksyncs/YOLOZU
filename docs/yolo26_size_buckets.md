# YOLO26 size buckets (n/s/m/l/x): params + FLOPs envelopes

YOLOZU’s `yolo26{n,s,m,l,x}` buckets are meant to match the **compute envelopes** of the corresponding YOLO26 size buckets.

This repo intentionally does **not** vendor YOLO26 models. Instead, we:
1) measure a **reference ONNX** for each bucket, and
2) ensure YOLOZU models land in the same params/FLOPs envelope.

## Target envelopes (source of truth)

Config: `yolozu/data/configs/yolo26_size_buckets.json`

The file contains:
- `targets.yolo26{n,s,m,l,x}.params`: parameter count (int)
- `targets.yolo26{n,s,m,l,x}.flops`: FLOPs at `imgsz=640` (int)
- `tolerance.params_ratio` / `tolerance.flops_ratio`: envelope width (default: ±10%)

Until you fill in the targets, the buckets are defined but not enforced.

## Reproducible measurement method (ONNX)

We provide a minimal profiler for ONNX models:

```bash
# Needs: pip install 'yolozu[onnxrt]'  (brings in onnx)
python3 tools/profile_onnx_compute.py --onnx /path/to/yolo26n.onnx --imgsz 640
```

The profiler reports:
- `params`: count of ONNX initializers (weights)
- `macs` / `flops`: approximate MACs/FLOPs for Conv/Gemm/MatMul only

Notes / caveats:
- FLOPs are approximate (counts only Conv/Gemm/MatMul and relies on ONNX shape inference).
- FLOPs definition is pinned in the config (`flops_definition`) so comparisons stay consistent.

## Workflow (recommended)

1) Measure YOLO26 reference models (`yolo26n/s/m/l/x`) and fill in `yolozu/data/configs/yolo26_size_buckets.json`.
2) Measure YOLOZU exported ONNX models for each bucket.
3) Check that YOLOZU params/FLOPs fall within the tolerance envelope for the corresponding bucket.

