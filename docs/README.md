# YOLOZU docs

YOLOZU is an Apache-2.0-only, **contract-first evaluation harness**.
This `docs/` folder is organized by **what you want to do**, not by module names.

## What makes YOLOZU different (the “sell”)

1) **Bring-your-own inference + contract-first evaluation**: export `predictions.json` from any
   backend (PyTorch/ONNXRuntime/TensorRT/C++/Rust) and compare apples-to-apples.
2) **Safe TTT (test-time training)**: presets + guard rails + reset policies so you can try TTT
   without silently corrupting your model or your metrics.
3) **Apache-2.0-only ops**: a license policy + checks to keep the toolchain clean for OSS/commercial
   use.

Links:
- Bring-your-own inference: [external_inference.md](external_inference.md)
- TTT protocol: [ttt_protocol.md](ttt_protocol.md)
- License policy: [license_policy.md](license_policy.md)
- AI-first guide: [ai_first.md](ai_first.md)
- Migration helpers: [migrate.md](migrate.md)
- Import adapters (canonical schema): [import_adapters.md](import_adapters.md)
- Quantization: [quantization.md](quantization.md)

## Start here (4 entry points)

### 1) Evaluate from `predictions.json` (bring-your-own inference)

Run inference anywhere (PyTorch / ONNXRuntime / TensorRT / C++ / Rust) and export the same
`predictions.json`. YOLOZU validates and scores it consistently (COCO mAP, scenario reports).

Shortest command:

```bash
python3 tools/eval_suite.py --dataset /path/to/coco-yolo --predictions-glob 'reports/pred_*.json'
```

Read next:
- [External inference backends](external_inference.md)
- [Training / inference / export (overview)](training_inference_export.md)
- [Tools index](tools_index.md)

### 2) Train → Export → Evaluate (RT-DETR training pipeline)

The repo includes an RT-DETR-based training pipeline (`rtdetr_pose/`) with data/loss/export wiring,
plus a **production-style run contract** (fixed artifact paths, full resume, export + parity gate).

Shortest command:

```bash
python3 tools/run_yolo26n_smoke_rtdetr_pose.py
```

Read next:
- [Training / inference / export](training_inference_export.md)
- [Run contract (training artifacts)](run_contract.md)
- [Real model interface](real_model_interface.md)
- [Continual learning (rtdetr_pose)](continual_learning.md)

### 3) Contracts (schemas that make results comparable)

Contracts are the product: stable JSON artifacts that let you compare apples-to-apples across
backends and environments.

Shortest command:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
```

Read next:
- [Predictions JSON schema (v1)](predictions_schema.md)
- [Schema governance (version lifecycle + migration)](schema_governance.md)
- [RFC workflow + golden compatibility assets](rfc_workflow.md)
- [Contract-first benchmark publication loop](benchmark_publication.md)
- [CoTTA design spec (YOLOZU phase 1)](cotta_design_spec.md)
- [CoTTA drift validation protocol](cotta_validation.md)
- [EATA design spec (YOLOZU phase 1)](eata_design_spec.md)
- [EATA stability/efficiency benchmark](eata_benchmark.md)
- [SAR design spec (YOLOZU phase 1)](sar_design_spec.md)
- [Adapter contract (v1)](adapter_contract.md)
- [Adapter templates + onboarding](adapter_templates.md)
- [Backend parity matrix automation](backend_parity_matrix.md)
- [Doctor diagnostics for environment drift](doctor_diagnostics.md)

### 4) Protocols & Bench (YOLO26 protocol + sweeps)

Protocols pin evaluation settings (split / image size / bbox format / metric key).
Bench + sweeps help track regressions over time and compare many runs.

Shortest command:

```bash
python3 tools/eval_suite.py --protocol yolo26 --dataset /path/to/coco-yolo --predictions-glob '/path/to/preds*.json'
```

Read next:
- [YOLO26 evaluation protocol](yolo26_eval_protocol.md)
- [Hyperparameter sweep harness](hpo_sweep.md)
- [Latency/FPS benchmark harness](benchmark_latency.md)
- [License policy (Apache-2.0-only guard)](license_policy.md)
