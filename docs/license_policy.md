# License Policy (Apache-2.0 / No Copyleft Code)

This repository is intended to be **Apache-2.0** code only.

## Rules

- Do **not** vendor or depend on GPL/AGPL code in this repository.
- To compare against external baselines (e.g., YOLO26), run them in a separate environment and **only import predictions JSON** into this repo for evaluation.
- Keep datasets and model weights out of git.

## COCO / coco128

The `coco128` helper dataset is fetched from **official COCO hosting** and converted to YOLO-format labels locally.
Datasets have their own licenses; using them does not change the license of this repository.

## Quick Checks

- Run `python3 tools/check_license_policy.py` before pushing.
- The unit test `python3 -m unittest tests/test_license_policy.py` enforces basic guardrails (e.g., no `ultralytics` fetch URL, presence of `LICENSE`).
- CI runs `tools/check_license_policy.py`, `ruff`, and `python -m unittest` on push/PR.

## Commercial-use due diligence (best-effort, not legal advice)

This repo keeps its **code** Apache-2.0-only, but commercial usage risk can still come from:
- **Dependencies** (Python packages, CUDA/TensorRT/system libs, Docker base images)
- **Datasets** (image licenses vary; some datasets are research-only)
- **Model weights** (separate licenses; keep out of git)

To help audit Python dependencies, generate a license report from the *current environment*:

```bash
python3 tools/report_dependency_licenses.py --output reports/dependency_licenses.json
```

To use it as a guardrail (fail if copyleft-suspect licenses are detected):

```bash
python3 tools/report_dependency_licenses.py --fail-on-copyleft
```

Notes:
- This is **best-effort**: it only inspects installed Python distributions' metadata.
- It does **not** audit CUDA/TensorRT/system libraries, datasets, or model weights.
- For real commercial deployment, you should also do a formal review with legal counsel.
