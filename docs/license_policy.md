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

