# Smoke assets (offline)

This folder is bundled so users can run a full dry-run smoke check without network downloads.

Contents:
- `images/val/*.jpg` (10 small images)
- `labels/val/*.txt` (YOLO bbox labels)
- `labels/val/classes.json`
- `predictions/predictions_dummy.json` (schema version 1)

Generate/update deterministically from local `data/coco128` with:

```bash
python3 tools/generate_smoke_assets.py
```

Run the end-to-end smoke flow:

```bash
bash scripts/smoke.sh
```
