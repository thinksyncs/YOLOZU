# Manifest coverage audit (YOLOZU-1m2)

- Source of truth: `tools/manifest.json`
- Scope: declarative fields (`platform`, `inputs`, `effects`, `outputs`, `examples`) and structural consistency
- Status: `PASS` (no known gaps)

## Validation checks

```bash
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json
python3 tools/validate_tool_manifest.py --manifest tools/manifest.json --require-declarative
```

Both commands pass on current `main`.

## Notes

- Historical gaps identified in `YOLOZU-1m2.2` were normalized in `YOLOZU-1m2.4`.
- Future manifest updates should keep strict declarative mode green.
