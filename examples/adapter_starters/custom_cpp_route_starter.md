# Custom C++ route starter

1. Run inference in your C++ service or binary.
2. Emit YOLOZU predictions JSON using this per-image entry format:

```json
{
  "image": "/abs/path/to/image.jpg",
  "detections": [
    {
      "class_id": 0,
      "score": 0.9,
      "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
    }
  ]
}
```

3. Validate output:

```bash
python3 tools/validate_predictions.py /path/to/predictions.json --strict
```

4. Add the output file to adapter parity suite:

```bash
python3 tools/adapter_parity_suite.py \
  --adapter-predictions custom_cpp=/path/to/predictions.json \
  --adapter-predictions rtdetr=/path/to/reference_predictions.json \
  --output reports/adapter_parity_suite.json
```
