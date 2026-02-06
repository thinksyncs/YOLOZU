# Adapter contract (v1)

Adapters power `tools/export_predictions.py --adapter <name>`.

## Required behavior
An adapter must implement:
- `predict(records: list[dict]) -> list[dict]`
  - Each entry must be `{image, detections}`
  - Detections must include `class_id`, `score`, and `bbox` (`cx,cy,w,h`)

## Optional behavior
- `predict` may include extra keys per detection (mask/depth/pose/intrinsics)
- `records` are built from YOLO-format datasets via `yolozu.dataset.build_manifest`

## TTT (Test-Time Training) hooks
Adapters may optionally support test-time training (Tent, MIM) by implementing:

- `supports_ttt() -> bool`
  - Returns `True` if the adapter supports TTT, `False` otherwise
  - Default: `False`

- `get_model() -> torch.nn.Module | None`
  - Returns the underlying PyTorch model for adaptation
  - Called by TTT integration to access model parameters
  - Default: `None`

- `build_loader(records, *, batch_size: int = 1) -> Iterable[torch.Tensor]`
  - Builds a data loader that yields batches of preprocessed tensors
  - Each batch should be ready for model forward pass (no additional preprocessing)
  - Used by TTT to create adaptation batches before inference
  - Default: raises `RuntimeError("this adapter does not support TTT")`

### Example TTT adapter implementation

```python
class MyAdapter(ModelAdapter):
    def supports_ttt(self) -> bool:
        return True
    
    def get_model(self):
        self._ensure_backend()
        return self._model
    
    def build_loader(self, records, *, batch_size: int = 1):
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i+batch_size]
            tensors = [self._preprocess(r) for r in batch_records]
            yield torch.stack(tensors)
```

### Usage with TTT

When TTT is enabled (`--ttt`), the integration:
1. Calls `adapter.get_model()` to access the model
2. Calls `adapter.build_loader(records)` to create adaptation batches
3. Runs TTT adaptation (Tent entropy minimization or MIM)
4. Calls `adapter.predict(records)` on the adapted model

See [docs/ttt_integration_plan.md](ttt_integration_plan.md) for more details.

## Stability
- `predict` signature and output schema are **stable**.
- New optional fields may be added without breaking old clients.
- TTT hooks are **stable** as of v1.

## Versioning
Adapters should be compatible with predictions schema `v1`.
