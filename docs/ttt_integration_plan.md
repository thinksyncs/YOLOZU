# TTT Integration Notes (implemented)

See also phase-1 CoTTA rollout requirements: [docs/cotta_design_spec.md](cotta_design_spec.md).

## Goals
- Enable test-time training (TTT) without breaking the current `tools/export_predictions.py` flow.
- Keep default behavior unchanged when TTT is disabled.
- Support two methods: **Tent** and **MIM**.
- Provide clear logging (`updated_param_count`, losses, MIM `mask_ratio`) and deterministic runs via seed.

## Modules
- **yolozu/tta/ttt_mim.py** (already exists)
  - Functions: `run_ttt_mim()` and helpers for masking, recon loss, update filtering.
  - Output: `TTTMIMResult(losses, mask_ratio, updated_param_count)`.
- **yolozu/tta/tent.py** (already exists)
  - `TentRunner` with `updated_param_count` logging.
- **yolozu/tta/integration.py**
  - `run_ttt(adapter, records, config)` to adapt the model before prediction.
  - Handles data loading and preprocessing via adapter hooks.
- **yolozu/tta/config.py**
  - `TTTConfig` dataclass for shared CLI/config defaults.

## Adapter Interface Additions
To avoid changes in prediction flow, add **optional** methods to adapters:
- `build_loader(records, *, batch_size)`
  - Returns an iterable of preprocessed tensors suitable for the model.
- `get_model()`
  - Returns the underlying torch model for TTT updates.

Default adapters that do not implement these methods will not support TTT.

## Export Predictions Integration
Add CLI flags in `tools/export_predictions.py` (defaults keep existing behavior):
- `--ttt` (bool): enable TTT
- `--ttt-method {tent,mim,cotta}` (default: tent)
- `--ttt-steps` (int, default: 1)
- `--ttt-lr` (float, default: 1e-4)
- `--ttt-batch-size` (int, default: 1)
- `--ttt-max-batches` (int, optional cap)
- `--ttt-update-filter {all,norm_only,adapter_only}` (default: all)
- MIM-specific: `--ttt-mask-prob`, `--ttt-patch-size`, `--ttt-mask-value`
- `--ttt-log-out` (path): write JSON log

### Flow in `export_predictions.py`
1. Build adapter and dataset manifest.
2. **If `--ttt` enabled**:
   - Acquire model via `adapter.get_model()`.
   - Build loader via `adapter.build_loader()`.
   - Run TTT (`run_ttt_mim` or `TentRunner`) for configured steps.
   - Capture metrics (`updated_param_count`, loss summaries).
3. Run `adapter.predict(records)` as usual.
4. Apply TTA (existing flow).
5. Write output + optional TTT log JSON.

TTT should be strictly pre-prediction to keep output schema unchanged.

## Logging
- Add `ttt` block under meta when `--wrap` is used:
  - `enabled`, `method`, `steps`, `lr`, `batch_size`,
  - `report`: `updated_param_count`, `mask_ratio` (MIM), and `losses`.
- `--ttt-log-out` writes a lightweight JSON log, similar to TTA.

## Backwards Compatibility
- No change when `--ttt` is not set.
- If adapter lacks `get_model()` or `build_loader()`, raise a clear error.

## Tests
- **Unit tests** for integration helper:
  - MIM path uses dummy model + synthetic loader.
  - Tent path uses dummy model + simple logits.
- **CLI smoke test** for `export_predictions.py` with dummy adapter (TTT disabled by default).

## Incremental Delivery
1. Add `TTTConfig` + `integration.py` runner wrappers.
2. Add adapter hooks (RTDETRPoseAdapter + DummyAdapter stubs).
3. Add CLI flags and logging in `export_predictions.py`.
4. Add tests and docs.
