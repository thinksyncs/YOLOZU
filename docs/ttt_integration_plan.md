# Test-Time Training (TTT) Integration Plan

See also the Phase 1 CoTTA rollout requirements: [docs/cotta_design_spec.md](cotta_design_spec.md).
Planned next method design: [docs/eata_design_spec.md](eata_design_spec.md).
Planned SAR rollout constraints: [docs/sar_design_spec.md](sar_design_spec.md).

## Objectives
- **Seamless Integration:** Enable Test-Time Training (TTT) without disrupting the existing `tools/export_predictions.py` workflow.
- **Zero-Impact Defaults:** Ensure that the default behavior remains strictly unchanged when TTT is disabled.
- **Method Support:** Initially support two primary adaptation methods: **Tent** and **MIM** (Masked Image Modeling).
- **Observability & Reproducibility:** Provide comprehensive logging (e.g., `updated_param_count`, loss metrics, MIM `mask_ratio`) and guarantee deterministic execution via strict seed control.

## Core Modules
- **`yolozu/tta/ttt_mim.py`** (Existing)
  - **Functions:** `run_ttt_mim()` alongside helper functions for masking, reconstruction loss computation, and update filtering.
  - **Output:** Returns a `TTTMIMResult` object containing `losses`, `mask_ratio`, and `updated_param_count`.
- **`yolozu/tta/tent.py`** (Existing)
  - Implements the `TentRunner` class, which includes built-in logging for `updated_param_count`.
- **`yolozu/tta/integration.py`**
  - Exposes `run_ttt(adapter, records, config)` to adapt the model weights prior to the prediction phase.
  - Manages data loading and preprocessing by leveraging adapter-specific hooks.
- **`yolozu/tta/config.py`**
  - Defines the `TTTConfig` dataclass to centralize shared CLI arguments and configuration defaults.

## Adapter Interface Extensions
To prevent structural changes to the prediction flow, the following **optional** methods will be introduced to the adapter interface:
- `build_loader(records, *, batch_size)`
  - **Returns:** An iterable of preprocessed tensors formatted appropriately for the underlying model.
- `get_model()`
  - **Returns:** The underlying PyTorch model instance required for TTT parameter updates.

*Note: Default adapters that omit these methods will inherently lack TTT support and will raise a clear exception if TTT is requested.*

## Export Predictions Integration
The following CLI flags will be added to `tools/export_predictions.py` (default values preserve existing behavior):
- `--ttt` (bool): Enables Test-Time Training.
- `--ttt-method {tent,mim,cotta,eata,sar}` (default: `tent`)
- `--ttt-steps` (int, default: `1`)
- `--ttt-lr` (float, default: `1e-4`)
- `--ttt-batch-size` (int, default: `1`)
- `--ttt-max-batches` (int, optional): Imposes a hard cap on the number of adaptation batches.
- `--ttt-update-filter {all,norm_only,adapter_only}` (default: `all`)
- **MIM-specific flags:** `--ttt-mask-prob`, `--ttt-patch-size`, `--ttt-mask-value`
- `--ttt-log-out` (path): Specifies the destination for the JSON-formatted TTT log.

### Execution Flow in `export_predictions.py`
1. Initialize the adapter and construct the dataset manifest.
2. **If `--ttt` is enabled:**
   - Retrieve the model instance via `adapter.get_model()`.
   - Construct the data loader via `adapter.build_loader()`.
   - Execute the TTT routine (`run_ttt_mim` or `TentRunner`) for the configured number of steps.
   - Capture relevant metrics (e.g., `updated_param_count`, loss summaries).
3. Execute `adapter.predict(records)` using the standard inference pathway.
4. Apply Test-Time Augmentation (TTA) if configured (existing flow).
5. Serialize the output predictions and optionally write the TTT JSON log.

*Crucially, TTT must execute strictly prior to the prediction phase to ensure the output schema remains entirely unaffected.*

## Logging Strategy
- When the `--wrap` flag is utilized, append a `ttt` block under the `meta` section:
  - **Configuration:** `enabled`, `method`, `steps`, `lr`, `batch_size`.
  - **Report:** `updated_param_count`, `mask_ratio` (for MIM), and `losses`.
- The `--ttt-log-out` flag will generate a lightweight, standalone JSON log, mirroring the existing TTA logging paradigm.

## Backwards Compatibility
- The system will exhibit no behavioral changes unless the `--ttt` flag is explicitly provided.
- If an adapter lacks the `get_model()` or `build_loader()` implementations, the system will fail fast and raise an informative error.

## Testing Strategy
- **Unit Tests** for the integration helper:
  - The MIM pathway will be validated using a dummy model coupled with a synthetic data loader.
  - The Tent pathway will be validated using a dummy model and simplified logit outputs.
- **CLI Smoke Tests:** Validate `export_predictions.py` using a dummy adapter to ensure TTT remains disabled by default and fails gracefully when unsupported.

## Incremental Delivery Plan
1. Introduce `TTTConfig` and the `integration.py` runner wrappers.
2. Implement adapter hooks (specifically for `RTDETRPoseAdapter` and `DummyAdapter` stubs).
3. Integrate CLI flags and logging mechanisms into `export_predictions.py`.
4. Finalize unit tests and update relevant documentation.

## References
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. In *International Conference on Learning Representations (ICLR)*.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
