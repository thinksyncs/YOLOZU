# TODO (PyTorch Model + TensorRT + Metrics/Loss + Datasets)

Spec reference: `rt_detr_6dof_geom_mim_spec_en_v0_4.md`

## Implementation status (at a glance)

### Data / validation
- [x] YOLO-format dataset manifest builder
- [x] Sidecar metadata support (mask/depth/pose/intrinsics paths + basic content checks)
- [x] Dataset audit tool + validator
- [x] Dataset loader returns full per-instance GT per spec (`M`, `D_obj`, `R_gt`, `t_gt`, `K_gt`)
- [x] Multi-object samples (multiple instances per image) end-to-end support
- [x] Range/units checks (depth meters, mask binary, bbox normalized) + clear conventions
- [ ] Deterministic splits + shuffling + reproducibility hooks (seeded)

### Model (RT-DETR)
- [x] Minimal DETR-style scaffold wired to pose heads (shape-correct)
- [x] Backbone stub: simple strided conv stages (not RT-DETR-grade)
- [x] Neck stub: per-scale 1x1 projections to `hidden_dim`
- [x] Decoder stub: `torch.nn.TransformerDecoder` with learned queries
- [x] Positional embedding: 2D sin/cos
- [ ] Full RT-DETR backbone + neck (parity with spec / real RT-DETR)
- [ ] RT-DETR hybrid encoder / multi-scale fusion (if required for parity)
- [ ] RT-DETR training-time tricks (e.g., denoising queries) if targeting published behavior

### Heads / geometry
- [x] HeadFast outputs (cls/bbox/log_z/rot6D + optional uncertainty)
- [x] CenterOffsetHead (Δu, Δv) (simple linear head)
- [x] GlobalKHead (δfx, δfy, δcx, δcy) (simple linear head)
- [x] rot6D -> R conversion helper
- [ ] Translation recovery from bbox/offsets + corrected intrinsics (spec §2)

### Losses / metrics
- [x] Baseline losses (classification/box/log-depth/rot + basic regularizers)
- [x] Symmetry-aware metrics (geodesic, ADD-S) + unit tests (metrics-level)
- [x] Matching-aware training losses wired to real matcher outputs
- [ ] Full detector metrics aggregation (mAP/Recall) wired to real outputs

### Training / inference / export
- [x] ONNX export wrapper for the current minimal model (opset 17)
- [ ] Training pipeline (matcher, augmentation, MIM teacher, staged schedule)
- [ ] Inference path (translation recovery, template verification, constraints gating)
- [ ] TensorRT export + engine build + parity tests + benchmarks

Status (2026-01-18)
- Scaffold created in `rtdetr_pose/` with dataset loader, validator, and model stubs.
- Dataset manifest supports metadata sidecar (`.json`) for mask/depth/pose/intrinsics; validator checks shapes and paths.
- SIM/Blender sidecar schema added (M/D_obj/R_gt/t_gt/K_gt + cad_points) and content checks for mask/depth/bbox/projection.
- Reports for baseline/gates/scenarios are scaffolded (dummy metrics; no real model yet).
- rtdetr_pose scaffolding files are tracked in git; generated caches/reports are ignored.
- Full RT-DETR, training loop, and TensorRT conversion are not implemented yet.
- Baseline DETR-style backbone/decoder implemented; replace with full RT-DETR for parity.

Current priorities (auto)
1) Stage 1/4 (training-first): make the dataset loader return full GT fields + build a minimal training loop that runs end-to-end.
2) Stage 2/3: upgrade the model from the current scaffold toward RT-DETR parity while keeping losses/metrics wired.
3) Stage 5/6 (later): inference path + TensorRT export/parity benchmarks.

### Training-first next steps (recommended order)
- [x] Dataset returns per-instance GT: `M`, `D_obj`, `R_gt`, `t_gt`, `K_gt` (+ optional `K_gt'`, `cad_points`)
  - [x] Normalize sidecar keys into canonical fields (`R_gt`, `t_gt`, `K_gt`, `M`, `D_obj`)
  - [x] Preserve paths vs inlined arrays without eager decoding
  - [x] Record per-sample availability stats (pose/intrinsics/mask/depth)
- [ ] Batch collation for variable #instances and masks/depth
- [ ] Collate keeps per-instance counts (for matcher) and pads/query-aligns labels/bboxes
- [x] Minimal trainer entrypoint (1 epoch over coco128; logs loss scalars)
- [x] Trainer prints GT availability summary for debugging
- [x] Matching (Hungarian) + staged cost terms (start with cls/box, then add z/rot)
- [ ] Checkpointing + config-driven runs (resume/repro)
- [x] Loss/metric integration test: one training step + backward + no NaNs

Notes (2026-01-21)
- Training scaffold can now consume full GT availability for mask/depth via `gt_M_mask`/`gt_D_obj_mask` (propagated through Hungarian alignment as `M_mask`/`D_obj_mask`).
- Optional: when `t_gt` is missing, `tools/train_minimal.py` can derive `z` (and `t` if `K_gt` exists) from `D_obj` at bbox center via `--z-from-dobj` (arrays inline by default; paths require `--load-aux`).
- [ ] Inference-only utilities (later): decoding + constraints gate + template verify

## Stage 0) Repo + environment alignment
- [x] Decide codebase location (new repo under `/Users/akira/YOLOZU` or existing).
- [x] Lock framework versions (PyTorch, CUDA, TensorRT, onnxruntime) in `docs/versions.md`.
- [x] Define experiment config structure (YAML/JSON) and checkpoints layout (scaffolded).

## Stage 1) Dataset + validation (per spec §4)
- [ ] Implement dataset loader with required GT fields (scaffolded for YOLO bboxes only):
  - `I`, `class_gt`, `bbox_gt`, `M`, `D_obj`, `R_gt`, `t_gt`, `K_gt`
- [ ] Add DataSet type checks (scaffolded for image/labels):
  - shapes, ranges, missing depth mask handling
  - bbox/mask consistency; `D_obj` masked by `M`
  - project CAD points with `(R_gt, t_gt, K_gt)` to validate bbox/mask
- [x] Metadata sidecar (`.json`) support for mask/depth/pose/intrinsics paths.
- [x] Validator checks pose/intrinsics shapes; mask/depth path existence and array shape match.
- [x] Optional SIM jitter fields (`K_gt'`) validation (intrinsics_prime).
- [x] Sidecar accepts `M/D_obj/R_gt/t_gt/K_gt/cad_points` and performs content checks when enabled.
- [x] Build a small "dataset audit" CLI and summary report.
- [x] Add range/units checks (depth meters, mask binary) and clarify multi-object behavior.

### SIM/Blender sidecar schema (labels/*.json)
- `M` or `mask_path`: silhouette mask (array or path; 2D).
- `D_obj` or `depth_path`: object-only depth (array or path; 2D).
- `R_gt`: 3x3 rotation matrix.
- `t_gt`: translation vector (len=3).
- `K_gt`: intrinsics (`{fx,fy,cx,cy}` or 3x3 matrix).
- `K_gt'` (optional): corrected intrinsics.
- `cad_points` (optional): list of 3D CAD points for projection checks.

Remaining decisions (整理)
- GT coordinate convention (camera vs object frame) and units (meters vs millimeters).
- Mask/depth file formats and value ranges (png/npy/json; depth scale).
- Multi-object samples: whether `cad_points` is per-class or per-instance.
- Whether `K_gt'` is required in SIM exports or optional with fallback to `K_gt`.

Start next (着手)
- Lock framework versions and export opset.
- Implement RT-DETR backbone/decoder (export-safe) and wire HeadFast outputs.
- Extend losses/metrics to full spec and add symmetry invariance tests.

## Stage 2) Model architecture (per spec §3)
- [x] Implement RT-DETR-like backbone/decoder with HeadFast outputs:
  - class logits, bbox, `log_z`, `rot6D`, optional `log_sigma_z/rot`
- [ ] Replace baseline backbone/decoder with full RT-DETR for parity with spec.
- [x] Implement CenterOffsetHead (`Δu, Δv`) and GlobalKHead (`δf/δc`) (stub).
- [x] Implement rotation conversion `rot6D -> R ∈ SO(3)`.
- [x] Ensure export-friendly ops for TensorRT (avoid meshgrid; export wrapper in place).

## Stage 3) Losses + metrics (per spec §5, §7, §8)
- [x] Losses (baseline implementations):
  - `L_cls`, `L_box`, `L_z` (log-depth), `L_rot_sym`
  - `L_off`, `L_K`, optional `L_mim`
  - `L_z_prior`, `L_plane`, `L_upright` regularizers
- [x] Metrics (baseline implementations):
  - depth error
  - symmetry-aware geodesic, ADD-S
- [x] Unit tests for symmetry + metric invariance (metrics-level).
- [ ] Wire metric aggregation/mAP/Recall with real detector outputs.

## Stage 4) Training pipeline (per spec §6, §10, §11)
- [ ] Data augmentation + SIM jitter integration.
- [ ] Hungarian matching with staged cost terms.
- [ ] MIM teacher + masking + loss schedule.
- [ ] Staged training: offsets first, then GlobalKHead.

## Stage 5) Inference + constraints (per spec §2, §8, §9)
- [ ] Translation recovery using corrected `K'` + offsets.
- [ ] Symmetry-aware template verification (Top-K only).
- [ ] Constraints gating (depth prior, plane, upright) using `constraints.yaml`.
- [ ] Low-FP gate via `score_tmp_sym < τ`.

## Stage 6) TensorRT export + parity
- [ ] Export to ONNX and build TensorRT engine.
- [ ] Parity tests: PyTorch vs TensorRT outputs within tolerance.
- [ ] Benchmark fps and latency vs target (>= 30 fps).

## Stage 7) Evaluation + scenario suite (per spec §6)
- [ ] Scenario suite run (symmetry, tabletop, depth extremes, jitter).
- [ ] Single report output: fps, mAP/Recall, depth error, pose error, rejection rate.
- [ ] CI smoke run on tiny COCO subset.

## Stage 8) Documentation + handoff
- [ ] Update spec/checklist with actual implementation references.
- [ ] Document training/inference commands and export steps.
