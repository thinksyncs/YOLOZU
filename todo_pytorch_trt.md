# TODO (PyTorch Model + TensorRT + Metrics/Loss + Datasets)

Spec reference: `rt_detr_6dof_geom_mim_spec_en_v0_4.md`

Status (2026-01-18)
- Scaffold created in `rtdetr_pose/` with dataset loader, validator, and model stubs.
- Dataset manifest supports metadata sidecar (`.json`) for mask/depth/pose/intrinsics; validator checks shapes and paths.
- SIM/Blender sidecar schema added (M/D_obj/R_gt/t_gt/K_gt + cad_points) and content checks for mask/depth/bbox/projection.
- Reports for baseline/gates/scenarios are scaffolded (dummy metrics; no real model yet).
- Full RT-DETR, training loop, and TensorRT conversion are not implemented yet.

Current priorities (auto)
1) Stage 1: finish SIM/Blender GT loader + validation (mask/depth/pose/intrinsics) and projection checks.
2) Stage 2: replace stub model with real RT-DETR backbone/decoder + export-safe ops.
3) Stage 3: implement full losses/metrics with symmetry tests.
4) Stage 5/6: inference path + TensorRT export/parity benchmarks.

## Stage 0) Repo + environment alignment
- [x] Decide codebase location (new repo under `/Users/akira/YOLOZU` or existing).
- [ ] Lock framework versions (PyTorch, CUDA, TensorRT, onnxruntime).
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

### SIM/Blender sidecar schema (labels/*.json)
- `M` or `mask_path`: silhouette mask (array or path; 2D).
- `D_obj` or `depth_path`: object-only depth (array or path; 2D).
- `R_gt`: 3x3 rotation matrix.
- `t_gt`: translation vector (len=3).
- `K_gt`: intrinsics (`{fx,fy,cx,cy}` or 3x3 matrix).
- `K_gt'` (optional): corrected intrinsics.
- `cad_points` (optional): list of 3D CAD points for projection checks.

## Stage 2) Model architecture (per spec §3)
- [ ] Implement RT-DETR backbone/neck/decoder with HeadFast outputs (stub only):
  - class logits, bbox, `log_z`, `rot6D`, optional `log_sigma_z/rot`
- [x] Implement CenterOffsetHead (`Δu, Δv`) and GlobalKHead (`δf/δc`) (stub).
- [x] Implement rotation conversion `rot6D -> R ∈ SO(3)`.
- [ ] Ensure export-friendly ops for TensorRT.

## Stage 3) Losses + metrics (per spec §5, §7, §8)
- [ ] Losses (partial scaffolding):
  - `L_cls`, `L_box`, `L_z` (log-depth), `L_rot_sym`
  - `L_off`, `L_K`, optional `L_mim`
  - `L_z_prior`, `L_plane`, `L_upright` regularizers
- [ ] Metrics (partial scaffolding):
  - mAP/Recall, depth error
  - symmetry-aware geodesic, ADD-S
- [ ] Unit tests for symmetry + metric invariance.

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
