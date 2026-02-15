# Specification (EN) — Real-Time RT-DETR for 2D Detection + Depth + 6DoF Pose  
**with Geometry-Aligned MIM, Symmetry Handling, Handheld Intrinsics Compensation, Commonsense Constraints, and Physics-Augmentation**  
Version: **v0.4-en**

> This document is written to minimize ambiguity. Terms, coordinate frames, and when modules run (train-only vs inference) are explicitly stated.

---

## 0. Glossary and conventions (read first)

### 0.1 Coordinate frames
- **Camera frame**: right-handed. The exact convention must be fixed in code and dataset. Recommended:
  - `+Z` points forward (into the scene),
  - `+X` points right in the image,
  - `+Y` points down in the image (consistent with pixel coordinates).
- **Object frame**: fixed per CAD model (mesh coordinate system). Must be consistent across dataset generation and evaluation.

### 0.2 Intrinsics
- `K = (fx, fy, cx, cy)` are the pinhole intrinsics (no distortion in the core equations).
- If distortion exists, it must be either:
  - (A) removed by preprocessing (undistort), or
  - (B) modeled as an optional augmentation / correction module, but keep the main geometry equations consistent.

### 0.3 Pose parameters
- Pose is `(R, t)` where:
  - `R ∈ SO(3)` is object rotation in camera frame (object → camera),
  - `t = (X, Y, Z)` is object translation in camera frame.

### 0.4 “Object-only depth”
- `D_obj` is a depth image defined **only on the object silhouette**.
- Background depth is **missing** (stored as `0`, `NaN`, or masked out).
- All losses using `D_obj` must ignore missing pixels.

### 0.5 Train-only vs inference
- **Inference graph (TensorRT)** must remain minimal and stable.
- Expensive or auxiliary modules (MIM teacher, heavy augmentation, some template operations) run **train-time only** or **outside** the TensorRT graph.

---

## 1. Goals and requirements

### 1.1 Product goal
- Provide real-time object detection + depth + 6DoF pose estimation from monocular RGB.
- Designed for external deployment (SaaS/API).

### 1.2 Performance target
- End-to-end inference (including postprocess and optional template verification) should achieve:
  - **≥ 30 fps** (≤ 33 ms/frame).

### 1.3 Licensing direction
- Prefer Apache-2.0–compatible baselines (RT-DETR family).
- Avoid AGPL components in the inference path.

### 1.4 Symmetry & commonsense constraints checklist
- [ ] `symmetry.json` exists and is validated at load time.
- [ ] `constraints.yaml` exists with enable flags for each constraint.
- [ ] Symmetry-aware rotation loss and metrics are enabled.
- [ ] Template verification uses symmetry-aware scoring.
- [ ] Depth prior regularizer is implemented and ablated.
- [ ] Table-plane and upright constraints are implemented and ablated.
- [ ] Inference gates (low-FP mode) are configurable and tested.

---

## 2. Inputs and outputs

### 2.1 Model inputs
- RGB image `I ∈ R^{H×W×3}`.
- Nominal intrinsics `K = (fx, fy, cx, cy)` (known).
- No additional inputs at inference.

### 2.2 Model outputs (per-frame, per-query)
For each predicted object `i`:
- `class_i`, `score_det_i`
- `bbox_i = (cx_i, cy_i, w_i, h_i)` (image coordinates)
- `log_z_i` (or `inv_z_i`) where `z_i = exp(log_z_i)` is object center depth
- `rot6D_i` (preferred) to represent `R_i`
- optional uncertainties: `log_sigma_z_i`, `log_sigma_rot_i`
- **CenterOffsetHead (recommended)**: `Δu_i, Δv_i` (per-query)
- **GlobalKHead (recommended for handheld)**: `δfx, δfy, δcx, δcy` (per-frame)

### 2.3 Corrected intrinsics (handheld compensation)
GlobalKHead predicts small corrections:
- `fx' = fx * (1 + δfx)`
- `fy' = fy * (1 + δfy)`
- `cx' = cx + δcx`
- `cy' = cy + δcy`
Define corrected `K' = (fx', fy', cx', cy')`.

### 2.4 Translation recovery (postprocess; no direct learning of tx,ty by default)
To avoid unstable direct regression of `t_xy`, translation is **recovered** from image center + depth + intrinsics:

- Let bbox center be `(u, v) = (cx_i, cy_i)`.
- Apply center correction: `u' = u + Δu_i`, `v' = v + Δv_i`.
- Recover translation in camera frame:
  - `Z = z_i`
  - `X = (u' - cx') / fx' * Z`
  - `Y = (v' - cy') / fy' * Z`

**Important clarification**:
- `cx, cy` in `K` are the **principal point**.
- `t = (X,Y,Z)` is the **3D translation**. They are different concepts.

---

## 3. Model architecture

### 3.1 Base
- RT-DETR backbone + neck + transformer decoder.

### 3.2 Inference heads (TensorRT graph; must remain lightweight)
**HeadFast**:
- classification logits
- 2D box regression
- depth head: `log_z` (or `inv_z`)
- rotation head: `rot6D` → `R ∈ SO(3)` via orthonormalization
- optional: uncertainty heads

**CenterOffsetHead** (per-query):
- outputs `Δu, Δv` to correct “bbox center ≠ projected 3D center” bias

**GlobalKHead** (per-frame):
- outputs `δfx, δfy, δcx, δcy` (small corrections)

### 3.3 Train-only modules (must be disabled/removed for TensorRT export)
- Geometry-Aligned MIM: `Masker` + `Decoder_mim` + `RenderTeacher`
- Optional keypoint head (train-only) if needed for additional geometric supervision
- Heavy augmentation modules (HardAug)
- Any expensive template search (inference must be Top-K only)

---

## 4. Dataset specification (Blender/SIM)

### 4.1 Required ground truth per image
- `I` RGB
- `class_gt`
- `bbox_gt`
- `M` silhouette mask
- `D_obj` object-only depth (background missing)
- `R_gt`, `t_gt` (object pose)
- `K_gt` (and optionally `K_gt'` if intrinsics jitter is synthesized)

### 4.2 Handheld simulation (recommended)
Add realistic intrinsics/extrinsics variation:
- Intrinsics drift: `δf`, `δc` (zoom/crop/principal-point drift)
- Extrinsics jitter: camera pose perturbation (handheld shake)
- Optional rolling shutter parameterization (if you have videos)

### 4.3 Consistency checks (must be automated)
- Project CAD points using `(R_gt, t_gt, K_gt)` and validate:
  - bbox overlap with `bbox_gt`
  - silhouette overlap with `M`
- Validate `D_obj` mask consistency: `D_obj` non-missing pixels must coincide with `M`.

---

## 5. Loss functions

### 5.1 Detection
- `L_cls` for classification
- `L_box` for box regression (e.g., L1 + GIoU)

### 5.2 Depth (object center depth)
Preferred:
- `L_z = | log(z_pred) - log(z_gt) |`

Optional heteroscedastic form:
- `e_z = log(z_pred) - log(z_gt)`
- `L_z = |e_z| * exp(-s_z) + s_z`, where `s_z = log_sigma_z`

### 5.3 Rotation (base, before symmetry)
Preferred with rot6D:
- Convert to `R_pred`, then use geodesic distance:
  - `d(R1,R2) = arccos((trace(R1^T R2) - 1)/2)`

### 5.4 CenterOffsetHead
If GT offset is available (SIM), define:
- `L_off = || [Δu, Δv] - [Δu_gt, Δv_gt] ||_1`

If GT offset is not available (real images), use weak supervision:
- reprojection consistency (train-only keypoints) and/or template-based consistency.

### 5.5 GlobalKHead
If GT intrinsics correction is available (SIM), define:
- `L_K = || [δfx, δfy, δcx, δcy] - [δfx_gt, δfy_gt, δcx_gt, δcy_gt] ||_1`

If GT is not available (real images), optional weak supervision:
- table-plane consistency, orthogonality/lines (if applicable), template consistency.

### 5.6 Total loss
`L = L_cls + λ_box L_box + λ_z L_z + λ_rot L_rot + λ_off L_off + λ_K L_K + λ_mim L_mim (+ λ_kp L_kp)`

---

## 6. Geometry-Aligned MIM (train-only)

### 6.1 Objective
Improve sample efficiency and Sim2Real robustness by forcing masked feature reconstruction to match a **geometry-derived teacher feature**.

### 6.2 Teacher tensor (from segmentation + object-only depth)
Given `M` and `D_obj`:
- Choose a reference depth: `z_ref = median(D_obj[M==1])`
- Normalize:
  - `D_norm = M * ( log(D_obj + ε) - log(z_ref + ε) )`
- Teacher input:
  - `T = concat(M, D_norm)` (2 channels)
  - Optional: add `edge(M)` as a 3rd channel.

### 6.3 Teacher feature
- `F_rend = RenderTeacher(T)` (small CNN)
- Use `stopgrad(F_rend)` to stabilize.

### 6.4 Masked feature inpainting
- Apply block masking to neck feature map (prefer **P5**, optionally P4).
- Reconstruct with `Decoder_mim` to produce `F_pred`.
- MIM loss (only on masked locations):
  - `L_mim = || (F_pred - F_rend) * MaskFeat ||_1`
- Optional: boundary weighting using mask edges.

---

## 7. Symmetry handling (critical; avoids “ill-posed teacher”)

### 7.1 Why this exists
For rotationally symmetric objects, pose labels are **not unique**. Training with a naïve rotation loss becomes ill-posed.

### 7.2 Symmetry metadata (preferred; no symmetry head initially)
Maintain `symmetry.json`:

```json
{
  "class_id_or_name": {
    "type": "none | Cn | Dn | Cinf | mirror",
    "n": 4,
    "axis": [0, 0, 1],
    "notes": "optional"
  }
}
```

- `Cn`: discrete n-fold rotational symmetry
- `Cinf`: continuous rotation symmetry around `axis` (e.g., cylinder)

### 7.3 Symmetry-aware rotation loss (must be used for both training and evaluation)
Define:
- `L_rot_sym = min_{S ∈ G} d(R_pred, R_gt · S)`

- For discrete groups: enumerate `S`.
- For continuous yaw symmetry: use a yaw-invariant distance or minimize over a small yaw sample set.

### 7.4 Symmetry-aware template verification
Define:
- `score_tmp_sym = max_{S ∈ G} score_tmp(render(R_pred · S), obs)`

This prevents rejecting correct poses that are equivalent under symmetry.

### 7.5 When to add a SymmetryHead (optional)
Only if symmetry varies within a class or unknown classes are frequent:
- outputs `sym_type`, `n`, `axis`.
Otherwise, rely on `symmetry.json` for determinism and simplicity.

---

## 8. Commonsense geometric constraints (train regularizers + inference gating)

### 8.1 Depth prior from size + bbox + intrinsics
Compute `z_prior(bbox, size, K')` and add weak regularization:
- `L_z_prior = | log(z_pred) - log(z_prior) |` (small weight)

### 8.2 Table-plane constraint (if calibrated plane exists)
Plane `π: n·X + d = 0` in camera frame.
- Train-time: `L_plane` pushes the object base point (or contact point) near the plane.
- Inference: reject if the recovered `t` is physically below the plane.

### 8.3 Upright constraint
For tabletop objects, roll/pitch may be constrained.
- Train-time: `L_upright` per class (penalty outside range)
- Inference: optional gate or score penalty.

### 8.4 Inference-time gating / rescoring (lightweight)
- Reject implausible depth/plane violations.
- Final score:
  - `S = w_det*score_det + w_tmp*score_tmp_sym - w_unc*(sigma_z + sigma_rot)`

---

## 9. Template (rule-based) fusion (inference + optional training)

### 9.1 Inference-time policy (must remain real-time)
- Apply template verification to **Top-K** candidates only.
- Avoid global search; no full-image template scanning.

### 9.2 Recommended scoring
- Silhouette: Chamfer distance using distance transform
- Depth: masked depth difference on object region (ignore missing pixels)
- Combine via `score_tmp`.

### 9.3 Training-time usage (optional)
- Use template score as sample weighting or as an additional consistency term.
- Keep expensive computations out of the inference graph.

---

## 10. Training schedule (efficiency + stability)

### 10.1 Staged introduction for handheld heads
- Stage A: enable **CenterOffsetHead**, disable GlobalKHead (stabilize translation).
- Stage B: enable **GlobalKHead** (if SIM provides `K_gt'`; start with small `λ_K`).

### 10.2 MIM schedule
- Stage A: train detection + depth (+ weak rotation).
- Stage B: enable Geometry-Aligned MIM (`λ_mim` ~ 0.05–0.2).
- Stage C: disable MIM; fine-tune task losses only.

### 10.3 Task-aligned matching (Hungarian)
- Keep Hungarian matching.
- Add pose/template terms to matching cost **later** in training to avoid early instability:
  - `cost = w_cls*C_cls + w_box*C_box + w_pose*C_pose (+ w_tmp*C_tmp)`

---

## 11. AugmentationEngine (train-only; realistic + efficient)

### 11.1 Purpose
Increase robustness (Sim2Real, handheld) while keeping training efficient.

### 11.2 Parameter distribution learning (test images → p(θ))
Estimate imaging parameters from real test images/videos and fit a distribution `p(θ)`:
- exposure/WB/gamma
- shot+read noise
- PSF blur / defocus
- compression
- vignette / chromatic aberration
- handheld jitter / rolling shutter (if videos)

At training: sample `θ ~ p(θ)`.

### 11.3 Three-tier pipeline (Fast / Phys / Hard)
- **FastAug (always)**: crop/resize/flip, mild affine, mild color jitter (GPU)
- **PhysAug (probabilistic)**:
  - PSF convolution (defocus/blur)
  - noise + simplified ISP (Monte Carlo approximations)
  - depth-dependent haze via `I = I0*exp(-β d) + A(1-exp(-β d))`
- **HardAug (rare + cached)**:
  - motion blur via time integration (Monte Carlo)
  - rolling shutter warps

### 11.4 Label consistency (non-negotiable)
If a geometric transform is applied, update bbox/seg/depth/perspective consistently.
Always preserve object-only depth missing regions; exclude missing pixels from losses.

### 11.5 Efficiency controls
- probability schedules (stronger later)
- caching for HardAug
- low-res approximation for PSF/noise where acceptable
- optional importance sampling (apply heavier aug to hard/uncertain examples)

---

## 12. TensorRT export and inference optimization

- TensorRT graph includes only: RT-DETR + HeadFast + CenterOffsetHead + GlobalKHead.
- Exclude all train-only modules.
- Prefer fixed-shape export where possible.
- Default FP16; consider INT8 later.

---

## 13. Evaluation and ablations

### 13.1 Metrics
- 2D detection: mAP, Recall@FP
- Depth: AbsRel, RMSE (center depth and/or object pixels)
- Pose: ADD(-S), rotation error (deg), translation error (cm)
- Runtime: E2E latency, fps, stability (seed variance)

### 13.2 Required ablation ladder
Base → +CenterOffset → +GlobalK → +SymLoss → +MIM → +AugEngine → +TemplateGate → +Commonsense

---

## 14. Repository deliverables (minimum)
- `spec_en.md` (this file)
- `symmetry.json`
- `constraints.yaml`
- `augment.yaml`
- automated dataset consistency checks
- training config(s) for staged schedule
- TensorRT export script ensuring train-only components are excluded
