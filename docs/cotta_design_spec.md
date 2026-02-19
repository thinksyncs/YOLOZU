# CoTTA design spec for YOLOZU (phase 1)

This document defines the first production-safe CoTTA integration scope for YOLOZU.

## Goal

Suppress long-run online drift and forgetting in test-time adaptation while keeping rollout risk low.

Phase 1 prioritizes operationally safe components:

- EMA teacher model
- multi-augmentation prediction averaging
- stochastic restoration limited to safe parameter subsets (LoRA/Norm)

## Update scope (phase 1)

### Allowed trainable targets

- LoRA parameters
- normalization affine parameters (LN/GN/BN affine where used)

### Explicitly disallowed in phase 1

- full backbone weight adaptation
- unrestricted full-model optimizer updates

Rationale: start with low-blast-radius adaptation and prevent irreversible drift.

## Teacher model specification

- Teacher is defined as EMA(student):
  - `theta_teacher <- m * theta_teacher + (1 - m) * theta_student`
- Teacher update runs after each adaptation step.
- EMA momentum `m` is configurable.
- Teacher parameters are never directly optimized by gradient descent.

## Multi-augmentation prediction averaging

### Supported initial augmentation set

- identity
- horizontal flip

Optional future augmentations (not phase-1 default): scale jitter / mild color jitter.

### Aggregation behavior

- Run student (or teacher-selected path) on each augmentation branch.
- Map predictions back to common image coordinates.
- Aggregate by confidence-aware averaging before final postprocess/NMS path.
- Aggregation strategy must be deterministic given seed/config.

## Stochastic restoration (safe mode)

- Restoration is applied only to allowed trainable targets.
- For each eligible parameter element, restore from source snapshot with probability `p_restore`.
- Source snapshot defaults to initial pre-adaptation model state for the session.
- Restoration executes on configured cadence (e.g., every N adaptation steps).

## Safety boundaries and guardrails

The following guardrails are required for phase 1 rollout:

- gradient norm clipping (`max_grad_norm`)
- per-step update norm cap (`max_update_norm`)
- cumulative update norm cap (`max_total_update_norm`)
- divergence stop condition via loss-ratio threshold (`max_loss_ratio`)
- optional immediate fallback to baseline weights on guardrail breach

Failure behavior:

- adaptation step is aborted on hard breach
- event is logged to TTT diagnostics output
- model state is restored according to fallback policy

## Configuration knobs (phase 1)

Minimum required knobs:

- `ttt.method: cotta`
- `ttt.cotta.ema_momentum`
- `ttt.cotta.augmentations` (initial: `identity`, `hflip`)
- `ttt.cotta.aggregation` (initial default: confidence-weighted mean)
- `ttt.cotta.restore_prob`
- `ttt.cotta.restore_interval`
- `ttt.update_filter` (must support `norm_only`, `lora_only`, and combined safe subset)
- `ttt.max_grad_norm`
- `ttt.max_update_norm`
- `ttt.max_total_update_norm`
- `ttt.max_loss_ratio`

## Logging and observability

TTT logs should include:

- run/session id
- method (`cotta`)
- update scope summary (which parameter groups were active)
- teacher EMA momentum
- augmentation set and aggregation mode
- restoration probability + applied restore count
- guardrail metrics and triggered events

## Rollout policy

1. Default-off (`--ttt` required).
2. Phase-1 safe preset enables:
   - LoRA/Norm-only updates
   - conservative restore probability
   - strict guardrails
3. Run on fixed subset first; compare baseline vs CoTTA under same protocol.

## Out of scope (phase 1)

- full-parameter CoTTA updates
- heavy augmentation ensembles
- backend-specific optimized CoTTA kernels

## Exit criteria to move beyond phase 1

- no repeated guardrail breaches in smoke/protocol runs
- measurable drift suppression on fixed evaluation slices
- no unacceptable latency blow-up versus baseline TTT path
