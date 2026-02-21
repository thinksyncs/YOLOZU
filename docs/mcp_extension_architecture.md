# MCP extension architecture (YOLOZU v2)

This document fixes the extension layering and operational policy.

## Layer design

1. Core layer (`yolozu.integrations.layers.core`)
   - request/response shaping
   - error normalization
   - short summary generation
2. YOLOZU API layer (`yolozu.integrations.layers.api`)
   - CLI/API wrapper execution
   - argument/path guards
   - allowlist enforcement
3. Job layer (`yolozu.integrations.layers.jobs`)
   - async execution with `job_id`
   - queue / running / completed / failed / cancelled tracking
4. Artifact layer (`yolozu.integrations.layers.artifacts`)
   - run listing/description
   - metadata (`git_sha`, `manifest_sha256`, runtime info)

## Long-running jobs

Long tools should return quickly with `job_id`.

Current API surface:
- `jobs.list`
- `jobs.status(job_id)`
- `jobs.cancel(job_id)`
- `runs.list`
- `runs.describe(run_id)`

## Security policy

- Only allowlisted `yolozu` top-level subcommands are executable.
- Path traversal (`..`) is rejected.
- Absolute paths outside workspace are rejected.

## CI no-abandon rule

When MCP-related changes are pushed:
1. run local checks (`manifest`, unit tests)
2. push and monitor latest CI
3. if failure appears, patch and re-run immediately
