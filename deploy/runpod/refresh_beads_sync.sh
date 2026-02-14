#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

echo "[runpod] Refreshing origin/beads-sync (robust for --single-branch clones)"
git fetch origin +refs/heads/beads-sync:refs/remotes/origin/beads-sync

BEADS_WT="${REPO_ROOT}/.git/beads-worktrees/beads-sync"
if [[ ! -d "${BEADS_WT}" ]]; then
  echo "[runpod] beads worktree missing; creating via bd sync (no push)"
  bd sync --accept-rebase --no-push
fi

echo "[runpod] Resetting beads worktree to origin/beads-sync"
git -C "${BEADS_WT}" reset --hard origin/beads-sync

echo "[runpod] Importing latest JSONL into bd DB"
bd sync --import-only

echo "[runpod] Current bd list:"
bd list || true
