#!/usr/bin/env bash
set -euo pipefail

# Refresh local bd database from the remote `beads-sync` branch without
# switching branches or writing to `.beads/issues.jsonl` in the working tree.
#
# This avoids a RunPod quirk where `git fetch origin beads-sync` may only update
# FETCH_HEAD (leaving `origin/beads-sync` stale).

REMOTE="${REMOTE:-origin}"
SYNC_BRANCH="${SYNC_BRANCH:-beads-sync}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

git fetch "${REMOTE}" "+refs/heads/${SYNC_BRANCH}:refs/remotes/${REMOTE}/${SYNC_BRANCH}"

tmp="$(mktemp -t beads-sync.XXXXXX.jsonl)"
trap 'rm -f "${tmp}"' EXIT

git show "refs/remotes/${REMOTE}/${SYNC_BRANCH}:.beads/issues.jsonl" > "${tmp}"

# Import requires direct DB access (bd automatically uses --no-daemon).
bd import -i "${tmp}" --force >/dev/null

echo "refreshed from ${REMOTE}/${SYNC_BRANCH}"

