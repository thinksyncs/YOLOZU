# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim               # Claim work (assignee + in_progress)
bd update <id> --status in_progress  # Alternative (no auto-claim)
bd close <id>         # Complete work
bd sync               # Sync with git
```

## GitHub Issues Linking

Use `external-ref` to link a Beads issue to a GitHub Issue number.

```bash
bd update <id> --external-ref gh-123
```

## Multi-environment / Team Workflow (2台開発)

- まず `git pull --rebase`（`.beads/*.jsonl` もここで更新される）
- 着手するissueは `bd update <id> --claim`（同時編集を避ける）
- 競合したら `bd resolve-conflicts` → `bd sync`
- Beadsの共有は `bd sync` の `beads-sync` ブランチで行う（全clone共通）

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
