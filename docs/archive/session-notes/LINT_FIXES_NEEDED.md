# Linting Fixes Needed for Open PRs

## Summary
This document lists the linting errors found in currently open PRs that need to be fixed before they can be merged.

## PR #8: copilot/implement-gradient-clipping

**Status**: Failed CI - Lint Step

**Error**: F401 - Unused import

**File**: `rtdetr_pose/tests/test_train_minimal_integration.py`

**Line**: 2

**Issue**: `json` module is imported but never used

**Fix**: Remove the unused import on line 2

```diff
import importlib.util
-import json
import tempfile
import unittest
from pathlib import Path
```

**CI Log Reference**: Workflow run 21734491317

## How to Apply Fixes

For PR maintainers:

1. Checkout the PR branch
2. Apply the fix shown above
3. Run `ruff check .` to verify
4. Commit and push

For reviewers:

The fix is straightforward and can be automated with:
```bash
ruff check --fix .
```

## Additional Notes

- All other checked PRs (#6, #7) either pass linting or have not been fully analyzed yet
- The main branch (d36c5e7) passes all linting checks
- This PR (#9) documents these findings and passes linting itself
