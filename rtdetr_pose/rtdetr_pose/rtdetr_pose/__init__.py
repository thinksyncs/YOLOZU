"""Compatibility shim for legacy imports.

Some environments resolve the inner package as `rtdetr_pose` (e.g., when
`repo_root/rtdetr_pose` is on `sys.path`). Older code/tests may still import
`rtdetr_pose.rtdetr_pose.*`. This subpackage keeps those imports working by
re-exporting modules from the parent package.
"""

