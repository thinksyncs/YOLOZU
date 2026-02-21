# Packaged manifests

This folder contains JSON manifests shipped inside the `yolozu` wheel.

- `tools_manifest.json`: a copy of the repository tool manifest (`tools/manifest.json`) for automation and discovery.

Access via:

```bash
yolozu resources list
yolozu resources cat manifest/tools_manifest.json
```
