# Migration helpers (`yolozu migrate`)

YOLOZU aims to accept common dataset layouts “as-is” and convert them into a stable internal contract.
The `yolozu migrate` subcommands generate small **descriptor artifacts** so downstream commands can treat
external datasets uniformly.

## 1) Ultralytics / YOLO (detect/pose/segment)

Ultralytics datasets are typically described by a `data.yaml`:

```yaml
path: /path/to/dataset_root
train: images/train
val: images/val
```

### `dataset.json` wrapper

`yolozu migrate dataset --from ultralytics` writes a `dataset.json` that points at the resolved
`images_dir` / `labels_dir` and optionally pins how to parse label txt files.

Example (segment / polygon labels):

```bash
yolozu migrate dataset \
  --from ultralytics \
  --data /path/to/data.yaml \
  --args /path/to/ultralytics/runs/.../args.yaml \
  --task segment \
  --output data/ultra_segment_wrapper
```

Outputs:
- `data/ultra_segment_wrapper/dataset.json`

`dataset.json` keys (subset):
- `images_dir`: absolute/relative path to `.../images/<split>`
- `labels_dir`: absolute/relative path to `.../labels/<split>`
- `split`: default split name
- `label_format`: `detect` (default) or `segment` (YOLO polygon labels)

You can then pass the wrapper directory anywhere YOLOZU expects a dataset root:

```bash
yolozu validate dataset data/ultra_segment_wrapper
```

### YOLO polygon labels

When `label_format=segment`, YOLOZU expects label txt lines like:

```
class_id x1 y1 x2 y2 ... xn yn
```

YOLOZU derives a bbox from the polygon and keeps the raw polygon under `label["polygon"]`.

## 2) VOC / Cityscapes / ADE20K (semantic segmentation)

These datasets use class-id PNG masks rather than YOLO bbox labels.

`yolozu migrate seg-dataset` generates a semantic-segmentation dataset descriptor JSON compatible with:
- `yolozu.segmentation_dataset.load_seg_dataset_descriptor()`
- `tools/eval_segmentation.py`

Examples:

```bash
yolozu migrate seg-dataset --from voc --root /path/to/VOCdevkit/VOC2012 --split val --output reports/voc_seg_dataset.json
yolozu migrate seg-dataset --from cityscapes --root /path/to/cityscapes --split val --output reports/cityscapes_seg_dataset.json
yolozu migrate seg-dataset --from ade20k --root /path/to/ade20k --split val --output reports/ade20k_seg_dataset.json
```

The output descriptor contains:
- `samples`: `[{ "id": "...", "image": "...", "mask": "..." }, ...]`
- `classes` (when known)
- `ignore_index`
- `path_type`: `absolute` (default) or `relative`

