# Instance Segmentation Demo (PNG masks)

This is a tiny **synthetic** instance segmentation demo dataset + predictions to showcase YOLOZU's
PNG-mask contract and `mask mAP` evaluator.

## Run

```bash
python3 tools/eval_instance_segmentation.py \
  --dataset examples/instance_seg_demo/dataset \
  --split val2017 \
  --predictions examples/instance_seg_demo/predictions/instance_seg_predictions.json \
  --pred-root examples/instance_seg_demo/predictions \
  --classes examples/instance_seg_demo/classes.txt \
  --output reports/instance_seg_demo_eval.json \
  --html reports/instance_seg_demo_eval.html \
  --overlays-dir reports/instance_seg_demo_overlays \
  --max-overlays 10
```

Open `reports/instance_seg_demo_eval.html` to view overlays.

