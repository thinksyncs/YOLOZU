import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.dataset import build_manifest  # noqa: E402
from yolozu.instance_segmentation_eval import extract_gt_instances_from_record, load_mask_bool  # noqa: E402
from yolozu.instance_segmentation_eval import evaluate_instance_map  # noqa: E402
from yolozu.instance_segmentation_predictions import iter_instances, load_instance_segmentation_predictions_entries  # noqa: E402
from yolozu.metrics_report import build_report, write_json  # noqa: E402


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate instance segmentation predictions (mask mAP over PNG masks).")
    p.add_argument("--dataset", required=True, help="YOLO-format dataset root (images/ + labels/).")
    p.add_argument("--split", default=None, help="Split under images/ and labels/ (default: auto).")
    p.add_argument("--predictions", required=True, help="Instance segmentation predictions JSON.")
    p.add_argument("--pred-root", default=None, help="Optional root to resolve relative prediction mask paths.")
    p.add_argument("--classes", default=None, help="Optional classes.txt/classes.json for class_id→name.")
    p.add_argument("--output", default="reports/instance_seg_eval.json", help="Output JSON report path.")
    p.add_argument("--html", default=None, help="Optional HTML report path.")
    p.add_argument("--title", default="YOLOZU instance segmentation eval report", help="HTML title.")
    p.add_argument("--overlays-dir", default=None, help="Optional directory to write overlay images for HTML.")
    p.add_argument("--max-overlays", type=int, default=0, help="Max overlays to render (default: 0).")
    p.add_argument("--overlay-max-size", type=int, default=768, help="Max size (max(H,W)) for overlay images (default: 768).")
    p.add_argument("--overlay-alpha", type=float, default=0.5, help="Mask overlay alpha (default: 0.5).")

    p.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for predictions (default: 0.0).")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap for number of images to evaluate.")
    p.add_argument(
        "--allow-rgb-masks",
        action="store_true",
        help="Allow 3-channel masks (uses channel 0; intended for grayscale stored as RGB).",
    )
    return p.parse_args(argv)


def _class_colors(num_classes: int):
    import colorsys

    colors: list[tuple[int, int, int]] = []
    for i in range(int(num_classes)):
        h = (float(i) * 0.6180339887498949) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.70, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _load_class_id_to_name(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return {int(i): str(name) for i, name in enumerate(lines)}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(data, dict):
        names = data.get("class_names") or data.get("names")
        if isinstance(names, list):
            out: dict[int, str] = {}
            for i, name in enumerate(names):
                if name is None:
                    continue
                out[int(i)] = str(name)
            return out
        id_to_name = data.get("id_to_name")
        if isinstance(id_to_name, dict):
            out: dict[int, str] = {}
            for k, v in id_to_name.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out

    return {}


def _overlay_instances(image_rgb, instances: list[dict[str, Any]], *, colors, alpha: float, allow_rgb_masks: bool):
    import numpy as np
    from PIL import Image

    img = image_rgb.convert("RGB")
    base = np.asarray(img).astype("float32")
    out = base.copy()

    a = float(alpha)
    for inst in instances:
        try:
            class_id = int(inst.get("class_id", 0))
        except Exception:
            continue
        color = colors[int(class_id) % len(colors)] if colors else (255, 0, 0)
        mask_val = inst.get("mask")
        if mask_val is None:
            continue
        try:
            m = load_mask_bool(mask_val, allow_rgb=allow_rgb_masks)
            m = np.asarray(m, dtype=bool)
        except Exception:
            continue
        if m.shape[0] != out.shape[0] or m.shape[1] != out.shape[1]:
            continue
        c = np.array(color, dtype="float32")
        out[m] = out[m] * (1.0 - a) + c * a

    return Image.fromarray(out.clip(0, 255).astype("uint8"), mode="RGB")


def _resize_max(image_rgb, *, max_size: int):
    from PIL import Image

    img = image_rgb.convert("RGB")
    w, h = img.size
    m = max(int(w), int(h))
    if m <= int(max_size):
        return img
    scale = float(max_size) / float(m)
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    return img.resize((new_w, new_h), resample=Image.BILINEAR)


def _concat_h(images):
    from PIL import Image

    imgs = [im.convert("RGB") for im in images if im is not None]
    if not imgs:
        raise ValueError("no images to concat")
    heights = [im.size[1] for im in imgs]
    widths = [im.size[0] for im in imgs]
    h = max(heights)
    w = sum(widths)
    out = Image.new("RGB", (w, h), color=(0, 0, 0))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.size[0]
    return out


def _write_html(*, html_path: Path, title: str, report: dict[str, Any], overlays: list[dict[str, Any]] | None):
    metrics = report.get("metrics") if isinstance(report, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}
    meta = report.get("meta") if isinstance(report, dict) else None
    if not isinstance(meta, dict):
        meta = {}

    classes = metrics.get("classes")
    if not isinstance(classes, list):
        classes = []

    def rel(p: str) -> str:
        try:
            return str(Path(p).relative_to(html_path.parent))
        except Exception:
            return str(p)

    lines: list[str] = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '  <meta charset="utf-8" />',
        f"  <title>{title}</title>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:16px;}",
        "    .meta{color:#666; font-size:12px; overflow-wrap:anywhere;}",
        "    table{border-collapse:collapse; width:100%; margin-top:12px;}",
        "    th,td{border:1px solid #ddd; padding:6px 8px; font-size:12px; text-align:right;}",
        "    th:first-child, td:first-child{text-align:left;}",
        "    .grid{display:grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap:16px; margin-top:16px;}",
        "    .card{border:1px solid #ddd; border-radius:8px; padding:8px;}",
        "    img{max-width:100%; height:auto; border-radius:6px;}",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>dataset: {meta.get('dataset')}</p>",
        f"<p class='meta'>split: {meta.get('split')}</p>",
        f"<p class='meta'>predictions: {meta.get('predictions')}</p>",
        f"<p class='meta'>output_json: {rel(str(meta.get('output_json', '')))}</p>",
        f"<p class='meta'>images_evaluated: {meta.get('images_evaluated')}</p>",
        f"<p class='meta'>map50: {metrics.get('map50')}</p>",
        f"<p class='meta'>map50_95: {metrics.get('map50_95')}</p>",
        "<h2>Per-class AP</h2>",
        "<table>",
        "<tr><th>class_id</th><th>name</th><th>ap@0.50</th><th>mean_ap</th></tr>",
    ]

    for c in classes:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        name = c.get("name")
        ap50 = c.get("ap@0.50")
        mean_ap = c.get("map50_95")
        lines.append(f"<tr><td>{cid}</td><td>{name}</td><td>{ap50}</td><td>{mean_ap}</td></tr>")

    lines.extend(["</table>"])

    if overlays:
        lines.append("<h2>Overlays</h2>")
        lines.append("<div class='grid'>")
        for it in overlays:
            if not isinstance(it, dict):
                continue
            overlay_path = it.get("overlay")
            if not isinstance(overlay_path, str) or not overlay_path:
                continue
            lines.extend(
                [
                    "<div class='card'>",
                    f"  <img src='{rel(overlay_path)}' />",
                    f"  <div class='meta'>image: {it.get('image')}</div>",
                    "</div>",
                ]
            )
        lines.append("</div>")

    lines.extend(["</body>", "</html>"])
    html_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root

    pred_json_path = Path(args.predictions)
    if not pred_json_path.is_absolute():
        pred_json_path = repo_root / pred_json_path

    pred_root = Path(args.pred_root) if args.pred_root else pred_json_path.parent
    if not pred_root.is_absolute():
        pred_root = repo_root / pred_root

    class_names: dict[int, str] = {}
    classes_path: Path | None = None
    if args.classes:
        classes_path = Path(args.classes)
        if not classes_path.is_absolute():
            classes_path = repo_root / classes_path
        class_names = _load_class_id_to_name(classes_path)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    manifest = build_manifest(dataset_root, split=args.split)
    records = manifest["images"]
    if args.max_images is not None:
        records = records[: int(args.max_images)]

    entries, pred_meta = load_instance_segmentation_predictions_entries(pred_json_path)

    result = evaluate_instance_map(
        records=records,
        predictions_entries=entries,
        min_score=float(args.min_score),
        pred_root=pred_root,
        allow_rgb_masks=bool(args.allow_rgb_masks),
    )

    per_class_rows: list[dict[str, Any]] = []
    for cid in sorted(result.per_class.keys()):
        ap_map = result.per_class.get(int(cid), {}) or {}
        vals = [float(v) for v in ap_map.values() if isinstance(v, (int, float))]
        name = class_names.get(int(cid))
        per_class_rows.append(
            {
                "id": int(cid),
                "name": str(name) if name is not None else None,
                **ap_map,
                "map50": float(ap_map.get("ap@0.50", 0.0)) if ap_map else 0.0,
                "map50_95": float(sum(vals) / float(len(vals))) if vals else 0.0,
            }
        )

    report = build_report(
        losses={},
        metrics={
            "map50": float(result.map50),
            "map50_95": float(result.map50_95),
            "classes": per_class_rows,
            "counts": dict(result.counts),
        },
        meta={
            "dataset": str(dataset_root),
            "split": str(manifest.get("split")),
            "predictions": str(pred_json_path),
            "pred_root": str(pred_root),
            "classes": str(classes_path) if classes_path is not None else None,
            "output_json": str(output_path),
            "images_evaluated": int(len(records)),
            "min_score": float(args.min_score),
            "allow_rgb_masks": bool(args.allow_rgb_masks),
            "predictions_meta": pred_meta,
            "warnings": list(result.warnings),
        },
    )
    write_json(output_path, report)
    print(output_path)

    overlays_index: list[dict[str, Any]] | None = None
    if args.html:
        overlays_index = []
        if args.overlays_dir and int(args.max_overlays) > 0:
            try:
                import numpy as np
                from PIL import Image
            except Exception as exc:  # pragma: no cover
                raise SystemExit(f"numpy + Pillow required for overlays: {exc}") from exc

            overlays_dir = Path(args.overlays_dir)
            if not overlays_dir.is_absolute():
                overlays_dir = repo_root / overlays_dir
            overlays_dir.mkdir(parents=True, exist_ok=True)

            # Build a lightweight prediction index: image key -> instances.
            pred_index: dict[str, list[dict[str, Any]]] = {}
            for inst in iter_instances(entries):
                image = inst.get("image")
                if not isinstance(image, str) or not image:
                    continue
                try:
                    score = float(inst.get("score", 1.0))
                except Exception:
                    score = 1.0
                if score < float(args.min_score):
                    continue
                mask = inst.get("mask")
                if isinstance(mask, (str, Path)):
                    p = Path(mask)
                    if not p.is_absolute():
                        p = pred_root / p
                    inst = dict(inst)
                    inst["mask"] = str(p)
                pred_index.setdefault(image, []).append(inst)
                base = image.split("/")[-1]
                if base and base not in pred_index:
                    pred_index[base] = pred_index[image]

            # Render first N images that have either GT or preds.
            max_class_id = 0
            if result.per_class:
                max_class_id = max(int(k) for k in result.per_class.keys())
            colors = _class_colors(max(1, int(max_class_id) + 1))
            for idx, rec in enumerate(records):
                if len(overlays_index) >= int(args.max_overlays):
                    break
                image_path = rec.get("image")
                if not isinstance(image_path, str) or not image_path:
                    continue

                gt_instances, _ = extract_gt_instances_from_record(rec, allow_rgb_masks=bool(args.allow_rgb_masks))
                pred_instances = pred_index.get(image_path) or pred_index.get(image_path.split("/")[-1]) or []
                if not gt_instances and not pred_instances:
                    continue

                try:
                    img = Image.open(image_path).convert("RGB")
                except Exception:
                    continue

                # Downscale image for overlays only.
                img_r = _resize_max(img, max_size=int(args.overlay_max_size))
                scale_x = float(img_r.size[0]) / float(img.size[0]) if img.size[0] else 1.0
                scale_y = float(img_r.size[1]) / float(img.size[1]) if img.size[1] else 1.0

                def _resize_mask_bool(mask_bool):
                    m = np.asarray(mask_bool, dtype=bool)
                    if m.shape[0] != img.size[1] or m.shape[1] != img.size[0]:
                        return None
                    if scale_x == 1.0 and scale_y == 1.0:
                        return m
                    im = Image.fromarray((m.astype("uint8") * 255), mode="L")
                    im_r = im.resize(img_r.size, resample=Image.NEAREST)
                    return (np.asarray(im_r).astype("int64", copy=False) != 0)

                # Inline-resize GT masks (they're already loaded arrays).
                gt_r: list[dict[str, Any]] = []
                for inst in gt_instances:
                    m = inst.get("mask")
                    if m is None:
                        continue
                    m_r = _resize_mask_bool(m)
                    if m_r is None:
                        continue
                    gt_r.append({"class_id": inst.get("class_id", 0), "mask": m_r})

                # Pred masks are loaded from disk; resize after load inside overlay function by passing arrays directly.
                pred_r: list[dict[str, Any]] = []
                for inst in pred_instances:
                    mask_path = inst.get("mask")
                    if mask_path is None:
                        continue
                    try:
                        m = load_mask_bool(mask_path, allow_rgb=bool(args.allow_rgb_masks))
                    except Exception:
                        continue
                    m_r = _resize_mask_bool(m)
                    if m_r is None:
                        continue
                    pred_r.append(
                        {
                            "class_id": inst.get("class_id", 0),
                            "mask": m_r,
                        }
                    )

                ov_gt = _overlay_instances(img_r, gt_r, colors=colors, alpha=float(args.overlay_alpha), allow_rgb_masks=bool(args.allow_rgb_masks))
                ov_pred = _overlay_instances(img_r, pred_r, colors=colors, alpha=float(args.overlay_alpha), allow_rgb_masks=bool(args.allow_rgb_masks))

                combined = _concat_h([img_r, ov_gt, ov_pred])
                stem = Path(image_path).stem
                digest = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:8]
                out_path = overlays_dir / f"{idx:06d}_{stem}_{digest}.png"
                try:
                    combined.save(out_path)
                except Exception:
                    continue

                overlays_index.append({"image": image_path.split("/")[-1], "overlay": str(out_path)})

        html_path = Path(args.html)
        if not html_path.is_absolute():
            html_path = repo_root / html_path
        _write_html(html_path=html_path, title=str(args.title), report=report, overlays=overlays_index if overlays_index else None)
        print(html_path)


if __name__ == "__main__":
    main()
