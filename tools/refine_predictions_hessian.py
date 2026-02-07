import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Input predictions JSON.")
    p.add_argument("--output", required=True, help="Output predictions JSON.")
    p.add_argument("--wrap", action="store_true", help="Wrap output as {predictions:[...], meta:{...}}.")
    p.add_argument("--refine-offsets", action="store_true", help="Refine per-detection offsets (experimental).")
    p.add_argument("--dry-run", action="store_true", help="Write schema-only output without changing values.")
    return p.parse_args(argv)


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj, None
    if isinstance(obj, dict) and isinstance(obj.get("predictions"), list):
        meta = obj.get("meta")
        return list(obj["predictions"]), meta if isinstance(meta, dict) else None
    raise ValueError("unsupported predictions format")


def _refine_entry(entry: dict[str, Any], *, refine_offsets: bool, dry_run: bool) -> dict[str, Any]:
    dets = entry.get("detections")
    if not isinstance(dets, list):
        return dict(entry)

    new_entry = dict(entry)
    new_dets = []
    for det in dets:
        if not isinstance(det, dict):
            new_dets.append(det)
            continue

        new_det = dict(det)
        refined = []
        warnings: list[str] = []

        if refine_offsets:
            offsets = new_det.get("offsets")
            if isinstance(offsets, list) and len(offsets) == 2:
                refined.append("offsets")
                delta = [0.0, 0.0]
                if not dry_run:
                    # Placeholder: keep values unchanged but record a stable refinement payload.
                    new_det["offsets"] = [float(offsets[0]) + float(delta[0]), float(offsets[1]) + float(delta[1])]
            else:
                warnings.append("missing_offsets")

        new_det["hessian_refinement"] = {
            "enabled": bool(refine_offsets),
            "refined": refined,
            "warnings": warnings,
        }
        new_dets.append(new_det)

    new_entry["detections"] = new_dets
    return new_entry


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    in_path = _resolve(args.predictions)
    out_path = _resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predictions, meta_in = _load_predictions(in_path)
    refined = [
        _refine_entry(p, refine_offsets=bool(args.refine_offsets), dry_run=bool(args.dry_run)) for p in predictions
    ]

    if args.wrap:
        payload: dict[str, Any] = {
            "predictions": refined,
            "meta": {
                "timestamp": _now_utc(),
                "tool": "refine_predictions_hessian",
                "refine_offsets": bool(args.refine_offsets),
                "dry_run": bool(args.dry_run),
                "input_meta": meta_in,
            },
        }
    else:
        payload = refined

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

