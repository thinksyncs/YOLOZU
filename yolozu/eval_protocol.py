from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROTOCOLS = {
    "yolo26": _REPO_ROOT / "protocols" / "yolo26_eval.json",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_eval_protocol(protocol_id: str) -> dict[str, Any]:
    path = _DEFAULT_PROTOCOLS.get(protocol_id)
    if path is None:
        raise ValueError(f"unknown protocol: {protocol_id}")
    doc = json.loads(path.read_text())
    validate_eval_protocol(doc)
    return doc


def validate_eval_protocol(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise ValueError("eval protocol must be an object")

    protocol_id = doc.get("id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("eval protocol.id must be a non-empty string")

    split = doc.get("split")
    if not isinstance(split, str) or not split:
        raise ValueError("eval protocol.split must be a non-empty string")

    bbox_format = doc.get("bbox_format")
    if bbox_format not in ("cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs"):
        raise ValueError("eval protocol.bbox_format must be a supported bbox format")

    metric_key = doc.get("metric_key")
    if not isinstance(metric_key, str) or not metric_key:
        raise ValueError("eval protocol.metric_key must be a non-empty string")

    imgsz = doc.get("imgsz")
    if imgsz is not None and not (isinstance(imgsz, int) and imgsz > 0):
        raise ValueError("eval protocol.imgsz must be a positive int or null")

    metric_label = doc.get("metric_label")
    if metric_label is not None and not isinstance(metric_label, str):
        raise ValueError("eval protocol.metric_label must be a string or null")

    reports = doc.get("reports", {})
    if reports is not None and not isinstance(reports, dict):
        raise ValueError("eval protocol.reports must be an object or null")


def apply_eval_protocol_args(args: Any, protocol: dict[str, Any]) -> Any:
    """Apply protocol-defined settings to an argparse namespace."""

    args.split = protocol.get("split", args.split)
    args.bbox_format = protocol.get("bbox_format", args.bbox_format)
    return args

