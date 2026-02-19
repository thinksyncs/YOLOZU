#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from yolozu.eval_protocol import eval_protocol_hash, validate_eval_protocol
from yolozu.instance_segmentation_predictions import validate_instance_segmentation_predictions_payload
from yolozu.predictions import validate_predictions_payload
from yolozu.segmentation_predictions import validate_segmentation_predictions_payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path_text: str) -> Path:
    p = Path(path_text)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _validate_contract(contract: str, payload: Any, *, strict: bool) -> None:
    if contract == "predictions_json":
        validate_predictions_payload(payload, strict=bool(strict))
        return
    if contract == "segmentation_predictions_json":
        validate_segmentation_predictions_payload(payload)
        return
    if contract == "instance_segmentation_predictions_json":
        validate_instance_segmentation_predictions_payload(payload)
        return
    raise ValueError(f"unsupported contract in golden manifest: {contract}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate versioned golden compatibility assets and protocol snapshots.")
    p.add_argument("--manifest", default="baselines/golden/v1/manifest.json", help="Golden manifest JSON path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = _resolve_path(str(args.manifest))
    if not manifest_path.is_file():
        raise SystemExit(f"golden manifest not found: {manifest_path}")

    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit("golden manifest must be a JSON object")

    protocol_obj = manifest.get("protocol") or {}
    protocol_path = _resolve_path(str(protocol_obj.get("path") or ""))
    expected_protocol_hash = str(protocol_obj.get("expected_sha256") or "")
    if not protocol_path.is_file():
        raise SystemExit(f"protocol file not found: {protocol_path}")

    protocol_doc = _load_json(protocol_path)
    validate_eval_protocol(protocol_doc)
    actual_protocol_hash = _sha256_file(protocol_path)
    if expected_protocol_hash and actual_protocol_hash != expected_protocol_hash:
        raise SystemExit(
            "protocol hash mismatch: "
            f"expected={expected_protocol_hash} actual={actual_protocol_hash} path={protocol_path}"
        )

    protocol_sem_hash_expected = str(protocol_obj.get("expected_semantic_hash") or "")
    protocol_sem_hash_actual = eval_protocol_hash(protocol_doc)
    if protocol_sem_hash_expected and protocol_sem_hash_actual != protocol_sem_hash_expected:
        raise SystemExit(
            "protocol semantic hash mismatch: "
            f"expected={protocol_sem_hash_expected} actual={protocol_sem_hash_actual}"
        )

    assets = manifest.get("assets") or []
    if not isinstance(assets, list) or not assets:
        raise SystemExit("golden manifest assets must be a non-empty list")

    checked = 0
    for item in assets:
        if not isinstance(item, dict):
            raise SystemExit("golden asset item must be an object")
        name = str(item.get("name") or "")
        rel_path = str(item.get("path") or "")
        contract = str(item.get("contract") or "")
        strict = bool(item.get("strict", True))
        expected_sha = str(item.get("expected_sha256") or "")
        if not name or not rel_path or not contract or not expected_sha:
            raise SystemExit(f"golden asset entry missing required fields: {item}")

        path = _resolve_path(rel_path)
        if not path.is_file():
            raise SystemExit(f"golden asset not found ({name}): {path}")

        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise SystemExit(
                "golden asset hash mismatch: "
                f"name={name} expected={expected_sha} actual={actual_sha} path={path}"
            )

        payload = _load_json(path)
        _validate_contract(contract, payload, strict=strict)
        checked += 1

    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(manifest_path),
                "assets_checked": checked,
                "protocol_path": str(protocol_path),
                "protocol_sha256": actual_protocol_hash,
                "protocol_semantic_hash": protocol_sem_hash_actual,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
