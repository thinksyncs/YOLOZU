from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        image = str(record["image"])
        detections: list[dict[str, Any]] = []
        predictions.append({"image": image, "detections": detections})
    return predictions


def main() -> None:
    raise SystemExit("Fill this starter with OpenCV-dnn inference and write YOLOZU predictions JSON.")


if __name__ == "__main__":
    main()
