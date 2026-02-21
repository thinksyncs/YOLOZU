from __future__ import annotations

from typing import Any


def export_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        image = str(record["image"])
        detections: list[dict[str, Any]] = []
        predictions.append({"image": image, "detections": detections})
    return predictions


def main() -> None:
    raise SystemExit("Fill this starter with Ultralytics inference and write YOLOZU predictions JSON.")


if __name__ == "__main__":
    main()
