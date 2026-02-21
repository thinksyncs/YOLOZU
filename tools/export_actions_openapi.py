from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLOZU Actions API OpenAPI schema to JSON.")
    parser.add_argument("--output", default="reports/actions_openapi.json", help="Output JSON path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2).")
    args = parser.parse_args()

    try:
        from yolozu.integrations.actions_api import app
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "dependency")
        if str(missing).startswith("yolozu"):
            print(f"Missing local module: {missing}. Ensure local package is installed (e.g. pip install -e .).")
        else:
            print(f"Missing optional dependency: {missing}. Install extras with: pip install '.[actions]'")
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(app.openapi(), indent=args.indent, ensure_ascii=False), encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
