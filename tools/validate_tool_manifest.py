import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_RUNNERS = {"python3", "bash"}
_IO_KINDS = {"file", "dir", "string", "number", "json", "stdout"}


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_tool(tool: Any, *, index: int) -> list[str]:
    errors: list[str] = []
    where = f"tools[{index}]"
    if not isinstance(tool, dict):
        return [f"{where}: expected object"]

    tool_id = tool.get("id")
    if not isinstance(tool_id, str) or not tool_id:
        errors.append(f"{where}.id: required string")
    elif not _ID_RE.match(tool_id):
        errors.append(f"{where}.id: invalid format '{tool_id}'")

    entrypoint = tool.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint:
        errors.append(f"{where}.entrypoint: required string")
    else:
        if entrypoint.startswith("/"):
            errors.append(f"{where}.entrypoint: must be repo-relative, got absolute path")
        if ".." in Path(entrypoint).parts:
            errors.append(f"{where}.entrypoint: must not contain '..'")
        ep = _resolve(entrypoint)
        if not ep.exists():
            errors.append(f"{where}.entrypoint: file not found: {entrypoint}")

    runner = tool.get("runner")
    if not isinstance(runner, str) or runner not in _RUNNERS:
        errors.append(f"{where}.runner: must be one of {sorted(_RUNNERS)}")

    summary = tool.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append(f"{where}.summary: required string")

    docs = tool.get("docs")
    if docs is not None:
        if not isinstance(docs, list):
            errors.append(f"{where}.docs: must be a list of repo-relative paths")
        else:
            for j, doc in enumerate(docs):
                if not isinstance(doc, str) or not doc:
                    errors.append(f"{where}.docs[{j}]: must be non-empty string")
                    continue
                if doc.startswith("/"):
                    errors.append(f"{where}.docs[{j}]: must be repo-relative, got absolute path")
                    continue
                p = _resolve(doc)
                if not p.exists():
                    errors.append(f"{where}.docs[{j}]: file not found: {doc}")

    for field in ("inputs", "outputs"):
        items = tool.get(field)
        if items is None:
            continue
        if not isinstance(items, list):
            errors.append(f"{where}.{field}: must be a list")
            continue
        for j, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{where}.{field}[{j}]: expected object")
                continue
            name = item.get("name")
            kind = item.get("kind")
            if not isinstance(name, str) or not name:
                errors.append(f"{where}.{field}[{j}].name: required string")
            if not isinstance(kind, str) or kind not in _IO_KINDS:
                errors.append(f"{where}.{field}[{j}].kind: must be one of {sorted(_IO_KINDS)}")
            flag = item.get("flag")
            if flag is not None and (not isinstance(flag, str) or (flag and not flag.startswith("--"))):
                errors.append(f"{where}.{field}[{j}].flag: expected '--foo' style flag string")

    examples = tool.get("examples")
    if examples is not None:
        if not isinstance(examples, list):
            errors.append(f"{where}.examples: must be a list")
        else:
            for j, ex in enumerate(examples):
                if not isinstance(ex, dict):
                    errors.append(f"{where}.examples[{j}]: expected object")
                    continue
                cmd = ex.get("command")
                if not isinstance(cmd, str) or not cmd.strip():
                    errors.append(f"{where}.examples[{j}].command: required string")

    return errors


def validate_manifest(obj: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["manifest: expected object"]

    mv = obj.get("manifest_version")
    if not isinstance(mv, int) or mv < 1:
        errors.append("manifest.manifest_version: required integer >= 1")

    tools = obj.get("tools")
    if not isinstance(tools, list) or not tools:
        errors.append("manifest.tools: required non-empty list")
        return errors

    seen: set[str] = set()
    for i, tool in enumerate(tools):
        errors.extend(_validate_tool(tool, index=i))
        tool_id = tool.get("id") if isinstance(tool, dict) else None
        if isinstance(tool_id, str) and tool_id:
            if tool_id in seen:
                errors.append(f"tools[{i}].id: duplicate id '{tool_id}'")
            seen.add(tool_id)

    return errors


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate tools/manifest.json structure and references.")
    p.add_argument("--manifest", default="tools/manifest.json", help="Manifest path (default: tools/manifest.json).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    manifest_path = _resolve(str(args.manifest))
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        obj = _load_json(manifest_path)
    except Exception as exc:
        print(f"error: failed to parse JSON: {manifest_path}: {exc}", file=sys.stderr)
        return 2

    errors = validate_manifest(obj)
    if errors:
        print("error: invalid tool manifest:", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 2

    print(f"OK: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

