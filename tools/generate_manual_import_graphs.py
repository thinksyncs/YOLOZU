"""Generate manual graphs (DOT + TikZ) without requiring Graphviz.

Outputs are written under manual/figures/:
- import_deps_packages.dot: package-level internal import graph (DOT)
- import_deps_packages.tikz: rendered TikZ diagram for LaTeX inclusion

Graphviz `dot` is NOT required. The manual includes the TikZ output.

Design intent:
- Keep the PDF readable: collapse to package-level nodes.
- Track only internal imports rooted in this repo.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ImportEdge:
    src: str
    dst: str


INTERNAL_ROOTS = {
    "rtdetr_pose",
    "yolozu",
    "tools",
    "tests",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def iter_python_files(root: Path) -> Iterable[Path]:
    for rel in ("tools", "rtdetr_pose", "yolozu", "tests"):
        base = root / rel
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            # skip build artifacts
            if any(part in {"__pycache__", "build", ".venv"} for part in p.parts):
                continue
            yield p


def module_name_for_path(root: Path, path: Path) -> str | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None

    parts = list(rel.parts)
    if not parts:
        return None

    if parts[0] == "rtdetr_pose" and len(parts) >= 2 and parts[1] == "rtdetr_pose":
        # rtdetr_pose/rtdetr_pose/foo/bar.py -> rtdetr_pose.foo.bar
        sub = parts[2:]
        if not sub:
            return "rtdetr_pose"
        sub[-1] = sub[-1].removesuffix(".py")
        return ".".join(["rtdetr_pose", *sub]).removesuffix(".__init__")

    if parts[0] in {"yolozu", "tests"}:
        sub = parts[1:]
        sub[-1] = sub[-1].removesuffix(".py")
        return ".".join([parts[0], *sub]).removesuffix(".__init__")

    if parts[0] == "tools":
        # tools scripts are not a real package, but for graph readability we still namespace them.
        if len(parts) == 2:
            return f"tools.{parts[1].removesuffix('.py')}"
        sub = parts[1:]
        sub[-1] = sub[-1].removesuffix(".py")
        return ".".join(["tools", *sub]).removesuffix(".__init__")

    return None


def resolve_import_from(base_module: str, level: int, module: str | None) -> str | None:
    if level <= 0:
        return module

    base_parts = base_module.split(".")
    # level=1 means "from . import" (same package)
    # level=2 means parent package, etc.
    cut = max(0, len(base_parts) - level)
    prefix = base_parts[:cut]
    if module:
        return ".".join(prefix + module.split(".")) if prefix else module
    return ".".join(prefix) if prefix else None


def top_package(mod: str) -> str:
    return mod.split(".", 1)[0]


def extract_internal_edges(root: Path) -> list[ImportEdge]:
    edges: list[ImportEdge] = []

    for path in iter_python_files(root):
        src_mod = module_name_for_path(root, path)
        if not src_mod:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            # skip generated/broken files
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name:
                        continue
                    dst_top = alias.name.split(".", 1)[0]
                    if dst_top in INTERNAL_ROOTS:
                        edges.append(ImportEdge(src=src_mod, dst=alias.name))

            elif isinstance(node, ast.ImportFrom):
                dst = resolve_import_from(src_mod, node.level, node.module)
                if not dst:
                    continue
                dst_top = dst.split(".", 1)[0]
                if dst_top in INTERNAL_ROOTS:
                    edges.append(ImportEdge(src=src_mod, dst=dst))

    return edges


def collapse_to_packages(edges: Iterable[ImportEdge]) -> list[ImportEdge]:
    collapsed = set()
    for e in edges:
        s = top_package(e.src)
        d = top_package(e.dst)
        if s == d:
            continue
        collapsed.add((s, d))
    return [ImportEdge(src=s, dst=d) for (s, d) in sorted(collapsed)]


def write_dot(path: Path, edges: Iterable[ImportEdge]) -> None:
    nodes = sorted({e.src for e in edges} | {e.dst for e in edges})
    lines: list[str] = []
    lines.append("digraph import_deps {")
    lines.append("  rankdir=LR;")
    lines.append("  bgcolor=\"transparent\";")
    lines.append("  node [shape=box, style=\"rounded,filled\", fillcolor=\"#f7f7f7\", color=\"#333333\", fontname=\"Helvetica\"]; ")
    lines.append("  edge [color=\"#555555\", penwidth=1.2, arrowsize=0.8];")

    for n in nodes:
        lines.append(f"  \"{n}\";")

    for e in edges:
        lines.append(f"  \"{e.src}\" -> \"{e.dst}\";")

    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tikz(path: Path, edges: Iterable[ImportEdge]) -> None:
    # Fixed layout for package-level graph to keep it readable.
    nodes = sorted({e.src for e in edges} | {e.dst for e in edges})
    order = [n for n in ("tests", "tools", "yolozu", "rtdetr_pose") if n in nodes]
    remaining = [n for n in nodes if n not in order]
    nodes = order + remaining

    def tikz_id(name: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)
        if not safe:
            safe = "node"
        if safe[0].isdigit():
            safe = f"n_{safe}"
        return f"pkg_{safe}"

    def tex_escape(text: str) -> str:
        # Minimal escaping for TikZ node labels.
        return (
            text.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("#", r"\#")
            .replace("&", r"\&")
        )

    # Coordinates.
    # For the common 4-node summary (tools/yolozu/rtdetr_pose/tests), use a stable 2-column grid.
    # This avoids the previous "always left" arrow illusion from using east->west anchors at x=0.
    if set(nodes) == {"tools", "yolozu", "rtdetr_pose", "tests"} and len(nodes) == 4:
        coords = {
            "tests": (0.0, 0.0),
            "tools": (0.0, -2.2),
            "yolozu": (6.0, 0.0),
            "rtdetr_pose": (6.0, -2.2),
        }
    else:
        # Fallback: vertical column.
        y_step = 1.3
        coords = {n: (0.0, -i * y_step) for i, n in enumerate(nodes)}

    lines: list[str] = []
    lines.append("% Auto-generated by tools/generate_manual_import_graphs.py")
    lines.append("\\begin{tikzpicture}[")
    lines.append("  font=\\small,")
    lines.append("  box/.style={draw, rounded corners, align=center, inner sep=5pt, fill=black!3},")
    lines.append("  arrow/.style={-{Latex[length=2mm]}, thick, draw=black!60},")
    lines.append("]")

    for n, (x, y) in coords.items():
        node_id = tikz_id(n)
        label = tex_escape(n)
        lines.append(f"  \\node[box] ({node_id}) at ({x:.1f},{y:.1f}) {{{label}}};")

    edge_set = {(e.src, e.dst) for e in edges}
    for e in edges:
        if e.src not in coords or e.dst not in coords:
            continue
        src_id = tikz_id(e.src)
        dst_id = tikz_id(e.dst)
        # If there is a reciprocal edge, bend to keep both visible.
        if (e.dst, e.src) in edge_set and e.src != e.dst:
            bend = "bend left=18" if e.src < e.dst else "bend right=18"
            lines.append(f"  \\draw[arrow, {bend}] ({src_id}) to ({dst_id});")
        else:
            lines.append(f"  \\draw[arrow] ({src_id}) -- ({dst_id});")

    lines.append("\\end{tikzpicture}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    out_dir = root / "manual" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = extract_internal_edges(root)
    pkg_edges = collapse_to_packages(edges)

    dot_out = out_dir / "import_deps_packages.dot"
    tikz_out = out_dir / "import_deps_packages.tikz"

    write_dot(dot_out, pkg_edges)
    write_tikz(tikz_out, pkg_edges)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
