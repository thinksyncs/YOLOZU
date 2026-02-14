import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpStats:
    nodes: int = 0
    macs: int = 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True, help="Path to ONNX model.")
    p.add_argument("--imgsz", type=int, default=640, help="Input resolution used when model has dynamic H/W (default: 640).")
    p.add_argument("--output", default=None, help="Optional JSON output path (default: print to stdout).")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    return p.parse_args(argv)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _dims_to_list(shape) -> list[int | None]:
    dims: list[int | None] = []
    for d in shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        else:
            dims.append(None)
    return dims


def _shape_index(model) -> dict[str, list[int | None]]:
    shapes: dict[str, list[int | None]] = {}

    for init in model.graph.initializer:
        if init.name and init.dims:
            shapes[init.name] = [int(x) for x in init.dims]

    def add_vi(vi):
        try:
            t = vi.type.tensor_type
            if not t.HasField("shape"):
                return
            if not vi.name:
                return
            shapes[vi.name] = _dims_to_list(t.shape)
        except Exception:
            return

    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        add_vi(vi)

    return shapes


def _set_first_input_shape(model, *, imgsz: int) -> None:
    if not model.graph.input:
        return
    vi = model.graph.input[0]
    t = vi.type.tensor_type
    if not t.HasField("shape"):
        return
    dims = t.shape.dim
    if len(dims) < 4:
        return
    if not dims[0].dim_value:
        dims[0].dim_value = 1
    if not dims[1].dim_value:
        dims[1].dim_value = 3
    if not dims[2].dim_value:
        dims[2].dim_value = int(imgsz)
    if not dims[3].dim_value:
        dims[3].dim_value = int(imgsz)


def _count_params(model) -> int:
    total = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= int(d)
        total += int(n)
    return int(total)


def _conv_macs(*, out_shape: list[int | None], w_shape: list[int | None], group: int) -> int | None:
    if len(out_shape) < 4 or len(w_shape) < 4:
        return None
    n, cout, hout, wout = out_shape[:4]
    w_cout, w_cin_g, kh, kw = w_shape[:4]
    n = _coerce_int(n)
    cout = _coerce_int(cout)
    hout = _coerce_int(hout)
    wout = _coerce_int(wout)
    w_cout = _coerce_int(w_cout)
    w_cin_g = _coerce_int(w_cin_g)
    kh = _coerce_int(kh)
    kw = _coerce_int(kw)
    if None in (n, cout, hout, wout, w_cout, w_cin_g, kh, kw):
        return None
    if int(cout) != int(w_cout):
        return None
    k = int(w_cin_g) * int(kh) * int(kw)
    return int(n) * int(cout) * int(hout) * int(wout) * int(k)


def _matmul_macs(a_shape: list[int | None], b_shape: list[int | None]) -> int | None:
    if len(a_shape) < 2 or len(b_shape) < 2:
        return None
    k_a = _coerce_int(a_shape[-1])
    k_b = _coerce_int(b_shape[-2])
    if k_a is None or k_b is None or int(k_a) != int(k_b):
        return None
    m = _coerce_int(a_shape[-2])
    n = _coerce_int(b_shape[-1])
    if m is None or n is None:
        return None
    batch = 1
    for d in a_shape[:-2]:
        di = _coerce_int(d)
        if di is None:
            return None
        batch *= int(di)
    return int(batch) * int(m) * int(n) * int(k_a)


def profile_onnx(*, onnx_path: Path, imgsz: int) -> dict[str, Any]:
    try:
        import onnx  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnx is required (pip install 'yolozu[onnxrt]')") from exc

    model = onnx.load(str(onnx_path))
    _set_first_input_shape(model, imgsz=int(imgsz))
    warnings: list[str] = []

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        warnings.append(f"shape_inference_failed: {exc.__class__.__name__}")

    shapes = _shape_index(model)
    params = _count_params(model)

    stats: dict[str, OpStats] = {}

    for node in model.graph.node:
        op = str(getattr(node, "op_type", "") or "")
        if not op:
            continue
        if op not in {"Conv", "Gemm", "MatMul"}:
            continue

        if op == "Conv":
            if len(node.input) < 2 or len(node.output) < 1:
                continue
            w_name = node.input[1]
            out_name = node.output[0]
            w_shape = shapes.get(w_name)
            out_shape = shapes.get(out_name)
            if w_shape is None or out_shape is None:
                warnings.append(f"missing_shape: Conv {node.name or out_name}")
                continue
            group = 1
            for attr in node.attribute:
                if getattr(attr, "name", "") == "group":
                    group = int(getattr(attr, "i", 1) or 1)
            macs = _conv_macs(out_shape=out_shape, w_shape=w_shape, group=int(group))
            if macs is None:
                warnings.append(f"unknown_macs: Conv {node.name or out_name}")
                continue
        else:
            if len(node.input) < 2:
                continue
            a_shape = shapes.get(node.input[0])
            b_shape = shapes.get(node.input[1])
            if a_shape is None or b_shape is None:
                warnings.append(f"missing_shape: {op} {node.name or node.output[0] if node.output else '?'}")
                continue
            macs = _matmul_macs(a_shape, b_shape)
            if macs is None:
                warnings.append(f"unknown_macs: {op} {node.name or node.output[0] if node.output else '?'}")
                continue

        cur = stats.get(op, OpStats())
        stats[op] = OpStats(nodes=int(cur.nodes) + 1, macs=int(cur.macs) + int(macs))

    total_macs = int(sum(s.macs for s in stats.values()))
    total_flops = int(2 * total_macs)

    inputs = []
    for vi in model.graph.input:
        try:
            name = str(vi.name)
            shape = _dims_to_list(vi.type.tensor_type.shape) if vi.type.tensor_type.HasField("shape") else None
            inputs.append({"name": name, "shape": shape})
        except Exception:
            continue

    return {
        "schema_version": 1,
        "onnx": str(onnx_path),
        "imgsz": int(imgsz),
        "params": int(params),
        "macs": int(total_macs),
        "flops": int(total_flops),
        "ops": {k: {"nodes": int(v.nodes), "macs": int(v.macs), "flops": int(2 * v.macs)} for k, v in stats.items()},
        "inputs": inputs,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    onnx_path = Path(args.onnx)
    payload = profile_onnx(onnx_path=onnx_path, imgsz=int(args.imgsz))

    text = json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

