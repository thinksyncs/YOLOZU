def export_onnx(
    model,
    dummy_input,
    output_path,
    *,
    opset_version: int = 17,
    input_name: str = "images",
    dynamic_hw: bool = False,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for export") from exc
    try:
        import onnx  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("onnx is required for torch.onnx.export (pip install onnx)") from exc

    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            return (
                out["logits"],
                out["bbox"],
                out["log_z"],
                out["rot6d"],
                out["offsets"],
                out["k_delta"],
            )

    wrapper = ExportWrapper(model).eval()
    input_name = str(input_name) if input_name else "images"
    input_dynamic_axes: dict[int, str] = {0: "batch"}
    if bool(dynamic_hw):
        input_dynamic_axes[2] = "height"
        input_dynamic_axes[3] = "width"
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=[input_name],
        output_names=["logits", "bbox", "log_z", "rot6d", "offsets", "k_delta"],
        opset_version=int(opset_version),
        dynamic_axes={
            input_name: input_dynamic_axes,
            "logits": {0: "batch"},
            "bbox": {0: "batch"},
            "log_z": {0: "batch"},
            "rot6d": {0: "batch"},
            "offsets": {0: "batch"},
            "k_delta": {0: "batch"},
        },
    )
