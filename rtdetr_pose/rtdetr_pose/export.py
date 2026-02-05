def export_onnx(model, dummy_input, output_path, *, opset_version: int = 17):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for export") from exc

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
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits", "bbox", "log_z", "rot6d", "offsets", "k_delta"],
        opset_version=int(opset_version),
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "bbox": {0: "batch"},
            "log_z": {0: "batch"},
            "rot6d": {0: "batch"},
            "offsets": {0: "batch"},
            "k_delta": {0: "batch"},
        },
    )
