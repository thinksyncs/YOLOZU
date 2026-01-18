def export_onnx(model, dummy_input, output_path):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for export") from exc
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits", "bbox", "log_z", "rot6d", "offsets", "k_delta"],
        opset_version=17,
    )
