import tempfile
import unittest
from pathlib import Path

try:
    import onnx
    from onnx import TensorProto, helper
except Exception:  # pragma: no cover
    onnx = None  # type: ignore
    helper = None  # type: ignore
    TensorProto = None  # type: ignore


@unittest.skipIf(onnx is None, "onnx not installed")
class TestONNXRTQuantize(unittest.TestCase):
    def test_quantize_dynamic_writes_model(self):
        from yolozu.onnxrt_quantize import quantize_onnx_dynamic

        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=str(repo_root)) as td:
            root = Path(td)
            model_in = root / "model.onnx"
            model_out = root / "model_int8.onnx"

            # Tiny MatMul graph: Y = X @ W
            x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
            y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
            w_init = helper.make_tensor("W", TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])
            node = helper.make_node("MatMul", ["X", "W"], ["Y"])
            graph = helper.make_graph([node], "g", [x], [y], initializer=[w_init])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            onnx.save(model, model_in)

            out_path = quantize_onnx_dynamic(onnx_in=model_in, onnx_out=model_out, weight_type="qint8")
            self.assertTrue(out_path.exists())
            loaded = onnx.load(str(out_path))
            self.assertIsNotNone(loaded)


if __name__ == "__main__":
    unittest.main()

