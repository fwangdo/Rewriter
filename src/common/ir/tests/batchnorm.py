"""BatchNormalization test subgraphs for the rewriter pipeline.

BN formula: y = (x - mean) / sqrt(var + eps) * scale + bias
Converted:  y = x * scale_factor + bias_factor  (Mul + Add)

Cases:

1. basic_4d:     X[B,C,H,W]  — standard conv feature map
2. small_eps:    X[B,C,H,W]  — eps=1e-12 (numerically sensitive)
3. single_chan:  X[B,1,H,W]  — C=1 edge case
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def bn_basic_4d(B: int = 2, C: int = 8, H: int = 4, W: int = 4) -> onnx.ModelProto:
    """X[B,C,H,W] — standard BN with default eps=1e-5."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, C, H, W])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, C, H, W])

    scale = numpy_helper.from_array(np.random.randn(C).astype(np.float32) * 0.5 + 1.0, name="scale")
    bias = numpy_helper.from_array(np.random.randn(C).astype(np.float32) * 0.1, name="bias")
    mean = numpy_helper.from_array(np.random.randn(C).astype(np.float32) * 0.5, name="mean")
    var = numpy_helper.from_array(np.abs(np.random.randn(C)).astype(np.float32) + 0.1, name="var")

    node = helper.make_node("BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"])
    graph = helper.make_graph([node], "bn_basic_4d", [X], [Y],
                              initializer=[scale, bias, mean, var])
    return _make_model(graph)


def bn_small_eps(B: int = 2, C: int = 8, H: int = 4, W: int = 4) -> onnx.ModelProto:
    """X[B,C,H,W] — BN with eps=1e-12."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, C, H, W])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, C, H, W])

    scale = numpy_helper.from_array(np.random.randn(C).astype(np.float32) * 0.5 + 1.0, name="scale")
    bias = numpy_helper.from_array(np.random.randn(C).astype(np.float32) * 0.1, name="bias")
    mean = numpy_helper.from_array(np.random.randn(C).astype(np.float32) * 0.5, name="mean")
    var = numpy_helper.from_array(np.abs(np.random.randn(C)).astype(np.float32) + 0.1, name="var")

    node = helper.make_node("BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"],
                            epsilon=1e-12)
    graph = helper.make_graph([node], "bn_small_eps", [X], [Y],
                              initializer=[scale, bias, mean, var])
    return _make_model(graph)


def bn_single_channel(B: int = 2, H: int = 4, W: int = 4) -> onnx.ModelProto:
    """X[B,1,H,W] — C=1 edge case."""
    C = 1
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, C, H, W])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, C, H, W])

    scale = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="bias")
    mean = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="mean")
    var = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="var")

    node = helper.make_node("BatchNormalization", ["X", "scale", "bias", "mean", "var"], ["Y"])
    graph = helper.make_graph([node], "bn_single_channel", [X], [Y],
                              initializer=[scale, bias, mean, var])
    return _make_model(graph)


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    cases = [
        ("basic_4d", bn_basic_4d),
        ("small_eps", bn_small_eps),
        ("single_channel", bn_single_channel),
    ]
    for name, fn in cases:
        model = fn()
        print(f"{name}: {len(model.graph.node)} node(s), "
              f"inputs={[i.name for i in model.graph.input]}, "
              f"outputs={[o.name for o in model.graph.output]}")
