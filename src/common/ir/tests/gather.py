"""Gather test subgraphs for the rewriter pipeline.

Gather(data, indices, axis=0) selects rows from data by indices.
Cases:

1. both_static:         data[V,D] and indices[S] both initializers → precompute
2. scalar_static_idx:   data is runtime, indices is scalar initializer → Slice+Reshape
3. small_vocab:         data[V,D] initializer, indices runtime, small V → Equal+Mul+Add
4. large_vocab:         data[V,D] initializer, indices runtime, large V → chunked broadcast
5. dynamic:             both runtime (cannot convert — baseline)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def gather_both_static(V: int = 10, D: int = 4, S: int = 3) -> onnx.ModelProto:
    """data[V,D] and indices[S] both static — fully precomputable."""
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [S, D])
    data = numpy_helper.from_array(np.random.randn(V, D).astype(np.float32), name="data")
    indices = numpy_helper.from_array(np.array([0, 2, 5], dtype=np.int64)[:S], name="indices")

    node = helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0)
    graph = helper.make_graph([node], "gather_both_static", [], [Y], initializer=[data, indices])
    return _make_model(graph)


def gather_scalar_static_idx(V: int = 10, D: int = 4) -> onnx.ModelProto:
    """data[V,D] runtime, scalar static index → Slice+Reshape."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [V, D])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [D])
    idx = numpy_helper.from_array(np.array(3, dtype=np.int64), name="idx")

    node = helper.make_node("Gather", ["X", "idx"], ["Y"], axis=0)
    graph = helper.make_graph([node], "gather_scalar_static_idx", [X], [Y], initializer=[idx])
    return _make_model(graph)


def gather_small_vocab(V: int = 16, D: int = 8, S: int = 4) -> onnx.ModelProto:
    """data[V,D] static, indices[S] runtime, small V → Equal+Mul+Add chain."""
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [S])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [S, D])
    data = numpy_helper.from_array(np.random.randn(V, D).astype(np.float32), name="data")

    node = helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0)
    graph = helper.make_graph([node], "gather_small_vocab", [indices], [Y], initializer=[data])
    return _make_model(graph)


def gather_large_vocab(V: int = 3000, D: int = 8, S: int = 4) -> onnx.ModelProto:
    """data[V,D] static, indices[S] runtime, large V → chunked broadcast."""
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [S])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [S, D])
    data = numpy_helper.from_array(np.random.randn(V, D).astype(np.float32), name="data")

    node = helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0)
    graph = helper.make_graph([node], "gather_large_vocab", [indices], [Y], initializer=[data])
    return _make_model(graph)


def gather_dynamic(V: int = 10, D: int = 4, S: int = 3) -> onnx.ModelProto:
    """Both data and indices are runtime — cannot convert (baseline)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [V, D])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [S])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [S, D])

    node = helper.make_node("Gather", ["X", "indices"], ["Y"], axis=0)
    graph = helper.make_graph([node], "gather_dynamic", [X, indices], [Y])
    return _make_model(graph)


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    cases = [
        ("both_static", gather_both_static),
        ("scalar_static_idx", gather_scalar_static_idx),
        ("small_vocab", gather_small_vocab),
        ("large_vocab", gather_large_vocab),
        ("dynamic", gather_dynamic),
    ]
    for name, fn in cases:
        model = fn()
        init_names = {i.name for i in model.graph.initializer}
        inp_names = [i.name for i in model.graph.input if i.name not in init_names]
        print(f"{name}: {len(model.graph.node)} node(s), "
              f"inputs={inp_names}, "
              f"outputs={[o.name for o in model.graph.output]}")
