"""MatMul test subgraphs for the rewriter pipeline.

Each function builds a minimal ONNX subgraph containing a MatMul node.
Cases (excluding batched and channel-limit):

1. static_right_2d:  [M,K] @ W[K,N]
2. static_right_3d:  [B,M,K] @ W[K,N]
3. static_right_4d:  [B,H,M,K] @ W[K,N]
4. static_left_2d:   W[M,K] @ [K,N]
5. static_left_3d:   W[M,K] @ [B,K,N]
6. dynamic:          [B,M,K] @ [B,K,N]  (both runtime)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def matmul_static_right_2d(M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """X[M,K] @ W[K,N] — static right weight, 2D."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    W = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), name="W")

    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "matmul_static_right_2d", [X], [Y], initializer=[W])
    return _make_model(graph)


def matmul_static_right_3d(B: int = 2, M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """X[B,M,K] @ W[K,N] — static right weight, 3D activation."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, M, N])
    W = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), name="W")

    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "matmul_static_right_3d", [X], [Y], initializer=[W])
    return _make_model(graph)


def matmul_static_right_4d(B: int = 2, H: int = 3, M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """X[B,H,M,K] @ W[K,N] — static right weight, 4D activation."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, H, M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, H, M, N])
    W = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), name="W")

    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "matmul_static_right_4d", [X], [Y], initializer=[W])
    return _make_model(graph)


def matmul_static_left_2d(M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """W[M,K] @ X[K,N] — static left weight, 2D."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [K, N])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    W = numpy_helper.from_array(np.random.randn(M, K).astype(np.float32), name="W")

    node = helper.make_node("MatMul", ["W", "X"], ["Y"])
    graph = helper.make_graph([node], "matmul_static_left_2d", [X], [Y], initializer=[W])
    return _make_model(graph)


def matmul_static_left_3d(B: int = 2, M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """W[M,K] @ X[B,K,N] — static left weight, 3D activation."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [B, K, N])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, M, N])
    W = numpy_helper.from_array(np.random.randn(M, K).astype(np.float32), name="W")

    node = helper.make_node("MatMul", ["W", "X"], ["Y"])
    graph = helper.make_graph([node], "matmul_static_left_3d", [X], [Y], initializer=[W])
    return _make_model(graph)


def matmul_dynamic(B: int = 2, M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """A[B,M,K] @ B[B,K,N] — both inputs are runtime, no weights."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [B, M, K])
    Binp = helper.make_tensor_value_info("B", TensorProto.FLOAT, [B, K, N])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [B, M, N])

    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "matmul_dynamic", [A, Binp], [Y])
    return _make_model(graph)


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


# --- quick sanity check ---
if __name__ == "__main__":
    cases = [
        ("static_right_2d", matmul_static_right_2d),
        ("static_right_3d", matmul_static_right_3d),
        ("static_right_4d", matmul_static_right_4d),
        ("static_left_2d", matmul_static_left_2d),
        ("static_left_3d", matmul_static_left_3d),
        ("dynamic", matmul_dynamic),
    ]
    for name, fn in cases:
        model = fn()
        print(f"{name}: {len(model.graph.node)} node(s), "
              f"inputs={[i.name for i in model.graph.input]}, "
              f"outputs={[o.name for o in model.graph.output]}")
