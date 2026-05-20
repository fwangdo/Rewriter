"""Gemm test subgraphs for the rewriter pipeline.

Gemm computes: Y = alpha * (A @ B) + beta * C
Cases:

1. basic:           A[M,K] @ B[K,N] + C[N]        (transB=0, alpha=1, beta=1)
2. transB:          A[M,K] @ B[N,K]^T + C[N]      (transB=1)
3. transA:          A[K,M]^T @ B[K,N] + C[N]      (transA=1)
4. no_bias:         A[M,K] @ B[K,N]               (no C input)
5. alpha_beta:      0.5 * (A @ B^T) + 2.0 * C     (non-default scaling)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def gemm_basic(M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """A[M,K] @ B[K,N] + C[N] — default attrs."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    B = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), name="B")
    C = numpy_helper.from_array(np.random.randn(N).astype(np.float32), name="C")

    node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"])
    graph = helper.make_graph([node], "gemm_basic", [A], [Y], initializer=[B, C])
    return _make_model(graph)


def gemm_transB(M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """A[M,K] @ B[N,K]^T + C[N] — transB=1."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    B = numpy_helper.from_array(np.random.randn(N, K).astype(np.float32), name="B")
    C = numpy_helper.from_array(np.random.randn(N).astype(np.float32), name="C")

    node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], transB=1)
    graph = helper.make_graph([node], "gemm_transB", [A], [Y], initializer=[B, C])
    return _make_model(graph)


def gemm_transA(M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """A[K,M]^T @ B[K,N] + C[N] — transA=1."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [K, M])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    B = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), name="B")
    C = numpy_helper.from_array(np.random.randn(N).astype(np.float32), name="C")

    node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], transA=1)
    graph = helper.make_graph([node], "gemm_transA", [A], [Y], initializer=[B, C])
    return _make_model(graph)


def gemm_no_bias(M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """A[M,K] @ B[K,N] — no bias term."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    B = numpy_helper.from_array(np.random.randn(K, N).astype(np.float32), name="B")

    node = helper.make_node("Gemm", ["A", "B"], ["Y"])
    graph = helper.make_graph([node], "gemm_no_bias", [A], [Y], initializer=[B])
    return _make_model(graph)


def gemm_alpha_beta(M: int = 4, K: int = 8, N: int = 6) -> onnx.ModelProto:
    """0.5 * (A @ B^T) + 2.0 * C — non-default alpha/beta, transB=1."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    B = numpy_helper.from_array(np.random.randn(N, K).astype(np.float32), name="B")
    C = numpy_helper.from_array(np.random.randn(N).astype(np.float32), name="C")

    node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"],
                            alpha=0.5, beta=2.0, transB=1)
    graph = helper.make_graph([node], "gemm_alpha_beta", [A], [Y], initializer=[B, C])
    return _make_model(graph)


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    cases = [
        ("basic", gemm_basic),
        ("transB", gemm_transB),
        ("transA", gemm_transA),
        ("no_bias", gemm_no_bias),
        ("alpha_beta", gemm_alpha_beta),
    ]
    for name, fn in cases:
        model = fn()
        print(f"{name}: {len(model.graph.node)} node(s), "
              f"inputs={[i.name for i in model.graph.input]}, "
              f"outputs={[o.name for o in model.graph.output]}")
