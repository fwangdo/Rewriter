"""Lowering definitions for individual ONNX ops.

Each function takes symbolic inputs and returns a primitive op tree.
e.g. lower_matmul(A[M,K], B[K,N]) -> ReduceSum(BroadcastMul(A, B), axis=K)

Reference: ONNX spec reference implementations.
"""
