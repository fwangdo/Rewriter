"""Qualcomm ONNX Runtime QNN EP operator pool.

Source:
https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators

This module intentionally exposes only op-name sets.  Backend-specific
constraints are kept as comments for now; predicate-level legality can be added
after the basic contract is wired into the pipeline.
"""

from __future__ import annotations


QNN_SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "Abs",
        "Add",
        "And",
        "ArgMax",
        "ArgMin",
        "Asin",
        "Atan",
        "AveragePool",
        "BatchNormalization",
        "Cast",
        "Ceil",
        "Clip",
        "Concat",
        "Conv",
        "ConvTranspose",
        "Cos",
        "CumSum",
        "DepthToSpace",
        "DequantizeLinear",
        "Div",
        "Einsum",
        "Elu",
        "Equal",
        "Exp",
        "Expand",
        "Flatten",
        "Floor",
        "Gather", # limited
        "GatherElements",
        "GatherND",
        "Gelu",
        "Gemm",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "Greater",
        "GreaterOrEqual",
        "GridSample",
        "HardSigmoid",
        "HardSwish",
        "InstanceNormalization",
        "Inverse",
        "LRN",
        "LSTM",
        "LayerNormalization",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "Log",
        "LogSoftmax",
        "LpNormalization", # limited. 
        "MatMul",
        "Max",
        "MaxPool",
        "Mean",
        "Min",
        "Mod",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "PRelu",
        "Pad",
        "Pow",
        "QuantizeLinear",
        "RandomUniformLike",
        "Reciprocal",
        "ReduceL2",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "Relu",
        "Resize",
        "Round",
        "STFT",
        "ScatterElements",
        "ScatterND",
        "Sigmoid",
        "Sign",
        "Sin",
        "Slice",
        "Softmax",
        "SpaceToDepth",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Sum",
        "Tanh",
        "ThresholdedRelu",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "Upsample",
        "Where",
    }
)

QNN_LIMITED_OPS: frozenset[str] = frozenset(
    { "Gather", "LpNormalization" } 
)

QNN_CONTRIB_SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "com.microsoft:DequantizeLinear",
        "com.microsoft:Gelu",
        "com.microsoft:QuantizeLinear",
        "com.microsoft.MatMulNBits",
    }
)


QNN_ALL_SUPPORTED_OPS: frozenset[str] = (
    QNN_SUPPORTED_OPS | QNN_CONTRIB_SUPPORTED_OPS
)


# Constraint notes from the QNN EP supported-op table:
# - Gather: only supports positive indices.
# - LpNormalization: only p == 2.

#
# Additional modeling notes:
# - QNN EP may fall back to CPU unless fallback is disabled at ORT session level.
# - Dynamic shapes must be fixed for HTP execution.
# - Op membership is not sufficient for legality; dtype, shape, rank, attrs,
#   quantization form, and selected backend must eventually be checked.


__all__ = [
    "QNN_ALL_SUPPORTED_OPS",
    "QNN_CONTRIB_SUPPORTED_OPS",
    "QNN_SUPPORTED_OPS",
]
