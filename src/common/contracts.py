from __future__ import annotations

"""Shared operator contracts for ONNX rewrite and future superopt stages."""

BENCHMARK_ONNX_OPSET = 17
BENCHMARK_ONNX_MIN_OPSET = 13

VISION_SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "AveragePool",
        "Cast",
        "Concat",
        "Conv",
        "Div",
        "GlobalAveragePool",
        "HardSigmoid",
        "HardSwish",
        "MatMul",
        "MaxPool",
        "Mul",
        "ReduceMean",
        "Relu",
        "Reshape",
        "Resize",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Split",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Transpose",
    }
)

LLM_SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "Add",
        "Cast",
        "Concat",
        "Div",
        "Gather",
        "MatMul",
        "Mul",
        "ReduceMean",
        "Reshape",
        "Sigmoid",
        "Slice",
        "Softmax",
        "Sqrt",
        "Sub",
        "Tanh",
        "Transpose",
    }
)

VISION_OPS = VISION_SUPPORTED_OPS
LLM_OPS = LLM_SUPPORTED_OPS

UNION_SUPPORTED_OPS: frozenset[str] = frozenset(
    VISION_SUPPORTED_OPS
    | LLM_SUPPORTED_OPS
    | {
        "Clip",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Equal",
        "Erf",
        "Expand",
        "Flatten",
        "GatherElements",
        "Gelu",
        "IsNaN",
        "LeakyRelu",
        "Less",
        "LessOrEqual",
        "Max",
        "Min",
        "Mod",
        "Neg",
        "Pad",
        "Pow",
        "Range",
        "ReduceMax",
        "ReduceSum",
        "Shape",
        "Sin",
        "Tanh",
        "Tile",
        "TopK",
        "Unsqueeze",
        "Where",
    }
)

SUPPORTED_OPS: frozenset[str] = UNION_SUPPORTED_OPS

# Practical LLM contract reading:
# - META_OPS are shape/layout bookkeeping ops that real systems often try to
#   reduce aggressively, but may still tolerate in limited form.
# - MUST_REMOVE_OPS are the dynamic mask/cache ops we want to eliminate even
#   in a practical accelerator-style contract.
LLM_META_OPS: frozenset[str] = frozenset(
    {
        "Concat",
        "Gather",
        "Reshape",
        "Shape",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    }
)

LLM_MUST_REMOVE_OPS: frozenset[str] = frozenset(
    {
        "Cos",
        "Expand",
        "Range",
        "ScatterND",
        "Sin",
        "Trilu",
        "Where",
    }
)
