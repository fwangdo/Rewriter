from __future__ import annotations

from pathlib import Path


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
        "Transpose",
    }
)

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


_OPSET1_MOBILE_CNN = {
    "Add",
    "AveragePool",
    "Constant",
    "Conv",
    "Flatten",
    "Gemm",
    "GlobalAveragePool",
    "Relu",
    "Reshape",
}

_OPSET2_DETECTION_VISION = _OPSET1_MOBILE_CNN | {
    "Concat",
    "MaxPool",
    "Mul",
    "Pad",
    "Resize",
    "Sigmoid",
    "Slice",
    "Split",
    "Transpose",
}

_OPSET3_HYBRID_TRANSFORMER = _OPSET2_DETECTION_VISION | {
    "Div",
    "Erf",
    "Gather",
    "Gelu",
    "MatMul",
    "ReduceMean",
    "Shape",
    "Softmax",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Unsqueeze",
}

_OPSET4_DECODER_CORE = _OPSET3_HYBRID_TRANSFORMER | {
    "Cast",
    "Cos",
    "Equal",
    "Expand",
    "Neg",
    "Pow",
    "Sin",
    "Tanh",
    "Where",
}

_OPSET5_FULL_BENCHMARK = _OPSET4_DECODER_CORE | {
    "ConstantOfShape",
    "GatherElements",
    "HardSigmoid",
    "HardSwish",
    "IsNaN",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Max",
    "Min",
    "Mod",
    "Range",
    "ReduceMax",
    "ReduceSum",
    "Tile",
    "TopK",
}


LOGICAL_OPSETS: dict[str, frozenset[str]] = {
    "opset1_mobile_cnn": frozenset(_OPSET1_MOBILE_CNN),
    "opset2_detection_vision": frozenset(_OPSET2_DETECTION_VISION),
    "opset3_hybrid_transformer": frozenset(_OPSET3_HYBRID_TRANSFORMER),
    "opset4_decoder_core": frozenset(_OPSET4_DECODER_CORE),
    "opset5_full_benchmark": frozenset(_OPSET5_FULL_BENCHMARK),
}


PRIORITY_MODELS: dict[str, Path] = {
    "mobilevit_xxs": Path("benchmarks/onnx/vision/mobilevit_xxs/onnx/model.onnx"),
    "mobilenetv2": Path("benchmarks/onnx/vision/mobilenetv2/onnx/model.onnx"),
    "yolo26_nano": Path("benchmarks/onnx/vision/yolo26_nano/onnx/model.onnx"),
    "tinyllama_15m": Path("benchmarks/onnx/nlp/tinyllama_15m/onnx/model.onnx"),
    "pythia_70m": Path("benchmarks/onnx/nlp/pythia_70m/onnx/model.onnx"),
    "smollm_135m": Path("benchmarks/onnx/nlp/smollm_135m/onnx/model.onnx"),
}

EXTENDED_BENCHMARK_MODELS: dict[str, Path] = {}

ALL_BENCHMARK_MODELS: dict[str, Path] = {
    **PRIORITY_MODELS,
    **EXTENDED_BENCHMARK_MODELS,
}


BENCHMARK_DOWNLOAD_SPECS: dict[str, dict[str, object]] = {
    "mobilevit_xxs": {
        "repo_id": "apple/mobilevit-xx-small",
        "revision": "refs/pr/3",
        "local_dir": Path("benchmarks/onnx/vision/mobilevit_xxs"),
        "include": ("onnx/*", "config.json", "preprocessor_config.json"),
    },
    "mobilenetv2": {
        "repo_id": "onnxmodelzoo/mobilenetv2-12",
        "local_dir": Path("benchmarks/onnx/vision/mobilenetv2"),
        "include": ("mobilenetv2-12.onnx",),
    },
    "yolo26_nano": {
        "repo_id": "onnx-community/yolo26n-ONNX",
        "local_dir": Path("benchmarks/onnx/vision/yolo26_nano"),
        "include": ("onnx/model.onnx",),
    },
    "tinyllama_15m": {
        "repo_id": "nickypro/tinyllama-15M-fp32",
        "revision": "refs/pr/2",
        "local_dir": Path("benchmarks/onnx/nlp/tinyllama_15m"),
        "include": ("onnx/model.onnx", "onnx/model.onnx_data"),
    },
    "pythia_70m": {
        "repo_id": "Xenova/pythia-70m",
        "local_dir": Path("benchmarks/onnx/nlp/pythia_70m"),
        "include": ("onnx/model.onnx", "onnx/model.onnx_data"),
    },
    "smollm_135m": {
        "repo_id": "onnx-community/SmolLM-135M-ONNX",
        "local_dir": Path("benchmarks/onnx/nlp/smollm_135m"),
        "include": ("onnx/model.onnx", "onnx/model.onnx_data"),
    },
}
