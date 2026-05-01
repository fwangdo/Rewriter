from __future__ import annotations

from pathlib import Path


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
