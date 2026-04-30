"""Minimal ONNX rewrite scaffold focused on supported-op-only graphs."""

from .analysis import AuditSummary, audit_model, audit_path
from .checker.op_checker import OpChecker
from .core import OptimizationResult, UnsupportedOpError, optimize_model
from .runtime import LatencyResult, ValidationResult, build_inputs_for_model, compare_models, measure_latency
from .specs import (
    ALL_BENCHMARK_MODELS,
    BENCHMARK_ONNX_MIN_OPSET,
    BENCHMARK_DOWNLOAD_SPECS,
    BENCHMARK_ONNX_OPSET,
    EXTENDED_BENCHMARK_MODELS,
    LLM_SUPPORTED_OPS,
    LOGICAL_OPSETS,
    PRIORITY_MODELS,
    SUPPORTED_OPS,
    UNION_SUPPORTED_OPS,
    VISION_SUPPORTED_OPS,
)

__all__ = [
    "ALL_BENCHMARK_MODELS",
    "AuditSummary",
    "BENCHMARK_ONNX_MIN_OPSET",
    "BENCHMARK_DOWNLOAD_SPECS",
    "BENCHMARK_ONNX_OPSET",
    "EXTENDED_BENCHMARK_MODELS",
    "LatencyResult",
    "LLM_SUPPORTED_OPS",
    "LOGICAL_OPSETS",
    "OpChecker",
    "OptimizationResult",
    "PRIORITY_MODELS",
    "SUPPORTED_OPS",
    "UNION_SUPPORTED_OPS",
    "UnsupportedOpError",
    "ValidationResult",
    "VISION_SUPPORTED_OPS",
    "audit_model",
    "audit_path",
    "build_inputs_for_model",
    "compare_models",
    "measure_latency",
    "optimize_model",
]
