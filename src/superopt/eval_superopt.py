"""Evaluation CLI for superoptimization.

Runs the superopt pipeline on a model, then measures correctness
and optionally latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import onnx

from ..common.contracts import LLM_SUPPORTED_OPS, VISION_SUPPORTED_OPS
from .pipeline import superoptimize

CONTRACTS: dict[str, frozenset[str]] = {
    "vision": VISION_SUPPORTED_OPS,
    "llm": LLM_SUPPORTED_OPS,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate superoptimization on an ONNX model."
    )
    parser.add_argument("model", help="Path to input ONNX model")
    parser.add_argument(
        "--contract",
        choices=list(CONTRACTS.keys()),
        default="vision",
        help="Supported-op contract (default: vision)",
    )
    parser.add_argument("--output", help="Path for optimized model (default: temp file)")
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument(
        "--correctness",
        dest="correctness",
        action="store_true",
        default=True,
        help="Run correctness check (default: true)",
    )
    parser.add_argument(
        "--no-correctness",
        dest="correctness",
        action="store_false",
        help="Skip correctness check",
    )
    parser.add_argument(
        "--latency", action="store_true", default=False,
        help="Run latency comparison",
    )
    args = parser.parse_args(argv)

    supported_ops = CONTRACTS[args.contract]

    # Determine output path.
    if args.output:
        output_path = args.output
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        output_path = tmp.name
        tmp.close()

    # Run pipeline.
    result = superoptimize(
        args.model,
        output_path,
        supported_ops,
        max_iter=args.max_iter,
        max_nodes=args.max_nodes,
    )

    report: dict[str, object] = {
        "input": result.input_path,
        "output": result.output_path,
        "original_nodes": result.original_nodes,
        "optimized_nodes": result.optimized_nodes,
        "explore": {
            "iterations": result.explore_stats.iterations,
            "matches": result.explore_stats.total_matches,
            "applied": result.explore_stats.total_applied,
            "saturated": result.explore_stats.saturated,
        },
    }

    # Correctness check.
    if args.correctness:
        from ..onnx_rewrite.runtime.validation import compare_models

        val = compare_models(args.model, output_path)
        report["correctness"] = {
            "success": val.success,
            "max_abs_diff": val.max_abs_diff,
            "max_rel_diff": val.max_rel_diff,
            "cases_run": val.cases_run,
        }

    # Latency comparison.
    if args.latency:
        from ..onnx_rewrite.runtime.benchmark import measure_latency
        from ..onnx_rewrite.runtime.validation import build_inputs_for_model

        inputs = build_inputs_for_model(args.model)
        lat_before = measure_latency(args.model, inputs)
        lat_after = measure_latency(output_path, inputs)
        report["latency"] = {
            "before_median_ms": lat_before.median_ms,
            "after_median_ms": lat_after.median_ms,
            "speedup": lat_before.median_ms / lat_after.median_ms if lat_after.median_ms > 0 else 0.0,
        }

    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
