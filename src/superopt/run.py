"""Entry point: ONNX model → superoptimization → optimized ONNX model.

Usage:
    python -m src.superopt.run input.onnx -o output.onnx
    python -m src.superopt.run input.onnx --contract llm
"""

from __future__ import annotations

import argparse
import logging

from ..common.contracts import LLM_SUPPORTED_OPS, VISION_SUPPORTED_OPS
from .pipeline import superoptimize

CONTRACTS: dict[str, frozenset[str]] = {
    "vision": VISION_SUPPORTED_OPS,
    "llm": LLM_SUPPORTED_OPS,
}

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run superoptimization on an ONNX model.",
    )
    parser.add_argument("-i", "--input", help="Path to input ONNX model")
    parser.add_argument("-o", "--output", required=True, help="Path for optimized ONNX model")
    parser.add_argument(
        "--contract",
        choices=list(CONTRACTS.keys()),
        default="vision",
        help="Supported-op contract (default: vision)",
    )
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument(
        "--ilp-solver",
        choices=("scipy", "ortools_scip"),
        default="ortools_scip",
        help="ILP solver backend (default: ortools_scip)",
    )
    parser.add_argument("--ilp-time-limit", type=float, default=600,
                        help="ILP solver time limit in seconds (default: 600)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    result = superoptimize(
        args.input,
        args.output,
        CONTRACTS[args.contract],
        max_iter=args.max_iter,
        max_nodes=args.max_nodes,
        ilp_solver=args.ilp_solver,
        ilp_time_limit_s=args.ilp_time_limit,
    )

    logger.info(
        "done: %d → %d nodes",
        result.original_nodes,
        result.optimized_nodes,
    )


if __name__ == "__main__":
    main()
