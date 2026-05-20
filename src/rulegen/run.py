"""running rule generation"""

from __future__ import annotations

import argparse
import logging


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run superoptimization on an ONNX model.",
    )
    parser.add_argument("-i", "--input", help="Path to input ONNX model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    # TODO 

    # logger.info(
    #     "done: %d → %d nodes",
    # )


if __name__ == "__main__":
    main()
