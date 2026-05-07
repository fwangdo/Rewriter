"""Compatibility shim for older imports.

Keep shared contract/model definitions in `src/common` and
`src/onnx_rewrite/specs/catalog.py`.
"""

from __future__ import annotations

from src.common.contracts import LLM_OPS, SUPPORTED_OPS, VISION_OPS
from src.onnx_rewrite.specs.catalog import PRIORITY_MODELS

__all__ = ["LLM_OPS", "SUPPORTED_OPS", "VISION_OPS", "PRIORITY_MODELS"]
