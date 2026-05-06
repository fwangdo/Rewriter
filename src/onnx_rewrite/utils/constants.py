from __future__ import annotations

"""Compatibility shim for older imports.

Keep shared contract/model definitions in `src/common` and
`src/onnx_rewrite/specs/catalog.py`.
"""

from src.common.contracts import LLM_OPS, SUPPORTED_OPS, VISION_OPS
from src.onnx_rewrite.specs.catalog import PRIORITY_MODELS
