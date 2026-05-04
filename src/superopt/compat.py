"""Compatibility bridge to baseline onnx_rewrite passes.

Provides pre-pass and post-pass functions that reuse existing passes
for operations that are impractical as e-graph rules.

Imports pass modules directly to avoid triggering onnx_rewrite.__init__
which has dependencies on 'common.contracts' import path.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import onnx


def _import_pass(module_name: str):
    """Import a pass module from onnx_rewrite.passes without triggering __init__."""
    # Ensure src/ is on sys.path so 'common.contracts' resolves
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return importlib.import_module(f"onnx_rewrite.passes.{module_name}")


def run_pre_passes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run pre-passes on ONNX model before IR conversion.

    Lowers deep-pattern ops (DecoderMask, Trilu) that are impractical
    as e-graph rewrite rules due to pattern depth.
    """
    cf_mod = _import_pass("constant_folding")
    dm_mod = _import_pass("rewrite_decoder_mask")
    tr_mod = _import_pass("rewrite_trilu")

    model, _ = cf_mod.ConstantFolding().run(model)
    model, _ = dm_mod.RewriteDecoderMask().run(model)
    model, _ = tr_mod.RewriteTrilu().run(model)
    model, _ = cf_mod.ConstantFolding().run(model)
    return model


def run_post_passes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run post-passes on ONNX model after extraction.

    Folds remaining constants and cleans up dead nodes.
    """
    cf_mod = _import_pass("constant_folding")
    cl_mod = _import_pass("cleanup")

    model, _ = cf_mod.ConstantFolding().run(model)
    # Shape inference before cleanup so checker has complete type info.
    model = onnx.shape_inference.infer_shapes(model)
    # Ensure all graph outputs have shape fields (checker requires it).
    # Dynamic-shaped outputs (e.g. Concat) may lack shapes after inference.
    _ensure_output_shapes(model.graph)
    model, _ = cl_mod.Cleanup().run(model)
    return model


def _ensure_output_shapes(graph: onnx.GraphProto) -> None:
    """Fill in missing shape fields on graph outputs to satisfy the ONNX checker."""
    for output in graph.output:
        if not output.type.HasField("tensor_type"):
            output.type.tensor_type.elem_type = 1  # FLOAT
        tt = output.type.tensor_type
        if not tt.HasField("shape"):
            # Add an empty shape so the checker doesn't complain.
            tt.shape.SetInParent()
