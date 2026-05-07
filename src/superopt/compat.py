"""Bridge to baseline onnx_rewrite passes.

Provides pre-pass and post-pass functions that reuse existing passes
for operations that are impractical as e-graph rules.
"""

from __future__ import annotations

import onnx

from src.onnx_rewrite.passes.constant_folding import ConstantFolding
from src.onnx_rewrite.passes.cleanup import Cleanup
from src.onnx_rewrite.passes.rewrite_decoder_mask import RewriteDecoderMask
from src.onnx_rewrite.passes.rewrite_trilu import RewriteTrilu


def run_pre_passes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run pre-passes on ONNX model before IR conversion.

    Lowers deep-pattern ops (DecoderMask, Trilu) that are impractical
    as e-graph rewrite rules due to pattern depth.
    """
    model, _ = ConstantFolding().run(model)
    # model, _ = RewriteDecoderMask().run(model)
    # model, _ = RewriteTrilu().run(model)
    # model, _ = ConstantFolding().run(model)
    return model


def run_post_passes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run post-passes on ONNX model after extraction.

    Folds remaining constants and cleans up dead nodes.
    """
    model, _ = ConstantFolding().run(model)
    model = onnx.shape_inference.infer_shapes(model)
    _ensure_output_shapes(model.graph)
    model, _ = Cleanup().run(model)
    return model


def _ensure_output_shapes(graph: onnx.GraphProto) -> None:
    """Fill in missing shape fields on graph outputs to satisfy the ONNX checker."""
    for output in graph.output:
        if not output.type.HasField("tensor_type"):
            output.type.tensor_type.elem_type = 1  # FLOAT
        tt = output.type.tensor_type
        if not tt.HasField("shape"):
            tt.shape.SetInParent()
