"""Compatibility bridge to baseline onnx_rewrite passes.

Provides pre-pass and post-pass functions that reuse existing passes
for operations that are impractical as e-graph rules.

Imports pass modules directly to avoid triggering onnx_rewrite.__init__
which has dependencies on 'common.contracts' import path.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import onnx


_compat_initialized = False


def _ensure_compat():
    """Register onnx_rewrite stub packages so relative imports work
    without executing onnx_rewrite/__init__.py (which pulls in
    rewrite_bn.py that requires Python 3.11 syntax).
    """
    global _compat_initialized
    if _compat_initialized:
        return
    _compat_initialized = True

    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    ow_dir = Path(src_dir) / "onnx_rewrite"

    # Register stub packages (empty __init__ modules) for the parent packages
    # so that relative imports inside pass modules resolve correctly.
    for pkg_name, pkg_path in [
        ("onnx_rewrite", ow_dir),
        ("onnx_rewrite.passes", ow_dir / "passes"),
        ("onnx_rewrite.utils", ow_dir / "utils"),
    ]:
        if pkg_name not in sys.modules:
            import types
            stub = types.ModuleType(pkg_name)
            stub.__path__ = [str(pkg_path)]
            stub.__package__ = pkg_name
            sys.modules[pkg_name] = stub


def _import_pass(module_name: str):
    """Import a pass module from onnx_rewrite.passes without triggering __init__."""
    _ensure_compat()

    fqn = f"onnx_rewrite.passes.{module_name}"
    if fqn in sys.modules:
        return sys.modules[fqn]

    src_dir = str(Path(__file__).resolve().parent.parent)
    file_path = Path(src_dir) / "onnx_rewrite" / "passes" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(fqn, file_path,
                                                   submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "onnx_rewrite.passes"
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


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
