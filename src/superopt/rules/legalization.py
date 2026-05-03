"""Legalization rewrite rules.

These rules lower unsupported ops into supported equivalents.
They mirror the decompositions in onnx_rewrite/passes/ but are
encoded as e-graph equality rules.

Key difference from onnx_rewrite passes: in the e-graph, both
the original and decomposed forms coexist as equivalents.
The extraction phase decides which to pick based on the cost model
and legality constraints.
"""

from __future__ import annotations

from .base import RewriteRule
from ..egraph.enode import EClassId, ENode
from ..egraph.egraph import EGraph
from ..egraph.pattern import PatternNode, PatternVar, Subst


def get_legalization_rules() -> list[RewriteRule]:
    """Return legalization rewrite rules."""
    x = PatternVar("?x")
    a = PatternVar("?a")
    b = PatternVar("?b")
    y = PatternVar("?y")

    rules: list[RewriteRule] = []

    # --- neg_to_mul: Neg(x) → Mul(x, Constant(-1)) ---
    # Neg is not in LLM_SUPPORTED_OPS. Mul is.
    # We use apply_fn to synthesize the -1 constant enode.
    rules.append(RewriteRule(
        name="neg_to_mul",
        source=PatternNode("Neg", (x,)),
        target=PatternNode("Neg", (x,)),  # placeholder, apply_fn overrides
        apply_fn=_apply_neg_to_mul,
    ))

    # --- greater_to_less: Greater(a, b) → Less(b, a) ---
    # Greater is not in LLM_SUPPORTED_OPS. Less isn't either,
    # but this canonicalizes comparison direction.
    rules.append(RewriteRule(
        name="greater_to_less",
        source=PatternNode("Greater", (a, b)),
        target=PatternNode("Less", (b, a)),
    ))

    # --- sub_to_add_neg: Sub(x, y) → Add(x, Neg(y)) ---
    # Sub is in LLM_SUPPORTED_OPS, but this canonical form enables
    # neg_to_mul to chain: Add(x, Neg(y)) → Add(x, Mul(y, -1)).
    # The extractor picks whichever form is legal and cheapest.
    rules.append(RewriteRule(
        name="sub_to_add_neg",
        source=PatternNode("Sub", (x, y)),
        target=PatternNode("Add", (x, PatternNode("Neg", (y,)))),
    ))

    # --- squeeze_to_reshape: Squeeze(x, axes) → Reshape(x, shape) ---
    # Squeeze is not in LLM_SUPPORTED_OPS. Reshape is.
    # Needs apply_fn because target shape depends on source eclass shape.
    rules.append(RewriteRule(
        name="squeeze_to_reshape",
        source=PatternNode("Squeeze", (x, PatternVar("?axes"))),
        target=PatternNode("Squeeze", (x, PatternVar("?axes"))),  # placeholder
        check=_check_has_shape,
        apply_fn=_apply_squeeze_to_reshape,
    ))

    # --- unsqueeze_to_reshape: Unsqueeze(x, axes) → Reshape(x, shape) ---
    # Unsqueeze is not in LLM_SUPPORTED_OPS. Reshape is.
    rules.append(RewriteRule(
        name="unsqueeze_to_reshape",
        source=PatternNode("Unsqueeze", (x, PatternVar("?axes"))),
        target=PatternNode("Unsqueeze", (x, PatternVar("?axes"))),  # placeholder
        check=_check_has_shape,
        apply_fn=_apply_unsqueeze_to_reshape,
    ))

    return rules


# --- apply_fn implementations ---

def _add_scalar_constant(egraph: EGraph, value: float, dtype: int = 1) -> EClassId:
    """Add a scalar constant enode to the egraph, return its eclass id."""
    enode = ENode("weight", (), attrs=(("__name__", f"__const_{value}"),))
    return egraph.add(enode)


def _apply_neg_to_mul(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Neg(x) → Mul(x, -1_constant)."""
    x_cid = subst["?x"]
    neg1_cid = _add_scalar_constant(egraph, -1.0)
    mul_enode = ENode("Mul", (x_cid, neg1_cid))
    return egraph.add(mul_enode)


def _check_has_shape(egraph: EGraph, subst: Subst) -> bool:
    """Check that the matched source eclass has known shape."""
    x_cid = subst["?x"]
    return egraph.eclass(x_cid).data.shape is not None


def _apply_squeeze_to_reshape(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Squeeze(x, axes) → Reshape(x, target_shape).

    The target shape is the source eclass's shape (which is the
    Squeeze output shape, set during ir_to_egraph).
    """
    x_cid = subst["?x"]
    # match_cid is the Squeeze output eclass — its shape IS the target shape.
    target_shape = egraph.eclass(match_cid).data.shape
    if target_shape is None:
        # Fallback: compute from input shape by removing squeezed dims.
        # This shouldn't happen if check passed, but be safe.
        return match_cid

    shape_cid = _add_shape_constant(egraph, target_shape)
    reshape_enode = ENode("Reshape", (x_cid, shape_cid))
    return egraph.add(reshape_enode)


def _apply_unsqueeze_to_reshape(
    egraph: EGraph, match_cid: EClassId, subst: Subst,
) -> EClassId:
    """Unsqueeze(x, axes) → Reshape(x, target_shape).

    Same approach as squeeze: use the matched eclass's shape as target.
    """
    x_cid = subst["?x"]
    target_shape = egraph.eclass(match_cid).data.shape
    if target_shape is None:
        return match_cid

    shape_cid = _add_shape_constant(egraph, target_shape)
    reshape_enode = ENode("Reshape", (x_cid, shape_cid))
    return egraph.add(reshape_enode)


def _add_shape_constant(egraph: EGraph, shape: tuple[int, ...]) -> EClassId:
    """Add a shape constant (weight node) to the egraph."""
    enode = ENode("weight", (), attrs=(("__name__", f"__shape_{shape}"),))
    return egraph.add(enode)
