"""Convert common RuleSpec objects into superopt RewriteRule objects."""

from __future__ import annotations

from src.common.rules import PatternSpec, RuleSpec, VarCheck

from ..egraph.egraph import EGraph
from ..egraph.pattern import Pattern, PatternNode, PatternVar, Subst
from .base import RewriteRule


def rulespec_to_rewrite(spec: RuleSpec) -> RewriteRule:
    """Convert a backend-agnostic RuleSpec to the legacy RewriteRule shape."""
    if spec.target is None:
        raise ValueError(f"RuleSpec without target is not pattern-convertible: {spec.name}")
    return RewriteRule(
        name=spec.name,
        source=_patternspec_to_pattern(spec.source),
        target=_target_to_pattern(spec.target),
        check=_build_check(spec.checks),
    )


def rulespecs_to_rewrites(specs: list[RuleSpec]) -> list[RewriteRule]:
    return [rulespec_to_rewrite(spec) for spec in specs]


def _target_to_pattern(target: PatternSpec | str) -> Pattern:
    if isinstance(target, str):
        return PatternVar(target)
    return _patternspec_to_pattern(target)


def _patternspec_to_pattern(pattern: PatternSpec | str) -> Pattern:
    if isinstance(pattern, str):
        return PatternVar(pattern)
    children = tuple(_patternspec_to_pattern(arg) for arg in pattern.args)
    return PatternNode(pattern.op, children, attrs=pattern.attrs)


def _build_check(checks: tuple[VarCheck, ...]):
    if not checks:
        return None

    def _check(egraph: EGraph, subst: Subst) -> bool:
        for check in checks:
            cid = subst.get(check.var)
            if cid is None:
                return False
            data = egraph.eclass(cid).data
            if check.scalar_close is not None:
                if data.scalar_value is None:
                    return False
                if abs(data.scalar_value - check.scalar_close) > 1e-6:
                    return False
            if check.is_constant is not None and data.is_constant != check.is_constant:
                return False
            if check.has_shape is not None and (data.shape is not None) != check.has_shape:
                return False
        return True

    return _check
