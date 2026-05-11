"""Backend-agnostic rewrite rule specifications."""

from .arithmetic import get_arithmetic_specs
from .fusion import get_fusion_specs
from .layout import get_layout_specs
from .legalization import RuleSpec, get_legalization_specs
from .spec import GraphBuilder, PatternSpec, VarCheck


def get_all_specs() -> list[RuleSpec]:
    """Single source of truth for the shared rewrite rule set."""
    return (
        get_legalization_specs()
        + get_arithmetic_specs()
        + get_layout_specs()
        + get_fusion_specs()
    )


__all__ = [
    "GraphBuilder",
    "PatternSpec",
    "RuleSpec",
    "VarCheck",
    "get_all_specs",
    "get_arithmetic_specs",
    "get_fusion_specs",
    "get_layout_specs",
    "get_legalization_specs",
]
