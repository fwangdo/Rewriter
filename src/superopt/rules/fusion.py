"""Fusion rewrite rules (multi-pattern).

These rules merge multiple operations into fewer, more efficient ones.
This is where equality saturation shines: fusion and fission rules
can coexist without phase-ordering conflicts.

Reference: Tensat Figure 8, 9, 10, 11.
"""

from __future__ import annotations

from src.common.rules import get_fusion_specs

from .base import RewriteRule
from .wrapper import rulespecs_to_rewrites


def get_fusion_rules() -> list[RewriteRule]:
    """Return fusion rewrite rules.

    Note: multi-pattern rules (multiple matched outputs) require
    special handling in the exploration phase.  The rules below
    are single-pattern approximations.
    """
    return rulespecs_to_rewrites(get_fusion_specs())
