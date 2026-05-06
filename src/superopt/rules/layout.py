"""Layout rewrite rules.

Transpose fusion, reshape chain simplification, and other
layout/shape transformation rules.
"""

from __future__ import annotations

from src.common.rules import get_layout_specs

from .base import RewriteRule
from .wrapper import rulespecs_to_rewrites


def get_layout_rules() -> list[RewriteRule]:
    """Return layout-related rewrite rules."""
    return rulespecs_to_rewrites(get_layout_specs())
