"""Arithmetic rewrite rules.

Identity/zero elimination and simple algebraic simplifications.
These are basic equality rules that any e-graph optimizer should have.
"""

from __future__ import annotations

from src.common.rules import get_arithmetic_specs

from .base import RewriteRule
from .wrapper import rulespecs_to_rewrites


def get_arithmetic_rules() -> list[RewriteRule]:
    """Return the standard arithmetic rewrite rules."""
    return rulespecs_to_rewrites(get_arithmetic_specs())
