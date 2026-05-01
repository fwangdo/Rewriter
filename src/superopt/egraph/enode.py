"""E-node: a single operation in the e-graph.

An e-node is an operator applied to a list of children e-classes.
Two e-nodes are considered identical (and deduplicated) if they have
the same op, children, and attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Opaque integer ids.
ENodeId = int
EClassId = int


@dataclass(frozen=True, slots=True)
class ENode:
    """An operator node in the e-graph.

    ``children`` are canonical e-class ids of the inputs.
    ``attrs`` is a hashable representation of operator attributes
    (e.g. axis, perm, strides) so that e-nodes with different
    attributes are never merged.
    """

    op: str
    children: tuple[EClassId, ...]
    attrs: tuple[tuple[str, Any], ...] = ()

    def canonicalize(self, find: dict[EClassId, EClassId]) -> ENode:
        """Return a copy with children mapped through union-find."""
        new_children = tuple(find.get(c, c) for c in self.children)
        if new_children == self.children:
            return self
        return ENode(op=self.op, children=new_children, attrs=self.attrs)
