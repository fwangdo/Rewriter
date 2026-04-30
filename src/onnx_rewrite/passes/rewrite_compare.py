from __future__ import annotations

import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class RewriteCompare(Folder):
    """Rewrite comparison ops into supported equivalents when trivial."""

    def _rewrite_greater(self, node: onnx.NodeProto) -> None:
        prefix = self.get_prefix(node)
        left_name, right_name = node.input[:2]
        output_name = node.output[0]
        replacement = helper.make_node(
            cons.OP_LESS,
            [right_name, left_name],
            [output_name],
            name=self.node_name(prefix, "less"),
        )
        self.replace_node(node, [replacement])
        self.log.append(f" - Greater({prefix}) is rewritten as Less")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == "Greater":
                self._rewrite_greater(node)

        self.remove_marked_nodes()
        return model, self.log
