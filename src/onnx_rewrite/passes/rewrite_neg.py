from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class RewriteNeg(Folder):
    """Rewrite Neg into Mul(x, -1)."""

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        prefix = self.get_prefix(node)
        input_name = node.input[0]
        output_name = node.output[0]
        minus_one_name = self.tensor_name(prefix, "minus_one")
        self.add_init(self.graph, minus_one_name, np.array(-1.0, dtype=np.float32))
        replacement = helper.make_node(
            cons.OP_MUL,
            [input_name, minus_one_name],
            [output_name],
            name=self.node_name(prefix, "mul"),
        )
        self.replace_node(node, [replacement])
        self.log.append(f" - Neg({prefix}) is rewritten as Mul")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == cons.OP_NEG:
                self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
