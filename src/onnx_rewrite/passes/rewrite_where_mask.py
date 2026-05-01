from __future__ import annotations

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper

from ..utils import cons
from .folder import Folder


class RewriteWhereMask(Folder):
    """Rewrite simple Where-based mask builders into arithmetic form."""

    def _get_constant_of_shape_fill(self, node: onnx.NodeProto) -> float | None:
        if node.op_type != cons.OP_CONSTANT_OF_SHAPE:
            return None
        for attr in node.attribute:
            if attr.name == "value":
                value = numpy_helper.to_array(attr.t)
                if value.size != 1:
                    return None
                return float(value.reshape(-1)[0])
        return 0.0

    def _get_scalar_initializer(self, value_name: str) -> float | None:
        value = self.init_map.get(value_name)
        if value is None or value.size != 1:
            return None
        return float(np.asarray(value).reshape(-1)[0])

    def _rewrite_where(self, node: onnx.NodeProto) -> None:
        if len(node.input) != 3:
            return

        cond_name, true_name, false_name = node.input
        zero_value = self._get_scalar_initializer(true_name)
        if zero_value is None or abs(zero_value) > 1e-8:
            return

        const_of_shape = self.get_producer(false_name)
        if const_of_shape is None:
            return
        fill_value = self._get_constant_of_shape_fill(const_of_shape)
        if fill_value is None or fill_value > -1.0e30:
            return

        one_name = self.tensor_name(self.get_prefix(node), "one")
        neginf_name = self.tensor_name(self.get_prefix(node), "neg_inf")
        cond_float_name = self.tensor_name(self.get_prefix(node), "cond_float")
        inverse_name = self.tensor_name(self.get_prefix(node), "inverse")
        self.add_init(self.graph, one_name, np.asarray(1.0, dtype=np.float32))
        self.add_init(self.graph, neginf_name, np.asarray(fill_value, dtype=np.float32))

        replacements = [
            helper.make_node(
                cons.OP_CAST,
                [cond_name],
                [cond_float_name],
                name=self.node_name(self.get_prefix(node), "cond_cast"),
                to=1,  # FLOAT
            ),
            helper.make_node(
                cons.OP_SUB,
                [one_name, cond_float_name],
                [inverse_name],
                name=self.node_name(self.get_prefix(node), "inverse"),
            ),
            helper.make_node(
                cons.OP_MUL,
                [inverse_name, neginf_name],
                [node.output[0]],
                name=self.node_name(self.get_prefix(node), "mask"),
            ),
        ]

        self.replace_node(node, replacements)
        self.log.append(
            f" - WhereMask({self.get_prefix(node)}) is rewritten as Sub/Mul mask"
        )

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == cons.OP_WHERE:
                self._rewrite_where(node)

        self.remove_marked_nodes()
        return model, self.log
