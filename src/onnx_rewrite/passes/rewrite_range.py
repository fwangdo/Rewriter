from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class RewriteRange(Folder):
    """Lower simple Range(0, limit, 1) into Slice over a precomputed table."""

    MAX_RANGE_TABLE = 4096

    def _get_scalar_initializer(self, value_name: str) -> int | None:
        value = self.init_map.get(value_name)
        if value is None or value.size != 1:
            return None
        return int(np.asarray(value).reshape(-1)[0])

    def _rewrite_range(self, node: onnx.NodeProto) -> None:
        if len(node.input) != 3:
            return

        start_value = self._get_scalar_initializer(node.input[0])
        step_value = self._get_scalar_initializer(node.input[2])
        if start_value != 0 or step_value != 1:
            return

        prefix = self.get_prefix(node)
        table_name = self.tensor_name(prefix, "arange_table")
        starts_name = self.tensor_name(prefix, "slice_starts")
        ends_shape_name = self.tensor_name(prefix, "slice_ends_shape")
        ends_name = self.tensor_name(prefix, "slice_ends")
        axes_name = self.tensor_name(prefix, "slice_axes")
        steps_name = self.tensor_name(prefix, "slice_steps")

        self.add_init(
            self.graph,
            table_name,
            np.arange(self.MAX_RANGE_TABLE, dtype=np.int64),
        )
        self.add_init(self.graph, starts_name, np.asarray([0], dtype=np.int64))
        self.add_init(self.graph, ends_shape_name, np.asarray([1], dtype=np.int64))
        self.add_init(self.graph, axes_name, np.asarray([0], dtype=np.int64))
        self.add_init(self.graph, steps_name, np.asarray([1], dtype=np.int64))

        replacements = [
            helper.make_node(
                cons.OP_RESHAPE,
                [node.input[1], ends_shape_name],
                [ends_name],
                name=self.node_name(prefix, "ends_reshape"),
            ),
            helper.make_node(
                cons.OP_SLICE,
                [table_name, starts_name, ends_name, axes_name, steps_name],
                [node.output[0]],
                name=self.node_name(prefix, "slice"),
            )
        ]

        self.replace_node(node, replacements)
        self.log.append(
            f" - Range({prefix}) is rewritten as Slice over a static arange table"
        )

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == cons.OP_RANGE:
                self._rewrite_range(node)

        self.remove_marked_nodes()
        return model, self.log
