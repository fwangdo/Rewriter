from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class RewriteMetaReshape(Folder):
    """Rewrite some Unsqueeze/Squeeze shape plumbing into Reshape templates."""

    def _specialize_dim(self, dim: object) -> int | None:
        if isinstance(dim, int):
            return dim
        if not isinstance(dim, str):
            return None
        if dim == "batch_size":
            return 1
        if "past_sequence_length" in dim and "+ 1" not in dim:
            return 0
        return None

    def _build_template(self, output_shape: list[object]) -> list[int] | None:
        template: list[int] = []
        unknown_indices: list[int] = []

        for index, dim in enumerate(output_shape):
            specialized = self._specialize_dim(dim)
            if specialized is not None:
                template.append(specialized)
                continue
            unknown_indices.append(index)
            template.append(-1)

        if len(unknown_indices) > 1:
            return None
        return template

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        output_name = node.output[0]
        output_shape = self.shape_info.get(output_name)
        if output_shape is None:
            return

        template = self._build_template(output_shape)
        if template is None:
            return

        prefix = self.get_prefix(node)
        shape_name = self.tensor_name(prefix, "reshape_shape")
        self.add_init(self.graph, shape_name, np.asarray(template, dtype=np.int64))
        replacement = helper.make_node(
            cons.OP_RESHAPE,
            [node.input[0], shape_name],
            [output_name],
            name=self.node_name(prefix, "reshape"),
        )
        self.replace_node(node, [replacement])
        self.log.append(
            f" - {node.op_type}({prefix}) is rewritten as Reshape with template {template}"
        )

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type not in {cons.OP_UNSQUEEZE, cons.OP_SQUEEZE}:
                continue
            self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
