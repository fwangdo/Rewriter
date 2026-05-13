from __future__ import annotations

import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

from .folder import Folder


class RewriteTrilu(Folder):
    """Rewrite Trilu into Shape/Range/Where-based triangular masking."""

    def _get_scalar_initializer(self, value_name: str) -> int | None:
        value = self.init_map.get(value_name)
        if value is None or value.size != 1:
            return None
        return int(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def _is_upper(node: onnx.NodeProto) -> bool:
        for attr in node.attribute:
            if attr.name == "upper":
                return bool(attr.i)
        return True

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        output_shape = self.shape_info.get(node.output[0])
        if output_shape is None or len(output_shape) != 2:
            self.log.append(f" - Trilu({self.get_prefix(node)}) kept as Trilu")
            return

        prefix = self.get_prefix(node)
        input_name = node.input[0]
        diag_value = 0
        if len(node.input) > 1:
            maybe_diag = self._get_scalar_initializer(node.input[1])
            if maybe_diag is not None:
                diag_value = maybe_diag

        shape_name = self.tensor_name(prefix, "shape")
        row_dim_name = self.tensor_name(prefix, "row_dim")
        col_dim_name = self.tensor_name(prefix, "col_dim")
        zero_name = self.tensor_name(prefix, "zero")
        one_name = self.tensor_name(prefix, "one")
        row_axes_name = self.tensor_name(prefix, "row_axes")
        col_axes_name = self.tensor_name(prefix, "col_axes")
        row_start_name = self.tensor_name(prefix, "row_start")
        row_end_name = self.tensor_name(prefix, "row_end")
        col_start_name = self.tensor_name(prefix, "col_start")
        col_end_name = self.tensor_name(prefix, "col_end")
        row_range_name = self.tensor_name(prefix, "row_range")
        col_range_name = self.tensor_name(prefix, "col_range")
        row_unsqueezed_name = self.tensor_name(prefix, "row_unsqueezed")
        col_unsqueezed_name = self.tensor_name(prefix, "col_unsqueezed")
        row_expanded_name = self.tensor_name(prefix, "row_expanded")
        col_expanded_name = self.tensor_name(prefix, "col_expanded")
        diag_name = self.tensor_name(prefix, "diag")
        row_shifted_name = self.tensor_name(prefix, "row_shifted")
        zero_mask_name = self.tensor_name(prefix, "zero_mask")
        zero_tensor_name = self.tensor_name(prefix, "zero_tensor")

        self.add_init(self.graph, zero_name, np.asarray(0, dtype=np.int64))
        self.add_init(self.graph, one_name, np.asarray(1, dtype=np.int64))
        self.add_init(self.graph, row_start_name, np.asarray(0, dtype=np.int64))
        self.add_init(self.graph, col_start_name, np.asarray(0, dtype=np.int64))
        self.add_init(self.graph, diag_name, np.asarray(diag_value, dtype=np.int64))
        self.add_init(self.graph, row_axes_name, np.asarray([1], dtype=np.int64))
        self.add_init(self.graph, col_axes_name, np.asarray([0], dtype=np.int64))
        self.add_init(self.graph, row_end_name, np.asarray([1], dtype=np.int64))
        self.add_init(self.graph, col_end_name, np.asarray([1], dtype=np.int64))

        replacements = [
            helper.make_node(
                "Shape",
                [input_name],
                [shape_name],
                name=self.node_name(prefix, "shape"),
            ),
            helper.make_node(
                "Gather",
                [shape_name, zero_name],
                [row_dim_name],
                name=self.node_name(prefix, "row_dim"),
                axis=0,
            ),
            helper.make_node(
                "Gather",
                [shape_name, one_name],
                [col_dim_name],
                name=self.node_name(prefix, "col_dim"),
                axis=0,
            ),
            helper.make_node(
                "Range",
                [row_start_name, row_dim_name, one_name],
                [row_range_name],
                name=self.node_name(prefix, "row_range"),
            ),
            helper.make_node(
                "Range",
                [col_start_name, col_dim_name, one_name],
                [col_range_name],
                name=self.node_name(prefix, "col_range"),
            ),
            helper.make_node(
                "Unsqueeze",
                [row_range_name, row_axes_name],
                [row_unsqueezed_name],
                name=self.node_name(prefix, "row_unsqueeze"),
            ),
            helper.make_node(
                "Unsqueeze",
                [col_range_name, col_axes_name],
                [col_unsqueezed_name],
                name=self.node_name(prefix, "col_unsqueeze"),
            ),
            helper.make_node(
                "Expand",
                [row_unsqueezed_name, shape_name],
                [row_expanded_name],
                name=self.node_name(prefix, "row_expand"),
            ),
            helper.make_node(
                "Expand",
                [col_unsqueezed_name, shape_name],
                [col_expanded_name],
                name=self.node_name(prefix, "col_expand"),
            ),
            helper.make_node(
                "Add",
                [row_expanded_name, diag_name],
                [row_shifted_name],
                name=self.node_name(prefix, "row_shift"),
            ),
            helper.make_node(
                "Less",
                (
                    [col_expanded_name, row_shifted_name]
                    if self._is_upper(node)
                    else [row_shifted_name, col_expanded_name]
                ),
                [zero_mask_name],
                name=self.node_name(prefix, "zero_mask"),
            ),
            helper.make_node(
                "ConstantOfShape",
                [shape_name],
                [zero_tensor_name],
                name=self.node_name(prefix, "zero_tensor"),
                value=helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0]),
            ),
            helper.make_node(
                "Where",
                [zero_mask_name, zero_tensor_name, input_name],
                [node.output[0]],
                name=self.node_name(prefix, "mask"),
            ),
        ]

        self.replace_node(node, replacements)
        self.log.append(f" - Trilu({prefix}) is rewritten as Shape/Range/Where mask")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == "Trilu":
                self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
