from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class RewriteLayerNorm(Folder):
    """Rewrite LayerNormalization into primitive arithmetic ops."""

    @staticmethod
    def _get_attribute(node: onnx.NodeProto, name: str, default: float | int) -> float | int:
        for attr in node.attribute:
            if attr.name != name:
                continue
            if isinstance(default, float):
                return float(attr.f)
            return int(attr.i)
        return default

    def _add_scalar_initializer(self, name: str, value: float, dtype: np.dtype = np.float32) -> str:
        self.add_init(self.graph, name, np.array(value, dtype=dtype))
        return name

    def _add_axes_initializer(self, name: str, axis: int) -> str:
        self.add_init(self.graph, name, np.array([axis], dtype=np.int64))
        return name

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        prefix = self.get_prefix(node)
        input_name = node.input[0]
        scale_name = node.input[1]
        bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
        output_name = node.output[0]

        axis = int(self._get_attribute(node, "axis", -1))
        epsilon = float(self._get_attribute(node, "epsilon", 1e-5))

        axes_name = self._add_axes_initializer(self.tensor_name(prefix, "axes"), axis)
        epsilon_name = self._add_scalar_initializer(self.tensor_name(prefix, "epsilon"), epsilon)

        mean_name = self.tensor_name(prefix, "mean")
        centered_name = self.tensor_name(prefix, "centered")
        squared_name = self.tensor_name(prefix, "squared")
        var_name = self.tensor_name(prefix, "var")
        var_eps_name = self.tensor_name(prefix, "var_eps")
        std_name = self.tensor_name(prefix, "std")
        normalized_name = self.tensor_name(prefix, "normalized")
        scaled_name = output_name if bias_name is None else self.tensor_name(prefix, "scaled")

        nodes = [
            helper.make_node(
                cons.OP_REDUCE_MEAN,
                [input_name, axes_name],
                [mean_name],
                name=self.node_name(prefix, "mean"),
                keepdims=1,
            ),
            helper.make_node(
                cons.OP_SUB,
                [input_name, mean_name],
                [centered_name],
                name=self.node_name(prefix, "center"),
            ),
            helper.make_node(
                cons.OP_MUL,
                [centered_name, centered_name],
                [squared_name],
                name=self.node_name(prefix, "square"),
            ),
            helper.make_node(
                cons.OP_REDUCE_MEAN,
                [squared_name, axes_name],
                [var_name],
                name=self.node_name(prefix, "var"),
                keepdims=1,
            ),
            helper.make_node(
                cons.OP_ADD,
                [var_name, epsilon_name],
                [var_eps_name],
                name=self.node_name(prefix, "var_eps"),
            ),
            helper.make_node(
                cons.OP_SQRT,
                [var_eps_name],
                [std_name],
                name=self.node_name(prefix, "std"),
            ),
            helper.make_node(
                cons.OP_DIV,
                [centered_name, std_name],
                [normalized_name],
                name=self.node_name(prefix, "normalize"),
            ),
            helper.make_node(
                cons.OP_MUL,
                [normalized_name, scale_name],
                [scaled_name],
                name=self.node_name(prefix, "scale"),
            ),
        ]

        if bias_name is not None:
            nodes.append(
                helper.make_node(
                    cons.OP_ADD,
                    [scaled_name, bias_name],
                    [output_name],
                    name=self.node_name(prefix, "bias"),
                )
            )

        self.replace_node(node, nodes)
        self.log.append(f" - LayerNormalization({prefix}) is rewritten as ReduceMean/Sub/Mul/Add/Sqrt/Div")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == cons.OP_LAYER_NORMALIZATION:
                self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
