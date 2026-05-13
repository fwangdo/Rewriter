from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


class RewriteClip(Folder):
    """Rewrite Clip into supported Max/Min chains when bounds are static."""

    def _get_attr_bound(self, node: onnx.NodeProto, attr_name: str) -> float | None:
        for attr in node.attribute:
            if attr.name == attr_name:
                return float(attr.f)
        return None

    def _get_input_bound(self, input_name: str) -> float | None:
        if not input_name:
            return None
        bound = self.init_map.get(input_name)
        if bound is None or bound.size != 1:
            return None
        return float(bound.reshape(-1)[0])

    def _ensure_bound_initializer(self, prefix: str, role: str, value: float) -> str:
        name = self.tensor_name(prefix, role)
        self.add_init(self.graph, name, np.array(value, dtype=np.float32))
        self.init_map[name] = np.array(value, dtype=np.float32)
        return name

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        prefix = self.get_prefix(node)
        input_name = node.input[0]
        output_name = node.output[0]

        min_value = None
        max_value = None

        if len(node.input) >= 2:
            min_value = self._get_input_bound(node.input[1])
        if len(node.input) >= 3:
            max_value = self._get_input_bound(node.input[2])

        if min_value is None:
            min_value = self._get_attr_bound(node, "min")
        if max_value is None:
            max_value = self._get_attr_bound(node, "max")

        if min_value is None and max_value is None:
            self.log.append(f" - Clip({prefix}) kept as Clip (dynamic bounds)")
            return

        nodes: list[onnx.NodeProto] = []
        current_name = input_name

        if min_value is not None:
            min_name = self._ensure_bound_initializer(prefix, "clip_min", min_value)
            max_out = output_name if max_value is None else self.tensor_name(prefix, "after_min")
            nodes.append(
                helper.make_node(
                    cons.OP_MAX,
                    [current_name, min_name],
                    [max_out],
                    name=self.node_name(prefix, "max"),
                )
            )
            current_name = max_out

        if max_value is not None:
            max_name = self._ensure_bound_initializer(prefix, "clip_max", max_value)
            nodes.append(
                helper.make_node(
                    cons.OP_MIN,
                    [current_name, max_name],
                    [output_name],
                    name=self.node_name(prefix, "min"),
                )
            )

        self.replace_node(node, nodes)
        self.log.append(f" - Clip({prefix}) is rewritten as Max/Min")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == cons.OP_CLIP:
                self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
