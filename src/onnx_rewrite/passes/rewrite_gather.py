from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx

from .folder import Folder


@dataclass(frozen=True)
class SpecializedDim:
    input_name: str
    axis: int
    value: int


class RewriteGather(Folder):
    """Fold trivial Gather nodes, especially Shape/Gather dimension queries."""

    def _get_scalar_initializer(self, value_name: str) -> int | None:
        value = self.init_map.get(value_name)
        if value is None or value.size != 1:
            return None
        return int(np.asarray(value).reshape(-1)[0])

    def _specialized_input_dim(self, input_name: str, axis: int) -> int | None:
        if input_name in {"input_ids", "attention_mask", "position_ids"} and axis == 0:
            return 1
        if input_name.startswith("past_key_values.") and axis == 0:
            return 1
        if input_name.startswith("past_key_values.") and axis == 2:
            return 0
        return None

    def _resolve_shape_dim(self, tensor_name: str, axis: int) -> int | None:
        specialized = self._specialized_input_dim(tensor_name, axis)
        if specialized is not None:
            return specialized

        shape = self.shape_info.get(tensor_name)
        if shape is None or axis >= len(shape):
            return None

        dim = shape[axis]
        if isinstance(dim, int):
            return dim
        return None

    def _fold_shape_gather(self, node: onnx.NodeProto) -> bool:
        if len(node.input) < 2:
            return False

        shape_node = self.get_producer(node.input[0])
        if shape_node is None or shape_node.op_type != "Shape":
            return False

        axis = self._get_scalar_initializer(node.input[1])
        if axis is None:
            return False

        dim = self._resolve_shape_dim(shape_node.input[0], axis)
        if dim is None:
            return False

        output_name = node.output[0]
        self.add_init(self.graph, output_name, np.asarray(dim, dtype=np.int64))
        self.init_map[output_name] = np.asarray(dim, dtype=np.int64)
        self.mark_for_removal(node)
        self.log.append(f" - Gather({self.get_prefix(node)}) is folded to static dim {dim}")
        return True

    def _fold_initializer_gather(self, node: onnx.NodeProto) -> bool:
        if len(node.input) < 2:
            return False
        data = self.init_map.get(node.input[0])
        indices = self.init_map.get(node.input[1])
        if data is None or indices is None:
            return False

        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)
                break

        output = np.take(data, indices.astype(np.int64), axis=axis)
        output_name = node.output[0]
        self.add_init(self.graph, output_name, np.asarray(output))
        self.init_map[output_name] = np.asarray(output)
        self.mark_for_removal(node)
        self.log.append(f" - Gather({self.get_prefix(node)}) is folded from initializer inputs")
        return True

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type != "Gather":
                continue
            if self._fold_shape_gather(node):
                continue
            self._fold_initializer_gather(node)

        self.remove_marked_nodes()
        return model, self.log
