from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper

from ..utils import cons
from .folder import Folder


@dataclass(frozen=True)
class ReshapeShapePart:
    kind: str
    value: int


class RewriteReshapeShape(Folder):
    """Rewrite common Shape/Gather/Unsqueeze/Concat builders into static Reshape templates."""

    def _get_scalar_initializer(self, value_name: str) -> int | None:
        value = self.init_map.get(value_name)
        if value is None or value.size != 1:
            return None
        return int(np.asarray(value).reshape(-1)[0])

    def _match_shape_part(self, value_name: str) -> ReshapeShapePart | None:
        if value_name in self.init_map:
            scalar = self._get_scalar_initializer(value_name)
            if scalar is None:
                return None
            return ReshapeShapePart("const", scalar)

        unsqueeze = self.get_producer(value_name)
        if unsqueeze is None or unsqueeze.op_type != cons.OP_UNSQUEEZE or len(unsqueeze.input) < 2:
            return None

        axis = self._get_scalar_initializer(unsqueeze.input[1])
        if axis != 0:
            return None

        gather = self.get_producer(unsqueeze.input[0])
        if gather is None or gather.op_type != cons.OP_GATHER or len(gather.input) < 2:
            return None

        index = self._get_scalar_initializer(gather.input[1])
        if index is None:
            return None

        shape = self.get_producer(gather.input[0])
        if shape is None or shape.op_type != cons.OP_SHAPE:
            return None

        return ReshapeShapePart("dim", index)

    def _match_concat_template(self, concat: onnx.NodeProto) -> list[int] | None:
        template: list[int] = []
        for position, input_name in enumerate(concat.input):
            part = self._match_shape_part(input_name)
            if part is None:
                return None
            if part.kind == "const":
                template.append(part.value)
                continue
            if part.value != position:
                return None
            template.append(0)
        return template

    def _rewrite_reshape(self, node: onnx.NodeProto) -> None:
        if len(node.input) < 2:
            return

        concat = self.get_producer(node.input[1])
        if concat is None or concat.op_type != cons.OP_CONCAT:
            return

        template = self._match_concat_template(concat)
        if template is None:
            return

        shape_name = self.tensor_name(self.get_prefix(node), "shape_template")
        self.add_init(self.graph, shape_name, np.asarray(template, dtype=np.int64))
        self.init_map[shape_name] = numpy_helper.to_array(self.graph.initializer[-1])
        node.input[1] = shape_name
        self.log.append(
            f" - Reshape({self.get_prefix(node)}) shape builder is replaced with template {template}"
        )

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in self.graph.node:
            if node.op_type == cons.OP_RESHAPE:
                self._rewrite_reshape(node)

        return model, self.log
