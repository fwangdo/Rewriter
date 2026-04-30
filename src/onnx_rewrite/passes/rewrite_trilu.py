from __future__ import annotations

import numpy as np
import onnx

from .folder import Folder


class RewriteTrilu(Folder):
    """Eliminate Trilu when its input is a zero tensor."""

    def _is_zero_constant_of_shape(self, value_name: str) -> bool:
        producer = self.get_producer(value_name)
        if producer is None or producer.op_type != "ConstantOfShape":
            return False

        for attr in producer.attribute:
            if attr.name != "value":
                continue
            data = onnx.numpy_helper.to_array(attr.t)
            return bool(data.size == 1 and float(data.reshape(-1)[0]) == 0.0)

        return True

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        prefix = self.get_prefix(node)
        input_name = node.input[0]
        output_name = node.output[0]

        if not self._is_zero_constant_of_shape(input_name):
            self.log.append(f" - Trilu({prefix}) kept as Trilu")
            return

        for consumer in self.get_consumers(output_name):
            for index, name in enumerate(consumer.input):
                if name == output_name:
                    consumer.input[index] = input_name

        for graph_output in self.graph.output:
            if graph_output.name == output_name:
                graph_output.name = input_name

        self.mark_for_removal(node)
        self.log.append(f" - Trilu({prefix}) is removed (zero input)")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == "Trilu":
                self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
