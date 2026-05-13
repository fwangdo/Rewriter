from __future__ import annotations

import onnx

from .folder import Folder


class RewriteScatterND(Folder):
    """Remove dense identity-style ScatterND updates."""

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        if len(node.input) != 3:
            return

        data_shape = self.shape_info.get(node.input[0])
        indices_shape = self.shape_info.get(node.input[1])
        updates_shape = self.shape_info.get(node.input[2])
        output_shape = self.shape_info.get(node.output[0])

        if not data_shape or not indices_shape or not updates_shape or not output_shape:
            return

        if data_shape != output_shape or data_shape != updates_shape:
            return

        if len(indices_shape) != len(data_shape) + 1:
            return

        if indices_shape[-1] != len(data_shape):
            return

        indices_producer = self.get_producer(node.input[1])
        data_producer = self.get_producer(node.input[0])
        updates_producer = self.get_producer(node.input[2])

        if indices_producer is None or indices_producer.op_type != "Concat":
            return
        if data_producer is None or updates_producer is None:
            return

        prefix = self.get_prefix(node)
        for consumer in self.get_consumers(node.output[0]):
            for index, name in enumerate(consumer.input):
                if name == node.output[0]:
                    consumer.input[index] = node.input[2]

        for graph_output in self.graph.output:
            if graph_output.name == node.output[0]:
                graph_output.name = node.input[2]

        self.mark_for_removal(node)
        self.log.append(f" - ScatterND({prefix}) is removed (identity updates passthrough)")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == "ScatterND":
                self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
