"""Get all information about onnx"""

from __future__ import annotations

import onnx 
import argparse

from .constant_folding import ConstantFolding 


class GraphParser:  


    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.graph = model.graph

        self.inputs: dict[str, onnx.ValueInfoProto] = dict()  # runtime inputs only
        self.outputs: dict[str, onnx.ValueInfoProto] = dict()
        self.initializers: dict[str, onnx.TensorProto] = dict()  # weights/constants
        self.nodes: dict[str, onnx.NodeProto] = dict()  # tensor name -> producing node

        self._prepare()
        self._parse()
        return


    def _prepare(self) -> None:
        obj = ConstantFolding()
        obj.prepare(self.model)
        new_model, logs = obj.run(self.model)
        self.model = new_model
        self.graph = new_model.graph
        return 


    def _parse(self) -> None:
        init_names = {i.name for i in self.graph.initializer}

        for vi in self.graph.input:
            if vi.name not in init_names:
                self.inputs[vi.name] = vi

        for vi in self.graph.output:
            self.outputs[vi.name] = vi

        for init in self.graph.initializer:
            self.initializers[init.name] = init

        for node in self.graph.node:
            for output_name in node.output:
                self.nodes[output_name] = node
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    args = parser.parse_args()

    model = onnx.load(args.path)
    obj = GraphParser(model)
    return 