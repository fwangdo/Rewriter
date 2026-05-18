# legality checker.
from __future__ import annotations
from dataclasses import dataclass, field

from src.common.spec.qnn import QNN_SUPPORTED_OPS, QNN_LIMITED_OPS
from src.common.spec.constant import *
from src.common.parser.graph_parser import GraphParser

import numpy as np
import onnx

from enum import Enum


@dataclass
class LegalityResult:
    legal_ops: dict[str, int] = field(default_factory=dict)
    illegal_ops: dict[str, int] = field(default_factory=dict)
    limited_ops: dict[str, int] = field(default_factory=dict)


class LegalitySignal(Enum):
    LEGAL = "Legal"
    ILLEGAL = "Illegal"
    LIMITED = "Limited"


class LegalityChecker:

    def __init__(self, model: onnx.ModelProto, parser: GraphParser) -> None:
        self.model = model
        self.parser = parser
        return

    # --- limited op checks ---

    def _check_gather(self, node: onnx.NodeProto) -> LegalitySignal:
        # should be positive indices.
        indices_name = node.input[1]
        if indices_name in self.parser.initializers:
            arr = onnx.numpy_helper.to_array(self.parser.initializers[indices_name])
            if (arr >= 0).all():
                return LegalitySignal.LEGAL
            else: 
                return LegalitySignal.ILLEGAL
        # dynamic indices — cannot verify statically
        return LegalitySignal.ILLEGAL


    def _check_lp_norm(self, node: onnx.NodeProto) -> LegalitySignal:
        # p should be 2.
        for attr in node.attribute:
            if attr.name == "p":
                if attr.i == 2:
                    return LegalitySignal.LEGAL
                else: 
                    return LegalitySignal.ILLEGAL
        # default p=2 in ONNX spec
        return LegalitySignal.LEGAL


    def _handle_limited_ops(self, node: onnx.NodeProto) -> LegalitySignal:
        op_type = node.op_type
        if op_type == GATHER:
            return self._check_gather(node)
        elif op_type == LP_NORMALIZATION:
            return self._check_lp_norm(node)
        else:
            raise Exception(f'[ERROR]: {op_type} is considered as limited operation.')

    # --- main check ---

    def check(self, node: onnx.NodeProto) -> LegalitySignal:
        op_name = node.op_type
        if op_name in QNN_LIMITED_OPS:
            return self._handle_limited_ops(node)
        elif op_name in QNN_SUPPORTED_OPS:
            return LegalitySignal.LEGAL
        else:
            return LegalitySignal.ILLEGAL

    def _record(self, signal: LegalitySignal, node: onnx.NodeProto, result: LegalityResult) -> None:
        op = node.op_type
        match signal:
            case LegalitySignal.LEGAL:
                result.legal_ops[op] = result.legal_ops.get(op, 0) + 1
            case LegalitySignal.ILLEGAL:
                result.illegal_ops[op] = result.illegal_ops.get(op, 0) + 1
            case LegalitySignal.LIMITED:
                result.limited_ops[op] = result.limited_ops.get(op, 0) + 1


    def run(self) -> tuple[LegalityResult, dict[str, onnx.NodeProto]]:
        result = LegalityResult()
        illegal: dict[str, onnx.NodeProto] = dict()

        for node in self.model.graph.node:
            signal = self.check(node)
            self._record(signal, node, result)
            if signal != LegalitySignal.LEGAL:
                for output_name in node.output:
                    illegal[output_name] = node

        return result, illegal