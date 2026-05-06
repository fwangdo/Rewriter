"""Run common RuleSpec rewrites on ONNX graphs."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnx
from onnx import helper

from src.common.rules import GraphBuilder, PatternSpec, RuleSpec, VarCheck

from .folder import Folder


class RuleRunner(Folder):
    """Apply backend-agnostic single-node RuleSpec rules to an ONNX graph."""

    def __init__(self, specs: list[RuleSpec]) -> None:
        super().__init__()
        self.specs = [
            spec
            for spec in specs
            if _is_single_node_source(spec.source)
            and (spec.target is not None or spec.build_fn is not None)
        ]

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)
        for node in list(self.graph.node):
            if node in self.nodes_to_remove:
                continue
            for spec in self.specs:
                subst = self._match_node(node, spec.source)
                if subst is None:
                    continue
                if not self._passes_checks(spec.checks, subst):
                    continue
                changed = self._apply(node, spec, subst)
                if not changed:
                    continue
                self.log.append(f"{spec.name}: rewrote {node.name or node.output[0]}")
                break
        self.remove_marked_nodes()
        return model, self.log

    def _match_node(
        self,
        node: onnx.NodeProto,
        pattern: PatternSpec,
    ) -> dict[str, str] | None:
        if node.op_type != pattern.op or len(node.input) != len(pattern.args):
            return None
        if pattern.attrs is not None:
            node_attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
            for key, expected in pattern.attrs:
                if node_attrs.get(key) != expected:
                    return None
        subst: dict[str, str] = {}
        for arg, input_name in zip(pattern.args, node.input):
            if not isinstance(arg, str):
                return None
            if not arg.startswith("?"):
                return None
            prev = subst.get(arg)
            if prev is not None and prev != input_name:
                return None
            subst[arg] = input_name
        return subst

    def _passes_checks(self, checks: tuple[VarCheck, ...], subst: dict[str, str]) -> bool:
        for check in checks:
            value_name = subst.get(check.var)
            if value_name is None:
                return False
            scalar_value = self._scalar_value(value_name)
            if check.scalar_close is not None:
                if scalar_value is None or abs(scalar_value - check.scalar_close) > 1e-6:
                    return False
            if check.scalar_abs_lt is not None:
                if scalar_value is None or abs(scalar_value) >= check.scalar_abs_lt:
                    return False
            if check.scalar_lte is not None:
                if scalar_value is None or scalar_value > check.scalar_lte:
                    return False
            if check.is_constant is not None and (value_name in self.init_map) != check.is_constant:
                return False
            if check.has_shape is not None and (value_name in self.shape_info) != check.has_shape:
                return False
        return True

    def _scalar_value(self, value_name: str) -> float | None:
        value = self.init_map.get(value_name)
        if value is None or value.size != 1:
            return None
        return float(value.reshape(-1)[0])

    def _apply(
        self,
        node: onnx.NodeProto,
        spec: RuleSpec,
        subst: dict[str, str],
    ) -> bool:
        if spec.build_fn is not None:
            return self._apply_build_fn(node, spec, subst)

        assert spec.target is not None
        if isinstance(spec.target, str):
            replacement = subst[spec.target]
            self._replace_value(node.output[0], replacement)
            self.mark_for_removal(node)
            return True

        new_nodes: list[onnx.NodeProto] = []
        final_value = self._emit_target(
            spec.target,
            subst,
            node,
            new_nodes,
            output_name=node.output[0],
        )
        if final_value != node.output[0]:
            self._replace_value(node.output[0], final_value)
        self.replace_node(node, new_nodes)
        return True

    def _apply_build_fn(
        self,
        node: onnx.NodeProto,
        spec: RuleSpec,
        subst: dict[str, str],
    ) -> bool:
        assert spec.build_fn is not None
        builder = OnnxGraphBuilder(self, node, subst)
        final_value = spec.build_fn(builder, dict(subst))
        if not isinstance(final_value, str):
            raise TypeError(f"{spec.name} returned non-ONNX value handle: {type(final_value).__name__}")
        if not builder.nodes and final_value == node.output[0]:
            return False
        if final_value != node.output[0]:
            self._replace_value(node.output[0], final_value)
        self.replace_node(node, builder.nodes)
        return True

    def _emit_target(
        self,
        target: PatternSpec | str,
        subst: dict[str, str],
        source_node: onnx.NodeProto,
        out_nodes: list[onnx.NodeProto],
        output_name: str | None = None,
    ) -> str:
        if isinstance(target, str):
            return subst[target]

        inputs = [
            self._emit_target(arg, subst, source_node, out_nodes)
            if isinstance(arg, PatternSpec)
            else subst[arg]
            for arg in target.args
        ]
        prefix = self.get_prefix(source_node)
        out_name = output_name or self.tensor_name(prefix, f"{target.op.lower()}_{len(out_nodes)}")
        attrs = dict(target.attrs or ())
        out_nodes.append(
            helper.make_node(
                target.op,
                inputs,
                [out_name],
                name=self.node_name(prefix, f"{target.op.lower()}_{len(out_nodes)}"),
                **_onnx_attrs(attrs),
            )
        )
        return out_name

    def _replace_value(self, old: str, new: str) -> None:
        for consumer in self.get_consumers(old):
            for index, input_name in enumerate(consumer.input):
                if input_name == old:
                    consumer.input[index] = new
        for output in self.graph.output:
            if output.name == old:
                output.name = new


def _onnx_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {key: list(value) if isinstance(value, tuple) else value for key, value in attrs.items()}


def _is_single_node_source(pattern: PatternSpec) -> bool:
    return all(isinstance(arg, str) for arg in pattern.args)


class OnnxGraphBuilder(GraphBuilder):
    """GraphBuilder implementation that emits ONNX nodes and initializers."""

    def __init__(self, runner: RuleRunner, source_node: onnx.NodeProto, subst: dict[str, str]) -> None:
        self.runner = runner
        self.source_node = source_node
        self.subst = subst
        self.nodes: list[onnx.NodeProto] = []
        self._index = 0

    def add_op(
        self,
        op: str,
        inputs: list[Any],
        attrs: dict[str, Any] | None = None,
    ) -> str:
        output_name = self._tensor_name(op.lower())
        self.nodes.append(
            helper.make_node(
                op,
                [_as_name(value) for value in inputs],
                [output_name],
                name=self._node_name(op.lower()),
                **_onnx_attrs(attrs or {}),
            )
        )
        return output_name

    def add_scalar(self, value: float, name: str = "") -> str:
        return self.add_array(np.array(value, dtype=np.float32), name or f"const_{value}")

    def add_array(
        self,
        arr: np.ndarray,
        name: str,
        dtype_code: int = 1,
    ) -> str:
        del dtype_code
        prefix = self.runner.get_prefix(self.source_node)
        tensor_name = self.runner.tensor_name(prefix, self._clean_name(name))
        self.runner.add_init(self.runner.graph, tensor_name, np.ascontiguousarray(arr))
        self.runner.init_map[tensor_name] = np.ascontiguousarray(arr)
        return tensor_name

    def get_weight_data(self, var: str) -> np.ndarray | None:
        value = self.subst[var]
        data = self.runner.init_map.get(value)
        return None if data is None else data.copy()

    def get_shape(self, var: str) -> tuple[int, ...] | None:
        shape = self.runner.shape_info.get(self.subst[var])
        if shape is None or not all(isinstance(dim, int) for dim in shape):
            return None
        return tuple(shape)

    def get_matched_shape(self) -> tuple[int, ...] | None:
        shape = self.runner.shape_info.get(self.source_node.output[0])
        if shape is None or not all(isinstance(dim, int) for dim in shape):
            return None
        return tuple(shape)

    def get_matched_attr(self, key: str) -> Any:
        for attr in self.source_node.attribute:
            if attr.name == key:
                value = helper.get_attribute_value(attr)
                if isinstance(value, list):
                    return tuple(value)
                return value
        return None

    def get_match(self) -> str:
        return self.source_node.output[0]

    def _tensor_name(self, role: str) -> str:
        name = self.runner.tensor_name(self.runner.get_prefix(self.source_node), f"{role}_{self._index}")
        self._index += 1
        return name

    def _node_name(self, role: str) -> str:
        return self.runner.node_name(self.runner.get_prefix(self.source_node), f"{role}_{self._index}")

    @staticmethod
    def _clean_name(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _as_name(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected ONNX tensor name, got {type(value).__name__}")
    return value
