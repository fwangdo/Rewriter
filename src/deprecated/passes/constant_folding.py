from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.helper import tensor_dtype_to_np_dtype

from ..utils import cons
from .folder import Folder


class ConstantFolding(Folder):
    """Fold ONNX constant-producing nodes into graph initializers."""

    @staticmethod
    def _get_constant_value(node: onnx.NodeProto) -> np.ndarray | None:
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
            if attr.name == "value_float":
                return np.array(attr.f, dtype=np.float32)
            if attr.name == "value_int":
                return np.array(attr.i, dtype=np.int64)
            if attr.name == "value_floats":
                return np.array(list(attr.floats), dtype=np.float32)
            if attr.name == "value_ints":
                return np.array(list(attr.ints), dtype=np.int64)
        return None

    def _rewrite_constant(self, node: onnx.NodeProto) -> None:
        output_name = node.output[0]
        value = self._get_constant_value(node)
        if value is not None:
            self.add_init(self.graph, output_name, value)
            self.init_map[output_name] = value
            self.mark_for_removal(node)
            self.log.append(f" - Constant({node.name}) is folded into initializer")

    def _rewrite_constant_of_shape(self, node: onnx.NodeProto) -> None:
        # precompute. 
        # before: np -> constant of shape -> something
        # after: np -> something. 
        shape_input = node.input[0]
        if shape_input in self.init_map:
            shape = self.init_map[shape_input].astype(np.int64)

            fill_value = 0.0
            fill_dtype = np.float32
            for attr in node.attribute:
                if attr.name == "value":
                    t = numpy_helper.to_array(attr.t)
                    fill_value = t.item()
                    fill_dtype = t.dtype

            output_array = np.full(shape, fill_value, dtype=fill_dtype)
            output_name = node.output[0]
            self.add_init(self.graph, output_name, output_array)
            self.init_map[output_name] = output_array
            self.mark_for_removal(node)
            self.log.append(f" - ConstantOfShape({node.name}) is folded into initializer")

    @staticmethod
    def _resolve_reshape_shape(data_shape: tuple[int, ...], shape_value: np.ndarray) -> tuple[int, ...]:
        raw_shape = [int(item) for item in np.asarray(shape_value, dtype=np.int64).reshape(-1)]
        resolved: list[int] = []
        infer_index: int | None = None
        known_product = 1

        for index, dim in enumerate(raw_shape):
            if dim == 0:
                resolved_dim = int(data_shape[index])
                resolved.append(resolved_dim)
                known_product *= resolved_dim
            elif dim == -1:
                if infer_index is not None:
                    raise ValueError("multiple -1 dims are not supported")
                infer_index = len(resolved)
                resolved.append(-1)
            else:
                resolved.append(dim)
                known_product *= dim

        if infer_index is not None:
            total = int(np.prod(data_shape, dtype=np.int64))
            resolved[infer_index] = total // known_product

        return tuple(resolved)

    @staticmethod
    def _scatter_nd(data: np.ndarray, indices: np.ndarray, updates: np.ndarray) -> np.ndarray:
        output = np.array(data, copy=True)
        indices = np.asarray(indices, dtype=np.int64)
        updates = np.asarray(updates)

        if indices.ndim == 0:
            raise ValueError("scatter_nd indices rank must be at least 1")

        index_depth = int(indices.shape[-1])
        if index_depth > output.ndim:
            raise ValueError("scatter_nd index depth exceeds data rank")

        outer_shape = tuple(indices.shape[:-1])
        slice_shape = tuple(output.shape[index_depth:])
        expected_updates_shape = outer_shape + slice_shape
        updates = np.reshape(updates, expected_updates_shape)

        coord_iter = np.ndindex(outer_shape) if outer_shape else [()]
        for coord in coord_iter:
            index = tuple(int(item) for item in indices[coord])
            output[index] = updates[coord]

        return output

    def _evaluate_constant_node(self, node: onnx.NodeProto) -> np.ndarray | None:
        inputs = [self.init_map.get(name) for name in node.input]
        if any(value is None for value in inputs):
            return None

        values = [np.asarray(value) for value in inputs if value is not None]

        if node.op_type == cons.OP_SHAPE:
            return np.asarray(values[0].shape, dtype=np.int64)

        if node.op_type == cons.OP_CAST:
            to_type = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_type = int(attr.i)
                    break
            if to_type is None:
                return None
            return values[0].astype(tensor_dtype_to_np_dtype(to_type))

        if node.op_type == cons.OP_UNSQUEEZE:
            data = values[0]
            axes = sorted(int(axis) for axis in np.asarray(values[1], dtype=np.int64).reshape(-1))
            output = data
            for axis in axes:
                output = np.expand_dims(output, axis=axis)
            return output

        if node.op_type == cons.OP_SQUEEZE:
            data = values[0]
            if len(values) > 1:
                axes = tuple(int(axis) for axis in np.asarray(values[1], dtype=np.int64).reshape(-1))
                return np.squeeze(data, axis=axes)
            return np.squeeze(data)

        if node.op_type == cons.OP_CONCAT:
            axis = 0
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = int(attr.i)
                    break
            return np.concatenate(values, axis=axis)

        if node.op_type == cons.OP_TRANSPOSE:
            perm = None
            for attr in node.attribute:
                if attr.name == "perm":
                    perm = list(attr.ints)
                    break
            return np.transpose(values[0], axes=perm)

        if node.op_type == cons.OP_RESHAPE:
            data = values[0]
            output_shape = self._resolve_reshape_shape(data.shape, values[1])
            return np.reshape(data, output_shape)

        if node.op_type == cons.OP_SLICE:
            data = values[0]
            starts = [int(v) for v in np.asarray(values[1], dtype=np.int64).reshape(-1)]
            ends = [int(v) for v in np.asarray(values[2], dtype=np.int64).reshape(-1)]
            axes = list(range(len(starts)))
            if len(values) > 3:
                axes = [int(v) for v in np.asarray(values[3], dtype=np.int64).reshape(-1)]
            steps = [1] * len(starts)
            if len(values) > 4:
                steps = [int(v) for v in np.asarray(values[4], dtype=np.int64).reshape(-1)]

            index = [slice(None)] * data.ndim
            for axis, start, end, step in zip(axes, starts, ends, steps):
                index[axis] = slice(start, end, step)
            return data[tuple(index)]

        if node.op_type == cons.OP_ADD:
            return values[0] + values[1]
        if node.op_type == cons.OP_SUB:
            return values[0] - values[1]
        if node.op_type == cons.OP_MUL:
            return values[0] * values[1]
        if node.op_type == cons.OP_DIV:
            return values[0] / values[1]
        if node.op_type == cons.OP_EQUAL:
            return np.equal(values[0], values[1])
        if node.op_type == cons.OP_LESS:
            return np.less(values[0], values[1])
        if node.op_type == cons.OP_WHERE:
            return np.where(values[0], values[1], values[2])
        if node.op_type == cons.OP_EXPAND:
            shape = tuple(int(item) for item in np.asarray(values[1], dtype=np.int64).reshape(-1))
            return np.broadcast_to(values[0], shape)
        if node.op_type == cons.OP_RANGE:
            start = np.asarray(values[0]).reshape(-1)[0]
            limit = np.asarray(values[1]).reshape(-1)[0]
            delta = np.asarray(values[2]).reshape(-1)[0]
            return np.arange(start, limit, delta, dtype=values[0].dtype)

        if node.op_type == "Trilu":
            data = values[0]
            diagonal = 0
            if len(values) > 1:
                diagonal = int(np.asarray(values[1]).reshape(-1)[0])
            upper = True
            for attr in node.attribute:
                if attr.name == "upper":
                    upper = bool(attr.i)
                    break
            if upper:
                return np.triu(data, k=diagonal)
            return np.tril(data, k=diagonal)

        if node.op_type == "ScatterND":
            return self._scatter_nd(values[0], values[1], values[2])

        return None

    def _rewrite_evaluable_node(self, node: onnx.NodeProto) -> None:
        try:
            value = self._evaluate_constant_node(node)
        except (ValueError, TypeError, RuntimeError, ArithmeticError):
            return
        if value is None:
            return
        output_name = node.output[0]
        self.add_init(self.graph, output_name, np.asarray(value))
        self.init_map[output_name] = np.asarray(value)
        self.mark_for_removal(node)
        self.log.append(f" - {node.op_type}({node.name}) is folded into initializer")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        all_logs: list[str] = []
        previous_deleted = -1
        while self.deleted_node != previous_deleted:
            previous_deleted = self.deleted_node
            self.prepare(model)
            graph = self.require_graph()

            for node in list(graph.node):
                if node.op_type == cons.OP_CONSTANT:
                    self._rewrite_constant(node)
                elif node.op_type == cons.OP_CONSTANT_OF_SHAPE:
                    self._rewrite_constant_of_shape(node)
                elif node.op_type in {
                    cons.OP_ADD,
                    cons.OP_CAST,
                    cons.OP_CONCAT,
                    cons.OP_DIV,
                    cons.OP_EQUAL,
                    cons.OP_EXPAND,
                    cons.OP_LESS,
                    cons.OP_MUL,
                    cons.OP_RANGE,
                    cons.OP_RESHAPE,
                    cons.OP_SLICE,
                    cons.OP_SHAPE,
                    cons.OP_TRANSPOSE,
                    cons.OP_SQUEEZE,
                    cons.OP_SUB,
                    cons.OP_UNSQUEEZE,
                    "ScatterND",
                    "Trilu",
                    cons.OP_WHERE,
                }:
                    self._rewrite_evaluable_node(node)

            self.remove_marked_nodes()
            all_logs.extend(self.log)

        self.log = all_logs
        return model, self.log
