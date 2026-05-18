"""ONNX-backed per-op shape inference.

Uses ``onnx.shape_inference.infer_node_outputs`` so that shape rules
stay aligned with the ONNX spec without manual reimplementation.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, defs, helper, numpy_helper, shape_inference


def infer_reduce_mean(
    input_shape: tuple[int, ...],
    input_dtype: int,
    axes: list[int],
    keepdims: bool = True,
    opset_version: int = 18,
) -> tuple[tuple[int, ...], int]:
    """Infer output shape/dtype of ReduceMean via ONNX shape inference.

    Parameters
    ----------
    input_shape : shape of the input tensor.
    input_dtype : ONNX TensorProto.DataType (e.g. 1 for FLOAT).
    axes : reduction axes (may be negative).
    keepdims : whether to keep reduced dimensions as size 1.
    opset_version : ONNX opset version from the source model.

    Returns
    -------
    (output_shape, output_dtype)
    """
    schema = defs.get_schema("ReduceMean", opset_version)
    node_inputs = ["x", "axes"] if opset_version >= 18 else ["x"]
    node_attrs = {"keepdims": 1 if keepdims else 0}
    if opset_version < 18:
        node_attrs["axes"] = axes
    node = helper.make_node(
        "ReduceMean",
        inputs=node_inputs,
        outputs=["y"],
        **node_attrs,
    )
    axes_array = np.array(axes, dtype=np.int64)
    input_types = {
        "x": helper.make_tensor_type_proto(input_dtype, list(input_shape)),
    }
    input_data = {}
    if opset_version >= 18:
        input_types["axes"] = helper.make_tensor_type_proto(
            TensorProto.INT64, list(axes_array.shape)
        )
        input_data["axes"] = numpy_helper.from_array(axes_array, name="axes")
    output_types = shape_inference.infer_node_outputs(
        schema, node, input_types, input_data,
    )
    out_type = output_types["y"]
    out_shape = tuple(
        d.dim_value for d in out_type.tensor_type.shape.dim
    )
    out_dtype = int(out_type.tensor_type.elem_type)
    return out_shape, out_dtype
