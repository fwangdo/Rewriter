# Backend Operator Contracts

This document records backend-style operator contracts that are more precise
than a plain supported-op set.  The Python constants in `contracts.py` are still
simple op pools; this file is for future predicate-based legality modeling.

## TI TIDL / TVM 11.2 Contract

Source: <https://software-dl.ti.com/codegen/docs/tvm/tvm_tidl_users_guide/supported-operators.html>

TI TVM partitions an ONNX model between:

- TIDL-supported layers, accelerated by the C7 NPU.
- Non-TIDL layers, handled by TVM-generated C7 code or Arm fallback.

This means the contract is not just "op is supported"; many ops are supported
only under attribute, shape, dtype, or placement constraints.

### Core Compute

| ONNX op | Backend layer | Key legality constraints |
|---|---|---|
| `Conv` | `TIDL_ConvolutionLayer` | horizontal/vertical stride must match; kernel/stride combinations are limited; dilation constraints apply |
| `ConvTranspose` | `TIDL_Deconv2DLayer` | limited kernel set; default dilation only |
| `Gemm`, `MatMul` | `TIDL_InnerProductLayer` | `Gemm`: `transA=0`, `alpha=1.0`, `beta=1.0`; bias must be `[1,N]` or `[N]`; some mixed signed/unsigned `MatMul` cases unsupported |
| `BatchNormalization` | `TIDL_BatchNormLayer` | `training_mode=1` unsupported; scale/bias/mean/variance must be constant 1-D tensors |
| `LayerNormalization` | `TIDL_LayerNormLayer` | width axis only; scale and bias must be `[1,N]` or `[N]` |

### Elementwise And Activations

| ONNX op | Backend layer | Key legality constraints |
|---|---|---|
| `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `Sum` | `TIDL_EltWiseLayer` | exactly two inputs; dimensions must match or be broadcastable |
| `Pow` | `TIDL_PowLayer` | exponent must be a constant scalar tensor |
| `Clip` | `TIDL_ClipLayer` | `min <= 0` and `max > 0` only |
| `Softmax` | `TIDL_SoftMaxLayer` | width and height axes only |
| `Relu`, `PRelu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `HardSigmoid`, `HardSwish`, `Elu`, `Mish` | activation layers | op-specific broadcast/parameter constraints may apply |
| `Abs`, `Neg`, `Sqrt`, `Exp`, `Log`, `Floor` | scalar/math layers | supported as standalone math ops |

### Shape And Layout

| ONNX op | Backend layer | Key legality constraints |
|---|---|---|
| `Reshape` | `TIDL_ReshapeLayer` | variable shape unsupported; input/output volume must match |
| `Flatten` | `TIDL_FlattenLayer` | supported |
| `Squeeze` | `TIDL_SqueezeLayer` | supported |
| `Unsqueeze` | `TIDL_UnsqueezeLayer` | output rank must be <= 6 |
| `Transpose` | `TIDL_TransposeLayer` | for rank > 4, some width-dimension permutations are unsupported |
| `Concat` | `TIDL_ConcatLayer` | axis values `-3`, `-2`, `-1` only; batch axis unsupported |
| `Slice`, `Split` | `TIDL_SliceLayer` | 4-D input only; batch size must be 1; non-unit stride is limited |
| `Pad` | `TIDL_PadLayer` | constant pad mode with zero value only; width/height axes only |
| `Expand` | `TIDL_ExpandLayer` | shape tensor must be constant |

### Indexing, Reduction, Quantization

| ONNX op | Backend layer | Key legality constraints |
|---|---|---|
| `Gather` | `TIDL_GatherLayer` | indices must be 1-D; input rank must be > 1; data cannot be constant; only indices can be constant |
| `ScatterND`, `ScatterElements` | scatter layer | limited reductions/axes; input data must be a zero tensor |
| `ReduceMean`, `ReduceSum` | reduction layers | supported |
| `ReduceMin`, `ReduceMax` | reduction layer | height axis only; `keepdims=1` only |
| `ArgMin`, `ArgMax` | arg layer | `keepdims=1`; axis `-3` only |
| `Cast` | `TIDL_CastLayer` | terminal nodes only |
| `QuantizeLinear`, `DequantizeLinear` | quantization layers | QDQ models only; axis constraints apply |

### Fusion-Only Operators

| ONNX op | Backend layer | Key legality constraints |
|---|---|---|
| `Erf`, `Identity` | `TIDL_IdentityLayer` | accelerated only as part of GELU fusion |
| `DropOut` | `TIDL_DropOutLayer` | not supported standalone |

## Modeling Notes

This contract cannot be represented accurately as `frozenset[str]`.
Future legality checks should use op predicates:

```text
is_supported(node, value_info, initializers) -> bool
```

Examples:

- `MatMul` is not merely supported/unsupported; dtype and input type constraints matter.
- `Cast` is legal only at graph inputs/outputs.
- `Gather` depends on whether `data` or `indices` are constant.
- `Concat`, `Slice`, `Softmax`, and reductions depend on axis and rank.
- `Conv` legality depends on group, dilation, stride, and kernel constraints.

For superoptimization, this implies that the cost model should distinguish:

- unsupported op
- supported op with illegal attributes
- supported op that falls back outside the intended accelerator path
- fully accelerated op
