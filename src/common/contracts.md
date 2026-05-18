# Backend Operator Contract

This document uses Qualcomm ONNX Runtime QNN Execution Provider as the
reference backend contract.  The Python constants in `contracts.py` are still
plain op pools; this document records the backend assumptions that should later
be modeled as legality predicates.

## Qualcomm QNN Execution Provider

Source:
<https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators>

QNN EP lowers ONNX graphs through ONNX Runtime into Qualcomm AI Runtime
(QAIRT/QNN).  The main accelerator target is the HTP backend.  The CPU backend
is a reference implementation and should not be treated as the performance
target.

Important modeling points:

- QNN EP can fall back to CPU unless CPU fallback is disabled.
- HTP supports both floating-point and quantized flows, but operator dtype
  support varies by op and backend.
- Dynamic shapes are not supported for HTP execution; symbolic input shapes
  must be fixed before deployment.
- The ONNX op list is a first approximation.  Full legality also depends on
  dtype, shape, rank, attributes, quantization form, and backend selection.

## Supported ONNX Operator Pool

The following pool is derived from the QNN EP supported ONNX operators list.
It is intentionally represented as an op-name pool here; predicate-level
constraints should be added separately.

### Tensor Arithmetic And Logic

| ONNX op |
|---|
| `Abs` |
| `Add` |
| `And` |
| `Asin` |
| `Atan` |
| `Ceil` |
| `Cos` |
| `Div` |
| `Elu` |
| `Equal` |
| `Exp` |
| `Floor` |
| `Greater` |
| `GreaterOrEqual` |
| `Less` |
| `LessOrEqual` |
| `Log` |
| `Max` |
| `Mean` |
| `Min` |
| `Mod` |
| `Mul` |
| `Neg` |
| `Not` |
| `Or` |
| `Pow` |
| `Reciprocal` |
| `Round` |
| `Sign` |
| `Sin` |
| `Sqrt` |
| `Sub` |
| `Sum` |
| `Tanh` |
| `Where` |

### Neural Network Compute

| ONNX op | Notes |
|---|---|
| `AveragePool` |  |
| `BatchNormalization` | fp16 supported since QNN EP 1.18.0 |
| `Clip` | fp16 supported since QNN EP 1.18.0 |
| `Conv` | 3D supported since QNN EP 1.18.0 |
| `ConvTranspose` | 3D supported since QNN EP 1.18.0 |
| `Gelu` |  |
| `Gemm` |  |
| `GlobalAveragePool` |  |
| `GlobalMaxPool` |  |
| `GridSample` |  |
| `HardSigmoid` |  |
| `HardSwish` |  |
| `InstanceNormalization` |  |
| `LRN` |  |
| `LSTM` |  |
| `LayerNormalization` |  |
| `LeakyRelu` |  |
| `LogSoftmax` |  |
| `LpNormalization` | only `p == 2` |
| `MatMul` | HTP typed support is limited in the QNN EP note |
| `MaxPool` |  |
| `PRelu` | fp16/int32 supported since QNN EP 1.18.0 |
| `Relu` |  |
| `Sigmoid` |  |
| `Softmax` |  |
| `ThresholdedRelu` |  |

### Shape, Layout, Indexing

| ONNX op | Notes |
|---|---|
| `ArgMax` |  |
| `ArgMin` |  |
| `Cast` |  |
| `Concat` |  |
| `CumSum` |  |
| `DepthToSpace` |  |
| `Einsum` |  |
| `Expand` |  |
| `Flatten` |  |
| `Gather` | only supports positive indices |
| `GatherElements` |  |
| `GatherND` |  |
| `Inverse` |  |
| `Pad` |  |
| `RandomUniformLike` |  |
| `Resize` |  |
| `STFT` |  |
| `ScatterElements` |  |
| `ScatterND` |  |
| `Slice` |  |
| `SpaceToDepth` |  |
| `Split` |  |
| `Squeeze` |  |
| `Tile` |  |
| `TopK` |  |
| `Transpose` |  |
| `Unsqueeze` |  |
| `Upsample` |  |

### Quantization And Contrib Ops

| ONNX op | Notes |
|---|---|
| `DequantizeLinear` |  |
| `QuantizeLinear` |  |
| `com.microsoft:DequantizeLinear` | 16-bit integer dequantization support |
| `com.microsoft:Gelu` |  |
| `com.microsoft:QuantizeLinear` | 16-bit integer quantization support |
| `com.microsoft.MatMulNBits` | supported bits == 4 on GPU backend |

## Modeling Notes

This contract should not remain a raw `frozenset[str]` long term.  The useful
legality surface is predicate-based:

```text
is_supported(node, value_info, initializers, backend) -> bool
```

Examples:

- `MatMul` is listed, but HTP dtype support is constrained.
- `Gather` is listed, but only positive indices are supported.
- `LpNormalization` is listed only for `p == 2`.
- `BatchNormalization`, `Clip`, `Conv`, `ConvTranspose`, and `PRelu` have
  version-specific dtype/rank notes.
- `Shape` and `Range` are not listed in the QNN EP supported ONNX operator
  pool, so LLM shape-flow rewrites should lower or remove them.
- `Unsqueeze` is listed by QNN EP, but it may still be useful to eliminate it
  when targeting a stricter internal contract.

For this project, the practical contract should distinguish:

- op not listed by QNN EP
- op listed but illegal under dtype/shape/attribute constraints
- op accepted by QNN EP but falling back to CPU
- op accepted and expected to run on HTP
