# Backend Operator Contracts

This document defines the practical backend contract used by the benchmark.
The project is LLM-first, so the LLM contract follows the public ONNX Runtime
QNN Execution Provider supported-op list.  Vision coverage is intentionally
weaker and remains a secondary target.

## QNN-Style LLM Contract

Source: <https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md#supported-onnx-operators>

`LLM_SUPPORTED_OPS` is a QNN-inspired ONNX op pool.  It is not a complete
Qualcomm HTP legality model; it ignores dtype and quantization requirements so
that the rewrite task can focus on graph legality and structural lowering.

### Scope

- Use the public QNN EP supported ONNX operators as the LLM op pool.
- Ignore dtype, quantization, and calibration requirements.
- Treat dynamic shape fixing as a rewrite / preprocessing responsibility.
- Treat unsupported ONNX ops as legalization targets.

### Explicit Non-Dtype Constraints

| Constraint | Modeling choice |
|---|---|
| Dynamic input shapes are unsupported | rewritten models should use fixed shapes |
| `Gather` supports positive indices only | future legality should validate constant/index ranges when available |
| `LpNormalization` supports `p == 2` | future legality should check the attribute |
| `Loop` and `If` are unsupported | control-flow ops are outside the supported contract |

### Important LLM Consequences

The contract supports core decoder blocks directly: `MatMul`, `Gemm`,
`LayerNormalization`, `Softmax`, `Gather`, elementwise math, layout ops, and
comparisons are in the supported pool.

The contract still leaves useful rewrite pressure:

- `Trilu` is not supported and should be lowered or eliminated.
- `Shape`, `Range`, and `ConstantOfShape` are not supported and should be
  folded, specialized, or rewritten.
- Dynamic-shape paths should be resolved before final legality checking.
- Quantization-specific ops are allowed as op names, but quantization
  correctness is not part of the current project scope.

## Modeling Notes

This contract cannot be represented accurately as `frozenset[str]`.
Future legality checks should use op predicates:

```text
is_supported(node, value_info, initializers) -> bool
```

Examples:

- `Gather` depends on whether indices are statically known and non-negative.
- `LpNormalization` depends on the `p` attribute.
- Dynamic shape legality depends on whether symbolic dimensions were fixed.
- Exact QNN HTP legality would also depend on dtype and quantization state.

For superoptimization, this implies that the cost model should distinguish:

- unsupported op
- supported op with illegal attributes
- supported op that falls back outside the intended accelerator path
- fully accelerated op
