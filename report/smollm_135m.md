# smollm_135m

## Status

- artifact: `benchmarks/onnx/nlp/smollm_135m/onnx/model.onnx`
- actual ONNX opset: `14`
- baseline target contract: `SUPPORTED_OPS`
- current baseline pipeline result: `supported-op-only`
- current correctness status: `pass`

## Before Rewrite

- total nodes: `2844`
- strict unsupported ops:
  - `ConstantOfShape=1`
  - `Cos=1`
  - `Equal=64`
  - `Expand=67`
  - `Greater=1`
  - `Neg=60`
  - `Pow=61`
  - `Range=5`
  - `ScatterND=1`
  - `Shape=127`
  - `Sin=1`
  - `Trilu=1`
  - `Unsqueeze=283`
  - `Where=64`

## Applied Rewrite

- `RewriteGather`
  - static dimension gathers를 initializer로 folding
- `RewriteCompare`
  - `Greater -> Less`
- `RewriteReshapeShape`
  - repeated attention reshape shape-builder를 template initializer로 치환
  - 대표 template:
    - `[0, 0, -1, 64]`
- `RewriteNeg`
  - `Neg -> Mul(-1)`
- `RewriteRange`
  - simple `Range(0, limit, 1)`을 static arange table `Slice`로 치환
- `RewriteTrilu`
  - triangular mask를 `Shape/Range/Less/Where` 기반 supported-op graph로 치환
- `RewriteScatterND`
  - dense identity-style scatter를 updates passthrough로 치환
- `Cleanup`
  - dead node 제거
  - unused initializer 제거
  - topological sort
  - ONNX checker 통과

## After Rewrite

- total nodes: `2824`
- baseline unsupported ops: `{}`
- correctness: `pass`
- max abs diff: `0.0`

## Conclusion

- smollm는 baseline `SUPPORTED_OPS` 기준 supported-op-only와 correctness를 만족한다.
- strict `LLM_SUPPORTED_OPS` 기준은 별도 후속 목표다.
