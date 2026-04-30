# smollm_135m

## Status

- artifact: `benchmarks/onnx/nlp/smollm_135m/onnx/model.onnx`
- actual ONNX opset: `14`
- strict target contract: `LLM_SUPPORTED_OPS`
- current strict pipeline result: `not yet supported-op-only`
- current correctness status: `not yet verified after strict-oriented cleanup`

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

- `RewriteCompare`
  - `Greater -> Less`
- `RewriteReshapeShape`
  - repeated attention reshape shape-builder를 template initializer로 치환
  - 대표 template:
    - `[0, 0, -1, 64]`
- `RewriteNeg`
  - `Neg -> Mul(-1)`
- `RewritePow`
  - norm 경로의 `Pow(x, 2)`를 `Mul(x, x)`로 치환
- `Cleanup`
  - dead node 제거
  - unused initializer 제거
  - topological sort
  - ONNX checker 통과

## After Rewrite

- total nodes: `2574`
- strict unsupported ops:
  - `ConstantOfShape=1`
  - `Cos=1`
  - `Equal=64`
  - `Expand=67`
  - `Less=1`
  - `Range=5`
  - `ScatterND=1`
  - `Shape=97`
  - `Sin=1`
  - `Trilu=1`
  - `Unsqueeze=223`
  - `Where=64`

## Conclusion

- smollm는 아직 초기 triage 단계다.
- `RoPE (Sin/Cos)`, `Trilu`, `ScatterND`, 그리고 대량의 mask plumbing이 남아 있다.
- 현재 우선순위는 tinyllama/pythia에서 decoder mask rewrite 축을 먼저 안정화한 뒤, 같은 패턴을 smollm로 확장하는 것이다.
