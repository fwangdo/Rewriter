# llm_status

## Current Snapshot

- vision 3종은 correctness까지 완료
- LLM 3종은 baseline `SUPPORTED_OPS` 기준 supported-op-only + correctness 완료
- strict `LLM_SUPPORTED_OPS` legality는 baseline 이후 별도 목표다

## Strict LLM Follow-Up

- baseline contract는 `SUPPORTED_OPS` union을 사용한다
- strict LLM contract는 `Shape`, `Unsqueeze`, `ConstantOfShape`, `Less`, `Pow`, `Erf` 같은 잔여 op를 더 줄여야 한다
- strict legality 작업은 correctness가 이미 고정된 baseline 위에서 별도 진행한다

## Latest Baseline LLM Results

- `tinyllama_15m`
  - baseline unsupported: `{}`
  - correctness: `pass`
  - max abs diff: `0.0`
- `pythia_70m`
  - baseline unsupported: `{}`
  - correctness: `pass`
  - max abs diff: `0.0`
- `smollm_135m`
  - baseline unsupported: `{}`
  - correctness: `pass`
  - max abs diff: `0.0`

## Practical Reading

- `tinyllama_15m`, `pythia_70m`, `smollm_135m` 모두 baseline supported-op-only graph를 만든다
- `smollm_135m`의 `Trilu`와 `ScatterND` blocker는 baseline에서 제거됐다
- strict contract 기준으로는 meta op와 activation op를 더 줄이는 작업이 남아 있다

## Practical Contract Note

- 지금 strict LLM contract는 `Shape`까지 금지해서 다소 공격적이다
- 실무형 해석에서는 `Shape`, `Gather(shape)`, `Unsqueeze`, `Squeeze`, `Concat`, `Reshape`를 soft-allowed meta op로 보는 편이 더 자연스럽다
- 반대로 `Where`, `Range`, `Expand`, `ScatterND`, `Trilu`는 계속 must-remove 쪽에 가깝다
