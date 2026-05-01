# llm_status

## Current Snapshot

- vision 3종은 correctness까지 완료
- LLM 3종은 아직 strict `LLM_SUPPORTED_OPS` legality 미완료
- practical 1차 목표는 `LLM_MUST_REMOVE_OPS = 0`을 correctness 유지 상태에서 달성하는 것이다
- `tinyllama_15m`, `pythia_70m`는 현재 이 practical goal을 달성했다
- strict legality 기준으로는 여전히 `tinyllama_15m`가 가장 쉽다

## Why `tinyllama_15m` Is The Easiest

- `pythia_70m`는 아직 exact `GELU`의 `Erf=6` blocker가 남아 있다
- `smollm_135m`는 `Sin/Cos`, `Trilu`, `ScatterND`, 대량의 mask plumbing이 남아 있다
- 반면 `tinyllama_15m`는 activation blocker 없이 causal mask + shape plumbing 쪽이 주된 잔여물이다

## Latest Strict LLM Histograms

- `tinyllama_15m`
  - `ConstantOfShape=2`
  - `Less=1`
  - `Shape=39`
  - `Unsqueeze=14`
  - must-remove: `{}`
  - correctness: `pass`
- `pythia_70m`
  - `ConstantOfShape=2`
  - `Erf=6`
  - `Less=1`
  - `Equal=2`
  - `Shape=22`
  - `Unsqueeze=2`
  - must-remove: `{}`
  - correctness: `pass`
- `smollm_135m`
  - `ConstantOfShape=1`
  - `Cos=1`
  - `Equal=63`
  - `Expand=67`
  - `Less=1`
  - `Range=5`
  - `ScatterND=1`
  - `Shape=125`
  - `Sin=1`
  - `Trilu=1`
  - `Unsqueeze=5`
  - `Where=63`

## Practical Reading

- `tinyllama_15m`는 `Range`까지 제거되어 현재 must-remove 관점에서는 비워졌다
- `pythia_70m`도 decoder mask rewrite를 일반화해 `Where/Expand/Range`를 제거했다
- 따라서 practical 기준의 다음 실제 blocker는 `smollm_135m`의 `Sin/Cos`, `Trilu`, `ScatterND`, 대량의 `Where/Expand`다
- strict contract 기준으로는 두 모델 모두 여전히 `Shape`, `Unsqueeze`, `ConstantOfShape`, `Less`, 그리고 `pythia`의 `Erf`가 남아 있다

## Practical Contract Note

- 지금 strict LLM contract는 `Shape`까지 금지해서 다소 공격적이다
- 실무형 해석에서는 `Shape`, `Gather(shape)`, `Unsqueeze`, `Squeeze`, `Concat`, `Reshape`를 soft-allowed meta op로 보는 편이 더 자연스럽다
- 반대로 `Where`, `Range`, `Expand`, `ScatterND`, `Trilu`는 계속 must-remove 쪽에 가깝다
