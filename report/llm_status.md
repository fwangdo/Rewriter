# llm_status

## Current Snapshot

- vision 3종은 correctness까지 완료
- LLM 3종은 아직 strict `LLM_SUPPORTED_OPS` legality 미완료
- 현재 가장 쉬운 모델은 `tinyllama_15m`

## Why `tinyllama_15m` Is The Easiest

- `pythia_70m`는 아직 exact `GELU`의 `Erf=6` blocker가 남아 있다
- `smollm_135m`는 `Sin/Cos`, `Trilu`, `ScatterND`, 대량의 mask plumbing이 남아 있다
- 반면 `tinyllama_15m`는 activation blocker 없이 causal mask + shape plumbing 쪽이 주된 잔여물이다

## Latest Strict LLM Histograms

- `tinyllama_15m`
  - `ConstantOfShape=4`
  - `Equal=2`
  - `Expand=2`
  - `Less=1`
  - `Range=1`
  - `Shape=42`
  - `Unsqueeze=14`
  - `Where=5`
  - correctness: `pass`
- `pythia_70m`
  - `ConstantOfShape=2`
  - `Equal=2`
  - `Erf=6`
  - `Expand=2`
  - `Less=1`
  - `Range=1`
  - `Shape=22`
  - `Unsqueeze=2`
  - `Where=5`
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

- `tinyllama_15m`는 산술과 상당수 shape-builder는 이미 정리됐다
- 하지만 strict contract를 끝내려면 결국 decoder mask subgraph를 통째로 정리해야 한다
- 현재 contract에서는 이 축이 가장 큰 실제 blocker다
