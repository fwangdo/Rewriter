# ONNX Rewrite TODO

## Current Goal

현재 목표는 `baseline 성능 확보`다.

여기서 baseline은 "약한 데모"가 아니라, **시니어 엔지니어라면 먼저 넣었을 법한 well-known rule-based rewrite를 충분히 갖춘 강한 baseline**을 뜻한다.

즉 지금 해야 할 일은 아래 세 가지를 하나의 패키지로 끝내는 것이다.

- benchmark 6종을 가능한 한 `supported op only` graph로 내리는 rule-based rewrite 구현
- 그 rewrite를 모두 적용한 결과를 `ONNX Runtime`에서 실행해 correctness / latency 검증
- 입력 다양성을 확보한 validation harness로 baseline 수치를 신뢰할 수 있게 만들기

현재 기준으로는 vision 3종 correctness는 이미 확보했고,
남은 핵심은 LLM 3종을 같은 수준까지 끌어올리는 것이다.

LLM 쪽 완료 기준은 반드시 `LLM_SUPPORTED_OPS` 기준 legality를 먼저 만족한 뒤 correctness를 통과하는 것이다.
union scaffold 기준 correctness만 확보된 상태는 완료로 보지 않는다.

## Current Progress

- [x] `mobilenetv2` end-to-end correctness 확보
- [x] `mobilevit_xxs` end-to-end correctness 확보
- [x] `yolo26_nano` end-to-end correctness 확보
- [x] vision용 `Clip` rewrite 추가
- [x] vision용 `LayerNormalization` decomposition 추가
- [x] vision pipeline에서 불필요한 `MatMul` lowering 제거
- [x] `tinyllama_15m` union-contract correctness 확보
- [x] `pythia_70m` union-contract correctness 확보
- [x] LLM `Reshape` shape-builder cleanup 추가
- [ ] `tinyllama_15m` strict `LLM_SUPPORTED_OPS` legality + correctness 확보
- [ ] `pythia_70m` strict `LLM_SUPPORTED_OPS` legality + correctness 확보
- [ ] `smollm_135m` end-to-end correctness 확보

## Success Criteria

아래 조건을 만족하면 1차 목표 달성으로 본다.

- 6개 benchmark 모델에 대해 rewrite pipeline이 실행된다
- 가능한 모델은 최종 graph가 `supported op only`를 만족한다
- rewrite 후 모델이 `ONNX Runtime`에서 실제로 실행된다
- 다양한 입력 샘플에 대해 원본과 rewritten 모델의 output 차이를 측정한다
- 동일 입력군에 대해 원본과 rewritten 모델의 latency를 측정한다
- 모델별로 `지원됨 / 미지원 blocker / correctness / latency` 상태가 report로 정리된다

## Benchmark Scope

현재 final benchmark는 아래 6종으로 고정한다.

- `mobilevit_xxs`
- `mobilenetv2`
- `yolo26_nano`
- `tinyllama_15m`
- `pythia_70m`
- `smollm_135m`

이 목록과 경로는 `specs/catalog.py`를 source-of-truth로 사용한다.

## Baseline Principles

### 1. Rewrite baseline은 강해야 한다

아래 원칙으로 구현한다.

- "supported op만 남기기 위해 필요한 rewrite"는 적극적으로 넣는다
- "이미 널리 알려진 표준적 decomposition / lowering / cleanup"은 빠뜨리지 않는다
- benchmark에서 반복적으로 보이는 패턴은 예외 없이 pass로 흡수한다
- 단, 실험적이거나 불안정한 heuristic보다 재현 가능한 정석 rewrite를 우선한다

### 2. 평가 baseline도 강해야 한다

수치 하나만 찍는 평가로 끝내지 않는다. 최소 검증 축은 아래 세 가지다.

- `다양한 입력`
- `correctness`
- `latency`

### 3. LLM contract는 의도적으로 빡빡해야 한다

LLM target contract는 decoder dense math의 최소 primitive만 남기고,
아래 항목들은 rewrite가 실제로 필요하도록 supported ops에서 제외한다.

- `LayerNorm / RMSNorm`
- `Pow`
- `Sin / Cos`
- `Range / Where / Expand`
- `Unsqueeze / Squeeze / Shape`
- `ConstantOfShape`
- `Equal / Less / Neg`
- `TopK`
- `Trilu / ScatterND`

## Workstreams

## 1. Rule-Based Rewrite Completion

목표: benchmark 6종을 `supported op only`에 최대한 가깝게 내리는 강한 baseline rewrite를 갖춘다.

### 1.1 Core lowering / decomposition

[ ] `BatchNormalization` 제거
[x] `Gemm` rewrite 정리
[ ] `MatMul` rewrite 정리
[ ] `Pow` rewrite 정리
[ ] `Gather` rewrite 정리
[x] `Identity` elimination
[x] `Constant` folding
[x] `ConstantOfShape` folding
[ ] `Shape`-driven trivial cleanup
[x] common `Reshape` shape-builder cleanup

### 1.2 Senior-level baseline에 포함해야 할 well-known rewrite

아래는 "있으면 좋은 것"이 아니라 baseline 강도를 위해 가능한 범위에서 반드시 검토해야 할 항목이다.

[x] `LayerNorm` 계열 분해 패턴 정리
[ ] `RMSNorm` 계열 산술 정리
[ ] `GELU` 패턴 정리
[ ] `SwiGLU / SiLU` 계열 패턴 정리
[ ] `RoPE` canonicalization
[ ] causal mask construction rewrite
[ ] KV-cache update / index-scatter canonicalization
[ ] `Transpose + Reshape + Unsqueeze + Squeeze` cleanup
[ ] 상수 broadcast / reshape chain 단순화
[ ] attention mask 주변 `Cast / Equal / Where / Expand` 정리
[ ] redundant `Concat / Split / Slice` cleanup 가능성 점검
[ ] `Resize` 주변 detection graph cleanup 가능성 점검

중요한 기준은 "논문감 novelty"가 아니라, **benchmark를 supported-op contract 안으로 넣는 데 실질적으로 도움 되는가**다.

### 1.3 Unsupported op triage

[x] 각 benchmark별 unsupported op histogram 다시 수집
[ ] `rewrite로 제거할 op`와 `supported op에 남겨둘 op`를 명시적으로 구분
[ ] 남는 blocker에 대해 `구현 예정 / 범위 밖 / benchmark 예외`로 상태 라벨 부여

## 2. Benchmark-by-Benchmark Execution

목표: rewrite 개발을 abstract하게 하지 말고, 6개 benchmark를 기준으로 밀어붙인다.

### 2.1 Vision

[x] `mobilenetv2`를 `supported op only` baseline으로 먼저 안정화
[x] `mobilevit_xxs`에서 hybrid vision + attention 경로 안정화
[x] `yolo26_nano`에서 detection topology와 `Resize/Concat/Split` 경로 안정화

### 2.2 Decoder LLM

[ ] `tinyllama_15m`에서 `RoPE + RMSNorm + rank-4 attention` 경로를 strict `LLM_SUPPORTED_OPS` 기준으로 안정화
[ ] `pythia_70m`에서 `parallel attention + LayerNorm + GELU` 경로를 strict `LLM_SUPPORTED_OPS` 기준으로 안정화
[ ] `smollm_135m`에서 `GQA` 경로를 strict `LLM_SUPPORTED_OPS` 기준으로 안정화

현재 strict LLM triage:

- `tinyllama_15m`
  - `Shape: 49 -> 37`
  - `Unsqueeze: 111 -> 63`
  - 남은 핵심: `Range / Less / Where / Expand / ConstantOfShape`
- `pythia_70m`
  - `Shape: 27 -> 15`
  - `Unsqueeze: 52 -> 27`
  - 남은 핵심: exact `GELU`의 `Erf=6`, 그리고 `Range / Less / Where / Expand / ConstantOfShape`
- `smollm_135m`
  - `Shape: 127 -> 97`
  - `Unsqueeze: 283 -> 223`
  - 남은 핵심: `Sin / Cos`, `Trilu`, `ScatterND`, 대량의 mask plumbing

### 2.3 모델별 완료 조건

각 모델은 아래 순서로 본다.

[ ] unsupported op audit
[ ] rewrite 적용
[ ] rewritten graph structural sanity check
[ ] ORT 실행 성공
[ ] correctness 측정
[ ] latency 측정
[ ] 결과 report 반영

## 3. Validation Harness

목표: "rewrite가 돌아갔다"가 아니라, "다양한 입력에서도 원본과 비슷하게 동작한다"를 확인한다.

### 3.1 Input diversity

[ ] 모델군별 입력 생성 정책 문서화
[ ] 같은 shape 안에서도 여러 랜덤 seed를 지원
[ ] 입력 분포를 최소 2종 이상 지원
[ ] edge-case 성격의 입력도 일부 포함

### 3.2 Vision input policy

[ ] 기본 image tensor shape 정책 정리
[ ] random normal / uniform / bounded positive 입력 비교
[ ] zero-heavy / near-constant 입력 추가
[ ] detection 모델용 shape 정책 정리

### 3.3 NLP input policy

[ ] token id 생성 범위 정책 정리
[ ] sequence length 다양화
[ ] attention mask 다양화
[ ] decoder prompt length 다양화
[ ] pathological token pattern 반복 입력 추가

### 3.4 Correctness metrics

[ ] `max abs diff` 측정
[ ] `mean abs diff` 측정
[ ] `relative error` 또는 분모 안정화된 비율 지표 검토
[ ] output tensor가 여러 개인 모델의 집계 방식 정의
[ ] 모델별 허용 tolerance 기준 정리

핵심은 "한 번 맞았다"가 아니라, **입력군 전반에서 얼마나 안정적으로 같은 출력을 내는가**다.

## 4. Latency Measurement

목표: rewrite의 품질을 ORT latency로 정량화한다.

### 4.1 Runtime policy

[x] `ONNX Runtime CPU` 기준 고정
[x] warmup / repeat / median / p95 정책 고정
[x] benchmark 실행 시 thread / provider 설정 고정
[x] 원본과 rewritten 모델에 동일 입력 배치 적용

### 4.2 Reporting

[x] 모델별 원본 latency 기록
[x] 모델별 rewritten latency 기록
[x] speedup / slowdown 비율 기록
[x] correctness와 latency를 한 표에서 같이 보이도록 정리

## 5. Reporting and Audit

목표: "무엇이 되었고 무엇이 안 되었는지"를 benchmark 기준으로 바로 보이게 한다.

[x] benchmark별 unsupported op before/after 표 작성
[x] benchmark별 correctness 요약 표 작성
[x] benchmark별 latency 요약 표 작성
[ ] pass별 기여도 또는 대표 적용 사례 기록
[ ] 아직 미지원인 op / 패턴 목록 정리

## Recommended Execution Order

아래 순서로 진행하는 것이 가장 현실적이다.

1. `mobilenetv2`를 완전한 vision baseline으로 끝낸다
2. `mobilevit_xxs`로 transformer 연산이 섞인 vision hybrid를 처리한다
3. `yolo26_nano`로 detection topology를 처리한다
4. `tinyllama_15m`로 decoder baseline을 연다
5. `pythia_70m`로 decoder family 다양성을 확보한다
6. `smollm_135m`로 GQA를 처리한다

이 순서를 따르는 이유는 다음과 같다.

- simplest vision -> complex vision -> detection -> decoder로 난이도가 점진적으로 올라간다
- rewrite bug가 생겼을 때 어느 계층에서 깨졌는지 추적하기 쉽다
- baseline latency report도 점진적으로 쌓을 수 있다

## Immediate Next Tasks

지금 바로 해야 할 일은 아래다.

[x] 6개 benchmark에 대해 unsupported op audit를 최신 기준으로 다시 돌린다
[ ] audit 결과를 기준으로 `rewrite must-have list`를 확정한다
[x] `mobilenetv2`를 첫 final-benchmark baseline-complete 모델로 만든다
[x] ORT correctness / latency harness를 다양한 입력 기준으로 고정한다
[x] 결과를 모델별 표로 남긴다

현재 immediate next는 아래다.

[ ] `tinyllama_15m` strict `LLM_SUPPORTED_OPS` unsupported op 감소
[ ] `pythia_70m` strict `LLM_SUPPORTED_OPS` unsupported op 감소
[ ] `smollm_135m` strict `LLM_SUPPORTED_OPS` blocker 정리
[ ] LLM contract 기준의 unsupported op triage와 rewrite 우선순위 확정
