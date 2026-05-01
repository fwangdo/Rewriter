# Legality-First Superopt 분석

## 현재 결론

현재 superopt의 1차 목표는 "더 빠른 graph를 찾는다"보다
"자동합성 과정에서 legality를 만족하는 graph를 안정적으로 뽑아낸다"에 둔다.

핵심 thesis:

- 순차 rewrite는 legality를 만족하는 조합을 쉽게 놓친다
- equality saturation은 legality를 만족하는 동등 후보를 더 오래 유지할 수 있다
- extraction에서 legality를 hard constraint 또는 큰 penalty로 넣는 것이
  이 프로젝트의 핵심 차별점이다

즉 당분간은 `optimization-second, legality-first`로 간다.

## Tensat에서 가져오는 것

Tensat (MLSys 2021)은 equality saturation을 tensor graph superoptimization에 적용한 연구다.
핵심 구조는 두 단계다.

1. **Exploration**: 입력 그래프로 e-graph를 초기화하고, rewrite rule을 반복 적용해 동등 그래프 공간을 확장한다.
2. **Extraction**: e-graph에서 cost가 최소인 concrete graph를 추출한다.

우리가 Tensat에서 직접 가져오는 핵심은:

- e-graph 자료구조
- match → apply → rebuild 탐색 루프
- cycle filtering
- greedy / ILP extraction 구조

## 우리가 바꾸는 것

### 1. legality가 cost보다 앞선다

Tensat은 latency-first다.
우리는 1차 milestone에서 legality를 더 강하게 본다.

- hard constraint:
  `selected op ∈ supported_ops`
- 또는 very large penalty:
  `cost += legality_penalty`

초기 scaffold는 둘 다 받되, 기본은 legality-first greedy extraction으로 둔다.

### 2. baseline 전체 선적용은 하지 않는다

baseline을 먼저 너무 많이 때리면 superopt가 찾을 수 있는 후보가 줄 수 있다.
그래서 pre-pass는 `light cleanup`으로 제한한다.

- constant folding
- identity 제거
- trivial shape/meta cleanup
- graph noise 제거
- optional must-remove cleanup 일부

반면 aggressive decomposition이나 backend-form canonicalization은
가능한 한 superopt 뒤로 민다.

### 3. cleanup은 superopt의 일부다

cleanup은 pre-pass에만 있는 것이 아니라 superopt 루프 안에도 있다.

- pre-cleanup: 탐색 전에 noise 감소
- iterative cleanup: exploration 중 rebuild/normalization
- post-cleanup: extraction 후 dangling/noop 제거

## 현실적인 난점

### 1. ONNX attribute 복잡도

ONNX의 `Conv`, `Resize`, `Slice`, `Transpose`는 attribute 조합이 복잡하다.
따라서 e-node에는 attribute를 hashable tuple로 넣고,
rule precondition에서 세부 검사를 하도록 설계해야 한다.

### 2. initializer 조작

weight fusion, concat, projection merge 같은 건
rule instantiation 시 실제 initializer를 계산해야 한다.
scaffold에서는 이 경로를 열어 두되, 초기는 leaf/constant reuse 위주로 제한한다.

### 3. data-dependent shape와 dynamic mask

LLM의 `Shape/Gather/Unsqueeze/Where/Expand/ScatterND/Trilu`는
Tensat식 pure optimization과 잘 안 맞는다.
그래서 superopt의 직접 입력은 가능한 한 `light cleanup` 이후 graph로 둔다.

### 4. Python 성능

큰 LLM에서는 Python e-graph가 느릴 수 있다.
초기 검증은 작은 vision graph와 tiny LLM 서브그래프 중심으로 한다.

## 권장 파이프라인

```text
ONNX
  -> pre-cleanup(light)
  -> ONNX -> IR
  -> E-graph init
  -> exploration + iterative cleanup
  -> legality-aware extraction
  -> IR -> ONNX
  -> post-cleanup
```

## 1차 milestone

- `mobilenetv2` 또는 작은 hybrid vision graph에서
  legality-aware extraction이 실제로 동작한다
- baseline보다 더 빠른 graph를 못 찾더라도 괜찮다
- 우선은 "unsupported op를 자동으로 피해 가는 후보 선택"을 보이는 것이 목표다

## 결론

superopt는 계속 가져가되, thesis를 좁혀야 한다.

- full Tensat port가 아니다
- latency-first optimizer도 아니다
- 현재 프로젝트에서의 superopt는
  **light cleanup + legality-aware equality saturation scaffold**가 핵심이다
