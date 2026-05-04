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

## 연구 문제: Weight-Aware E-Graph Abstraction

### 문제

현재 e-graph는 두 가지 추상화 레벨을 암묵적으로 혼용하고 있다.

1. **E-class 구분 (값 레벨)**: weight마다 고유한 e-class. `__name__` attr로 구분.
2. **패턴 매칭 (구조 레벨)**: `?w`로 아무 weight나 바인딩. 구조만 보고 규칙 적용.
3. **apply_fn (값 참조)**: 규칙 적용 시 `?w`로 바인딩된 e-class에서 실제 weight 값을 꺼내 새 weight를 계산.

이 경계가 규칙마다 ad-hoc으로 결정되어 있다.

### 코드에서의 발현

`legalization.py`의 규칙들을 보면 세 가지 패턴이 공존한다:

**Type A — 순수 구조 변환** (weight 불투명):
```python
# Clip(x, min, max) → Min(Max(x, min), max)
source=PatternNode("Clip", (x, clip_min, clip_max))
target=PatternNode("Min", (PatternNode("Max", (x, clip_min)), clip_max))
```
`?min`, `?max`가 weight인지 runtime인지 모름. 구조만 변환.

**Type B — 구조 매칭 + 값 조건 검사** (weight 값을 check_fn에서 확인):
```python
# Pow(x, e) → Mul(x, x)  (단, e의 scalar_value ≈ 2.0)
source=PatternNode("Pow", (x, e))
check=_check_pow_exp(2.0)   # ?e의 scalar_value를 읽음
```
`?e`로 아무 것이나 매칭하지만, check_fn이 값을 꺼내본다.

**Type C — 구조 매칭 + 값 기반 새 weight 합성** (weight 값을 읽고 변환):
```python
# BN(x, s, b, m, v) → Mul(x, scale_factor) + Add(bias_factor)
source=PatternNode("BatchNormalization", (x, s, bn_b, bn_m, bn_v))
apply_fn=_apply_bn_decompose  # ?s, ?bn_b, ?bn_m, ?bn_v의 실제 ndarray를 읽어 계산
```
`?s`, `?bn_b` 등은 패턴에서는 "아무 값"이지만, apply_fn은 실제 데이터를 꺼내서
fused weight를 합성한다.

### 스케일 문제

weight마다 고유 e-class → 구조적으로 동일한 transformer layer N개가 전부 별개 e-class 트리로 복제된다. 모델이 커지면:
- leaf e-class 수 = O(initializer 수)
- 매 iteration 매칭 수 = O(layer 수 × 규칙 수)
- max_nodes 한도에 도달하기 전에 의미 있는 최적화 탐색 불가

mobilevit_xxs에서 이미 관찰됨: 417 노드 → 230k matches, 27k IR nodes, 175초.

### 연구 방향

"구조 레벨에서 e-graph를 공유하고, 추출 시 값 바인딩을 복원"할 수 있다면:
- e-graph 크기가 layer 수에 무관하게 유지
- 탐색 공간이 구조적 다양성에만 비례
- 값 의존 규칙(Type C)은 추출 후 별도 패스로 분리 가능

핵심 난제: 추출 시 각 위치에 어떤 weight를 매핑할지 결정하는 문제.

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
