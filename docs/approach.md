# Approach: Loop-Level Decomposition + E-Graph Equality Saturation for Automatic Rewrite Rule Discovery

## 1. Problem

딥러닝 모델은 ONNX 등의 표준 형식으로 표현되며, MatMul, Conv, Gemm, Gather, BatchNormalization 등의 고수준 연산(operator)으로 구성된다. 특정 하드웨어(혹은 백엔드)는 이 중 일부만 지원한다. 예를 들어 Conv는 지원하지만 MatMul은 지원하지 않는 경우, 모델 내의 MatMul을 수학적으로 동일한 Conv 표현으로 변환해야 한다.

핵심 문제: **이러한 변환 규칙(rewrite rule)을 사람이 수동으로 작성하지 않고, 자동으로 발견할 수 있는가?**

## 2. 기존 접근과 한계

### TASO (Tensor Algebra SuperOptimizer)
- MatMul, Conv 등을 **분해하지 않고 primitive operator로 유지**한다.
- 사람이 operator의 대수적 성질(associativity, linearity, bilinearity 등)을 43개 axiom으로 작성한다.
- Z3 SMT solver가 이 axiom 범위 내에서 graph substitution의 정당성을 검증한다.
- **한계**: 발견 가능한 substitution이 사람이 작성한 axiom의 조합 범위로 제한된다. MatMul → Conv 같은 변환은 axiom에 해당 관계가 명시되어 있지 않으면 발견 불가능하다.

### STENSO (Symbolic Program Synthesis)
- 역시 `np.dot`, `np.tensordot`을 **primitive로 유지**한다.
- 프로그램을 symbolic execution으로 수학식(다항식)으로 변환하고, SymPy를 이용해 동치인 다른 프로그램을 sketch synthesis로 탐색한다.
- **한계**: SymPy의 symbolic math engine에 의존하며, 탐색 공간이 sketch 문법으로 제한된다.

### 공통 한계
두 접근 모두 **operator 내부를 분해하지 않는다.** MatMul과 Conv가 내부적으로 같은 연산 구조(indexed sum of products)를 가진다는 사실은, operator를 통째로 두는 한 시스템이 스스로 발견할 수 없다. 사람이 axiom이나 sketch로 알려줘야 한다.

## 3. 본 접근: Loop-Level Decomposition + E-Graph

### 핵심 아이디어

> **고수준 연산을 loop/index level의 primitive로 완전 분해한 뒤, e-graph equality saturation으로 정규화하면, 서로 다른 고수준 연산이 같은 canonical form에 도달하여 equivalence가 자동으로 드러난다.**

사람이 "MatMul은 associative이다" 같은 성질을 선언할 필요 없이, MatMul의 정의 자체를 분해하면 그 성질이 구조적으로 나타난다.

### 3.1. Loop-Level IR 설계

현재 프로젝트의 기존 IR은 NumPy-like 수준(add, mul, reshape, reduce_sum 등)이다. 이 수준에서는 MatMul을 symbolic shape 없이 표현할 수 없다 — `reduce_sum(axis=?)` 에서 어떤 축을 reduce할지가 shape에 의존하기 때문이다.

따라서 IR을 loop/index level로 내린다. 필요한 primitive:

| Primitive | 의미 | 예시 |
|---|---|---|
| `loop(var, range, body)` | variable에 대한 반복 | `loop(k, K, ...)` |
| `load(tensor, [index_expr...])` | 텐서의 특정 위치 읽기 | `load(A, [i, k])` |
| `store(tensor, [index_expr...], value)` | 텐서의 특정 위치 쓰기 | `store(C, [i, j], value)` |
| `affine_expr` | index variable에 대한 affine 연산 | `i * stride + offset`, `oh + kh` |
| `mul`, `add` | 스칼라 산술 | element-level 연산 |
| `reduce_sum(var, range, body)` | variable에 대한 합산 | `Σ_k body(k)` |

이 primitive로 MatMul과 Conv를 표현하면:

```
# MatMul: C[i,j] = Σ_k A[i,k] * B[k,j]
store(C, [i, j],
    reduce_sum(k, K,
        mul(load(A, [i, k]), load(B, [k, j]))))

# Conv2D: O[n,oc,oh,ow] = Σ_ic Σ_kh Σ_kw I[n,ic,oh+kh,ow+kw] * W[oc,ic,kh,kw]
store(O, [n, oc, oh, ow],
    reduce_sum(ic, IC,
        reduce_sum(kh, KH,
            reduce_sum(kw, KW,
                mul(load(I, [n, ic, oh+kh, ow+kw]),
                    load(W, [oc, ic, kh, kw]))))))
```

**두 연산 모두 "affine index를 가진 load들의 곱을 특정 축에 대해 합산"하는 동일한 구조**임이 IR 수준에서 드러난다. 차이는 loop variable의 개수와 index expression의 형태뿐이다.

### 3.2. Lowering: ONNX → Loop-Level IR

ONNX 모델의 각 operator를 loop-level IR로 변환(lowering)한다.

- lowering은 SIR graph의 `add_node` 또는 `lower()` 함수를 통해 수행된다.
- lowering table에 등록된 op(MatMul, Conv, Gemm 등)은 loop-level primitive로 분해된다.
- lowering table에 없는 op(add, mul 등 이미 primitive인 op)은 그대로 통과한다.

```python
def lower(sir_graph, sir_node):
    if sir_node.op in lowering_table:
        return lowering_table[sir_node.op](sir_graph, sir_node)
    # primitive — 그대로 노드 등록
    sir_graph._add_raw_node(sir_node)
```

MatMul의 4가지 케이스(2D×2D, 1D×2D, 2D×1D, batched ND)에 대해 각각 lowering 규칙을 정의한다. 1D 케이스는 unsqueeze → 2D matmul → squeeze로 분해되며, batched 케이스는 batch dimension을 보존하는 loop으로 표현된다.

### 3.3. E-Graph Saturation with Normalization Rules

분해된 loop-level IR을 e-graph에 넣고, **정규화 rewrite rule**로 equality saturation을 수행한다.

rewrite rule은 operator의 "성질"을 선언하는 것이 아니라, **수식을 canonical form으로 만드는 정규화 규칙**이다:

**산술 정규화:**
- `add(a, b) → add(b, a)` (commutativity — canonical ordering)
- `mul(a, b) → mul(b, a)` (commutativity — canonical ordering)
- `add(add(a, b), c) → add(a, add(b, c))` (associativity — right-association)
- `mul(mul(a, b), c) → mul(a, mul(b, c))` (associativity — right-association)

**Reduce 정규화:**
- `reduce_sum(i, reduce_sum(j, body))` → loop 순서 정규화 (canonical ordering of loop variables)
- `reduce_sum(i, add(a, b))` → `add(reduce_sum(i, a), reduce_sum(i, b))` (distributivity)
- `reduce_sum(i, mul(c, body))` → `mul(c, reduce_sum(i, body))` (i가 c에 없을 때, loop-invariant hoisting)

**Index 정규화:**
- `(i + 0) → i` (identity elimination)
- `(i * 1) → i`
- affine expression을 canonical form으로 (`coeff * var + offset`)

**핵심 메커니즘**: 이 정규화 규칙들은 개별 operator의 성질이 아니라 **기본 산술과 합산의 보편적 성질**이다. 규칙의 수가 적고 (TASO의 43개 operator-specific axiom 대비), domain-independent하다. 그런데 이 규칙만으로 서로 다른 고수준 op의 분해 결과가 같은 canonical form에 도달할 수 있다.

### 3.4. Equivalence Discovery

E-graph saturation 후, **같은 e-class에 속하는 서로 다른 expression은 동치**이다.

MatMul과 Conv를 각각 lowering한 결과가, 특정 index 조건(예: kernel size = input size, channel 구조 일치)에서 같은 e-class에 merge되면, "이 조건에서 MatMul ↔ Conv 변환이 가능하다"는 rewrite rule이 자동으로 발견된 것이다.

이 과정은 **SMT solver도, symbolic math engine도 사용하지 않는다.** e-graph의 congruence closure가 syntactic 수준에서 동등성을 판별한다. 정규화 규칙이 충분하면, 의미적으로 같은 두 expression은 같은 canonical form에 도달하고, e-graph이 이를 자동으로 merge한다.

### 3.5. Lifting: Loop-Level IR → ONNX

발견된 equivalence를 실제 사용 가능한 rewrite rule로 만들려면, loop-level IR expression을 다시 고수준 ONNX op으로 복원(lifting)해야 한다.

1. E-graph에서 extraction을 수행하여 최적 expression을 선택한다.
2. extraction 결과는 기존 infrastructure의 IRGraph/IRNode (concrete named graph)로 변환된다.
3. IRGraph → ONNX 변환(`ir_to_onnx`)을 통해 ONNX 모델로 복원한다.
4. **lifting 성공 = 해당 supported operation으로 표현 가능함을 증명**한 것이다.

## 4. 시스템 아키텍처

```
                        ┌─────────────────────────────┐
                        │         ONNX Model           │
                        └──────────────┬──────────────┘
                                       │
                              Lowering (per op)
                                       │
                        ┌──────────────▼──────────────┐
                        │    Loop-Level IR (SIRNode)   │
                        │  load, store, reduce_sum,    │
                        │  loop, affine index, mul/add │
                        └──────────────┬──────────────┘
                                       │
                              ir_to_egraph()
                                       │
                        ┌──────────────▼──────────────┐
                        │          E-Graph             │
                        │   normalization rewrite      │
                        │   rules로 saturation         │
                        │                              │
                        │   → 같은 e-class에 merge된   │
                        │     expression = 동치         │
                        └──────────────┬──────────────┘
                                       │
                              Extraction
                                       │
                        ┌──────────────▼──────────────┐
                        │   IRGraph (concrete graph)   │
                        └──────────────┬──────────────┘
                                       │
                              Lifting (ir_to_onnx)
                                       │
                        ┌──────────────▼──────────────┐
                        │  Rewritten ONNX Model        │
                        │  (supported ops only)         │
                        └─────────────────────────────┘
```

### 모듈 구조

| 모듈 | 위치 | 역할 |
|---|---|---|
| **Loop-Level IR 정의** | `src/common/ir/` | SIRNode, loop/index primitive 정의 |
| **E-Graph 엔진** | `src/common/egraph/` | EGraph, ENode, pattern matching, analysis (공용) |
| **Explore/Extract** | `src/common/explore/`, `extract/` | saturation, extraction (공용) |
| **Normalization Rules** | `src/common/rules/` | 정규화 rewrite rule (공용) |
| **Lowering** | `src/rulegen/lowering/` | ONNX op → loop-level IR 변환 규칙 |
| **Lifting** | `src/rulegen/lifting/` | loop-level IR → ONNX op 복원 |
| **Rulegen Pipeline** | `src/rulegen/pipeline.py` | 전체 rule generation 파이프라인 |
| **Superopt Pipeline** | `src/superopt/pipeline.py` | 발견된 rule을 적용하는 phase ordering |

### Superopt와 Rulegen의 관계

두 시스템은 e-graph 엔진(`common/egraph/`, `explore/`, `extract/`)을 공유하지만 목적이 다르다:

- **Rulegen**: 고수준 op을 loop-level로 분해 → e-graph로 equivalence 발견 → 새 rewrite rule 생성
- **Superopt**: 발견된 rewrite rule + 기존 rule을 적용하여 concrete 모델을 최적화 (phase ordering)

Rulegen은 symbolic IR(shape 무관, 구조 중심)을 사용하고, Superopt는 concrete IR(실제 shape, dtype, initializer 포함)을 사용한다.

## 5. 핵심 Challenge

### E-Graph 폭발
Loop-level로 분해하면 term 수가 급격히 증가한다. MatMul 하나가 수십 개의 loop/load/mul/reduce 노드로 분해되고, 정규화 rule 적용 시 e-graph이 빠르게 커진다.

### 정규화 규칙의 완전성
정규화 규칙이 부족하면, 의미적으로 같은 두 expression이 다른 canonical form에 도달하여 equivalence를 놓칠 수 있다. 규칙 집합의 **confluent** 여부 (어떤 순서로 적용해도 같은 결과에 도달하는지)가 중요하다.

### Symbolic Shape 처리
Lowering 시점에 구체적인 shape을 모를 수 있다. Loop range와 index expression이 symbolic variable로 표현되어야 하며, e-graph에서 symbolic variable 간의 관계(예: "MatMul의 K == Conv의 IC × KH × KW")를 추론할 수 있어야 한다.

### Lifting의 어려움
E-graph에서 발견된 loop-level equivalence를 다시 고수준 ONNX op으로 복원하는 것은 pattern matching 문제이며, 임의의 loop nest가 어떤 ONNX op에 대응되는지 인식하는 것이 쉽지 않다.

## 6. 기존 연구 대비 차별점

| | TASO | STENSO | 본 접근 |
|---|---|---|---|
| **Op 분해 수준** | 분해 안 함 (graph-level) | 분해 안 함 (program-level) | Loop/index level까지 완전 분해 |
| **Equivalence 판별** | Z3 + 수동 axiom | SymPy symbolic execution | E-graph + 정규화 rule (syntactic) |
| **Rule 작성** | 사람이 43개 axiom 작성 | 사람이 sketch 문법 작성 | 정규화 rule만 작성 (domain-independent) |
| **발견 범위** | Axiom 조합 범위 | Sketch 문법 범위 | 분해 구조에서 드러나는 모든 equivalence |
| **비용** | 작은 탐색 공간 | 중간 | 큰 탐색 공간 (e-graph 폭발 위험) |
