# Superopt TODO

## 1. Tensat 핵심 아이디어

### 1.1 문제: Phase-Ordering Problem

기존 graph rewrite 시스템(TASO 포함)은 rewrite rule을 **순차적으로** 적용한다.
어떤 rule을 먼저 적용하느냐에 따라 다른 rule이 적용 가능해지거나 불가능해진다.
이것이 phase-ordering problem이다.

예시: `(a × 2) / 2`에 대해
- `x × 2 → x ≪ 1`을 먼저 적용하면 → `(a ≪ 1) / 2` → 더 이상 단순화 불가
- `x × 2 / 2 → x`를 먼저 적용하면 → `a` → 최적

순차 탐색은 이 선택을 heuristic으로 결정하므로 최적을 놓친다.

### 1.2 해법: Equality Saturation

Equality saturation은 **모든 가능한 rewrite를 동시에 적용**한다.

핵심 데이터 구조는 **e-graph**다.

```
e-graph = { e-class₁, e-class₂, ..., e-classₘ }
e-class = { e-node₁, e-node₂, ..., e-nodeₖ }
e-node = op(child-e-class₁, child-e-class₂, ...)
```

- **e-node**: 하나의 연산. children은 e-class를 가리킨다.
- **e-class**: 의미적으로 동등한 e-node들의 집합.
- **e-graph**: e-class들의 집합. 지수적으로 많은 동등 프로그램을 압축 표현한다.

rewrite rule `l → r`을 적용할 때:
1. e-graph에서 패턴 `l`에 매칭되는 위치를 찾는다
2. `r`에 해당하는 e-node를 **기존 것을 파괴하지 않고** 추가한다
3. `l`이 속한 e-class와 `r`을 merge한다

이 과정을 반복하면 e-graph는 점점 더 많은 동등 프로그램을 표현하게 된다.
saturation(더 이상 새 정보가 없을 때) 또는 timeout에 도달하면 멈춘다.

### 1.3 Two-Phase 구조

Tensat의 최적화는 두 단계로 나뉜다.

**Phase 1: Exploration**
- 입력 tensor graph로 e-graph를 초기화한다
- 매 iteration마다 모든 rewrite rule의 match를 찾고 적용한다
- cycle filtering으로 순환 그래프 생성을 방지한다
- saturation 또는 `N_max`(e-node 수 상한) 또는 `k_max`(iteration 상한)에 도달하면 종료

**Phase 2: Extraction**
- e-graph에서 **최적의 concrete graph**를 꺼낸다
- 각 e-class에서 e-node 하나씩을 골라 전체 cost를 최소화한다
- Greedy extraction: e-class별로 subtree cost가 가장 작은 e-node를 선택. 빠르지만 subgraph 공유를 무시.
- ILP extraction: 전체 e-graph를 정수 선형 계획(ILP)으로 풀어 globally optimal 선택. cycle constraint 제거 + topological order constraint로 확장 가능.

### 1.4 Tensor Graph 표현

Tensat은 tensor graph를 아래처럼 표현한다.

- 각 operator `oᵢ`는 하나의 node `nᵢ`에 대응
- node는 output tensor를 표현
- node의 children은 input tensor를 표현
- 전체 그래프는 DAG
- multi-output graph는 `noop` node로 root를 하나로 합침

### 1.5 Rewrite Rule 표현

Rewrite rule은 S-expression으로 표현한다.

```
source: (matmul ?input_1 ?input_2), (matmul ?input_1 ?input_3)
target: (split_0 (split 1 (matmul ?input_1 (concat_2 1 ?input_2 ?input_3)))),
        (split_1 (split 1 (matmul ?input_1 (concat_2 1 ?input_2 ?input_3))))
```

- `?` prefix는 variable node (어떤 concrete node든 매칭 가능)
- single-pattern rule: 출력이 하나인 rule. egg의 기본 지원.
- multi-pattern rule: 출력이 여러 개인 rule. canonicalization + compatible match 검증이 필요.

### 1.6 Shape Checking

rewrite match가 syntactic하게 맞더라도 shape이 안 맞을 수 있다.
Tensat은 egg의 **e-class analysis** feature를 사용해 각 e-class에 shape/layout/split 정보를 붙이고, rewrite 적용 전에 shape compatibility를 검증한다.

### 1.7 Cycle Filtering

valid rewrite가 e-graph에 cycle을 만들 수 있다.
추출된 그래프는 DAG여야 하므로 cycle을 방지해야 한다.

- **Vanilla**: 매 match마다 전체 e-graph를 순회해 cycle 여부 판별. `O(n_m · N)`.
- **Efficient**: iteration 시작 시 descendant map을 한 번 계산. pre-filtering으로 대부분의 cycle을 빠르게 걸러내고, post-processing으로 남은 cycle을 DFS로 해결. 최대 2000x 빠름.

### 1.8 Cost Model

- operator별 독립 cost (input size와 parameter에 따른 runtime)
- graph 전체 cost = node cost의 합
- GPU 환경 가정 (operator를 하나씩 순차 실행)
- 우리 프로젝트에서는 ORT CPU latency가 cost model 역할

### 1.9 핵심 결과

- TASO 대비 최대 16% 추가 speedup (원본 대비 최대 68.9% speedup)
- 최적화 시간은 TASO 대비 평균 48x 빠름
- ILP extraction이 greedy보다 더 좋은 그래프를 찾음 (BERT, NasNet-A에서 유의미한 차이)
- multi-pattern rule iteration 수 `k_multi`를 늘리면 더 좋은 결과를 얻지만 e-graph가 지수적으로 커짐

---

## 2. 우리 프로젝트에의 적용 설계

### 2.1 핵심 차별점: Legalization-Aware Superoptimization

Tensat은 **고정된 backend**를 가정하고 latency만 최적화한다.
우리는 **operator 제약이 변하는 환경**에서 legalization과 optimization을 **동시에** 수행한다.

구체적으로:
- Tensat의 cost model은 순수 latency: `cost(graph) = Σ runtime(node)`
- 우리의 cost model은 latency + legality penalty:
  `cost(graph) = Σ runtime(node) + λ · Σ penalty(node ∉ supported_ops)`
- 또는 extraction constraint에 legality를 hard constraint로 넣을 수 있다:
  `∀ selected node i: op(i) ∈ supported_ops`

이것이 핵심 contribution이다.

### 2.2 전체 파이프라인

```
ONNX model
  │
  ▼
[ONNX → Internal IR 변환]     ← onnx_to_ir()
  │
  ▼
[E-Graph 초기화]               ← ir_to_egraph()
  │
  ▼
[Exploration Phase]            ← explore(egraph, rules, limits)
  │  - rewrite rule 매칭
  │  - e-node 추가 (non-destructive)
  │  - cycle filtering
  │  - shape checking
  ▼
[Extraction Phase]             ← extract(egraph, cost_model, constraints)
  │  - greedy 또는 ILP
  │  - legality constraint 포함
  ▼
[Internal IR → ONNX 변환]     ← ir_to_onnx()
  │
  ▼
Optimized ONNX model
  │
  ▼
[Evaluation]                   ← eval_superopt()
  - ORT correctness (vs 원본)
  - ORT latency (vs 원본, vs baseline rewrite)
```

### 2.3 디렉토리 구조

```
src/
├── common/
│   ├── contracts.py          # VISION/LLM_SUPPORTED_OPS (기존, 공유)
│   └── __init__.py
├── onnx_rewrite/             # Stage 1: rule-based baseline (기존)
│   └── ...
└── superopt/                 # Stage 2: e-graph superoptimization (신규)
    ├── __init__.py
    ├── todo.md               # 이 문서
    ├── ir/
    │   ├── __init__.py
    │   ├── node.py           # e-graph용 internal IR node 정의
    │   ├── graph.py          # IR graph (DAG) 표현
    │   └── convert.py        # ONNX ↔ IR 변환
    ├── egraph/
    │   ├── __init__.py
    │   ├── eclass.py         # e-class 구현
    │   ├── enode.py          # e-node 구현
    │   ├── egraph.py         # e-graph 구현 (union-find 기반)
    │   ├── pattern.py        # S-expr pattern matching
    │   └── analysis.py       # e-class analysis (shape, type 전파)
    ├── rules/
    │   ├── __init__.py
    │   ├── base.py           # RewriteRule base class
    │   ├── arithmetic.py     # x+0→x, x*1→x, x*0→0, ...
    │   ├── layout.py         # transpose fusion, reshape chain, ...
    │   ├── fusion.py         # conv+bn, matmul+bias, ...
    │   └── legalization.py   # Pow→Mul, LayerNorm decomp, GELU→Tanh, ...
    ├── explore/
    │   ├── __init__.py
    │   ├── explorer.py       # exploration phase 메인 루프
    │   ├── matcher.py        # single/multi-pattern matching
    │   └── cycle.py          # cycle filtering (efficient variant)
    ├── extract/
    │   ├── __init__.py
    │   ├── cost.py           # cost model (ORT latency proxy + legality)
    │   ├── greedy.py         # greedy extraction
    │   └── ilp.py            # ILP extraction (optional, SCIP/OR-Tools)
    ├── pipeline.py           # end-to-end: ONNX → superopt → ONNX
    └── eval_superopt.py      # correctness / latency 측정 CLI
```

### 2.4 모듈별 설계

#### 2.4.1 IR (`ir/`)

ONNX protobuf를 직접 e-graph에 넣으면 너무 복잡하다.
중간 IR을 거쳐야 한다.

```python
@dataclass
class IRNode:
    op: str                    # "Add", "MatMul", "Conv", ...
    inputs: list[str]          # input tensor id 목록
    outputs: list[str]         # output tensor id 목록
    attrs: dict[str, Any]      # operator attributes (axis, perm, ...)
    shape: tuple[int, ...] | None  # output shape (shape inference 결과)

@dataclass
class IRGraph:
    nodes: list[IRNode]
    inputs: list[str]          # graph input tensor ids
    outputs: list[str]         # graph output tensor ids
    initializers: dict[str, np.ndarray]  # constant weights
```

핵심 설계 결정:
- **Tensat 방식**: node = output tensor. children = input tensors. 이 방식이 e-graph에 자연스럽다.
- weight/initializer는 leaf e-node로 표현 (Tensat의 `weight` node)
- graph input은 leaf e-node로 표현 (Tensat의 `input` node)
- multi-output node (e.g. Split)는 `split_0`, `split_1` 같은 projection node를 두어 단일 output으로 취급

ONNX → IR 변환 시:
- `onnx.shape_inference.infer_shapes()` 먼저 실행
- initializer를 별도 dict로 분리
- Constant node는 initializer로 fold
- graph output을 noop node로 합침

#### 2.4.2 E-Graph (`egraph/`)

egg (Rust)를 Python으로 포팅하거나, Python binding을 쓸 수 있다.
초기에는 **순수 Python 구현**으로 시작한다.

핵심 구조:

```python
class EClass:
    id: int
    nodes: set[ENodeId]       # 이 class에 속한 e-node들
    data: AnalysisData | None # shape, type 등 분석 결과

class ENode:
    op: str                    # operator 이름
    children: tuple[EClassId, ...]  # children e-class ids
    attrs: frozenset           # hashable attributes

class EGraph:
    classes: dict[EClassId, EClass]
    nodes: dict[ENodeId, ENode]
    union_find: UnionFind      # e-class merge 관리
    memo: dict[ENode, EClassId]  # dedup: 동일 e-node → 같은 class
```

핵심 연산:
- `add(enode) → EClassId`: e-node를 추가. 이미 존재하면 기존 class 반환.
- `merge(id1, id2)`: 두 e-class를 합침. union-find로 관리.
- `rebuild()`: merge 후 memo와 analysis를 재정합. (egg의 핵심 연산)
- `find(id) → EClassId`: canonical e-class id 반환.

#### 2.4.3 E-Class Analysis (`egraph/analysis.py`)

Tensat은 e-class마다 shape/layout/split 정보를 붙인다.
우리도 동일하게 한다.

```python
@dataclass
class AnalysisData:
    shape: tuple[int, ...] | None   # output tensor shape
    dtype: int | None               # onnx TensorProto.DataType
    is_constant: bool               # 상수 여부
```

analysis는 e-node의 children analysis로부터 bottom-up으로 계산한다.
merge 시 두 analysis를 join한다 (shape이 같아야 함, 다르면 conflict).

#### 2.4.4 Pattern Matching (`egraph/pattern.py`)

Tensat은 S-expression 패턴을 쓴다. 우리도 비슷하게 간다.

```python
@dataclass
class PatternVar:
    name: str                  # "?x", "?y", ...

@dataclass
class PatternNode:
    op: str
    children: list[Pattern]    # PatternVar | PatternNode
    attrs: dict | None         # optional attribute constraints

Pattern = PatternVar | PatternNode
```

matching은 e-graph를 순회하며 pattern을 e-class 단위로 매칭한다.
match 결과는 `dict[str, EClassId]` (variable → e-class binding).

#### 2.4.5 Rewrite Rules (`rules/`)

```python
@dataclass
class RewriteRule:
    name: str
    source: list[Pattern]      # source patterns (multi-pattern: len > 1)
    target: list[Pattern]      # target patterns
    check: Callable | None     # shape/attribute 호환성 검증 (optional)
```

카테고리별 rule:

**arithmetic.py** — 항등원/영원 제거, 산술 단순화
```
x + 0 → x
x * 1 → x
x * 0 → 0
x - x → 0
x / x → 1 (x ≠ 0)
```

**layout.py** — layout/shape 정리
```
transpose(transpose(x, p1), p2) → transpose(x, compose(p1, p2))
reshape(reshape(x, s1), s2) → reshape(x, s2)
```

**fusion.py** — operator fusion (multi-pattern)
```
matmul(x, w1) + matmul(x, w2) → split(matmul(x, concat(w1, w2)))
conv(x, w) + bias → conv_with_bias(x, w, bias)
```

**legalization.py** — unsupported op을 supported op으로 내림
```
Pow(x, 2) → Mul(x, x)
Neg(x) → Mul(x, -1)
LayerNorm(x) → ReduceMean + Sub + Mul + ...
GELU_exact(x) → GELU_tanh(x)
```

legalization rule은 기존 `onnx_rewrite/passes/`의 rule과 의미적으로 동일하지만,
e-graph rule 형태로 인코딩한다는 차이가 있다.

#### 2.4.6 Exploration (`explore/`)

```python
def explore(
    egraph: EGraph,
    rules: list[RewriteRule],
    max_iter: int = 15,
    max_nodes: int = 50000,
    k_multi: int = 1,
) -> EGraph:
    for iteration in range(max_iter):
        if len(egraph.nodes) >= max_nodes:
            break

        descendant_map = get_descendants(egraph)
        matches = search_all(egraph, rules)

        for match in matches:
            if will_create_cycle(match, descendant_map):
                continue
            apply(egraph, match)

        egraph.rebuild()

        if saturated(egraph):
            break

    return egraph
```

cycle filtering은 Tensat의 efficient variant를 구현한다:
1. iteration 시작 시 descendant map 계산
2. pre-filtering: match가 cycle을 만드는지 descendant map으로 빠르게 검증
3. post-processing: DFS로 실제 cycle을 찾아 filter list에 추가

#### 2.4.7 Extraction (`extract/`)

**Greedy extraction**:
```python
def extract_greedy(egraph: EGraph, root: EClassId, cost_fn) -> IRGraph:
    # 각 e-class에서 subtree cost가 최소인 e-node를 bottom-up으로 선택
    # 빠르지만 subgraph 공유를 무시
```

**ILP extraction** (선택적, 후순위):
```python
def extract_ilp(egraph: EGraph, root: EClassId, cost_fn, supported_ops) -> IRGraph:
    # 변수: x_i ∈ {0,1} (e-node i 선택 여부)
    # 목적: minimize Σ c_i · x_i
    # 제약:
    #   (1) root e-class에서 정확히 하나 선택
    #   (2) 선택된 node의 children e-class에서 각각 하나 이상 선택
    #   (3) topological order (cycle 방지)
    #   (4) ★ legality: 선택된 node의 op ∈ supported_ops (hard constraint)
```

**우리만의 확장: Legality constraint**

ILP에 legality를 넣는 방법은 두 가지다.

방법 A: **Hard constraint** — 선택된 모든 node의 op가 supported_ops에 속해야 한다.
```
∀ selected node i: op(i) ∈ supported_ops   →   x_i = 0 if op(i) ∉ supported_ops
```
장점: 추출 결과가 반드시 합법. 단점: feasible solution이 없을 수 있음.

방법 B: **Soft penalty** — unsupported op에 큰 cost penalty.
```
cost(i) = runtime(i) + λ · 𝟙[op(i) ∉ supported_ops]
```
장점: 항상 feasible. 단점: penalty λ 튜닝 필요, 완전 합법 보장 안 됨.

초기에는 **Greedy + hard filter** (supported op 아닌 e-node는 선택 후보에서 제외)로 시작한다.

#### 2.4.8 Cost Model (`extract/cost.py`)

```python
class CostModel:
    def node_cost(self, op: str, shape: tuple, attrs: dict) -> float:
        """operator 하나의 추정 cost를 반환"""

    def legality_penalty(self, op: str, supported_ops: frozenset) -> float:
        """supported op가 아니면 penalty 반환"""
```

초기 cost model은 단순하게:
- 모든 compute op: cost = 1.0 (uniform cost)
- initializer/input/noop: cost = 0.0
- unsupported op penalty: cost = INF (hard constraint 효과)

후기에는 ORT profiling 기반 learned cost model로 교체 가능.

#### 2.4.9 Pipeline (`pipeline.py`)

```python
def superoptimize(
    input_path: str,
    output_path: str,
    supported_ops: frozenset[str],
    max_iter: int = 15,
    max_nodes: int = 50000,
) -> SuperoptResult:
    # 1. ONNX 로드 + shape inference
    onnx_model = onnx.load(input_path)

    # 2. ONNX → IR
    ir_graph = onnx_to_ir(onnx_model)

    # 3. IR → E-Graph 초기화
    egraph, root = ir_to_egraph(ir_graph)

    # 4. Exploration
    rules = get_all_rules()
    egraph = explore(egraph, rules, max_iter, max_nodes)

    # 5. Extraction (with legality)
    cost_model = CostModel(supported_ops)
    opt_ir = extract_greedy(egraph, root, cost_model)

    # 6. IR → ONNX
    opt_model = ir_to_onnx(opt_ir, onnx_model)
    onnx.save(opt_model, output_path)

    return SuperoptResult(...)
```

#### 2.4.10 Evaluation (`eval_superopt.py`)

기존 `onnx_rewrite/runtime/`의 validation/benchmark 코드를 재사용한다.

측정 항목:
- correctness: `max_abs_diff(원본, superopt)`
- latency: `median_ms(원본)` vs `median_ms(superopt)` vs `median_ms(baseline_rewrite)`
- legality: `ops(superopt) ⊆ supported_ops` 여부
- 탐색 비용: exploration 시간, extraction 시간, e-graph 크기 (e-node 수, e-class 수)

---

## 3. 구현 순서

### Phase 0: Scaffold

[ ] `src/superopt/__init__.py` 생성
[ ] 디렉토리 구조 생성 (`ir/`, `egraph/`, `rules/`, `explore/`, `extract/`)
[ ] `src/common/contracts.py`에서 `SUPPORTED_OPS` import 확인

### Phase 1: IR + ONNX 변환

[ ] `IRNode`, `IRGraph` dataclass 정의 (`ir/node.py`, `ir/graph.py`)
[ ] `onnx_to_ir()`: ONNX model → IRGraph 변환 (`ir/convert.py`)
[ ] `ir_to_onnx()`: IRGraph → ONNX model 변환 (`ir/convert.py`)
[ ] 변환 round-trip 테스트: `onnx → ir → onnx` 후 ORT correctness 확인

### Phase 2: E-Graph Core

[ ] `UnionFind` 구현
[ ] `ENode`, `EClass`, `EGraph` 구현 (`egraph/`)
[ ] `add()`, `merge()`, `find()`, `rebuild()` 구현
[ ] `AnalysisData` 및 e-class analysis 구현
[ ] `ir_to_egraph()`: IRGraph → EGraph 초기화
[ ] `egraph_to_ir()`: EGraph에서 (trivial) IRGraph 추출 — extraction 전 sanity check용

### Phase 3: Pattern Matching + Rules

[ ] `Pattern`, `PatternVar`, `PatternNode` 정의 (`egraph/pattern.py`)
[ ] single-pattern search 구현: e-graph에서 패턴 매칭
[ ] match apply 구현: match 결과로 target e-node 추가 + merge
[ ] 기본 arithmetic rule 5개 정의 (`rules/arithmetic.py`)
[ ] 기본 legalization rule 3개 정의 (`rules/legalization.py`): Neg→Mul, Pow→Mul, x+0→x

### Phase 4: Exploration

[ ] exploration 메인 루프 구현 (`explore/explorer.py`)
[ ] vanilla cycle filtering 구현 (`explore/cycle.py`)
[ ] saturation 판정 구현
[ ] 작은 합성 그래프로 exploration 동작 확인

### Phase 5: Extraction

[ ] greedy extraction 구현 (`extract/greedy.py`)
[ ] uniform cost model 구현 (`extract/cost.py`)
[ ] legality hard filter 구현: unsupported op e-node는 선택 후보에서 제외
[ ] extraction 결과를 IRGraph로 변환
[ ] 작은 합성 그래프로 extraction 동작 확인

### Phase 6: End-to-End Pipeline

[ ] `pipeline.py`: ONNX → superopt → ONNX 전체 흐름 연결
[ ] `mobilenetv2`에서 end-to-end 실행
[ ] correctness 확인 (원본 vs superopt)
[ ] latency 측정 (원본 vs baseline vs superopt)

### Phase 7: Rule 확장 + 실험

[ ] legalization rule 확장: LayerNorm, GELU, RoPE 등
[ ] layout rule 추가: transpose fusion, reshape chain
[ ] fusion rule 추가: matmul merge (multi-pattern)
[ ] efficient cycle filtering 구현
[ ] benchmark 6종 전체 실험
[ ] baseline rewrite 대비 latency 비교 report

### Phase 8: ILP Extraction (선택적)

[ ] ILP formulation 구현 (`extract/ilp.py`)
[ ] legality를 ILP hard constraint로 추가
[ ] greedy vs ILP 비교 실험
[ ] ILP solver: `scipy.optimize.milp` 또는 `python-mip` 또는 `OR-Tools`

---

## 4. 핵심 설계 결정 (미확정)

### 4.1 Python vs Rust

Tensat은 Rust (egg 라이브러리)로 구현했다. 우리는:

- **초기**: 순수 Python. 프로토타이핑과 ONNX 생태계 호환이 중요.
- **후기**: 성능 병목이 생기면 egg의 Python binding (`egglog` 또는 직접 binding)을 검토.

판단 기준: benchmark 6종 중 가장 큰 `smollm_135m` (2844 nodes)에서
exploration이 합리적 시간 (< 5분) 내에 끝나는지 여부.

### 4.2 기존 onnx_rewrite 코드 재사용 범위

- `src/common/contracts.py`: 그대로 공유 (SUPPORTED_OPS, domain contracts)
- `onnx_rewrite/runtime/validation.py`: correctness 측정 재사용
- `onnx_rewrite/runtime/benchmark.py`: latency 측정 재사용
- `onnx_rewrite/passes/`: rule 의미는 참조하되, 코드 자체는 재사용하지 않음 (IR 체계가 다름)

### 4.3 e-graph에 ONNX attribute를 얼마나 넣을 것인가

Tensat은 stride, padding, axis 같은 attribute를 e-node의 일부로 표현한다.
ONNX는 attribute가 훨씬 다양하다 (kernel_shape, dilations, group, perm, ...).

초기 전략:
- e-node의 `attrs`에 hashable한 attribute tuple을 넣는다
- pattern matching에서는 op name만으로 1차 필터, attrs는 check function에서 검증
- 이렇게 하면 rule 작성이 간단해지고, attribute 폭발을 피할 수 있다

### 4.4 multi-output operator 처리

ONNX의 `Split` 같은 multi-output operator:
- Tensat 방식: `split` node + `split_0`, `split_1` projection node
- 우리도 동일하게 projection node로 풀어서 각 output을 별도 e-class로 관리

### 4.5 Dynamic shape 처리

현재 benchmark의 LLM 모델은 dynamic shape를 쓴다.
e-graph analysis에서 shape이 None인 경우를 허용하되,
shape-dependent rule (e.g. reshape fusion)은 concrete shape일 때만 적용한다.
