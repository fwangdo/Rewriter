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

Tensat은 **2-layer 전략** (Algorithm 2, Section 5.2)을 사용한다:

**Layer 1 — Pre-filtering (sound, not complete)**:
- iteration 시작 시 descendant map을 한 번 빌드. O(V+E).
- 각 match에 대해 O(1) 체크: target의 입력 e-class가 match e-class를 descendant로 갖는지.
- 대부분의 cycle을 빠르게 걸러내지만, 같은 iteration 내 이전 apply가 만든 새 관계
  때문에 cycle이 생길 수 있다 (논문이 이 한계를 명시적으로 언급).

**Layer 2 — Post-processing (completeness 보장)**:
- iteration 끝에 root로부터 DFS로 실제 cycle을 탐지.
- cycle 발견 시 가장 최근에 추가된 e-node(highest nid)를 **blacklist**.
- cycle이 없어질 때까지 반복.
- blacklist된 e-node는 extraction과 이후 descendant-map 빌드에서 제외.

두 layer를 합치면 pre-filtering이 O(V+E)로 대부분 잡고,
post-processing이 O(n_c · N)으로 나머지를 잡는다. 최대 2000x 빠름(Table 6).

**주의**: Layer 1만 구현하면 soundness만 보장되고 completeness가 보장되지 않는다.
Layer 2 없이는 cycle이 extraction 단계에서 ValueError를 일으킬 수 있다.

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

### Phase 0: Scaffold ✅

[x] `src/superopt/__init__.py` 생성
[x] 디렉토리 구조 생성 (`ir/`, `egraph/`, `rules/`, `explore/`, `extract/`)
[x] `src/common/contracts.py`에서 `SUPPORTED_OPS` import 확인

### Phase 1: IR + ONNX 변환 ✅

[x] `IRNode`, `IRGraph` dataclass 정의 (`ir/node.py`, `ir/graph.py`)
[x] `onnx_to_ir()`: ONNX model → IRGraph 변환 (`ir/convert.py`)
[x] `ir_to_onnx()`: IRGraph → ONNX model 변환 (`ir/convert.py`)
[x] multi-output node 처리 (`OP_PROJ` projection node)
[x] 변환 round-trip 테스트: `onnx → ir → onnx` 후 ORT correctness 확인

### Phase 2: E-Graph Core ✅

[x] Union-find 기반 `EGraph` 구현 (`egraph/egraph.py`)
[x] `ENode`, `EClass` 구현 (`egraph/enode.py`, `egraph/eclass.py`)
[x] `add()`, `merge()`, `find()`, `rebuild()` 구현 (egg-style repair)
[x] `AnalysisData` 및 e-class analysis 구현 (`egraph/analysis.py`)
[x] `ir_to_egraph()`: IRGraph → EGraph 초기화 (`pipeline.py`)

### Phase 3: Pattern Matching + Rules ✅ (기본 골격)

[x] `Pattern`, `PatternVar`, `PatternNode` 정의 (`egraph/pattern.py`)
[x] single-pattern search 구현
[x] match apply 구현 (`rules/base.py`: `apply_rule`, `_instantiate`)
[x] `RewriteRule` with `check` / `apply_fn` callbacks
[x] arithmetic rules: `add_comm`, `mul_comm`, `add_assoc_right`, `mul_assoc_right`
[x] layout rules: `reshape_reshape`, `transpose_transpose_identity` (perm=(0,1) only)
[x] fusion rules: `bias_add_commute`

### Phase 4: Exploration ✅

[x] exploration 메인 루프 구현 (`explore/explorer.py`)
[x] saturation 판정 구현
[x] `ExploreStats` 리포팅
[x] `check.sh` smoke test (tinyllama_15m) — steps 1-7 pass

### Phase 5: Extraction ✅

[x] greedy extraction 구현 (`extract/greedy.py`)
[x] uniform cost model + legality hard filter (`extract/cost.py`)
[x] extraction 결과를 IRGraph로 변환
[x] ILP extraction via `scipy.optimize.milp` (`extract/ilp.py`)

### Phase 6: End-to-End Pipeline ✅

[x] `pipeline.py`: ONNX → superopt → ONNX 전체 흐름 연결
[x] `tinyllama_15m`에서 end-to-end 실행 (`check.sh` 7단계 pass)

### Phase 7: Rule 확장 ✅

[x] 20개 규칙 구현 (arithmetic 4, layout 2, fusion 1, legalization 13 + no-bias variant)
[x] scalar_value 분석 (AnalysisData에 추가, Pow/Where/Range check에 사용)
[x] pre-pass (ConstantFolding → DecoderMask → Trilu → ConstantFolding)
[x] post-pass (ConstantFolding → ShapeInference → Cleanup)
[x] compat.py: baseline pass를 __init__.py 우회하여 import
[x] egraph.initializers: 원본 weight 데이터 접근 가능
[x] efficient cycle filtering — Tensat Algorithm 2 완전 구현:
      Layer 1: descendant map pre-filtering (O(V+E) per iter, sound)
      Layer 2: DFS post-processing + blacklist (completeness 보장)

### Phase 8: ILP Extraction ✅ (기본 구현)

[x] ILP formulation 구현 (`extract/ilp.py`)
[x] `scipy.optimize.milp` 기반
[ ] legality를 ILP hard constraint로 추가 (현재 greedy만 legality-aware)
[ ] greedy vs ILP 비교 실험

### Phase 9: ORT latency-first extraction ← **현재 단계**

**변경된 목표**: supported/unsupported op 구분은 extraction constraint로 쓰지 않는다.
모든 ONNX op를 허용하고, ONNX Runtime profiling에서 측정한 op별 평균 latency
cost를 사용해 e-graph 안에서 가장 낮은 cost의 graph를 extraction한다.

최종 성공 기준은 아래 순서다.

1. superopt output이 ORT에서 로드된다.
2. 원본 모델과 superopt output이 correctness tolerance 안에 든다.
3. end-to-end ORT CPU latency가 원본 또는 ORT optimizer 결과보다 개선된다.

#### 현재 확인 결과

| Model | 상태 | 확인 내용 |
|-------|------|-----------|
| mobilenetv2 | correctness PASS | 기존 latency 산출물과 재생성 산출물 모두 `max_abs_diff ~= 1e-15` |
| tinyllama_15m | correctness PASS after fix | invalid Reshape shape 6개를 제거했고, 재생성 결과 `max_abs_diff=3.17e-05` |
| pythia_70m | old artifact FAIL | 기존 산출물은 `Reshape` target에 `-1`이 여러 개 있어 ORT load 실패. tinyllama와 같은 원인으로 보임 |
| yolo26_nano | old artifact FAIL | 기존 산출물은 `Resize` optional input 복원 오류로 ORT load 실패 |
| mobilevit_xxs | unresolved | e-graph 폭발로 산출/검증 미완료 |
| smollm_135m | unresolved | protobuf 2GB 제한으로 산출/검증 미완료 |

#### 이번에 확인한 correctness blocker

- `Squeeze/Unsqueeze -> Reshape` legalization이 shape inference의 unknown dim을
  `-1`로 그대로 materialize했다.
- ONNX `Reshape` shape tensor는 `-1`을 최대 하나만 허용한다.
- `tinyllama_15m` old artifact에는 `[-1, 1, 1, -1]`, `[1, 1, -1, -1]`
  같은 invalid shape가 있었다.
- 해당 rule은 unknown dim이 2개 이상인 target shape에서는 적용하지 않도록 막았다.

#### 다음 할 일

[ ] pythia_70m 재생성 후 ORT correctness 확인
[ ] yolo26_nano `Resize` optional input round-trip 복원 수정
[ ] mobilevit_xxs: e-graph 폭발 해결. 규칙별 match 수 프로파일
[ ] smollm_135m: protobuf 2GB 제한 / IR node 폭발 원인 분석
[ ] latency benchmark를 correctness gate 이후에만 집계하도록 정리
[ ] paper evaluation section을 latency-first 목표에 맞게 업데이트

---

## 4. 핵심 설계 결정

### 4.1 Weight 처리의 추상화 레벨 (Issue #1)

**현재 방식**: 모든 weight를 `__name__`으로 구분하여 각각 고유한 e-class 부여.
패턴 매칭에서는 `?w`로 아무 weight나 바인딩하지만, apply_fn/check_fn에서
필요할 때 값을 꺼내 본다.

이 경계가 규칙마다 암묵적으로 결정되어 있다:

| 유형 | 예시 | weight 처리 |
|------|------|------------|
| 순수 구조 | `Clip(x,min,max)→Min(Max(x,min),max)` | 불투명 전달 |
| 값 조건 | `Pow(x,e)→Mul(x,x)` (e≈2) | check_fn에서 scalar_value 확인 |
| 값 합성 | `BN(x,s,b,m,v)→Mul+Add` | apply_fn에서 ndarray 읽고 새 weight 계산 |

**현재 시점의 기준**: 이 세 유형의 구분은 타당하며, 각 규칙이 어떤 유형인지
명시하는 것이 중요하다. 향후 이 구분을 코드 레벨에서 enforce할지는
스케일 문제가 실제로 bottleneck이 되는지 확인 후 결정.

**잠재적 연구 문제**: 구조적으로 동일한 layer가 weight 때문에 별개
e-class 트리로 복제되어 e-graph가 O(layer 수)로 커지는 문제.
구조 공유 + 추출 시 값 복원이 가능하면 해결되지만, feasibility 미확인.
(→ GitHub Issue #1)

### 4.2 기존 onnx_rewrite 코드 재사용 범위

- `src/common/contracts.py`: 그대로 공유 (SUPPORTED_OPS, domain contracts)
- `onnx_rewrite/passes/`: compat.py를 통해 pre/post-pass로 재사용
  (ConstantFolding, Cleanup, DecoderMask, Trilu)
- `onnx_rewrite/runtime/`: correctness/latency 측정 재사용 예정

### 4.3 Python 성능

- 현재 tinyllama_15m(761 e-class)는 수 초 내 완료.
- mobilevit_xxs(417 노드)에서 e-graph 폭발 발생 — 규칙 수/조합 문제이지
  Python 자체의 한계는 아직 아님.
- 대형 모델에서 bottleneck이 되면 egglog Python binding 검토.
