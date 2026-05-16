# Superopt TODO

## 0. New Goal: Compiler-Style Automated Legalization

수동 rule 작성(peephole optimization)을 넘어, 컴파일러 정석 구조로 legality 문제를 자동 해결한다.

### 0.1 핵심 구조: Lowering → Saturation → Lifting

LLVM의 컴파일 구조와 동형:

| LLVM | 우리 |
|---|---|
| C → LLVM IR (lowering) | ONNX op → primitive IR (lowering) |
| LLVM IR 최적화 (pass) | primitive IR에서 e-graph saturation |
| LLVM IR → machine inst (instruction selection) | primitive IR → ONNX op (lifting = pattern matching) |

**Lowering**: 모든 ONNX op을 primitive op으로 분해.
```
MatMul(A,B)  := reduce_sum(broadcast_mul(A, B), axis=k)
Conv1x1(X,W) := reduce_sum(broadcast_mul(X, W), axis=c)
```

**Saturation**: primitive level에서 e-graph가 등가 표현을 탐색.
동일한 `reduce_sum(broadcast_mul(...))` 패턴으로 분해되면 MatMul과 Conv가 자동으로 같은 e-class에 합류.

**Lifting**: lowering 정의를 뒤집어서 primitive 패턴 → ONNX op 인식 (tree pattern matching).
lowering이 정의되면 lifting은 자명. 추가 작업 아님.

### 0.2 기존 방식 대비 이점

**수동 rule 작성 (기존):**
- unsupported op m개 × "어떤 supported op 조합으로?" = 매번 사람이 탐색
- op 간 변환 발견이 어렵고 누락 위험
- 새 HW contract마다 rule set 재작성

**Lowering 기반 (새 방식):**
- 각 ONNX op의 lowering 정의 N개만 필요 (ONNX spec reference impl에서 기계적 도출 가능)
- op 간 equivalence는 e-graph saturation에서 자동 발견
- contract이 바뀌어도 supported op set만 교체하면 됨

방향이 뒤집힘:
- 기존: "이 illegal op을 뭘로 바꾸지?" (탐색 필요)
- 새 방식: "primitive로 내려간 뒤, legal op 중 매칭되는 게 있나?" (매칭만)

### 0.3 Solver-Free 구조

**ILP extraction 불필요** — legality만 따지면 최적화 문제가 아니라 feasibility 문제.
e-graph에서 supported op으로 lifting 가능한 조합이 하나라도 있으면 해결.

**Z3 verification 불필요** — correctness는 observational equivalence (random input 비교)로 충분.
TASO가 enumerate + Z3로 나눠서 한 걸, e-graph saturation이 한 번에 처리.

**Scope이 작음** — illegal op 하나 (+ 주변 subgraph)만 e-graph에 넣으므로:
- saturation 빠름 (max_nodes=1000이면 충분)
- TASO의 4 ops 한계를 넘어 깊은 equivalence 탐색 가능

### 0.4 Lifting = Tree Pattern Matching

primitive 조합을 ONNX op으로 인식하는 건 단순 hash lookup이 아니라 tree pattern matching.
```
reduce_sum                 ← depth 0
  └─ broadcast_mul         ← depth 1
       ├─ A
       └─ B
```
축과 shape에 따라 MatMul, Conv, AvgPool 등으로 달라짐.
이건 컴파일러의 instruction selection과 동일한 문제 — LLVM TableGen이 이미 풀어놓음.
depth 2~3 패턴 매칭이므로 5000 e-node에서도 충분히 빠름.

### 0.5 실용적 가치

칩 회사 시나리오: 새 NPU마다 supported op set이 달라짐.
- 현재: 엔지니어가 매번 수동 변환 rule 작성
- 이 도구: contract (supported op set)만 넣으면 변환이 자동 생성

nota 과제가 정확히 이 상황 — "이 HW에서 이 op만 됩니다, 모델 바꿔주세요"를 자동화.

### 0.6 선결 과제

1. **Primitive IR 설계**: ONNX op을 분해할 primitive op set 정의
   - 후보: `broadcast_mul`, `reduce_sum`, `broadcast_add`, `reshape`, `transpose`, `select`, `scatter` 등
   - ONNX op 자체를 더 세분화하는 타협도 가능 (예: Conv → im2col + MatMul + Reshape, 전부 ONNX op)
2. **Lowering 정의**: 주요 ONNX op 10~20개의 primitive 분해 (ONNX spec reference impl 참조)
3. **E-graph 연동**: 기존 e-graph 인프라에 primitive IR 노드 추가
4. **Lifting pattern**: lowering 정의의 역방향 tree pattern matcher 구현
5. **PoC 검증**: QNN contract 기준 illegal op 하나에 대해 end-to-end 자동 변환 시연

### 0.7 참고 연구

- **TASO (SOSP'19)**: subgraph enumeration + fingerprint + Z3 → 743 rules 자동 생성. 자체 IR, 4 ops 한계.
- **Trinity (ASPLOS'26)**: tile-level primitive axiom → e-graph saturation으로 FlashAttention 자동 발견. "촘촘한 primitive rule이면 e-graph으로 complex optimization이 emerge"를 실증.
- **STENSO**: symbolic execution + sketch synthesis. solver 기반, 소규모 kernel.
- **LLVM**: lowering → optimization → instruction selection 구조의 원형.

---

현재 핵심 문제는 e-graph 구현 자체보다 rewrite rule set이 아직 얇다는 점이다.
지금 규칙은 "ONNX op를 다른 op 조합으로 낮추는 legalization pack"에 가깝고,
e-graph가 여러 좋은 대안을 탐색할 만큼 optimization rule이 충분하지 않다.

## 1. 현재 판단

- active rule은 41개다.
- README/docs에는 42개로 적혀 있지만 `where_mask_decompose`는 코드에서 비활성화되어 있다.
- 대부분의 rule은 unsupported op lowering이다.
- 실제 optimization rule은 arithmetic 4개, layout 3개, fusion 1개뿐이다.
- 따라서 작은 모델에서 e-node / e-class 비율이 약 1.5 수준으로 낮게 나오는 것은 자연스럽다.
- 현재 rule set만으로 e-graph superoptimization의 강점을 충분히 보여주기는 어렵다.

## 2. Rule Correctness Audit

먼저 rule을 늘리기 전에 현재 rule이 항상 안전한지 확인한다.

- [ ] active rule 목록과 docs/README의 rule count를 일치시킨다.
- [ ] rule별 목적을 분류한다: legalization / canonicalization / cleanup / fusion / performance.
- [ ] rule별 필요 조건을 명시한다: dtype / shape / rank / broadcast / constant / opset.
- [ ] `Greater -> Less`의 출력 dtype 처리 확인.
- [ ] `Sub -> Add(Neg)`의 broadcast shape 처리 확인.
- [ ] `Range -> Slice(arange_table_4096)`의 limit 범위 조건 추가.
- [ ] `Where -> arithmetic`의 bool cast, dtype, broadcast safety 확인.
- [ ] `MatMul -> Conv`의 rank, layout, correctness 조건 재검토.
- [ ] risky rule은 hard enable하지 말고 guard를 강화한다.

## 3. Benchmark 기반 Gap 분석

필요한 rule은 감으로 추가하지 말고 benchmark에서 나온 실패와 빈도에서 출발한다.

- [ ] 모델별 original op histogram을 출력한다.
- [ ] 모델별 rewritten op histogram을 출력한다.
- [ ] contract 기준 unsupported op before/after를 출력한다.
- [ ] 제거 실패한 unsupported op를 rule 후보로 기록한다.
- [ ] 많이 등장하지만 rewrite 후보가 거의 없는 op family를 찾는다.
- [ ] e-node / e-class 비율이 낮은 모델에서 실제로 어떤 rule이 적용됐는지 출력한다.

## 4. High-Value Rule 후보

우선순위는 "많이 등장하고", "조건을 명확히 쓸 수 있고", "correctness 검증이 쉬운" 규칙이다.

- [ ] identity cleanup: `Add(x, 0)`, `Mul(x, 1)`, `Sub(x, 0)`, `Div(x, 1)`.
- [ ] annihilator cleanup: `Mul(x, 0)`, `Sub(x, x)` 등 안전한 경우만.
- [ ] constant folding: Add/Mul/Sub/Div/Reshape/Slice/Gather 계열의 constant-only case.
- [ ] Cast cleanup: redundant Cast, Cast(Cast(x)) collapse, same dtype Cast 제거.
- [ ] layout cleanup: arbitrary inverse Transpose cancel, Transpose identity 제거.
- [ ] shape-flow cleanup: Shape/Gather/Unsqueeze/Concat/Reshape로 만들어지는 static shape folding.
- [ ] reshape cleanup: Reshape identity, Reshape after Squeeze/Unsqueeze 정리.
- [ ] fusion: Conv+BN, MatMul+Add/Gemm canonicalization.
- [ ] LLM-specific: attention mask, position id, cache shape, Trilu/Where/Range 관련 rewrite.

## 5. E-Graph를 살리는 방향

e-graph가 강해지려면 한 노드에서 여러 derivation이 생기고, extraction이 그중 좋은 조합을 선택할 수 있어야 한다.

- [ ] 단순 lowering rule과 optimization alternative rule을 구분한다.
- [ ] lowering만 늘려서 node count를 키우는 방향을 피한다.
- [ ] 같은 의미를 유지하면서 서로 다른 비용 구조를 만드는 rule을 우선 추가한다.
- [ ] ORT나 backend가 이미 잘 fuse하는 패턴을 깨는 rule은 별도 policy로 관리한다.
- [ ] legal하지만 느린 graph와 illegal하지만 작은 graph를 cost/legality에서 분리해 평가한다.

## 6. Report에 남길 지표

rule set이 충분한지 판단하려면 결과표에 다음 지표가 필요하다.

- [ ] active rule count와 family별 count.
- [ ] model별 applied rule count.
- [ ] model별 applied rule family.
- [ ] e-classes, e-nodes, e-node / e-class.
- [ ] max_nodes 도달 여부.
- [ ] unsupported op before/after.
- [ ] correctness pass/fail.
- [ ] baseline 대비 node count와 legality 차이.

## 7. 당장 다음 작업

1. docs/README의 rule count를 실제 active rule count와 맞춘다.
2. rule audit 표를 만든다.
3. benchmark op histogram과 applied rule log를 추가한다.
4. cleanup/constant/layout rule부터 작게 추가한다.
5. LLM 모델에서 남는 unsupported op를 기준으로 다음 legalization rule을 정한다.
