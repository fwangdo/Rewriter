# Superopt TODO

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
