# Superopt 현실화 TODO

현재 목표는 논문용 research challenge를 키우는 것이 아니라, 포트폴리오와 현업
감각에 맞는 ONNX superoptimizer를 만드는 것이다.

핵심 목표:
- ONNX 모델을 입력으로 받는다.
- target backend가 요구하는 supported-op contract를 만족하는 graph를 만든다.
- e-graph는 rewrite 후보를 넓게 생성하는 엔진으로 사용한다.
- correctness와 backend validation을 통과한 후보만 최종 출력한다.
- latency는 cost model의 예측값이 아니라 실제 ORT 측정값으로 최종 선택한다.
- 개선 후보가 없으면 "no beneficial rewrite found"도 정상 결과로 인정한다.

## 1. Baseline 방향

Baseline은 무식하게 모든 rule을 켜는 방식으로 시작한다.

이유:
- e-graph는 rewrite ordering 문제를 줄이기 위해 쓰는 도구다.
- sound하다고 믿는 rewrite를 많이 넣고, e-graph가 여러 equivalent form을 동시에
  보존하게 하는 것이 기본 사용법에 가깝다.
- 문제는 "모든 rule을 켠다" 자체가 아니라, 모든 rule을 같은 신뢰도와 같은 목적의
  rewrite로 취급하는 것이다.
- 따라서 baseline에서는 전부 켜고, 문제가 생기는 지점에서 decision을 만든다.

Baseline success criteria:
- 6개 benchmark 모델이 모두 5분 안에 pipeline을 끝낸다.
- supported-op contract 검사 결과를 출력한다.
- correctness 실패 후보는 최종 출력하지 않는다.
- 후보별 graph 변화, correctness, latency, 선택/폐기 이유를 기록한다.

## 2. Target Contract 재도입

`모든 op 허용 + latency 최소` 목표는 너무 research스럽고, 현업형 optimizer로는
설명력이 약하다. supported ops를 다시 도입한다.

필요한 개념:
- `supported_ops`: target backend가 실행 가능한 op.
- `must_remove_ops`: 반드시 제거해야 하는 op.
- `preferred_ops`: 가능하면 남기거나 유도할 op.
- target profile:
  - `ort_cpu`
  - `mobile_cpu`
  - 향후 custom backend profile

TODO:
- [ ] target contract 자료구조 정의
- [ ] benchmark 6개 모델에 대해 current op histogram 출력
- [ ] target별 unsupported/must-remove histogram 출력
- [ ] final candidate가 contract를 만족하는지 검사
- [ ] contract 불만족 시 어떤 op가 남았는지 report에 기록

## 3. Rule Policy

Baseline에서는 모든 rule을 켠다. 이후 문제가 생기면 rule metadata와 policy로
결정을 남긴다.

Rule metadata 후보:
- `purpose`: legalization / canonicalization / cleanup / fusion_enabling / performance
- `risk`: exact / numerically_safe / tolerance_sensitive / approximate
- `requires`: shape known / constant known / rank fixed / dtype fixed
- `target_effect`: unsupported 제거 / preferred op 유도 / node 감소 / ORT fusion 유도
- `family`: arithmetic / layout / legalization / fusion / decomposition

TODO:
- [ ] 현재 rule 목록을 family 단위로 dump하는 명령 추가
- [ ] 모든 rule을 기본 ON으로 두는 baseline profile 추가
- [ ] rule별 metadata를 optional field로 추가
- [ ] correctness drift 발생 시 family 단위 bisect 가능하게 만들기
- [ ] numerical-risk rule은 tolerance policy와 연결

## 4. Candidate Generation / Selection

Cost model 하나로 바로 최종 graph를 고르는 구조는 취약하다. Cost model은 후보를
줄이는 heuristic이고, 최종 선택은 validation 결과로 한다.

Pipeline 목표:
```
ONNX
  -> pre-pass / shape inference
  -> e-graph saturation
  -> candidate extraction
  -> ONNX materialization
  -> ONNX checker
  -> supported-op contract check
  -> ORT load
  -> correctness check
  -> measured latency
  -> best valid candidate selection
```

TODO:
- [ ] original/pre-pass 결과도 candidate 0으로 포함
- [ ] e-graph extraction candidate 생성
- [ ] rule-family forced candidate 생성 여부 검토
- [ ] materialized graph hash dedup
- [ ] 후보 수 상한 도입
- [ ] 모델당 wall-clock timeout 도입
- [ ] valid 후보가 없을 때 fallback policy 명확화

## 5. Cost Model의 역할

Cost model은 정답이 아니다. 특히 op 평균 latency나 FLOPs proxy만으로는 ORT 실제
latency를 잘 예측하기 어렵다.

현실적인 역할:
- extraction heuristic
- 후보 pruning
- latency 측정 전 rough ranking

최종 선택 기준:
- ONNX checker PASS
- supported-op contract PASS
- ORT load PASS
- correctness PASS
- measured latency가 가장 낮음

TODO:
- [ ] cost-aware extraction이 실제로 cost callback을 쓰는지 검증
- [ ] callback overhead가 큰 모델에서 5분 안에 끝나도록 bound 설정
- [ ] estimated cost와 measured latency correlation 기록
- [ ] cost가 틀렸을 때도 validation으로 복구되는 구조 만들기

## 6. Correctness / Safety

현업형 superopt에서 correctness는 hard gate다. 틀린 후보는 아무리 빠르거나
contract를 만족해도 폐기한다.

TODO:
- [ ] model family별 deterministic input 생성 정리
- [ ] NLP input id 범위를 vocab 안으로 보장
- [ ] output count / shape / dtype / name order 검사
- [ ] model family별 tolerance 명시
- [ ] correctness 실패 후보의 max diff와 실패 output index 기록
- [ ] rule family bisect 도구 추가

## 7. Report / Portfolio Output

이 프로젝트의 가치는 "무엇을 고민했고 어떤 결정을 했는지"가 보여야 한다.

Report에 들어갈 것:
- model name
- original op histogram
- candidate op histogram
- unsupported/must-remove op before/after
- applied rule families
- candidate count
- rejected candidates and reasons
- selected candidate reason
- correctness result
- latency result
- time breakdown

TODO:
- [ ] JSON report schema 정의
- [ ] Markdown summary report 생성
- [ ] 6개 benchmark 전체 summary table 생성
- [ ] selected graph와 rejected graph 차이를 operation별로 비교

## 8. 코드 읽기 순서

현재 전제:
- egg 논문의 큰 개념은 이해하고 있다.
- e-class, e-node, union/find, merge, rebuild, canonicalization, upward merge의
  목적은 알고 있다.
- Tensat은 cycle 처리와 graph extraction 관점만 다시 복습하면 된다.

따라서 코드는 "자료구조를 처음부터 배우기"보다, 이 repo가 e-graph를 ONNX
superopt 제품 파이프라인으로 어떻게 연결하는지 보는 순서로 읽는다.

추천 순서:

1. Entry point / 전체 흐름
   - `src/superopt/pipeline.py`
   - 먼저 `superoptimize_topk()`를 읽는다.
   - 목표: ONNX load, pre-pass, legacy bridge, egglog backend, extraction,
     ONNX 저장이 어떤 순서로 이어지는지 잡는다.

2. IR 레이어
   - `src/common/ir/node.py`
   - `src/common/ir/graph.py`
   - `src/common/ir/convert.py`
   - 목표: ONNX graph를 e-graph에 넣기 전 어떤 중간 표현으로 낮추는지 본다.
   - 여기서 node id, inputs, attrs, shape, dtype, initializer 처리 방식을 이해한다.

3. Rule 정의
   - `src/superopt/rules/base.py`
   - `src/superopt/rules/legalization.py`
   - `src/superopt/rules/arithmetic.py`
   - `src/superopt/rules/layout.py`
   - `src/superopt/rules/fusion.py`
   - 목표: 어떤 rewrite가 있고, 단순 pattern rule과 `check`/`apply_fn` rule이 어떻게
     다른지 본다.
   - 현업형 baseline에서는 일단 모든 rule을 켜되, 문제가 생기는 rule family를
     decision으로 남긴다.

4. egglog backend
   - `src/superopt/backends/egglog.py`
   - 목표: IR이 egglog term으로 어떻게 encode/decode되는지 본다.
   - 특히 볼 것:
     - `_load_ir()`
     - `_make_expr()`
     - `_pattern_to_expr()`
     - `run_rules()`
     - `extract_best()` / `extract_topk()`
     - `_expr_to_ir()`
   - 여기서 현재 main path가 자체 e-graph가 아니라 egglog 중심이라는 점을 확인한다.

5. Legacy e-graph bridge
   - `src/superopt/egraph/egraph.py`
   - `src/superopt/explore/explorer.py`
   - `src/superopt/explore/matcher.py`
   - `src/superopt/extract/greedy.py`
   - 목표: 자체 e-graph 구현을 production path로 이해하지 말고, 아직 egglog-native로
     옮기지 못한 `check`/`apply_fn` rule을 materialize하는 bridge로 이해한다.
   - 이미 egg 개념을 알고 있으므로 자료구조 자체보다 "왜 아직 남아 있는가"를 본다.

6. Cost / extraction
   - `src/superopt/extract/cost.py`
   - `src/superopt/extract/greedy.py`
   - `src/superopt/extract/ilp.py`
   - 목표: cost model이 최종 답이 아니라 candidate ranking heuristic이라는 점을
     코드에서 확인한다.
   - Tensat 복습 포인트는 cycle이 있는 e-class graph에서 extraction을 어떻게
     well-founded하게 만드는지다.

7. Validation / benchmark
   - `src/superopt/bench_latency.py`
   - `src/superopt/bench_all.py`
   - `src/superopt/eval_superopt.py`
   - 목표: 현업형 superopt의 핵심인 correctness gate, latency measurement,
     실패 후보 폐기 정책을 어디에 둘지 본다.

8. Reportability / portfolio surface
   - 아직 가장 비어 있는 영역이다.
   - 앞으로 JSON/Markdown report를 만들어 graph 변화, supported-op 변화,
     correctness, latency, 선택 이유를 남겨야 한다.

읽을 때의 기준:
- e-graph 자료구조 자체보다 `ONNX -> IR -> e-graph -> candidates -> validation`
  연결을 먼저 본다.
- correctness를 깨는 rule은 research failure가 아니라 product decision 지점으로 본다.
- supported-op contract를 만족시키는 rewrite와 latency를 개선하는 rewrite를 구분한다.
- cost model이 틀릴 수 있음을 전제로 validation에서 복구하는 구조를 확인한다.

## 9. 당장 다음 작업

우선순위:
- [ ] 현재 남아 있는 egglog extraction 변경을 정리한다.
- [ ] 모든 모델이 5분 안에 끝나는 bounded baseline을 만든다.
- [ ] supported-op contract를 다시 도입한다.
- [ ] 모든 rule ON baseline을 기준으로 correctness/contract/latency report를 만든다.
- [ ] 문제가 생기는 rule family를 하나씩 decision으로 남긴다.
