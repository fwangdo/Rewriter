# ONNX Rewrite Explain

이 문서는 `front/onnx_rewrite` 구현을 **외워서 익숙해지기 위한 개인 학습 노트**다.

목표는 다음 세 가지다.

1. ONNX graph를 **value-name 기반 DAG**로 읽는 감각을 완전히 몸에 붙인다.
2. ONNX Runtime(ORT)으로 모델을 **직접 실행 / 비교 / 벤치마크**하는 방법을 익힌다.
3. 우리가 구현한 rewrite:
   - BN 제거
   - Conv-BN fusion
   - ConvTranspose-BN fusion
   - Gather 제거
   - Pow 제거
   - MatMul / Gemm 제거
   
   를 코드 수준에서 설명할 수 있게 만든다.

이 문서는 README보다 훨씬 자세하다.  
README가 “프로젝트 설명서”라면, 이 문서는 “내가 이 코드를 머리에 넣기 위한 해설서”다.

---

# 0. 먼저 가져갈 핵심 관점

`todo.md`에 적혀 있던 어려운 개념들을 먼저 한 번에 요약하면 이렇다.

1. ONNX는 **node name**보다 **tensor value name**이 더 중요하다.
2. `graph.input`과 `initializer`는 둘 다 node input으로 참조 가능한 **tensor value**다.
3. rewrite 설명은 항상:

```text
input tensor names -> intermediate tensor names -> output tensor name
```

순서로 읽어야 한다.

4. `Shape`, `Gather`, `Slice`, `Reshape`는 **값 계산(value computation)** 과 **shape 계산(shape computation)** 을 분리해서 봐야 한다.
5. broadcasting은 “대충 맞겠지”가 아니라, **어느 op에서 어떤 shape pair가 만나는지**를 써가며 봐야 한다.
6. static rewrite와 dynamic rewrite를 항상 구분해야 한다.
   - static: initializer 기반
   - dynamic: runtime tensor 기반
7. `Gather -> MatMul -> RewriteMatmul`처럼 **pass가 연쇄적으로 연결**될 수 있다.
8. correctness 검증은:
   - structural legality
   - ORT 실행 가능 여부
   - 원본/변환 수치 비교
   
   순서로 보는 게 맞다.

---

# 1. ONNX를 읽는 기본 사고방식

## 1.1 ONNX는 무엇인가

ONNX model은 대충 “연산 그래프”라고만 기억하면 부족하다.  
실제로 rewrite할 때는 아래처럼 생각해야 한다.

```text
ModelProto
  └─ GraphProto
      ├─ input        : 외부에서 들어오는 tensor value
      ├─ initializer  : graph 내부에 박힌 constant tensor value
      ├─ node         : op
      ├─ output       : 최종 결과 tensor value
      └─ value_info   : 중간 tensor에 대한 type/shape metadata
```

핵심은:

- `node`는 연산
- `input/output/initializer`는 tensor value
- 연결은 value name으로 된다

즉 ONNX는 다음 문장으로 외우면 된다.

> ONNX graph는 “tensor value name을 edge로 쓰는 DAG”다.

---

## 1.2 node name vs tensor output name

이건 반드시 확실히 이해해야 한다.

예를 들어:

```python
node = helper.make_node(
    "Add",
    ["x", "y"],
    ["z"],
    name="my_add",
)
```

여기서:

- `node.name == "my_add"`  
  디버깅용 이름

- `node.output[0] == "z"`  
  실제로 downstream이 참조하는 tensor value 이름

즉 다음 node가:

```python
helper.make_node("Relu", ["z"], ["relu_out"], name="my_relu")
```

처럼 `"z"`를 입력으로 받으면, 실제 연결은 `"my_add"`가 아니라 `"z"`를 통해 된다.

따라서 rewrite에서는:

- `node.name`은 사람이 보기 편하게 쓰는 보조 정보
- `node.input`, `node.output`의 tensor name이 실질적 연결 정보

라고 이해해야 한다.

이 감각이 없으면:

- producer 찾기
- consumer 찾기
- node 교체
- graph output rewire

가 다 헷갈린다.

---

## 1.3 graph.input과 initializer

이 부분도 중요하다.

### `graph.input`

외부에서 feed dict로 넣는 입력 tensor다.

예:

```text
input_ids
attention_mask
pixel_values
```

### `initializer`

모델 안에 박힌 constant tensor다.

예:

```text
Conv weight
Conv bias
embedding table
reshape shape tensor
constant scalar
```

중요한 점:

> 둘 다 node input으로 참조 가능한 tensor value다.

즉 node 입장에서는:

- 외부 입력인지
- 내부 constant인지

이름만 보면 바로 구분되지 않는다.

그래서 rewrite에서는 항상:

```python
if input_name in init_map:
    # static tensor
else:
    # runtime tensor
```

처럼 구분한다.

---

## 1.4 static tensor vs runtime tensor

rewrite할 때 이 구분은 거의 모든 판단의 시작점이다.

### static tensor

- `initializer`에 있음
- numpy로 바로 꺼낼 수 있음
- 값이 compile-time에 확정됨

예:

- Conv weight
- BN parameter
- Pow exponent scalar
- small embedding table

### runtime tensor

- 실행 시점에만 값이 정해짐
- input이거나, 어떤 node의 output

예:

- input ids
- activation
- dynamic shape tensor

암기 포인트:

> static이면 접을 수 있고, dynamic이면 구조를 유지한 채 동치 rewrite를 해야 한다.

---

# 2. ONNX 기본 API를 실제로 어떻게 쓰는가

## 2.1 모델 load / save / checker

가장 기본:

```python
import onnx

model = onnx.load("model.onnx")
graph = model.graph

onnx.checker.check_model(model)
onnx.save(model, "rewritten.onnx")
```

rewrite 루프의 기본 패턴은 항상:

1. load
2. graph 수정
3. checker
4. save

다.

---

## 2.2 node 순회

```python
for node in graph.node:
    print(node.name, node.op_type, list(node.input), list(node.output))
```

실제로는 op_type으로 분기한다.

```python
for node in list(graph.node):
    if node.op_type != "MatMul":
        continue
    ...
```

`list(graph.node)`로 감싸는 이유는 순회 중 삭제/추가를 직접 하지 않기 위해서다.

---

## 2.3 initializer를 numpy array로 바꾸기

```python
from onnx import numpy_helper

init_map = {
    init.name: numpy_helper.to_array(init)
    for init in graph.initializer
}
```

이 map 하나로 아래를 다 할 수 있다.

- weight static 여부 판단
- bias static 여부 판단
- BN parameter 읽기
- Pow exponent 읽기
- Gather data/indices static 여부 판단

즉 rewrite의 대부분은 사실상 이 `init_map` 위에서 돌아간다.

---

## 2.4 새 initializer 만들기

현재 코드에서는 [folder.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/folder.py)의 `add_init()`를 쓴다.

핵심은:

```python
self.add_init(graph, "new_name", np_array)
```

이게 하는 일:

1. 같은 이름 initializer가 있으면 제거
2. numpy array를 ONNX TensorProto로 만들어 graph.initializer에 추가

우리가 실제로 자주 추가하는 initializer 예:

- Conv fused weight
- Conv fused bias
- Unsqueeze axes tensor
- ReduceSum axes tensor
- Reshape shape tensor
- `1.0` constant
- vocab range tensor

---

## 2.5 새 node 만들기

기본:

```python
from onnx import helper

node = helper.make_node(
    "Conv",
    ["x", "w", "b"],
    ["y"],
    name="my_conv",
    kernel_shape=[1, 1],
    group=16,
)
```

자주 쓰는 패턴:

```python
helper.make_node(
    op_type,
    input_names,
    output_names,
    name=...,
    attr=value,
)
```

rewrite에서 많이 쓰는 op:

- `Conv`
- `ConvTranspose`
- `Add`
- `Mul`
- `Div`
- `Sqrt`
- `Equal`
- `Cast`
- `Reshape`
- `Transpose`
- `Unsqueeze`
- `ReduceSum`
- `Shape`
- `Concat`
- `Slice`

---

## 2.6 attribute 읽기

예를 들어 Conv의 `group`, BN의 `epsilon`, Gather의 `axis`를 읽을 때:

```python
for attr in node.attribute:
    if attr.name == "group":
        group = int(attr.i)
```

이런 식으로 본다.

현재 구현에서도 거의 이 패턴을 쓴다.

예:

```python
for attr in node.attribute:
    if attr.name == "epsilon":
        eps = float(attr.f)
```

암기:

- int attribute -> `attr.i`
- float attribute -> `attr.f`
- ints list -> `attr.ints`

---

## 2.7 graph 수정 패턴

현재 rewrite framework는 직접 `graph.node.remove(node)`를 순회 중에 하지 않는다.

대신:

1. `mark_for_removal(node)`
2. `append_nodes(new_nodes)`
3. 마지막에 `remove_marked_nodes()`

를 쓴다.

왜냐면 순회 중 remove는 버그를 만들기 쉽기 때문이다.

현재 [folder.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/folder.py)의 핵심 helper:

- `append_nodes`
- `mark_for_removal`
- `replace_node`
- `remove_marked_nodes`

실제로는:

```python
self.replace_node(old_node, new_nodes)
```

이 제일 많이 쓰인다.

---

# 3. ONNX shape / axis / broadcasting 감각

이 부분이 rewrite를 이해하는 핵심이다.

## 3.1 shape는 값과 별개다

예를 들어 `Shape(x)`는 `x`의 값을 계산하지 않는다.  
`x`의 **shape vector**를 만든다.

예:

```text
x.shape = [B, S, H]
Shape(x) -> [B, S, H]   # int64 tensor
```

즉 `Shape`, `Gather`, `Slice`, `Reshape`는 종종 **실제 activation 값이 아니라 shape 정보**를 다룬다.

이걸 구분 못하면 dynamic path 설명이 무너진다.

---

## 3.2 axis는 concrete example로 먼저 보자

예를 들어 tensor shape가:

```text
[B, S, H] = [2, 4, 8]
```

라고 하자.

- axis 0: batch 방향
- axis 1: sequence 방향
- axis 2: hidden 방향

그러면:

```text
Gather(data, indices, axis=0)
```

는 batch축 기준 선택이고,

```text
ReduceSum(..., axis=2)
```

는 hidden축을 합치는 것이다.

항상 general formula보다 **구체 shape 예시를 먼저 놓고** 읽는 게 낫다.

---

## 3.3 broadcasting

rewrite에서 broadcasting이 일어나는 대표 op:

- `Equal`
- `Mul`
- `Add`
- `Div`

중요한 건:

> “broadcast 된다”가 아니라, 정확히 어떤 shape pair가 만나서 어떻게 늘어나는지 봐야 한다.

예시 1:

```text
indices shape        = [B, S]
scalar_vocab_id      = []
Equal(indices, scalar_vocab_id)
result               = [B, S]
```

scalar가 `[B, S]`로 broadcast된다.

예시 2:

```text
mask shape           = [B, S]
Unsqueeze(-1)        -> [B, S, 1]
embedding_row shape  = [H]
Mul(mask, row)
result               = [B, S, H]
```

여기서는:

- `[B, S, 1]`
- `[H]`

가 만나서 `[B, S, H]`가 된다.

예시 3:

```text
Conv weight shape    = [C_out, C_in/group, kH, kW]
scale_factor shape   = [C_out]
```

이걸 fuse할 때는 그냥 곱하면 안 되고:

```python
scale_factor.reshape(C_out, 1, 1, 1)
```

로 output channel 축에 맞춰 broadcast시킨 뒤 곱해야 한다.

즉 broadcasting은 항상:

```text
어느 축에 맞춰 늘리고 싶은가?
```

를 먼저 정하고 reshape/unsqueeze를 해야 한다.

---

## 3.4 Unsqueeze의 axes 입력

이건 `todo.md`에도 적혀 있던 부분이다.

ONNX `Unsqueeze`는 보통 이렇게 생긴다.

```python
axes_name = self.tensor_name(prefix, "unsqueeze_axes")
self.add_init(graph, axes_name, np.array([3], dtype=np.int64))

helper.make_node(
    "Unsqueeze",
    [input_name, axes_name],
    [output_name],
    ...
)
```

즉 axes는:

- python scalar가 아니라
- `int64 tensor initializer`

로 들어가는 경우가 많다.

이 감각이 중요하다.

---

# 4. ONNX Runtime(ORT) 사용법

이 섹션은 “모델을 실제로 어떻게 돌리는가”를 설명한다.

## 4.1 가장 단순한 실행

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
outputs = session.run(None, feed_dict)
```

여기서:

- `session.run(None, inputs)`  
  `None`은 “모든 output을 달라”는 뜻

- `inputs`는:

```python
{
    "input_name": np_array,
    ...
}
```

형태다.

---

## 4.2 입력/출력 metadata 보기

현재 [runtime/rt_test.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/rt_test.py)가 이 용도로 좋다.

직접 보면:

```python
session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

for item in session.get_inputs():
    print(item.name, item.type, item.shape)

for item in session.get_outputs():
    print(item.name, item.type, item.shape)
```

이걸로:

- 입력 이름
- dtype
- symbolic shape

를 확인할 수 있다.

모델을 처음 볼 때 가장 먼저 할 일 중 하나다.

---

## 4.3 feed dict 만드는 법

예를 들어 BERT류에서:

```text
input_ids
attention_mask
token_type_ids
```

같은 입력이 있다면:

```python
inputs = {
    "input_ids": np.array([[101, 2023, 2003, 102]], dtype=np.int64),
    "attention_mask": np.array([[1, 1, 1, 1]], dtype=np.int64),
    "token_type_ids": np.array([[0, 0, 0, 0]], dtype=np.int64),
}
```

이런 식으로 넣는다.

vision 모델이면:

```python
inputs = {
    "input": np.random.randn(1, 3, 224, 224).astype(np.float32)
}
```

처럼 넣으면 된다.

가장 중요한 건:

1. 이름이 맞아야 한다.
2. dtype이 맞아야 한다.
3. shape가 맞아야 한다.

하나라도 틀리면 ORT가 바로 오류를 낸다.

---

## 4.4 실행 가능 여부부터 먼저 본다

rewrite 후 평가의 첫 단계는 “잘 돌리나?”다.

즉:

1. `onnx.checker.check_model`
2. `InferenceSession(...)`
3. `session.run(...)`

이 세 단계가 먼저다.

이후에야 correctness를 본다.

---

## 4.5 현재 프로젝트의 입력 생성 방식

현재 [runtime/validation.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/validation.py)는 입력을 자동으로 만든다.

핵심 함수:

- `_build_model_profile`
- `_resolve_input_shape`
- `_build_mask_input`
- `_build_int_input`
- `_build_float_input`
- `_generate_inputs_for_case`

### shape 해석

예를 들어 input shape가:

```text
['batch', 'sequence']
```

이면 `_resolve_input_shape(..., dynamic_size=2)`는:

```text
[2, 2]
```

로 바꾼다.

### mask 입력

이름에 `"mask"`가 들어가면:

- `ones`
- `random_binary`
- `prefix_drop`
- `checkerboard`

중 하나로 생성한다.

### integer 입력

`int64` / `int32` 입력은:

- `random_full`
- `edge_bias`
- `low_band`
- `repeated_token`

같은 모드로 생성한다.

이게 중요한 이유:

> correctness를 단일 random input 1회로 보면 dynamic path나 edge case를 거의 못 잡기 때문이다.

---

## 4.6 compare_models

현재 correctness 비교의 top-level은:

```python
compare_models(before_model_path, after_model_path)
```

이 함수는:

1. before ORT session 생성
2. after ORT session 생성
3. deterministic case pool 생성
4. 각 케이스마다 inputs 생성
5. before / after output 실행
6. `max_abs_diff` 계산
7. 전체 worst case를 뽑아 `ValidationResult` 반환

으로 동작한다.

즉 correctness는 단순히 한 번 실행해서 보는 것이 아니라, 여러 케이스의 최악값을 보는 것이다.

---

## 4.7 ORT latency 측정

현재 [runtime/benchmark.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/runtime/benchmark.py)는:

```python
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

로 session을 만든다.

핵심은:

- thread 수 고정
- sequential execution
- ORT optimization 켜기

다.

측정 방식:

1. warmup 여러 번
2. repeat 여러 번
3. 각 run의 milliseconds 저장
4. median / p95 계산

즉 benchmark는 median 중심으로 읽는다.

---

## 4.8 ORT를 직접 돌려보는 가장 단순한 실습

### 실습 1: 모델 구조 보기

```bash
python -m front.onnx_rewrite.runtime.rt_test \
  --input benchmarks/onnx/vision/resnet18.onnx
```

### 실습 2: rewrite 전 audit

```bash
python -m front.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --audit-only
```

### 실습 3: rewrite 실행

```bash
python -m front.onnx_rewrite.run_onnx_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --output /tmp/resnet18_rewritten.onnx
```

### 실습 4: rewrite + correctness + latency

```bash
python -m front.onnx_rewrite.eval_rewrite \
  --input benchmarks/onnx/vision/resnet18.onnx \
  --output /tmp/resnet18_rewritten.onnx \
  --report /tmp/resnet18_eval.json
```

이 정도를 직접 몇 번 반복해보면 ORT 사용법은 거의 감이 잡힌다.

---

# 5. 현재 rewrite framework 구조

핵심 파일:

- [passes/folder.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/folder.py)
- [passes/passer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/passer.py)
- [core/optimizer.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/core/optimizer.py)
- [checker/op_checker.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/checker/op_checker.py)

흐름:

1. model load
2. before unsupported summary
3. ordered passes 실행
4. onnx checker
5. after unsupported summary
6. unsupported 남으면 실패
7. save

즉 이 시스템은:

> legality-first rewrite pipeline

이다.

---

## 5.1 `Folder`가 하는 일

현재 모든 pass의 기반은 [folder.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/folder.py)다.

`prepare(model)`를 호출하면 아래가 준비된다.

- `self.model`
- `self.graph`
- `self.log`
- `self.init_map`
- `self.shape_info`
- `self.nodes_to_remove`
- `self.producer_by_output`
- `self.consumers_by_input`

즉 각 pass는 “ONNX graph를 읽고 수정하기 위한 작업장”을 `Folder`가 먼저 만들어주는 구조다.

이게 중요한 이유:

- pass마다 반복되는 boilerplate를 줄여준다
- state를 일관되게 유지한다
- helper 함수 이름을 통일할 수 있다

---

## 5.2 pass를 읽는 순서

코드를 읽을 때는 아래 순서로 보면 된다.

1. 이 pass가 어느 `op_type`을 잡는가?
2. `_build_context`나 `_plan`이 무엇을 모으는가?
3. static path와 dynamic path가 어떻게 나뉘는가?
4. `_emit_*`가 어떤 node list를 만드는가?
5. 마지막 output tensor name이 원래 output name과 어떻게 이어지는가?
6. 어떤 이유로 rewrite를 포기하는가?

암기 포인트:

> ONNX rewrite 코드는 “수집 -> 판단 -> 생성 -> 교체” 순서로 읽으면 된다.

---

# 6. BatchNormalization 제거

관련 파일:

- [passes/rewrite_bn.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/rewrite_bn.py)

## 6.1 BN 수식

BN은:

```text
y = (x - mean) / sqrt(var + eps) * scale + bias
```

로 정의된다.

이걸 정리하면:

```text
scale_factor = scale / sqrt(var + eps)
bias_factor  = bias - mean * scale_factor
y = x * scale_factor + bias_factor
```

즉 BN은 “채널별 affine transform”이다.

parameter shape:

```text
scale, bias, mean, var: [C]
```

여기서 `C`는 Conv 뒤 BN이면 사실상 `C_out`이다.

---

## 6.2 `BnNodeContext`

현재 BN pass는 context를 dataclass로 들고 다닌다.

```python
@dataclass(frozen=True)
class BnNodeContext:
    prefix: str
    node: onnx.NodeProto
    input_name: str
    output_name: str
    scale: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    eps: float
    pred: onnx.NodeProto | None
    pred_consumers: list[onnx.NodeProto]
```

이 구조 덕분에:

- BN parameter
- input/output tensor name
- predecessor 정보
- epsilon

을 한 번만 수집하면 된다.

이건 `nota-test` 스타일보다 확실히 읽기 쉽다.

---

## 6.3 BN alone -> depthwise 1x1 Conv

BN-alone path의 의미를 shape까지 써서 보자.

입력:

```text
x: [N, C, H, W]
```

BN 결과:

```text
y[:, c, h, w] = x[:, c, h, w] * scale_factor[c] + bias_factor[c]
```

이건 channel-wise 연산이고, 채널 간 mixing이 없다.

따라서 depthwise 1x1 Conv와 완전히 같다.

weight / bias:

```text
weight: [C, 1, 1, 1]
bias:   [C]
group:  C
```

왜 `group=C`인가?

- 각 output channel이 자기 input channel 하나만 보게 하려고
- BN은 cross-channel mixing이 없기 때문

현재 코드 핵심:

```python
channel_count = int(scale_factor.reshape(-1).shape[0])
conv_weight = scale_factor.reshape(channel_count, 1, 1, 1).astype(np.float32)
conv_bias = bias_factor.reshape(channel_count).astype(np.float32)
```

그리고:

```python
helper.make_node(
    cons.OP_CONV,
    [context.input_name, conv_weight_name, conv_bias_name],
    [context.output_name],
    name=self.node_name(context.prefix, "conv"),
    kernel_shape=[1, 1],
    group=channel_count,
)
```

이렇게 BN node를 Conv 하나로 바꾼다.

---

## 6.4 Conv-BN fusion

이건 BN-alone보다 더 중요하다.

Conv 출력:

```text
z[o, h, w] = sum_i x[i, ...] * W[o, i, ...] + b[o]
```

그 뒤 BN:

```text
y[o, h, w] = z[o, h, w] * scale_factor[o] + bias_factor[o]
```

합치면:

```text
y[o, h, w]
= sum_i x[i, ...] * (W[o, i, ...] * scale_factor[o])
 + (b[o] * scale_factor[o] + bias_factor[o])
```

즉:

```text
fused_weight[o, ...] = weight[o, ...] * scale_factor[o]
fused_bias[o] = conv_bias[o] * scale_factor[o] + bias_factor[o]
```

### 왜 `C_out` 기준인가

Conv 뒤 BN이면 BN이 보는 텐서는 Conv output이다.

Conv weight shape:

```text
[C_out, C_in / group, kH, kW]
```

여기서 output channel 축은 axis 0이다.

BN scale은 output channel마다 따로 있으므로, axis 0에 맞춰 곱해야 한다.

즉:

```python
scale_factor.reshape(C_out, 1, 1, 1)
```

처럼 reshape해서 곱해야 한다.

### group Conv에서는?

원래 weight:

```text
[C_out, C_in/group, kH, kW]
```

group 구조를 명시적으로 꺼내면:

```text
[group, C_out/group, C_in/group, kH, kW]
```

이렇게 된다.

여기서 BN scale `[C_out]`도:

```text
[group, C_out/group]
```

로 reshape하면 각 group의 output channel에 정확히 곱할 수 있다.

즉 이 부분의 핵심은:

> 원래 안 보이던 group 축을 reshape로 명시적으로 꺼내서, output-channel-per-group 축에 scale을 곱하는 것

---

## 6.5 ConvTranspose-BN fusion

ConvTranspose는 Conv와 거의 같지만 weight shape이 다르다.

ConvTranspose weight shape:

```text
[C_in, C_out / group, kH, kW]
```

즉 output channel 축이 axis 1이다.

Conv에서는 axis 0이었는데, 여기서는 axis 1이다.

그래서 BN scale은:

```text
fused_weight[:, o, ...] = weight[:, o, ...] * scale_factor[o]
```

처럼 곱해야 한다.

group 구조를 명시적으로 꺼내면:

```text
[group, C_in/group, C_out/group, kH, kW]
```

가 된다.

여기서 scale을 `[group, C_out/group]`로 reshape해서 axis 2에 맞춰 broadcast한다.

암기:

- `Conv`: output 축 axis 0
- `ConvTranspose`: output 축 axis 1

이게 BN fusion에서 제일 중요한 차이다.

---

# 7. Gather 제거

관련 파일:

- [passes/rewrite_gather.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/rewrite_gather.py)

현재 rewrite 중 가장 설명이 길어야 하는 pass다.

왜냐면 Gather는 패턴이 여러 개고, static / dynamic 차이가 크기 때문이다.

## 7.1 `GatherNodeContext`

```python
@dataclass(frozen=True)
class GatherNodeContext:
    prefix: str
    axis: int
    data_name: str
    index_name: str
    output_name: str
    data_array: np.ndarray | None
    index_array: np.ndarray | None
```

이 context는:

- Gather axis
- data / indices / output name
- data가 initializer인지
- indices가 initializer인지

를 한 번에 묶어준다.

이게 중요한 이유는 Gather가 아래처럼 case split이 많기 때문이다.

1. static gather
2. scalar index gather
3. vocab gather
4. dynamic axis-0 gather

---

## 7.2 Gather semantics를 concrete example로 이해하기

예를 들어:

```text
data shape    = [V, H]
indices shape = [B, S]
axis = 0
```

이면:

```text
Gather(data, indices, axis=0)
```

결과 shape는:

```text
[B, S, H]
```

즉 각 token id가 embedding row 하나를 고른다.

이걸 embedding lookup이라고 보면 된다.

---

## 7.3 static gather

`data`와 `indices`가 둘 다 static이면:

```python
folded = np.take(data_array, index_array.astype(np.int64), axis=context.axis)
```

로 접을 수 있다.

즉:

```text
Gather -> initializer
```

가 된다.

이건 가장 쉬운 case다.

---

## 7.4 scalar-index gather -> Slice + Reshape

`indices`가 scalar initializer면 Gather는 특정 축의 한 위치를 뽑는 것과 같다.

예:

```text
data shape   = [B, S, H]
axis         = 1
index        = 2
Gather(data, 2, axis=1)
output shape = [B, H]
```

이걸 두 단계로 본다.

1. `Slice`  
   `axis=1`에서 `[2:3]`만 남김  
   결과 shape: `[B, 1, H]`

2. `Reshape`  
   gathered axis를 제거  
   결과 shape: `[B, H]`

즉:

```text
Gather -> Slice + Reshape
```

왜 shape 계산과 value 계산을 분리해서 봐야 하는지 여기서 잘 보인다.

- `Slice`는 값 선택
- `Reshape`는 rank/shape 정리

---

## 7.5 embedding-style small vocab gather

조건:

- `axis == 0`
- `data_array is not None`
- vocab size가 작음

예:

```text
data shape    = [V, H]
indices shape = [B, S]
output shape  = [B, S, H]
```

아이디어:

각 vocab id `k`에 대해:

1. `Equal(indices, k)`  
   shape:

```text
[B, S] == [] -> [B, S]
```

2. `Cast` -> float mask  
   shape `[B, S]`

3. `Unsqueeze(-1)`  
   shape `[B, S, 1]`

4. embedding row `data[k]`와 `Mul`  
   shape:

```text
[B, S, 1] * [H] -> [B, S, H]
```

5. 모든 row contribution을 `Add`

즉:

```text
Gather(data, indices)
==
sum_k one_hot_like(indices == k) * embedding_row[k]
```

현재 코드는 이걸 `_emit_small_vocab_chain()`으로 만든다.

---

## 7.6 chunked gather

vocab이 크면 row를 하나씩 만드는 graph가 너무 커진다.

그래서 vocab을 chunk 단위로 나눈다.

예를 들어:

```text
V = 30000
chunk_size = 256
```

이면:

```text
chunk 0: [0..255]
chunk 1: [256..511]
...
```

각 chunk마다:

1. chunk range tensor 생성
2. `Equal(index_vector, range)`
3. `Cast`
4. `Unsqueeze`
5. chunk weight와 `Mul`
6. chunk axis에 `ReduceSum`

을 수행한다.

중간 tensor shape를 예로 들면:

```text
indices              : [B, S]
Unsqueeze(-1)        : [B, S, 1]
range                : [chunk]
Equal                : [B, S, chunk]
Cast                 : [B, S, chunk]
Unsqueeze(-1)        : [B, S, chunk, 1]
chunk_weight         : [chunk, H]
Mul                  : [B, S, chunk, H]
ReduceSum(axis=chunk): [B, S, H]
```

이렇게 각 chunk contribution을 만들고, 마지막에 chunk 결과들을 `Add`로 누적한다.

이 설명이 중요한 이유는 `todo.md`에 있던 다음 문장을 코드로 연결해 주기 때문이다.

> chunked gather는 `[B, S, C, H]` 중간 표현을 만들고 chunk axis를 줄여 `[B, S, H]`로 복원한다.

---

## 7.7 dynamic axis-0 gather

이건 가장 중요하다.

조건:

- `axis == 0`
- `data`는 dynamic tensor
- `indices`는 runtime tensor

이 경우 static folding이 불가능하므로, 수학적 동치 변환을 해야 한다.

### 핵심 아이디어

```text
Gather(data, indices, axis=0)
==
MatMul(one_hot(indices), data)
```

정확히는:

1. indices를 one-hot-like mask로 바꾼다
2. data를 2D로 flatten한다
3. `MatMul`
4. 원래 output shape로 `Reshape`

### 단계별로 보기

가정:

```text
data shape    = [V, H]
indices shape = [B, S]
output shape  = [B, S, H]
```

#### Step 0. range 생성

```text
range = [0, 1, ..., V-1]
shape = [V]
```

#### Step 1. indices unsqueeze

```text
indices           : [B, S]
indices_expanded  : [B, S, 1]
```

#### Step 2. Equal

```text
[B, S, 1] == [V] -> [B, S, V]
```

이게 one-hot-like mask다.

#### Step 3. Cast

```text
[B, S, V] bool -> [B, S, V] float
```

#### Step 4. mask flatten

```text
[B, S, V] -> [B*S, V]
```

#### Step 5. data flatten

```text
[V, H] -> [V, H]
```

혹은 suffix가 더 크면:

```text
[V, ...] -> [V, -1]
```

#### Step 6. MatMul

```text
[B*S, V] @ [V, H] -> [B*S, H]
```

#### Step 7. indices shape 읽기

```text
Shape(indices) -> [B, S]
```

그리고 suffix shape `[H]`를 붙여:

```text
[B, S] + [H] -> [B, S, H]
```

#### Step 8. Reshape

```text
[B*S, H] -> [B, S, H]
```

즉 dynamic Gather는:

```text
Equal + Cast + Reshape + MatMul + Shape + Concat + Reshape
```

로 바뀐다.

그리고 여기서 끝이 아니다.

> 이 `MatMul`은 다시 `RewriteMatmul`이 처리한다.

즉:

```text
Gather -> MatMul -> RewriteMatmul
```

이 chained rewrite가 pipeline의 핵심이다.

---

# 8. Pow 제거

관련 파일:

- [passes/rewrite_pow.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/rewrite_pow.py)

## 8.1 왜 Pow를 없애는가

`Pow`는 현재 supported set에 없다.

하지만 많은 모델에서:

- `x^2`
- `x^0.5`
- `x^-1`

같은 단순 형태로 자주 나온다.

그래서 exponent가 scalar initializer이면 supported op 조합으로 바꾸기 쉽다.

---

## 8.2 `PowNodeContext`

```python
@dataclass(frozen=True)
class PowNodeContext:
    prefix: str
    base_name: str
    exponent_name: str
    output_name: str
    exponent: float | None
```

여기서 핵심은 `exponent`가 미리 scalar constant인지 계산해 두는 것이다.

```python
def _get_scalar_exponent(self, exponent_name: str) -> float | None:
    exponent = self.init_map.get(exponent_name)
    if exponent is None or exponent.size != 1:
        return None
    return float(exponent.reshape(-1)[0])
```

즉:

- exponent가 initializer scalar면 rewrite 가능
- 아니면 dynamic exponent라서 keep

---

## 8.3 special-case rewrite

### `x^1`

identity다.

그래서 Pow node를 지우고 output name을 input name으로 redirect한다.

### `x^0.5`

```text
Pow(x, 0.5) == Sqrt(x)
```

### `x^-1`

```text
Pow(x, -1) == 1 / x
```

즉 `Div(1, x)`.

### `x^-0.5`

```text
Pow(x, -0.5) == 1 / Sqrt(x)
```

즉 `Sqrt + Div`.

### `x^2`, `x^3`, `x^4`

반복 `Mul`.

예:

```text
x^3 = (x * x) * x
```

이 rewrite는 구현은 간단하지만, unsupported set 닫는 데 중요하다.

---

# 9. MatMul 제거

관련 파일:

- [passes/rewrite_matmul.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/rewrite_matmul.py)

MatMul은 현재 transformer 계열 legality의 핵심 blocker다.

## 9.1 MatMul semantics

일반 수식:

```text
[... , M, K] @ [..., K, N] -> [..., M, N]
```

가장 중요한 직관:

> MatMul은 “마지막 두 축에서 K축을 따라 곱하고 더하는 연산”이다.

즉:

```text
output[..., m, n] = sum_k A[..., m, k] * B[..., k, n]
```

---

## 9.2 dynamic MatMul -> Mul + ReduceSum

둘 다 runtime tensor이면 Conv로 바꾸기 어렵다.

그래서 수학식 그대로 푼다.

예:

```text
A shape = [..., M, K]
B shape = [..., K, N]
```

1. A 마지막에 축 추가

```text
A -> [..., M, K, 1]
```

2. B에서 K 앞에 축 추가

```text
B -> [..., 1, K, N]
```

3. `Mul`

```text
[..., M, K, N]
```

4. K 축에 `ReduceSum`

```text
[..., M, N]
```

즉:

```text
MatMul -> Unsqueeze + Unsqueeze + Mul + ReduceSum
```

이게 `todo.md`에 적혀 있던:

> `ReduceMean * C`보다 `ReduceSum`이 직접적이고 현재 supported op set에 더 맞다

와 연결된다.

---

## 9.3 static weight MatMul -> Conv chain

`X @ W`에서 `W`가 static이면 1x1 Conv로 바꾸기 좋다.

왜냐면:

```text
W shape = [K, N]
```

를 Conv weight:

```text
[N, K, 1, 1]
```

로 볼 수 있기 때문이다.

다만 activation rank에 따라 layout 조작이 필요하다.

### rank 2 path

예:

```text
X shape = [M, K]
W shape = [K, N]
```

중간 과정:

1. `Reshape` -> `[1, K, M, 1]`
2. `Conv`
3. `Transpose`
4. `Reshape` -> `[M, N]`

### rank 3 path

예:

```text
X shape = [B, M, K]
```

중간 과정:

1. `Transpose` -> `[B, K, M]`
2. `Unsqueeze` -> `[B, K, M, 1]`
3. `Conv`
4. `Transpose`
5. `Reshape` -> `[B, M, N]`

### rank 4 path

예:

```text
X shape = [B, H, M, K]
```

중간 과정:

1. `Reshape`로 `[B*H, M, K]` 또는 equivalent rank-3 view
2. rank-3 Conv chain 수행
3. 다시 `[B, H, M, N]`로 복원

즉 rank 4 path는 결국 rank 3 path를 재사용하는 구조다.

---

## 9.4 left-static path

`W @ X`에서는 static tensor가 왼쪽에 있다.

이 경우 바로 Conv로 가기 어렵다.

그래서 아이디어는:

1. activation을 transpose
2. 내부적으로 right-static처럼 처리
3. output을 다시 transpose

즉:

> left-static MatMul은 transpose해서 right-static 문제로 바꾼다

로 이해하면 된다.

---

# 10. Gemm 제거

관련 파일:

- [passes/rewrite_gemm.py](/Users/hdy/code/portfolio/Gawee/front/onnx_rewrite/passes/rewrite_gemm.py)

## 10.1 Gemm semantics

`Gemm`은 대체로:

```text
alpha * A @ B + beta * C
```

형태다.

여기서 봐야 하는 attribute:

- `alpha`
- `beta`
- `transA`
- `transB`

그리고 optional bias input.

## 10.2 Conv chain으로 바꾸는 이유

`resnet18`의 마지막 FC 같은 경우 `Gemm`가 legality blocker다.

이걸 1x1 Conv chain으로 바꾸면 supported op set 안으로 내릴 수 있다.

핵심 과정:

1. weight 해석 (`transB`, `alpha`)
2. bias 해석 (`beta`)
3. 필요한 transpose (`transA`)
4. activation reshape / transpose
5. `Conv`
6. output reshape 복원

Gemm rewrite를 외울 때는:

> “Fully-connected를 1x1 Conv로 본다”

로 기억하면 된다.

---

# 11. ORT correctness / benchmark를 어떻게 해석할 것인가

이 부분도 중요하다.

## 11.1 structural legality

먼저 봐야 할 것:

1. graph가 `onnx.checker.check_model`을 통과하는가?
2. unsupported op가 남아 있는가?

이건 “구조가 합법적인가?”다.

## 11.2 ORT execution

그 다음:

1. `InferenceSession`이 열리는가?
2. `session.run`이 되는가?

이건 “실행 가능한가?”다.

## 11.3 correctness metric

그 다음:

1. 원본/변환 output shape가 같은가?
2. output 개수가 같은가?
3. `max_abs_diff`가 tolerance 이하인가?

이건 “수치적으로 충분히 같은가?”다.

이 순서를 외워야 한다.

> legality -> execution -> correctness

---

# 12. 지금 외워야 할 핵심 코드/공식

## 12.1 BN factor

```text
scale_factor = scale / sqrt(var + eps)
bias_factor  = bias - mean * scale_factor
```

## 12.2 BN-alone -> Conv

```text
weight shape = [C, 1, 1, 1]
bias shape   = [C]
group        = C
```

## 12.3 Conv-BN fusion

```text
Conv weight shape = [C_out, C_in/group, kH, kW]
fused_weight[o, ...] = weight[o, ...] * scale_factor[o]
fused_bias[o] = conv_bias[o] * scale_factor[o] + bias_factor[o]
```

## 12.4 ConvTranspose-BN fusion

```text
ConvTranspose weight shape = [C_in, C_out/group, kH, kW]
fused_weight[:, o, ...] = weight[:, o, ...] * scale_factor[o]
fused_bias[o] = conv_bias[o] * scale_factor[o] + bias_factor[o]
```

## 12.5 Gather(axis=0)

```text
Gather(data, indices)
==
Reshape(MatMul(one_hot_like(indices), flatten(data)), output_shape)
```

## 12.6 dynamic MatMul

```text
MatMul(A, B)
==
ReduceSum(Unsqueeze(A) * Unsqueeze(B), axis=K)
```

## 12.7 Pow

```text
x^0.5  = Sqrt(x)
x^-1   = 1 / x
x^-0.5 = 1 / Sqrt(x)
x^n    = repeated Mul
```

## 12.8 Gemm

```text
Gemm ≈ FC
FC ≈ 1x1 Conv
```

---

# 13. 마지막으로: 이 코드를 읽는 습관

앞으로 설명할 때는 항상 아래처럼 해야 한다.

1. node 설명은 `input tensor names -> output tensor names`로 먼저 푼다.
2. 중간 tensor shape를 단계별로 쓴다.
3. broadcasting이 일어나는 정확한 shape pair를 쓴다.
4. static rewrite와 dynamic rewrite를 구분한다.
5. shape 계산과 값 계산을 분리해서 설명한다.
6. chained rewrite는 pipeline 관점으로 설명한다.

즉 “대충 이런 느낌”으로 보면 안 되고, 항상:

```text
이 tensor가 어디서 왔고
shape가 뭐고
다음 op에서 어떻게 broadcast되고
어떤 output name으로 이어지는지
```

까지 써야 한다.

이 습관을 들이면 ONNX graph rewrite는 훨씬 빨리 익숙해진다.
