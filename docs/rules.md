# Shared Rewrite Rules

baseline과 superopt가 공유하는 rewrite 규칙 목록.
규칙은 `src/common/rules/`에 정의되며, 양쪽 모두 `get_all_specs()`로 로드한다.

## Legalization (34 rules)

| Name | Source → Target | Checks | Description |
|------|----------------|--------|-------------|
| eliminate_identity | Identity(x) → x | — | Identity 노드 제거 |
| greater_to_less | Greater(a, b) → Less(b, a) | — | Greater를 Less로 치환 |
| sub_to_add_neg | Sub(x, y) → Add(x, Neg(y)) | — | Sub를 Add+Neg로 분해 |
| neg_to_mul | Neg(x) → Mul(x, -1) | — | Neg를 Mul로 치환 |
| squeeze_to_reshape | Squeeze(x, axes) → Reshape(x, shape) | x has_shape | Squeeze를 Reshape로 치환 |
| unsqueeze_to_reshape | Unsqueeze(x, axes) → Reshape(x, shape) | x has_shape | Unsqueeze를 Reshape로 치환 |
| pow_to_identity | Pow(x, 1) → x | e scalar_close=1.0 | Pow(x,1) 제거 |
| pow_to_sqrt | Pow(x, 0.5) → Sqrt(x) | e scalar_close=0.5 | Pow를 Sqrt로 치환 |
| pow_to_mul | Pow(x, 2) → Mul(x, x) | e scalar_close=2.0 | Pow를 Mul로 치환 |
| pow_to_cube | Pow(x, 3) → Mul(Mul(x, x), x) | e scalar_close=3.0 | Pow를 세제곱으로 분해 |
| pow_to_reciprocal | Pow(x, -1) → Div(1, x) | e scalar_close=-1.0 | Pow를 역수로 치환 |
| pow_to_rsqrt | Pow(x, -0.5) → Div(1, Sqrt(x)) | e scalar_close=-0.5 | Pow를 역제곱근으로 치환 |
| layernorm_decompose | LayerNormalization(x, s, b) → ReduceMean+Sub+Mul+Sqrt+Div+Add | — | LayerNorm 분해 |
| where_mask_decompose | Where(cond, true, false) → Mul((1-Cast(cond)), false) | true scalar_abs_lt=1e-8, false scalar_lte=-1e30 | 마스크 패턴 Where 분해 |
| where_to_arithmetic | Where(cond, A, B) → Cast(cond)*A + (1-Cast(cond))*B | — | Where를 산술 연산으로 분해 |
| range_decompose | Range(start, limit, step) → Slice(arange_table, 0, limit) | start scalar_close=0.0, step scalar_close=1.0 | Range를 Slice 룩업으로 치환 |
| bn_decompose | BatchNormalization(x, s, b, m, v) → Mul(x, scale) + bias | — | BatchNorm 분해 |
| gemm_decompose | Gemm(a, w, b) → MatMul(a, w') + b' | — | Gemm를 MatMul+Add로 분해 |
| gemm_decompose_no_bias | Gemm(a, w) → MatMul(a, w') | — | bias 없는 Gemm를 MatMul로 분해 |
| matmul_to_conv | MatMul(a, w) → Conv(Reshape(a), w') | w is_constant | MatMul을 Conv로 변환 |
| shape_fold | Shape(x) → constant | x has_shape | 정적 Shape를 상수로 폴딩 |
| constantofshape_fold | ConstantOfShape(shape) → constant | shape is_constant | ConstantOfShape를 상수로 폴딩 |
| flatten_to_reshape | Flatten(x) → Reshape(x, [pre, post]) | x has_shape | Flatten을 Reshape로 치환 |
| expand_to_mul_ones | Expand(x, shape) → Mul(x, ones) | shape is_constant | Expand를 Mul 브로드캐스트로 치환 |
| cos_fold | Cos(x) → constant | x is_constant | 상수 Cos 폴딩 |
| sin_fold | Sin(x) → constant | x is_constant | 상수 Sin 폴딩 |
| pad_eliminate_zero | Pad(x, pads) → x | pads is_constant (all zero) | 제로 패딩 제거 |
| equal_fold | Equal(a, b) → constant | a, b is_constant | 상수 Equal 폴딩 |
| less_fold | Less(a, b) → constant | a, b is_constant | 상수 Less 폴딩 |
| not_to_sub | Not(x) → Sub(1, Cast(x)) | — | Not을 Sub로 치환 |
| abs_decompose | Abs(x) → Relu(x) + Relu(-x) | — | Abs를 Relu로 분해 |
| reciprocal_to_div | Reciprocal(x) → Div(1, x) | — | Reciprocal을 Div로 치환 |
| ceil_fold | Ceil(x) → constant | x is_constant | 상수 Ceil 폴딩 |
| floor_fold | Floor(x) → constant | x is_constant | 상수 Floor 폴딩 |

## Arithmetic (4 rules)

| Name | Source → Target | Checks | Description |
|------|----------------|--------|-------------|
| add_comm | Add(x, y) → Add(y, x) | — | 덧셈 교환법칙 |
| mul_comm | Mul(x, y) → Mul(y, x) | — | 곱셈 교환법칙 |
| add_assoc_right | Add(Add(x, y), z) → Add(x, Add(y, z)) | — | 덧셈 결합법칙 (우결합) |
| mul_assoc_right | Mul(Mul(x, y), z) → Mul(x, Mul(y, z)) | — | 곱셈 결합법칙 (우결합) |

## Layout (3 rules)

| Name | Source → Target | Checks | Description |
|------|----------------|--------|-------------|
| reshape_reshape | Reshape(Reshape(x, y), z) → Reshape(x, z) | — | 연속 Reshape 축소 |
| transpose_cancel_perm_0_1 | Transpose(Transpose(x, [0,1]), [0,1]) → x | perm=(0,1) | 동일 perm Transpose 상쇄 |
| transpose_cancel_perm_1_0 | Transpose(Transpose(x, [1,0]), [1,0]) → x | perm=(1,0) | 역 perm Transpose 상쇄 |

## Fusion (1 rule)

| Name | Source → Target | Checks | Description |
|------|----------------|--------|-------------|
| bias_add_commute | Add(MatMul(x, w), b) → Add(b, MatMul(x, w)) | — | bias Add 순서 교환 (fusion 준비) |
