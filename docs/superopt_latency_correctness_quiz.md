# Superopt Latency Correctness Quiz

Fill in the missing parts.

## 1. ONNX Reshape rule

ONNX `Reshape` uses `-1` to mean:

```text
TODO: write the meaning of -1 in a Reshape shape tensor.
```

A valid ONNX `Reshape` shape tensor may contain at most:

```text
TODO: write the maximum number of -1 dimensions.
```

## 2. Why metadata is not enough

In the IR, an unknown symbolic dimension may be represented as `-1`.

Explain why this is safe as shape metadata but dangerous when copied directly
into a `Reshape` initializer:

```text
TODO: explain the difference between metadata and executable operator input.
```

## 3. Implement the guard

Complete the helper:

```python
def is_valid_reshape_template(shape: tuple[int, ...]) -> bool:
    inferred_dims = TODO
    return TODO
```

Expected behavior:

```python
assert is_valid_reshape_template((1, 1, -1))
assert is_valid_reshape_template((-1, 1, 1))
assert not is_valid_reshape_template((-1, 1, 1, -1))
assert not is_valid_reshape_template((1, -1, -1))
```

## 4. Apply the guard

Complete the rewrite logic:

```python
def apply_unsqueeze_to_reshape(egraph, match_cid, subst):
    x_cid = subst["?x"]
    target_shape = egraph.eclass(match_cid).data.shape

    if target_shape is None:
        return TODO

    if not is_valid_reshape_template(target_shape):
        return TODO

    shape_cid = add_shape_constant(egraph, target_shape)
    return egraph.add(ENode("Reshape", (TODO, TODO)))
```

## 5. Correctness gate

Latency-first extraction can choose the cheapest candidate in the e-graph.
What still must be checked before reporting a speedup?

```text
TODO: list the required post-extraction checks.
```
