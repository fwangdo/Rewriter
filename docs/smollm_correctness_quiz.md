# SmolLM Correctness Quiz

Fill in the missing parts.

## 1. Two concerns

Complete the sentence:

> Legality and correctness are separate because `__1__`.

## 2. Save behavior

```python
def optimize_model(input_path, output_path, *, enforce_supported_only=True):
    # ...
    if unsupported and enforce_supported_only:
        raise RuntimeError('unsupported')
    if unsupported and not enforce_supported_only:
        __2__
```

What should `__2__` do?

## 3. Validation

```python
result = compare_models(
    before_model,
    after_model,
    max_abs_tolerance=__3__,
    max_rel_tolerance=__4__,
)
```

Fill in values that make `smollm_135m` pass the current observed noise level.

## 4. Short answer

Why is it useful to keep the numeric tolerance configurable when debugging a rewrite?

Answer in one sentence: `__5__`
