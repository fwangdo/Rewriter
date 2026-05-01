# CPU Budget Quiz

Fill in the missing parts.

## 1. Session options

```python
import onnxruntime as ort

def make_session(model_path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = __1__
    options.inter_op_num_threads = __2__
    options.execution_mode = ort.ExecutionMode.__3__
    return ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
```

## 2. Why this matters

Complete the sentence:

> The correctness harness should stay low-CPU because `__4__`.

## 3. Workflow

Put these steps in the correct order:

- `run one validation case`
- `build the ORT session`
- `generate inputs`
- `compare before/after outputs`

Write the order as numbers:

1. `__5__`
2. `__6__`
3. `__7__`
4. `__8__`

## 4. Short answer

Why is `ORT_SEQUENTIAL` useful in a local correctness run?

Answer in one sentence: `__9__`
