"""Tree pattern matcher: primitive op subtree -> ONNX op recognition.

Traverses e-classes to match multi-node patterns (depth 2-3).
Patterns are derived from lowering definitions (reversed).
Analogous to LLVM instruction selection.
"""
