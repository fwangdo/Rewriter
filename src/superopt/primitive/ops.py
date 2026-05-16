"""Primitive op class definitions for the low-level IR.

Each ONNX op is lowered into a composition of these primitives.
~20-30 frozen dataclasses (ReduceSum, BroadcastMul, Reshape, ...),
designed for Python match/case pattern matching.
"""
