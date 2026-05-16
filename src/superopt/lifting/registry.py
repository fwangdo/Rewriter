"""Lifting pattern registry filtered by target contract.

Given a supported op set, activates only the lifting patterns
for ops in that set. Drives the feasibility check:
"can this primitive subtree be expressed as a legal ONNX op?"
"""
