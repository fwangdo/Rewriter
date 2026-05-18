"""Rule discovery loop: the main orchestrator.

illegal op -> lowering -> e-graph saturation -> lifting -> rule registration -> verification.
Repeats until no illegal ops remain or no new rules are found.
"""
