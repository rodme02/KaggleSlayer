"""Trusted side of the harness: CV, registries, sandbox, telemetry.

Contains no LLM calls. Owns the parts of the pipeline whose correctness
must not be left to LLM judgment (leak-free CV, metric scoring, etc.).
"""
