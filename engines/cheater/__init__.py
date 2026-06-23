"""Adversarial test fixture: an engine whose train.py tries to fabricate a perfect metric.

Exists only to prove the frozen-scoring guard: the runner ignores train.py's
stdout and trusts only the frozen scorer, so the fake never wins. Not a real domain.
"""
