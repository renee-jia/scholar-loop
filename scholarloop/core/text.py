"""Tiny lexical helpers shared by the deterministic parts of the loop (skill ranking, novelty
checks). No embedding model: everything here is offline, cheap, and reproducible."""

from __future__ import annotations

import re

_STOP = {"the", "a", "an", "to", "of", "and", "or", "is", "in", "on", "for", "with", "at", "by",
         "be", "it", "this", "that", "from", "as", "are", "was", "not", "but", "than", "then",
         "when", "its", "into", "rather", "use", "using", "dont", "do", "you", "your", "can",
         "we", "our", "via", "based", "new", "novel", "approach", "method", "methods", "paper",
         "propose", "proposed", "results", "show", "model", "models"}


def terms(text: str) -> set[str]:
    """Lower-cased alphanumeric tokens longer than two chars, minus stop words."""
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in _STOP}


def containment(query: str, doc: str) -> float:
    """Fraction of the query's terms that also appear in the document (0 when the query is empty).
    Asymmetric on purpose: a short idea sitting entirely inside a long abstract scores high."""
    q = terms(query)
    if not q:
        return 0.0
    return len(q & terms(doc)) / len(q)


def jaccard(a: str, b: str) -> float:
    """Symmetric term overlap — the right measure when both texts are short (claim vs claim)."""
    ta, tb = terms(a), terms(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
