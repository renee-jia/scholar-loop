import pytest


@pytest.fixture(autouse=True)
def _canned_papers_pass_the_relevance_floor(monkeypatch):
    """Canned feeds return the same papers for any query, so the lexical relevance floor is off
    by default in tests. Tests of the floor itself pass `floor=` explicitly."""
    from scholarloop.ideation import litscout
    monkeypatch.setattr(litscout, "RELEVANCE_FLOOR", 0.0)
