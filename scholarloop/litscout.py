"""Lit Scout (DESIGN §4 L3) — literature grounding for the Reasoner.

Two cleanly separated halves, per the harness rule "retrieval is a deterministic tool,
extraction is an agent":
  - retrieval clients (`ArxivClient`, `OpenAlexClient`, `SemanticScholarClient`) — plain HTTP
    clients, no LLM. Each exposes `.search(topic, max_results) -> list[Paper]` and an injectable
    fetcher, so tests run against canned payloads with no network. The Lit Scout queries several
    sources, merges and de-duplicates them, and ranks by citation count so the highest-impact
    techniques surface first (impact grounding).
  - `LitScout(Agent)` — turns the retrieved papers into structured, testable `findings`
    (technique + source + predicted_effect) via the agent harness.

`scout(topic)` returns `(lit_context, lit_priors)`: a prose block the Reasoner reads, and a
list of "technique (source)" priors fed into the search-space analysis. This is what makes
ideas literature-grounded instead of blind local hill-climbing.
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from scholarloop.agent import Agent

_ATOM = "{http://www.w3.org/2005/Atom}"
_UA = "scholarloop/0.1 (https://github.com/renee-jia/scholar-loop)"


def _ssl_ctx() -> ssl.SSLContext | None:
    """certifi's CA bundle when available (some Python installs lack a usable system store)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def _http_get(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
        return r.read().decode("utf-8")


@dataclass
class Paper:
    title: str
    summary: str
    arxiv_id: str            # the citation id shown to the agent (arXiv id, DOI, or source id)
    url: str
    citations: int | None = None   # impact signal for ranking; None when the source omits it
    code_url: str | None = None    # populated only when a source actually links code


# --------------------------------------------------------------------------------------------
# Retrieval clients — one per source, all returning Paper. Fetchers are injectable for tests.
# --------------------------------------------------------------------------------------------
class ArxivClient:
    """Deterministic arXiv search. `fetcher(query, max_results) -> atom_xml` is injectable."""

    name = "arxiv"

    def __init__(self, fetcher=None, *, timeout: float = 20.0):
        self._fetch = fetcher or self._http_fetch
        self.timeout = timeout

    def _http_fetch(self, query: str, max_results: int) -> str:  # pragma: no cover - network
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": query, "max_results": max_results,
            "sortBy": "submittedDate", "sortOrder": "descending",
        })
        return _http_get(url, self.timeout)

    def search(self, topic: str, max_results: int = 5) -> list[Paper]:
        xml = self._fetch(f"all:{topic}", max_results)
        root = ET.fromstring(xml)
        papers: list[Paper] = []
        for e in root.findall(_ATOM + "entry"):
            id_text = (e.findtext(_ATOM + "id") or "").strip()
            papers.append(Paper(
                title=" ".join((e.findtext(_ATOM + "title") or "").split()),
                summary=" ".join((e.findtext(_ATOM + "summary") or "").split()),
                arxiv_id=id_text.rsplit("/", 1)[-1],
                url=id_text,
            ))
        return papers


def _reconstruct_abstract(inverted: dict | None) -> str:
    """OpenAlex ships abstracts as an inverted index {word: [positions]}; rebuild the prose."""
    if not inverted:
        return ""
    slots: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for p in positions:
            slots.append((p, word))
    slots.sort()
    return " ".join(w for _, w in slots)


def _openalex_cite_id(work: dict) -> tuple[str, str]:
    """Best (citation_id, url) for an OpenAlex work: prefer a detectable arXiv id, then DOI,
    then the OpenAlex short id."""
    locations = (work.get("locations") or []) + [work.get("primary_location") or {}]
    for loc in locations:
        landing = (loc or {}).get("landing_page_url") or ""
        if "arxiv.org/abs/" in landing:
            return "arXiv:" + landing.split("arxiv.org/abs/", 1)[1].split("v")[0].strip("/"), landing
    ids = work.get("ids") or {}
    if ids.get("doi"):
        return ids["doi"].replace("https://doi.org/", "doi:"), ids["doi"]
    oa = (ids.get("openalex") or work.get("id") or "")
    return ("openalex:" + oa.rsplit("/", 1)[-1]) if oa else "openalex:unknown", oa


class OpenAlexClient:
    """OpenAlex works search — reliable, no key needed, and carries citation counts.
    `fetcher(query, max_results) -> json_text` is injectable."""

    name = "openalex"

    def __init__(self, fetcher=None, *, timeout: float = 20.0, mailto: str = "reneejia368@gmail.com"):
        self._fetch = fetcher or self._http_fetch
        self.timeout = timeout
        self.mailto = mailto

    def _http_fetch(self, query: str, max_results: int) -> str:  # pragma: no cover - network
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
            "search": query, "per-page": max_results,
            "sort": "cited_by_count:desc", "mailto": self.mailto,
        })
        return _http_get(url, self.timeout)

    def search(self, topic: str, max_results: int = 5) -> list[Paper]:
        data = json.loads(self._fetch(topic, max_results))
        papers: list[Paper] = []
        for w in data.get("results", []):
            cite_id, url = _openalex_cite_id(w)
            papers.append(Paper(
                title=" ".join((w.get("title") or "").split()),
                summary=_reconstruct_abstract(w.get("abstract_inverted_index")),
                arxiv_id=cite_id, url=url,
                citations=w.get("cited_by_count"),
            ))
        return papers


class SemanticScholarClient:
    """Semantic Scholar graph search — citation counts + open-access PDF links. The free endpoint
    is rate-limited (HTTP 429) without an API key, so it is NOT a default source; add it explicitly
    when you have a key. `fetcher(query, max_results) -> json_text` is injectable."""

    name = "semantic_scholar"
    _FIELDS = "title,abstract,citationCount,externalIds,openAccessPdf"

    def __init__(self, fetcher=None, *, timeout: float = 20.0, api_key: str | None = None):
        self._fetch = fetcher or self._http_fetch
        self.timeout = timeout
        self.api_key = api_key

    def _http_fetch(self, query: str, max_results: int) -> str:  # pragma: no cover - network
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({
            "query": query, "limit": max_results, "fields": self._FIELDS,
        })
        req = urllib.request.Request(url, headers={"User-Agent": _UA,
                                                   **({"x-api-key": self.api_key} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout, context=_ssl_ctx()) as r:
            return r.read().decode("utf-8")

    def search(self, topic: str, max_results: int = 5) -> list[Paper]:
        data = json.loads(self._fetch(topic, max_results))
        papers: list[Paper] = []
        for p in data.get("data", []):
            ext = p.get("externalIds") or {}
            cite_id = ("arXiv:" + ext["ArXiv"]) if ext.get("ArXiv") else (
                ("doi:" + ext["DOI"]) if ext.get("DOI") else f"s2:{p.get('paperId', 'unknown')}")
            papers.append(Paper(
                title=" ".join((p.get("title") or "").split()),
                summary=" ".join((p.get("abstract") or "").split()),
                arxiv_id=cite_id, url=(p.get("openAccessPdf") or {}).get("url", ""),
                citations=p.get("citationCount"),
                code_url=(p.get("openAccessPdf") or {}).get("url"),
            ))
        return papers


def _dedup_key(p: Paper) -> str:
    """Same paper across sources collapses to one row: prefer the arXiv id (version-stripped),
    else the normalized title."""
    cid = (p.arxiv_id or "").lower()
    if cid.startswith("arxiv:"):
        num = cid.split(":", 1)[1]            # strip the prefix first, THEN drop the version suffix
        return "arxiv:" + num.split("v")[0]
    return "title:" + " ".join(p.title.lower().split())


def merge_papers(groups: list[list[Paper]], max_results: int) -> list[Paper]:
    """Merge per-source results: de-duplicate (keeping the row that carries a citation count),
    then rank by citations (descending; unknown last) and truncate. Pure, deterministic."""
    best: dict[str, Paper] = {}
    for group in groups:
        for p in group:
            k = _dedup_key(p)
            cur = best.get(k)
            if cur is None or (cur.citations is None and p.citations is not None) \
                    or (p.citations or -1) > (cur.citations or -1):
                # keep the richer record, but don't lose a code link the other copy had
                if cur and cur.code_url and not p.code_url:
                    p.code_url = cur.code_url
                best[k] = p
    ranked = sorted(best.values(), key=lambda p: (p.citations is None, -(p.citations or 0)))
    return ranked[:max_results]


# --------------------------------------------------------------------------------------------
# Extraction agent
# --------------------------------------------------------------------------------------------
FINDINGS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "technique": {"type": "string"},
                    "source": {"type": "string"},
                    "predicted_effect": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["technique", "source", "predicted_effect"],
            },
        },
    },
    "required": ["findings"],
}

_SYSTEM = (
    "You read ML papers and extract concrete, testable techniques (architecture changes, "
    "optimizers, schedules, augmentations) that could improve a model on the given topic. "
    "Cite the id shown in brackets as the source for each. Prefer techniques from higher-cited "
    "papers when they conflict. Do not invent techniques the papers don't support."
)


class LitScout(Agent):
    name = "lit_scout"
    system = _SYSTEM
    schema = FINDINGS_SCHEMA

    def __init__(self, llm, arxiv: ArxivClient | None = None, *, sources: list | None = None,
                 trace=None, max_results: int = 5):
        super().__init__(llm, trace=trace)
        if sources is not None:
            self.sources = list(sources)
        elif arxiv is not None:                       # backward-compatible single-source construction
            self.sources = [arxiv]
        else:
            self.sources = [ArxivClient(), OpenAlexClient()]   # default: arXiv + impact-ranked OpenAlex
        self.max_results = max_results

    def _gather(self, topic: str) -> list[Paper]:
        """Query every source, degrading per-source on failure, then merge + impact-rank."""
        groups: list[list[Paper]] = []
        for src in self.sources:
            try:
                groups.append(src.search(topic, self.max_results))
            except Exception as e:  # pragma: no cover - network
                print(f"lit_scout: {getattr(src, 'name', 'source')} fetch failed "
                      f"({type(e).__name__}), skipping it", file=sys.stderr)
        return merge_papers(groups, self.max_results)

    def build_prompt(self, ctx: dict) -> str:
        papers: list[Paper] = ctx["papers"]

        def tag(p: Paper) -> str:
            cites = f", {p.citations} citations" if p.citations is not None else ""
            code = f"\ncode: {p.code_url}" if p.code_url else ""
            return f"[{p.arxiv_id}{cites}] {p.title}\n{p.summary[:600]}{code}"

        body = "\n\n".join(tag(p) for p in papers)
        return (f"Topic: {ctx['topic']}\n\nFrom these papers, extract techniques that could "
                f"improve a model on this topic. For each: technique, source (the bracketed id), "
                f"predicted_effect, and a one-line rationale.\n\nPapers:\n{body}")

    def scout(self, topic: str) -> tuple[str, list[str]]:
        """Retrieve + extract. Returns (lit_context prose, lit_priors list).

        A total retrieval failure (every source down) degrades gracefully to no literature rather
        than killing the campaign — the loop can still reason from the ledger.
        """
        papers = self._gather(topic)
        if not papers:
            return "", []
        findings = self.run({"topic": topic, "papers": papers}).get("findings", [])
        lit_context = "\n".join(
            f"- {f['technique']} ({f['source']}): {f['predicted_effect']}"
            + (f" — {f.get('rationale')}" if f.get("rationale") else "")
            for f in findings)
        lit_priors = [f"{f['technique']} ({f['source']})" for f in findings]
        return lit_context, lit_priors
