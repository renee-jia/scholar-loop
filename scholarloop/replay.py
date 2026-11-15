"""Read a recorded run without a model: replay its stages, trace a sentence, try the checks.

Nothing here calls a model or trains anything. A recorded run is one `record.json` (written by
`capture`) holding what each stage produced, what the author decided, the approved statements
and the measurements behind them. Three things can be done with it:

    scholarloop replay            walk through the run stage by stage
    scholarloop trace "text"      from a sentence of the paper to its statement, runs and values
    scholarloop check "sentence"  ask the production sentence check whether the paper may say it

`check` runs the same function the writer's output is held to (`audit_prose`), on the approved
statements of the recorded run, so a reader can change a number, cite another statement or drop
the citation and see what is refused. It checks what code can check: that a sentence cites an
approved statement and uses only its numbers and sources. Whether a paraphrase keeps the meaning
is judged by a model reviewer in a live run and is not replayed here.

With no run given, the commands use the sample shipped with the package (`--sample NAME`).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import textwrap
from pathlib import Path

from scholarloop.review.dossier import STATEMENT_ID, approved_statements, audit_prose

SAMPLES = Path(__file__).parent / "samples"
DEFAULT_SAMPLE = "agnews"
WIDTH = 96


# ---- capture ------------------------------------------------------------------------------------

def _load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _trace_rows(run: Path) -> tuple[int, list[dict]]:
    """Agent calls in the run's traces, and the attempts a check sent back."""
    calls, rejected = set(), []
    for file in sorted(run.rglob("*.agents.jsonl")):
        if "reliability" in file.parts:               # the evaluation's own calls are not the run
            continue
        for line in file.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") != "finished":
                continue
            calls.add(row.get("call_id"))
            if row.get("status") == "rejected":
                rejected.append({"agent": row.get("agent"), "attempt": row.get("attempt"),
                                 "errors": [str(e)[:400] for e in row.get("errors", [])][:6]})
    return len(calls), rejected


def build_record(run, writing=None) -> dict:
    """Everything the replay shows, read from a run directory. `writing` is the writing
    directory when the paper was written apart from the run (default: the run itself)."""
    run = Path(run)
    writing = Path(writing) if writing else run
    review = _load(run / "scientific_review.json")
    state = _load(writing / "writing.json")
    if not review or not review.get("release") or not state or not state.get("sections"):
        raise ValueError(f"{run} is not a reviewed run with a drafted paper")
    dossier = review["dossier"]
    directions = _load(run / "directions.json", {})
    chosen = set(directions.get("chosen", []))
    handoff = _load(run / "handoff.json", {})
    picked = (handoff.get("idea") or {}).get("title")
    lit = _load(run / "lit_review.json", {})
    process = _load(run / "process.json", {})
    best = {}
    for ref, m in dossier["measurements"].items():
        best[m["run_id"]] = ref                       # later tiers come later in the record
    calls, rejected = _trace_rows(run)
    return {
        "version": 1,
        "name": run.name,
        "topic": handoff.get("topic") or lit.get("topic", ""),
        "steer": _load(run / "steer.json") or directions.get("steer", {}),
        "literature": {"papers": len(lit.get("papers", [])),
                       "themes": [t.get("name", "") for t in lit.get("themes", [])],
                       "gaps": len(lit.get("gaps", []))},
        "directions": [{"name": d.get("name", ""), "stance": d.get("stance", ""),
                        "chosen": d.get("id") in chosen} for d in directions.get("map", [])],
        "own_directions": [d for d in directions.get("own", [])] if isinstance(directions.get("own"), list) else [],
        "ideas": [{"title": i["title"], "elo": (i.get("tournament") or {}).get("elo"),
                   "picked": i["title"] == picked} for i in _load(run / "ideas.json", [])],
        "selection": handoff.get("selection_decision", {}),
        "experiments": [{"id": r["id"], "fidelity": r.get("fidelity"), "verdict": r.get("verdict"),
                         "value": dossier["measurements"][best[r["id"]]]["value"] if r["id"] in best else None,
                         "protocol": (r.get("protocol") or {}).get("condition")}
                        for r in dossier["runs"]],
        "metric": dossier["runs"][0].get("metric_name", "") if dossier["runs"] else "",
        "review_rounds": process.get("review_rounds", []),
        "release": {k: review["release"].get(k) for k in ("review_hash", "input_hash", "author_decision")},
        "dossier": {"argument": dossier["argument"], "measurements": dossier["measurements"],
                    "runs": [{k: r.get(k) for k in ("id", "fidelity", "verdict", "config", "candidate_id")}
                             for r in dossier["runs"]],
                    "papers": {k: {"title": v.get("title", ""), "url": v.get("url", "")}
                               for k, v in dossier["papers"].items()}},
        "paper": {"title": state["outline"]["title"],
                  "sections": [{"heading": s["heading"], "body": s["body"]} for s in state["sections"]]},
        "reviewers": [{"persona": r["persona"], "score": r["score"], "recommendation": r["recommendation"]}
                      for r in (state.get("review") or {}).get("reviews", [])],
        "agent_calls": calls,
        "rejected_attempts": rejected,
        "spend": {"usd": process.get("usd_spend_file"), "requests": process.get("requests_spend_file")},
    }


def capture(run, dest, writing=None) -> Path:
    """Write a recorded run to `dest`: record.json and the papers a reader may want to open."""
    run, dest = Path(run), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "record.json").write_text(json.dumps(build_record(run, writing), indent=1, ensure_ascii=False))
    for name in ("paper.md", "paper_traced.md", "review.md", "evidence_map.md"):
        source = (Path(writing) if writing else run) / name
        if source.exists():
            shutil.copyfile(source, dest / name)
    for name in ("scientific_review.md", "directions.md", "ideas.md", "lit_review.md"):
        if (run / name).exists():
            shutil.copyfile(run / name, dest / name)
    return dest


def load(location=None, sample: str = DEFAULT_SAMPLE) -> dict:
    """A record from a capture directory, a record.json, a raw run directory, or a sample."""
    path = Path(location) if location else SAMPLES / sample
    if path.is_file():
        return json.loads(path.read_text())
    if (path / "record.json").exists():
        return json.loads((path / "record.json").read_text())
    if not path.exists():
        raise FileNotFoundError(f"no recorded run at {path}; samples: {sorted(p.name for p in SAMPLES.glob('*') if p.is_dir())}")
    return build_record(path)


# ---- rendering ----------------------------------------------------------------------------------

def _wrap(text: str, indent: str = "  ") -> str:
    return "\n".join(textwrap.fill(line, WIDTH, initial_indent=indent, subsequent_indent=indent)
                     for line in str(text).splitlines() if line.strip())


def _sentences(body: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", body) if s.strip()]


def stages(record: dict) -> list[tuple[str, str]]:
    """The run as (heading, text) steps. `author` in a heading marks a decision a person made."""
    r, out = record, []
    steer = r.get("steer") or {}
    lines = [_wrap(steer.get("interest", ""))]
    lines += [_wrap(f"Hunch {i}: {h}") for i, h in enumerate(steer.get("hunches", []), 1)]
    out.append(("1. Steer (author)", "\n".join(x for x in lines if x)))
    lit = r["literature"]
    out.append(("2. Literature review", "\n".join(
        [f"  {lit['papers']} papers retrieved, {lit['gaps']} open gaps. Every identifier the review cites must be a",
         "  retrieved paper; a review citing anything else is sent back.", "  Themes:"]
        + [f"    - {t}" for t in lit["themes"]])))
    if r["directions"]:
        out.append(("3. Directions (author chooses)", "\n".join(
            [f"  [{'x' if d['chosen'] else ' '}] {d['name']}  ({d['stance']} the author's hunch)" for d in r["directions"]]
            + ["  Ideas are generated only inside the chosen directions."])))
    ideas = sorted(r["ideas"], key=lambda i: -(i["elo"] or 0))
    out.append(("4. Ideas and tournament (author picks)", "\n".join(
        [f"  [{'x' if i['picked'] else ' '}] {i['title'][:78]}" + (f"  Elo {i['elo']:.0f}" if i["elo"] else "")
         for i in ideas]
        + ["  Judges compare every pair twice with the order swapped; Elo is computed in code.",
           f"  Picked by: {r['selection'].get('actor', '?')} ({r['selection'].get('kind', '?')})"])))
    exps = r["experiments"]
    kept = sum(e["verdict"] == "kept" for e in exps)
    rows = [f"  {e['id'][-28:]:<28} {str(e['fidelity'] and e['fidelity'][-1]):<7} "
            f"{'' if e['value'] is None else format(e['value'], '.4f'):>9}  {e['verdict']}" for e in exps[-12:]]
    out.append(("5. Experiments", "\n".join(
        [f"  {len(exps)} runs, {kept} kept. Training writes an artifact; a separate process runs the frozen",
         f"  scorer on it, and that number ({r['metric']}) is the only one recorded.",
         f"  Last {len(rows)} runs:"] + rows)))
    arg = r["dossier"]["argument"]
    rounds = [f"  round {x['round']}: {x['status']:<18} new issues: {x['new_issues']['major']} major, "
              f"{x['new_issues']['minor']} minor; open {x['open_issues']}" for x in r["review_rounds"]]
    decision = r["release"].get("author_decision") or {}
    out.append(("6. Scientific review and release (author approves)", "\n".join(
        rounds + [f"  Approved: {len(arg['claims'])} claims, {len(arg.get('setup', []))} setup statements, "
                  f"{len(arg.get('related_work', []))} related-work statements, {len(arg['limitations'])} limitations.",
                  f"  Approved by {decision.get('actor', '?')} ({decision.get('kind', '?')}); bound to evidence hash",
                  f"  {r['release'].get('input_hash')}", "  A changed run, profile or scorer makes this approval stale."])))
    reviewers = ", ".join(f"{x['persona']} {x['score']}" for x in r["reviewers"])
    out.append(("7. Paper", "\n".join(
        [_wrap(r["paper"]["title"]),
         f"  {len(r['paper']['sections'])} sections, {sum(len(_sentences(s['body'])) for s in r['paper']['sections'])} sentences; "
         "every sentence cites an approved statement.",
         f"  Simulated reviewers (out of 10): {reviewers or 'not run'}"])))
    rej = r["rejected_attempts"]
    by = {}
    for x in rej:
        by[x["agent"]] = by.get(x["agent"], 0) + 1
    lines = [f"  {r['agent_calls']} agent calls; {len(rej)} outputs were refused by a check and returned to the model:"]
    lines += [f"    {agent}: {n}" for agent, n in sorted(by.items(), key=lambda kv: -kv[1])]
    for x in rej[:3]:
        lines += [f"  {x['agent']}, attempt {x['attempt']}:"] + [_wrap(e, "      ") for e in x["errors"][:2]]
    if r["spend"].get("usd"):
        lines.append(f"  Model spend recorded in this run directory, its evaluation included: "
                     f"${r['spend']['usd']:.2f} over {r['spend']['requests']} requests.")
    out.append(("8. The harness at work", "\n".join(lines)))
    return out


def example_sentence(record: dict) -> str:
    """A sentence of the paper that reports a number, to start `trace` and `check` from."""
    statements = approved_statements(record["dossier"])
    for section in record["paper"]["sections"]:
        for s in _sentences(section["body"]):
            ids = STATEMENT_ID.findall(s)
            if len(ids) == 1 and ids[0].startswith("claim_") and re.search(r"\d\.\d", s) \
                    and statements[ids[0]].get("measurements") and len(s) < 260:
                return s
    return _sentences(record["paper"]["sections"][0]["body"])[0]


def trace(record: dict, query: str) -> str:
    """The sentences of the paper matching `query`, each followed to its records."""
    dossier = record["dossier"]
    statements = approved_statements(dossier)
    runs = {r["id"]: r for r in dossier["runs"]}
    words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for section in record["paper"]["sections"]:
        for s in _sentences(section["body"]):
            if query.lower() in s.lower():
                scored.append((2.0, section["heading"], s))
            elif words:
                overlap = len(words & set(re.findall(r"\w+", s.lower()))) / len(words)
                if overlap >= 0.6:
                    scored.append((overlap, section["heading"], s))
    if not scored:
        return "No sentence of the paper matches. Try a few consecutive words from paper.md."
    out = []
    for _, heading, sentence in sorted(scored, key=lambda x: -x[0])[:1]:
        out += [f"Sentence ({heading}):", _wrap(sentence)]
        for cid in STATEMENT_ID.findall(sentence):
            st = statements.get(cid)
            if not st:
                continue
            out += [f"  -> approved statement [{cid}]:", _wrap(st["text"], "       ")]
            for ref in st.get("measurements", []):
                m = dossier["measurements"].get(ref)
                if m:
                    out.append(f"       measurement {ref} = {m['value']}"
                               + (f"  (per seed {m['seeds']})" if len(m.get("seeds", [])) > 1 else ""))
            for e in st.get("evidence", []):
                if e in runs:
                    run = runs[e]
                    out.append(f"       run {e}: {run['verdict']}, code version {str(run.get('candidate_id'))[:12]}, "
                               f"config {json.dumps(run.get('config'))[:120]}")
                elif e in dossier["papers"]:
                    out.append(f"       paper {e}: {dossier['papers'][e]['title'][:90]}")
                elif e not in ("profile",):
                    out.append(f"       record {str(e)[:64]}")
        out.append("")
    return "\n".join(out).rstrip()


def check(record: dict, sentence: str) -> list[str]:
    """What the production sentence check says about `sentence`; empty means it may be written."""
    return audit_prose(sentence, record["dossier"])


def variants(record: dict) -> list[tuple[str, str]]:
    """One accepted sentence of the paper and four ways to break it."""
    base = example_sentence(record)
    cid = STATEMENT_ID.findall(base)[0]
    number = re.search(r"\d+\.\d+", base).group()
    other = next((c["id"] for c in record["dossier"]["argument"]["claims"]
                  if c["id"] != cid and number not in c["text"]), cid)
    return [("as written in the paper", base),
            ("a number changed", base.replace(number, f"{float(number) - 1.5:.2f}", 1)),
            ("the citation removed", STATEMENT_ID.sub("", base).replace(" .", ".").strip()),
            ("an approved statement that does not carry this number", base.replace(f"[{cid}]", f"[{other}]")),
            ("a statement id that was never approved", base.replace(f"[{cid}]", "[claim_new_result]"))]


def render_check(record: dict, sentence: str) -> str:
    errors = check(record, sentence)
    head = "ACCEPTED" if not errors else "REFUSED"
    return "\n".join([head, _wrap(sentence)] + [_wrap("- " + e, "    ") for e in errors])


# ---- command line -------------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="scholarloop", description=__doc__.split("\n\n")[0],
                                 epilog="Live runs need a model key: see the README "
                                        "(python -m scholarloop.ideation, .review, .writing, .reliability).")
    sub = ap.add_subparsers(dest="command", required=True)

    def run_arg(p):
        p.add_argument("--run", help="a capture directory, record.json or run directory")
        p.add_argument("--sample", default=DEFAULT_SAMPLE, help="a recorded run shipped with the package")

    p = sub.add_parser("replay", help="walk through a recorded run stage by stage")
    run_arg(p)
    p.add_argument("--no-pause", action="store_true", help="print every stage without waiting")
    p = sub.add_parser("trace", help="follow a sentence of the paper to its statement, runs and values")
    run_arg(p)
    p.add_argument("text", nargs="?", help="words from the paper (default: an example sentence)")
    p = sub.add_parser("check", help="ask the sentence check whether the paper may say something")
    run_arg(p)
    p.add_argument("sentence", nargs="?", help="a sentence ending in [statement_id]; omit for examples "
                                               "and a prompt")
    p.add_argument("--examples", action="store_true", help="show the examples and exit")
    p = sub.add_parser("statements", help="list the approved statements a sentence may cite")
    run_arg(p)
    p = sub.add_parser("capture", help="record a finished run for replay")
    p.add_argument("run")
    p.add_argument("dest")
    p.add_argument("--writing", help="writing directory, when apart from the run")
    a = ap.parse_args(argv)

    if a.command == "capture":
        print(f"recorded {capture(a.run, a.dest, a.writing)}")
        return 0
    try:
        record = load(a.run, a.sample)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if a.command == "replay":
        print(f"Recorded run: {record['name']} — {record['topic']}\n"
              "Nothing below calls a model; it is read from the run's records.\n")
        for heading, text in stages(record):
            print(f"== {heading} ==\n{text}\n")
            if not a.no_pause and sys.stdin.isatty():
                try:
                    input("[Enter for the next stage] ")
                except EOFError:
                    pass
        example = example_sentence(record)
        print("Next:\n  scholarloop trace \"" + " ".join(example.split()[:6]) + "\"\n  scholarloop check --examples")
        return 0
    if a.command == "trace":
        print(trace(record, a.text or " ".join(example_sentence(record).split()[:8])))
        return 0
    if a.command == "statements":
        for cid, st in approved_statements(record["dossier"]).items():
            print(f"[{cid}] {st['text'][:150]}")
        return 0
    if a.sentence:
        print(render_check(record, a.sentence))
        return 0
    print("The sentence check the writer's output is held to, run on this run's approved statements.\n")
    for label, sentence in variants(record):
        print(f"-- {label}\n{render_check(record, sentence)}\n")
    if a.examples or not sys.stdin.isatty():
        return 0
    print("Type a sentence ending in [statement_id] (ids: scholarloop statements). Empty line to stop.")
    while True:
        try:
            sentence = input("> ").strip()
        except EOFError:
            break
        if not sentence:
            break
        print(render_check(record, sentence) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
