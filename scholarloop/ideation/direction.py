"""The author's steer and the direction map: what happens between the review and the ideas.

A literature review names gaps, but a gap is not yet a reason to work on something. Before any
idea is generated the author states a steer: what they want to find out, the hunches they
want tested, what to stay away from. The DirectionMapper then lays out a few research directions,
each grounded in reviewed papers and marked as developing, challenging or lying outside the
author's hunches. The author chooses among them (or writes their own), and only the chosen
directions are handed to the Ideator. Every idea must name the direction it belongs to.

The map is deliberately not a list of ideas: a direction is a line of inquiry with an open
question and a first experiment, and several different ideas can live inside one.
"""

from __future__ import annotations

from jsonschema import Draft7Validator

from scholarloop.core.agent import Agent
from scholarloop.review.contracts import STRINGS, TEXT, digest, obj

STEER_SCHEMA = obj({
    "interest": TEXT,          # what the author wants to find out, and why it matters to them
    "hunches": STRINGS,        # mechanisms or suspicions they want put to a test
    "avoid": STRINGS,          # directions that are out of scope or already known to be dull
    "contribution": TEXT,      # the kind of result they would value (method, negative result, ...)
    "known": STRINGS,          # what they already know or have tried, so it is not re-proposed
}, required=["interest"])
STANCES = ("develops", "challenges", "outside")
QUESTIONS = (
    ("interest", "What do you want to find out, and why does it matter to you?", False),
    ("hunches", "Hunches or mechanisms you want tested (separate several with ';', blank for none)", True),
    ("avoid", "Directions to stay away from (';' separated, blank for none)", True),
    ("contribution", "What kind of result would you value (new method, negative result, analysis ...)?", False),
    ("known", "What do you already know or have tried (';' separated, blank for none)?", True),
)


def validate_steer(steer) -> dict:
    """Normalise and validate an author's steer. Raises ValueError with every problem found."""
    if not isinstance(steer, dict):
        raise ValueError("A steer must be a JSON object")
    clean = {}
    for key, value in steer.items():
        if isinstance(value, str):
            value = " ".join(value.split())
        elif isinstance(value, list):
            value = [" ".join(v.split()) if isinstance(v, str) else v for v in value]
            value = [v for v in value if v != ""]
        if value not in ("", [], None):
            clean[key] = value
    errors = [e.message for e in Draft7Validator(STEER_SCHEMA).iter_errors(clean)]
    if not errors and len(clean["interest"].split()) < 4:
        errors.append("say what you want to find out in at least a few words")
    if errors:
        raise ValueError("Invalid steer: " + "; ".join(errors))
    return clean


def ask_steer(input_fn=input, print_fn=print) -> dict:
    """Collect a steer from a terminal. Re-asks the one required question until it is answered."""
    print_fn("      Before anything is generated, say where you want this to go.")
    steer = {}
    for key, question, many in QUESTIONS:
        answer = (input_fn(f"      {question}\n      > ") or "").strip()
        while key == "interest" and len(answer.split()) < 4:
            answer = (input_fn("      (a sentence, please) > ") or "").strip()
        if answer:
            steer[key] = [a.strip() for a in answer.split(";") if a.strip()] if many else answer
    return validate_steer(steer)


def steer_block(steer: dict) -> str:
    """The steer as prompt text. Hunches are numbered because directions refer to them."""
    lines = [f"What the author wants to find out: {steer['interest']}"]
    if steer.get("hunches"):
        lines.append("The author's hunches:")
        lines += [f"  H{i}. {h}" for i, h in enumerate(steer["hunches"], 1)]
    if steer.get("avoid"):
        lines.append("Stay away from: " + "; ".join(steer["avoid"]))
    if steer.get("contribution"):
        lines.append(f"Kind of result the author values: {steer['contribution']}")
    if steer.get("known"):
        lines.append("The author already knows or has tried: " + "; ".join(steer["known"]))
    return "\n".join(lines)


# The 3-5 bound is enforced in postcheck: the structured-output API accepts minItems of 0 or 1 only.
DIRECTIONS_SCHEMA = obj({"directions": {"type": "array", "items": obj({
    "name": TEXT,
    "thesis": TEXT,                 # the bet, in one or two sentences
    "stance": {"type": "string", "enum": list(STANCES)},
    "hunches": {"type": "array", "items": {"type": "integer"}},   # H-numbers this direction addresses
    "steer_link": TEXT,             # how it follows from, or argues with, what the author said
    "gap": TEXT,                    # the reviewed gap it stands on
    "sources": {**STRINGS, "minItems": 1},
    "open_question": TEXT,          # what nobody in the reviewed papers has answered
    "first_experiment": TEXT,       # the cheapest run in this setting that would be informative
    "risk": TEXT,                   # the most likely way this turns out uninteresting
})}})


class DirectionMapper(Agent):
    name = "direction_mapper"
    schema = DIRECTIONS_SCHEMA
    system = (
        "You help a researcher decide where to dig before any specific idea is written. From a "
        "literature review and the researcher's own steer, lay out three to five research "
        "directions. A direction is a line of inquiry, not an experiment: give its thesis, the "
        "reviewed gap it stands on, the papers it builds on (bracketed ids only), the open "
        "question, the cheapest informative first experiment in the given setting, and the most "
        "likely way it turns out dull. Take the researcher's hunches seriously: every hunch must "
        "be addressed by at least one direction, and steer_link must say concretely how the "
        "direction follows from or argues with what they wrote. Do not only agree. At least one "
        "direction must either challenge a hunch (say what would have to be true for it to be "
        "wrong, and how to find out) or come from outside the steer because the literature "
        "points there. Never propose what the researcher asked to avoid or already knows. "
        "Directions must differ in mechanism, not in wording."
    )

    def __init__(self, llm, profile_setting: str, *, trace=None):
        super().__init__(llm, trace=trace)
        self.setting = profile_setting

    def build_prompt(self, ctx: dict) -> str:
        parts = [f"Topic: {ctx['review'].topic}", f"Experimental setting: {self.setting}", "",
                 "The researcher's steer:", steer_block(ctx["steer"]), "",
                 "Literature review:", ctx["review"].to_prompt_block()]
        if ctx.get("previous"):
            parts += ["", "Directions on the previous map:"] + [f"- {p}" for p in ctx["previous"]]
        if ctx.get("feedback"):
            parts += ["", "The researcher's feedback on the previous map (follow it):", ctx["feedback"]]
        parts += ["", "Lay out 3-5 directions. In `hunches` list the H-numbers a direction addresses "
                  "(empty when stance is `outside`)."]
        return "\n".join(parts)

    def postcheck(self, output: dict, ctx: dict) -> list[str]:
        ids = {p.arxiv_id for p in ctx["review"].papers}
        n_hunches = len(ctx["steer"].get("hunches", []))
        errors, covered = [], set()
        if not 3 <= len(output["directions"]) <= 5:
            errors.append(f"lay out 3 to 5 directions (got {len(output['directions'])})")
        names = [" ".join(d["name"].lower().split()) for d in output["directions"]]
        if len(set(names)) != len(names):
            errors.append("directions must have distinct names")
        for i, d in enumerate(output["directions"], 1):
            unknown = sorted(s for s in d["sources"] if s not in ids)
            if unknown:
                errors.append(f"direction {i}: unknown paper ids {unknown} (cite only the bracketed ids)")
            bad = sorted(h for h in d["hunches"] if not 1 <= h <= n_hunches)
            if bad:
                errors.append(f"direction {i}: hunch numbers {bad} do not exist (the author gave {n_hunches})")
            if d["stance"] == "outside" and d["hunches"]:
                errors.append(f"direction {i}: an `outside` direction addresses no hunch; use develops/challenges")
            if d["stance"] != "outside" and n_hunches and not d["hunches"]:
                errors.append(f"direction {i}: name the hunch it {d['stance']}")
            covered.update(d["hunches"])
        missing = sorted(set(range(1, n_hunches + 1)) - covered)
        if missing:
            errors.append(f"no direction addresses hunch(es) {['H%d' % h for h in missing]}")
        if n_hunches and all(d["stance"] == "develops" for d in output["directions"]):
            errors.append("at least one direction must challenge a hunch or come from outside the steer")
        return errors

    def map(self, review, steer: dict, *, previous=(), feedback: str = "") -> list[dict]:
        out = self.run({"review": review, "steer": steer, "previous": list(previous), "feedback": feedback})
        return [with_id({**d, "origin": "mapper"}) for d in out["directions"]]


def with_id(direction: dict) -> dict:
    return {**direction, "id": "dir_" + digest({k: direction.get(k) for k in ("name", "thesis")})[:10]}


def own_direction(text: str) -> dict:
    """A direction the author wrote. It is not literature-grounded and says so."""
    text = " ".join(text.split())
    if len(text.split()) < 4:
        raise ValueError("Describe your own direction in at least a few words")
    name = text if len(text) <= 70 else text[:67].rsplit(" ", 1)[0] + "..."
    return with_id({"name": name, "thesis": text, "stance": "develops", "hunches": [],
                    "steer_link": "Written by the author.", "gap": "", "sources": [],
                    "open_question": "", "first_experiment": "", "risk": "", "origin": "author"})


def directions_block(directions: list[dict]) -> str:
    """The chosen directions as prompt text for the Ideator."""
    lines = []
    for d in directions:
        lines.append(f"- {d['id']} · {d['name']}: {d['thesis']}")
        if d.get("open_question"):
            lines.append(f"    open question: {d['open_question']}")
        if d.get("sources"):
            lines.append(f"    builds on: {', '.join('[%s]' % s for s in d['sources'])}")
        if d["origin"] == "author":
            lines.append("    (written by the author; find the reviewed papers that bear on it)")
    return "\n".join(lines)


def render_markdown(topic: str, steer: dict, directions: list[dict], chosen: list[str],
                    note: str = "") -> str:
    """`directions.md`, the page the author reads before choosing where ideas are generated."""
    out = [f"# Research directions — {topic}", "", "## Your steer", ""]
    out += ["> " + line for line in steer_block(steer).splitlines()]
    out += ["", "## Direction map", "",
            "_Directions are lines of inquiry, not experiments. Choose one or more; ideas are "
            "generated only inside the directions you choose._", "",
            "| # | direction | stance toward your hunches | builds on |", "|---|---|---|---|"]
    for i, d in enumerate(directions, 1):
        mark = "✅ " if d["id"] in chosen else ""
        hunches = " " + ", ".join(f"H{h}" for h in d["hunches"]) if d["hunches"] else ""
        stance = "yours" if d["origin"] == "author" else d["stance"] + hunches
        out.append(f"| {i} | {mark}**{d['name']}** | {stance} | "
                   f"{', '.join('`%s`' % s for s in d['sources']) or '—'} |")
    out.append("")
    for i, d in enumerate(directions, 1):
        out += [f"### {i}. {d['name']}" + ("  ✅ chosen" if d["id"] in chosen else ""), "", d["thesis"], ""]
        if d["origin"] == "author":
            out += ["- Written by you. Not yet tied to reviewed papers.", f"- **ID:** `{d['id']}`", ""]
            continue
        out += [f"- **How it relates to your steer:** {d['steer_link']}",
                f"- **Gap:** {d['gap']}",
                f"- **Open question:** {d['open_question']}",
                f"- **First experiment:** {d['first_experiment']}",
                f"- **How it could turn out dull:** {d['risk']}",
                f"- **ID:** `{d['id']}`", ""]
    if note:
        out += [f"> Your note on the choice: {note}", ""]
    if not chosen:
        out += ["*Nothing is chosen yet. Choose with `--directions 1,3`, add your own with "
                "`--own-direction \"...\"`, or ask for a new map with `--feedback \"...\"`.*", ""]
    return "\n".join(out)
