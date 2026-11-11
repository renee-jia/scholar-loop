"""Immutable idea nodes and append-only rounds/decisions inside session.json."""

from dataclasses import asdict
import time

from scholarloop.review.contracts import digest
from scholarloop.review.contracts import TEXT, STRINGS, obj
from jsonschema import Draft7Validator

BRIEF_SCHEMA = obj({"question": TEXT, "success_criteria": {**STRINGS, "minItems": 1},
                    "constraints": STRINGS, "non_goals": STRINGS, "resources": TEXT,
                    "literature_scope": TEXT})


def validate_brief(brief):
    errors = [e.message for e in Draft7Validator(BRIEF_SCHEMA).iter_errors(brief)]
    if errors:
        raise ValueError("Invalid research brief: " + "; ".join(errors))


def new_graph():
    return {"version": 1, "nodes": {}, "edges": [], "rounds": [], "decisions": [], "observations": []}


def context_hash(review, profile, brief=None):
    return digest({"review": review.to_dict(), "profile": asdict(profile), "brief": brief or {}})


def record_round(graph, ideas, *, review, profile, feedback, parents=(), relation="refines", audits=(), brief=None):
    for parent in parents:
        if parent not in graph["nodes"]:
            raise ValueError(f"Unknown parent idea: {parent}")
    ids = []
    round_no = len(graph["rounds"]) + 1
    for idea in ideas:
        raw = idea.to_dict()
        content = {k: v for k, v in raw.items() if k not in
                   {"idea_id", "selected", "status", "novelty", "novelty_review", "tournament"}}
        idea.idea_id = "idea_" + digest(content)[:24]
        previous = [d for d in graph["decisions"] if idea.idea_id in d["idea_ids"] and
                    d["action"] in {"reject", "revive"}]
        if previous and previous[-1]["action"] == "reject":
            idea.status = "rejected"
        if idea.idea_id in ids:
            continue
        ids.append(idea.idea_id)
        if idea.idea_id not in graph["nodes"]:
            graph["nodes"][idea.idea_id] = {"idea": idea.to_dict(), "created_round": round_no}
        for parent in parents:
            edge = {"from": parent, "to": idea.idea_id, "relation": relation, "round": round_no}
            if parent != idea.idea_id:
                graph["edges"].append(edge)
    graph["rounds"].append({"round": round_no, "idea_ids": ids,
                            "ideas": list({i.idea_id: i.to_dict() for i in ideas}.values()), "feedback": feedback,
                            "context_hash": context_hash(review, profile, brief), "audits": list(audits),
                            "brief": brief or {},
                            "review_snapshot": review.to_dict(), "timestamp": time.time()})


def record_decision(graph, action, ids, *, actor, reason, kind="human"):
    if not actor.strip() or not reason.strip() or not ids or len(set(ids)) != len(ids):
        raise ValueError("A decision needs an actor, reason and distinct idea ids")
    if set(ids) - set(graph["nodes"]):
        raise ValueError("Decision references an unknown idea id")
    if kind not in ("human", "automated"):
        raise ValueError("Decision actor kind must be human or automated")
    graph["decisions"].append({"action": action, "idea_ids": list(ids), "actor": actor,
                               "kind": kind, "reason": reason, "timestamp": time.time()})
