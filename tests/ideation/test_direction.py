"""The author's steer and the direction map: ideation starts from the author's input, the map
must engage with their hunches, and ideas are generated only inside the chosen directions."""

import json

import pytest

from scholarloop.core.agent import AgentError
from scholarloop.core.llm import MockLLM
from scholarloop.ideation import session as session_module
from scholarloop.ideation.direction import ask_steer, validate_steer
from scholarloop.ideation.litscout import ArxivClient
from scholarloop.ideation.session import ResearchSession
from tests.ideation.test_session import ATOM, PROFILE, QUERIES, REVIEW, _idea

SRC = "arXiv:1512.03385v1"
STEER = {"interest": "Find out whether skip connections matter in shallow nets",
         "hunches": ["Skips only help once depth hurts optimization", "Gains are a learning-rate artefact"],
         "avoid": ["bigger models"]}


def _dir(name, stance="develops", hunches=(1,), sources=(SRC,)):
    return {"name": name, "thesis": f"{name} thesis", "stance": stance, "hunches": list(hunches),
            "steer_link": "follows H1", "gap": "skips on shallow MLPs", "sources": list(sources),
            "open_question": "q", "first_experiment": "depth 2 vs 6 with and without skips", "risk": "null"}


MAP = {"directions": [_dir("Depth threshold"), _dir("Learning-rate confound", "challenges", (2,)),
                      _dir("Normalization instead", "outside", ())]}


def _session(tmp_path, jsons, **kw):
    return ResearchSession(MockLLM(jsons=jsons), PROFILE, tmp_path / "s", topic="mlp recipes",
                           sources=[ArxivClient(fetcher=lambda q, n: ATOM)], require_direction=True, **kw)


def _steered(idea, direction_id, link="tests H1"):
    return {**idea, "direction": direction_id, "steer_link": link}


def test_ideation_refuses_to_start_without_a_steer_and_a_chosen_direction(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, MAP])
    s.review()
    with pytest.raises(ValueError, match="starts from your steer"):
        s.ideate(1)
    s.set_steer(STEER, actor="renee")
    with pytest.raises(ValueError, match="Choose the directions"):
        s.ideate(1)
    assert len(s.llm.calls) == 2                                   # neither refusal spent a request
    s.map_directions()
    with pytest.raises(ValueError, match="Choose the directions"):
        s.ideate(1)
    pending = json.loads((s.out / "idea_pending_decisions.json").read_text())
    assert pending[0]["kind"] == "choose_directions" and len(pending[0]["direction_ids"]) == 3


def test_steer_reaches_the_query_planner_the_mapper_and_the_ideator(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, MAP])
    s.set_steer(STEER, actor="renee")
    s.review()
    assert "H2. Gains are a learning-rate artefact" in s.llm.calls[0]["prompt"]     # query planning
    board = s.map_directions()
    assert "Stay away from: bigger models" in s.llm.calls[2]["prompt"]
    chosen = s.choose_directions([2], actor="renee", note="keep it to one knob",
                                 own=["Compare skips against simply widening the layer"])
    assert [d["origin"] for d in chosen] == ["mapper", "author"]
    page = (s.out / "directions.md").read_text()
    assert "✅ **Learning-rate confound** | challenges H2" in page and "Written by you" in page

    mine = chosen[1]["id"]
    s.llm._jsons.extend([{"ideas": [_steered(_idea("LR-matched skips"), board[1]["id"]),
                                   _steered(_idea("Width control"), mine)]}])
    ideas = s.ideate(2)
    prompt = s.llm.calls[-1]["prompt"]
    assert "Directions the author CHOSE" in prompt and "keep it to one knob" in prompt
    assert board[0]["name"] not in prompt.split("Directions the author CHOSE")[1].split("Propose")[0]
    assert ideas[1].direction == mine
    md = (s.out / "ideas.md").read_text()
    assert "**Direction:** Learning-rate confound" in md and "**Follows your steer:** tests H1" in md
    handoff = s.select(1)
    assert handoff["author_direction"]["direction"]["name"] == "Learning-rate confound"
    assert handoff["author_direction"]["steer"]["avoid"] == ["bigger models"]


def test_ideas_outside_the_chosen_directions_are_rejected(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, MAP])
    s.set_steer(STEER, actor="renee")
    s.review(), s.map_directions()
    a, b = [d["id"] for d in s.choose_directions([1, 2], actor="renee")]
    other = s.state.direction_map[2]["id"]
    bad = [{"ideas": [_steered(_idea("A"), a), _steered(_idea("B"), other)]},      # unchosen direction
           {"ideas": [_steered(_idea("A"), a), _steered(_idea("B"), a)]},          # one chosen direction unused
           {"ideas": [_steered(_idea("A"), a), _steered(_idea("B"), b, link=" ")]}]  # no link to the steer
    s.llm._jsons.extend(bad)
    with pytest.raises(AgentError) as err:
        s.ideate(2)
    text = str(err.value) + " ".join(c["prompt"] for c in s.llm.calls[-2:])
    assert "must be one of the chosen ids" in text and "no idea belongs to chosen direction" in text
    assert "how it follows the author's steer" in str(err.value)


@pytest.mark.parametrize("mutate,message", [
    (lambda m: [m[0], _dir("Second look"), _dir("Third look", "outside", ())], "no direction addresses hunch"),
    (lambda m: [_dir("A", hunches=(1, 2)), _dir("B"), _dir("C")], "challenge a hunch or come from outside"),
    (lambda m: [m[0], m[1], _dir("C", "outside", (), sources=("arXiv:9999.00001",))], "unknown paper ids"),
    (lambda m: [m[0], m[1], _dir("C", "challenges", (7,))], "do not exist"),
    (lambda m: [m[0], m[1], _dir("depth  THRESHOLD", "outside", ())], "distinct names"),
    (lambda m: m[:2], "lay out 3 to 5 directions"),
])
def test_a_map_that_ignores_or_only_flatters_the_steer_is_rejected(tmp_path, mutate, message):
    s = _session(tmp_path, [QUERIES, REVIEW, {"directions": mutate(MAP["directions"])}, MAP])
    s.set_steer(STEER, actor="renee")
    s.review()
    assert [d["name"] for d in s.map_directions()] == [d["name"] for d in MAP["directions"]]
    assert message in s.llm.calls[-1]["prompt"]                    # the retry was told what was wrong


def test_a_new_steer_or_review_makes_the_map_and_the_candidates_stale(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, MAP])
    s.set_steer(STEER, actor="renee")
    s.review(), s.map_directions()
    d = s.choose_directions([1], actor="renee")[0]["id"]
    s.llm._jsons.append({"ideas": [_steered(_idea("A"), d)]})
    s.ideate(1)
    s.set_steer({**STEER, "hunches": ["Something else entirely is going on"]}, actor="renee")
    assert s.stage == "reviewed" and not s.ideas and not s.state.direction_map and not s.chosen
    assert [e["action"] for e in s.state.direction_log] == ["steer", "choose", "steer"]
    resumed = ResearchSession(MockLLM(), PROFILE, s.out, require_direction=True)
    assert resumed.state.steer["hunches"] == ["Something else entirely is going on"]


def test_waiving_the_steer_is_explicit_and_logged(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, {"ideas": [_idea("Skip MLP")]}])
    s.review()
    with pytest.raises(ValueError):
        s.waive_steer(actor="renee", reason=" ")
    s.waive_steer(actor="renee", reason="smoke test of the pipeline")
    assert s.ideate(1)[0].direction == "" and s.state.direction_log[-1]["action"] == "waive"


def test_steer_validation_and_terminal_intake():
    with pytest.raises(ValueError, match="at least a few words"):
        validate_steer({"interest": "regularization"})
    with pytest.raises(ValueError):
        validate_steer({"interest": "what helps small models most", "mood": "curious"})
    assert validate_steer({"interest": " what  helps small models most ", "hunches": ["", " a  b "]}) == {
        "interest": "what helps small models most", "hunches": ["a b"]}
    answers = iter(["skips", "do skips help shallow nets at all", "depth matters; lr confound", "", "", ""])
    steer = ask_steer(lambda _: next(answers), lambda *_: None)
    assert steer == {"interest": "do skips help shallow nets at all", "hunches": ["depth matters", "lr confound"]}


def test_interactive_asks_for_the_steer_then_the_direction_before_any_idea(tmp_path):
    s = _session(tmp_path, [QUERIES, REVIEW, MAP])
    answers = iter(["do skips help shallow nets at all", "depth matters; lr confound", "", "", "",
                    "a", "c 9", "o widen the layer instead of adding skips", "c 1 | one knob only", "s 1"])
    log = []

    def ask(prompt):
        answer = next(answers)
        if answer.startswith("c 1"):                               # the idea call comes after the choice
            first = s.state.direction_map[0]["id"]               # the author's own id is filled in below
            s.llm._jsons.append({"ideas": [_steered(_idea("A"), first), _steered(_idea("B"), "")]})
        return answer

    orig = s.choose_directions

    def choose(*args, **kw):
        out = orig(*args, **kw)
        s.llm._jsons[-1]["ideas"][1]["direction"] = out[1]["id"]
        return out

    s.choose_directions = choose
    handoff = s.interactive(n_ideas=2, input_fn=ask, print_fn=log.append)
    assert handoff["author_direction"]["note"] == "one knob only"
    assert any("Choose directions by their number" in line for line in log)
    assert len(s.llm.calls) == 4                                    # plan, review, map, ideas


def test_cli_stops_for_the_author_at_each_step(tmp_path, monkeypatch, capsys):
    out = tmp_path / "s"
    script = [QUERIES, REVIEW, MAP]
    llm = MockLLM(jsons=script)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr("scholarloop.core.llm.AnthropicLLM", lambda model=None: llm)
    monkeypatch.setattr(session_module, "ArxivClient", lambda: ArxivClient(fetcher=lambda q, n: ATOM))
    monkeypatch.setattr(session_module, "OpenAlexClient", lambda: ArxivClient(fetcher=lambda q, n: ATOM))
    base = ["--profile", "profiles/image-classification.yaml", "--out", str(out), "--no-tournament"]
    run = session_module._main

    assert run(base + ["--topic", "mlp recipes", "--auto"]) == 2                     # no steer: nothing runs
    assert "starts from your steer" in capsys.readouterr().err and not llm.calls
    args = ["--interest", STEER["interest"], "--hunch", STEER["hunches"][0], "--hunch", STEER["hunches"][1]]
    assert run(base + args + ["--stop-after", "ideas"]) == 0                         # stops at the map
    assert "choose: --directions" in capsys.readouterr().out and len(llm.calls) == 3
    assert not (out / "ideas.md").exists()

    board = json.loads((out / "directions.json").read_text())["map"]
    llm._jsons.append({"ideas": [_steered(_idea("A"), board[1]["id"])]})
    assert run(base + ["--directions", "2", "--n-ideas", "1", "--stop-after", "ideas"]) == 0
    assert "**Direction:** Learning-rate confound" in (out / "ideas.md").read_text()
    assert run(base + ["--select", "1"]) == 0 and (out / "handoff.json").exists()
