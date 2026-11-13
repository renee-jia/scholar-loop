import json

import pytest

from scholarloop.core.llm import MockLLM
from scholarloop.experiment.decision_policy import run_model_benchmark


def client(router=None):
    llm = MockLLM(router=router, jsons=None if router else [
        {"choice": "abstain", "rationale": "Insufficient evidence", "uncertainties": ["Cause unknown"]}])
    llm.model = "mock-policy"
    return llm


def test_model_arms_share_inputs_and_isolate_state_between_tasks(tmp_path):
    clients = []
    def factory():
        llm = client()
        clients.append(llm)
        return llm
    report = run_model_benchmark(tmp_path, factory, model="mock-policy", repeats=1)
    assert len(clients) == 8
    contexts = [json.loads(c.calls[0]['prompt']) for c in clients]
    assert contexts[:4] == contexts[4:]
    assert all(ctx['prior_decisions'] == [] for ctx in contexts)
    assert all('mechanism' not in ctx['task'] and 'observations' not in ctx['task'] for ctx in contexts)
    assert all(data['reserved_llm_calls'] == 4 for data in report['arms'].values())
    assert run_model_benchmark(tmp_path, lambda: pytest.fail('must reuse'), model="mock-policy", repeats=1) == report


def test_observations_and_prior_decisions_reach_next_call(tmp_path):
    clients = []
    def router(prompt):
        ctx = json.loads(prompt)
        return {"choice": "matched_compute" if not ctx['history'] else "abstain",
                "rationale": "Check the matched control", "uncertainties": ["Other causes"]}
    def factory():
        llm = client(router)
        clients.append(llm)
        return llm
    report = run_model_benchmark(tmp_path, factory, model="mock-policy", repeats=1, budget=1)
    assert all(r['spent'] == 1 for data in report['arms'].values() for r in data['episodes'])
    second = json.loads(clients[0].calls[1]['prompt'])
    assert second['history'][0]['action'] == 'matched_compute'
    assert second['prior_decisions'][0]['choice'] == 'matched_compute'


def test_retries_cannot_exceed_persistent_request_budget(tmp_path):
    clients = []
    def factory():
        llm = client(lambda prompt: {"invalid": True})
        clients.append(llm)
        return llm
    report = run_model_benchmark(tmp_path, factory, model="mock-policy", repeats=1, max_calls=1)
    assert all(len(c.calls) == 1 for c in clients)
    assert all(data['summary']['failed'] == 4 for data in report['arms'].values())
    assert all(data['reserved_llm_calls'] == 4 for data in report['arms'].values())


def test_frozen_identity_and_interrupted_episode_do_not_silently_restart(tmp_path):
    report = run_model_benchmark(tmp_path, client, model="mock-policy", repeats=1)
    with pytest.raises(ValueError, match='changed'):
        run_model_benchmark(tmp_path, client, model="another", repeats=1)
    task = report['arms']['plain']['episodes'][0]['task_id']
    (tmp_path / 'plain' / task / 'result.json').unlink()
    resumed = run_model_benchmark(tmp_path, lambda: pytest.fail('no new calls'), model="mock-policy", repeats=1)
    summary = resumed['arms']['plain']['summary']
    assert summary['planned_tasks'] == 4 and summary['interrupted'] == 1
    assert summary['unknown_cost_episodes'] == 1 and summary['total_cost_units'] is None


def test_wrong_client_model_fails_without_query(tmp_path):
    clients = []
    def factory():
        llm = client()
        clients.append(llm)
        return llm
    report = run_model_benchmark(tmp_path, factory, model="wrong", repeats=1)
    assert all(not c.calls for c in clients)
    assert all(data['summary']['failed'] == 4 for data in report['arms'].values())


def test_cached_episode_cannot_be_relabelled_as_another_arm(tmp_path):
    report = run_model_benchmark(tmp_path, client, model="mock-policy", repeats=1)
    task = report['arms']['plain']['episodes'][0]['task_id']
    path = tmp_path / 'plain' / task / 'result.json'
    row = json.loads(path.read_text())
    row['arm'] = 'structured'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Saved episode'):
        run_model_benchmark(tmp_path, lambda: pytest.fail('no calls'), model="mock-policy", repeats=1)
