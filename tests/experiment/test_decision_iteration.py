import copy
import json

import pytest

from scholarloop.core.agent import AgentTrace, AgentError
from scholarloop.core.llm import MockLLM
from scholarloop.experiment.decision_bench import make_tasks, public_task
from scholarloop.experiment.decision_iteration import IterationPolicy, synthetic_dossier
from scholarloop.experiment.decision_policy import run_model_benchmark
from tests.experiment.test_research_cycle import ASSESSMENT


def assessment(prompt):
    ctx = json.loads(prompt)
    result = copy.deepcopy(ASSESSMENT)
    result['question'] = ctx['question']
    result['evidence'] = [row['id'] for row in ctx['dossier']['runs']]
    result['claim_scope'] = 'Synthetic development observations only.'
    result['choice'] = 'matched_compute' if not ctx['history'] else 'abstain'
    result['next_step'].update(kind='ablation' if not ctx['history'] else 'stop',
                               max_training_attempts=1 if not ctx['history'] else 0,
                               target_issue_ids=[i['issue_id'] for i in ctx['tracked_issues']])
    result['issue_updates'] = [{'issue_id': i['issue_id'], 'status': 'open',
        'rationale': 'Other explanations remain.', 'evidence': result['evidence']}
        for i in ctx['tracked_issues']]
    return result


def test_real_reviewer_component_replays_issues_and_observations(tmp_path):
    clients = []
    def factory():
        llm = MockLLM(router=assessment)
        llm.model = 'mock-iteration'
        clients.append(llm)
        return llm
    report = run_model_benchmark(tmp_path, factory, model='mock-iteration', repeats=1,
                                 budget=1, arms=('iteration',))
    arm = report['arms']['iteration']
    assert arm['reserved_llm_calls'] == 8
    assert arm['summary']['abstentions'] == 4
    assert arm['summary']['total_cost_units'] == 4
    for llm in clients:
        first, second = [json.loads(call['prompt']) for call in llm.calls]
        assert first['tracked_issues'] == []
        assert second['tracked_issues'][0]['opened_round'] == 1
        assert len(second['dossier']['runs']) == 2
        assert second['history'][0]['assessment']['choice'] == 'matched_compute'
        assert second['dossier']['source_kind'] == 'synthetic_development_observations'
    paths = list(tmp_path.glob('iteration/*/synthetic_iterations.json'))
    assert len(paths) == 4
    assert all(json.loads(p.read_text())['source_kind'] == 'synthetic_component_evaluation_only' for p in paths)
    assert not list(tmp_path.rglob('scientific_review.json'))
    assert not list(tmp_path.rglob('experiments.jsonl'))


@pytest.mark.parametrize('mode', ['invented', 'real_source', 'action_mismatch', 'terminal_mismatch', 'budget'])
def test_adapter_rejects_unobserved_evidence_and_unbound_choices(tmp_path, mode):
    task = public_task(make_tasks()[0])
    ctx = {'task': task, 'history': [], 'remaining_budget': 1}
    # Retry prompts are not pure JSON; derive the intentionally invalid response once.
    base = {'question': task['question'], 'dossier': synthetic_dossier(ctx), 'history': [], 'tracked_issues': []}
    result = assessment(json.dumps(base))
    if mode == 'invented':
        result['evidence'] = ['synthetic:unobserved']
    elif mode == 'real_source':
        result['evidence'] = ['train']
    elif mode == 'action_mismatch':
        result['next_step']['kind'] = 'boundary_test'
    elif mode == 'terminal_mismatch':
        result['choice'] = 'genuine'
    else:
        ctx['remaining_budget'] = 0
    path = tmp_path / 'synthetic_iterations.json'
    policy = IterationPolicy(MockLLM(jsons=[result] * 3), AgentTrace(), path)
    with pytest.raises(AgentError):
        policy(ctx)
    assert policy.prior == [] and not path.exists()


def test_repeated_simulated_evidence_is_not_counted_as_new_observation():
    ctx = {'task': public_task(make_tasks()[0]), 'history': [
        {'action': 'matched_compute', 'gain': 8, 'evidence_id': 'same'},
        {'action': 'matched_compute', 'gain': 8, 'evidence_id': 'same'}]}
    assert len(synthetic_dossier(ctx)['runs']) == 2
    ctx['history'][1]['gain'] = 0
    with pytest.raises(ValueError, match='conflicting'):
        synthetic_dossier(ctx)
