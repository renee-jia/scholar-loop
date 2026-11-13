import copy

import pytest

from scholarloop.experiment.decision_bench import (
    ACTIONS, baseline, make_tasks, public_task, run_benchmark, run_episode, summarize,
)


def test_balanced_reproducible_tasks_have_no_explicit_answers_in_policy_view():
    tasks = make_tasks(2, 12)
    assert tasks == make_tasks(2, 12)
    assert tasks != make_tasks(2, 13)
    assert len({t['id'] for t in tasks}) == 8
    for task in tasks:
        view = public_task(task)
        assert 'mechanism' not in view and 'observations' not in view
        assert 6 < view['initial_gain'] < 10
    assert all(sum(t['mechanism'] == m for t in tasks) == 2
               for m in ('leakage', 'compute', 'subgroup', 'genuine'))


def test_full_control_budget_identifies_all_fixture_mechanisms():
    rows = [run_episode(t, baseline('fixed'), 5) for t in make_tasks()]
    assert all(r['correct'] and r['spent'] <= 5 for r in rows)
    assert summarize(rows)['accuracy'] == 1


def test_zero_budget_abstention_is_not_success():
    rows = [run_episode(t, baseline('fixed'), 0) for t in make_tasks()]
    summary = summarize(rows)
    assert summary['accuracy'] == 0 and summary['coverage'] == 0
    assert summary['selective_error'] is None
    assert summary['abstentions'] == len(rows)


def test_invalid_or_unaffordable_actions_never_execute():
    task = make_tasks()[0]
    row = run_episode(task, lambda ctx: {'action': 'clean_split'}, 1)
    assert row['status'] == 'budget_exhausted' and row['spent'] == 0 and row['history'] == []
    row = run_episode(task, lambda ctx: {'action': 'invented'}, 5)
    assert row['status'] == 'failed' and not row['correct']


def test_repeated_actions_cost_budget_but_not_new_evidence():
    row = run_episode(make_tasks()[0], lambda ctx: {'action': 'matched_compute'}, 2)
    assert row['spent'] == 2 and row['status'] == 'budget_exhausted'
    assert row['history'][0] == row['history'][1]


def test_policy_mutation_cannot_change_truth_or_recorded_evidence():
    task = make_tasks()[0]
    original = copy.deepcopy(task)
    def mutate(ctx):
        ctx['task']['actions'].clear()
        ctx['history'].append({'fake': True})
        return {'decision': 'abstain'}
    row = run_episode(task, mutate, 5)
    assert row['history'] == [] and task == original and len(ACTIONS) == 3


def test_failure_preserves_cost_and_denominator():
    def fail(ctx):
        if ctx['history']:
            raise RuntimeError('failed after observation')
        return {'action': 'matched_compute'}
    rows = [run_episode(t, fail, 3) for t in make_tasks(1)]
    summary = summarize(rows)
    assert summary['failed'] == summary['planned_tasks'] == 4
    assert summary['accuracy'] == 0 and summary['total_cost_units'] == 4


def test_accept_all_exposes_false_acceptance_instead_of_favoring_coverage():
    summary = summarize([run_episode(t, baseline('accept'), 0) for t in make_tasks(1)])
    assert summary['accuracy'] == 0.25 and summary['coverage'] == 1
    assert summary['false_genuine_acceptance'] == 1


def test_frozen_plan_and_free_repeatability(tmp_path):
    report = run_benchmark(tmp_path / 'bench', repeats=1)
    assert run_benchmark(tmp_path / 'bench', repeats=1) == report
    assert all(arm['summary']['planned_tasks'] == 4 for arm in report['arms'].values())
    with pytest.raises(ValueError, match='changed'):
        run_benchmark(tmp_path / 'bench', repeats=1, budget=4)
