import argparse

import pytest

from tools.benchmark import _summary, parse_seeds


def test_seed_range_is_stop_exclusive():
    assert parse_seeds('0:10') == list(range(10))
    assert parse_seeds('2:8:2') == [2, 4, 6]
    assert parse_seeds('1,3,42') == [1, 3, 42]


@pytest.mark.parametrize('value', ['', 'a', '1:2:3:4', '3:3'])
def test_invalid_seed_spec(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seeds(value)


def test_failures_remain_in_summary_denominator():
    rows = [
        {'status': 'ok', 'net_arpu_gain': 10},
        {'status': 'ok', 'net_arpu_gain': -5},
        {'status': 'timeout', 'net_arpu_gain': None},
    ]
    summary = _summary(rows, True)
    assert summary['runs'] == 3
    assert summary['scored_runs'] == 2
    assert summary['positive_runs'] == 1
    assert summary['positive_rate_all_runs'] == pytest.approx(1 / 3)
    assert summary['failed_runs'] == summary['timeouts'] == 1


def test_large_worker_output_and_arbitrary_cwd(tmp_path, monkeypatch):
    from tools.benchmark import _run_worker
    (tmp_path / 'loud_review_agent.py').write_text(
        "class Agent:\n"
        "    def act(self, env):\n"
        "        print('x' * 1000000)\n"
        "        return [{'target_tariff': 'tariff_9', 'channel': 'push', 'filter_arpu_segment': 'LOW'}]\n",
        encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = _run_worker('submission', 'loud_review_agent:Agent', 42, 15)
    assert result['ok'], result
    assert len(result['output']) >= 1000000


def test_worker_timeout_is_reported(tmp_path, monkeypatch):
    from tools.benchmark import _run_worker
    (tmp_path / 'slow_review_agent.py').write_text(
        "import time\nclass Agent:\n    def act(self, env):\n        time.sleep(30)\n        return []\n",
        encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    result = _run_worker('submission', 'slow_review_agent:Agent', 42, 1)
    assert not result['ok'] and result['timeout']


def test_empty_plan_is_not_a_success(tmp_path, monkeypatch):
    from tools.benchmark import _run_row
    (tmp_path / 'empty_review_agent.py').write_text(
        "class Agent:\n"
        "    def act(self, env):\n"
        "        env.run_pilot(target_tariff='tariff_9', channel='push', n_customers=10)\n"
        "        return []\n", encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    result = _run_row('empty_review_agent:Agent', 42, 15)
    assert result['status'] == 'invalid_plan'
    assert result['preflight_valid'] is False
