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
