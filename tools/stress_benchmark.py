"""Run the team agent on labelled synthetic stress environments.

These scenarios test mechanics and robustness; they are not organizer hidden tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment import make_environment
from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET
from scoring_core import score_campaigns, sanitize_campaigns
from validation.plan import validate_plan
from tools.benchmark import _git_metadata, parse_seeds

TARIFFS = ('tariff_1', 'tariff_8', 'tariff_10')
SEGMENTS = ('LOW', 'MID', 'HIGH')


def _tariffs() -> pd.DataFrame:
    return pd.DataFrame({
        'tariff_plan_code': TARIFFS,
        'price_tariff': [0.0, 6153.1, 7887.8],
        'Data_in_PKG': [0, 25600, 35840],
        'Min_another_operator_in_PKG': [0, 100, 160],
        'Min_another_operator_and_city_in_PKG': [0, 100, 160],
    })


def _profile(scenario: str) -> pd.DataFrame:
    rows = []
    next_id = 1
    for current in TARIFFS:
        for segment in SEGMENTS:
            n = 80
            if scenario == 'rare_good' and (current, segment) == ('tariff_1', 'LOW'):
                n = 12
            if scenario == 'small_and_empty':
                n = 5 if (current, segment) == ('tariff_1', 'LOW') else 0
                if (current, segment) == ('tariff_10', 'HIGH'):
                    n = 120
            value = {'LOW': 700.0, 'MID': 3500.0, 'HIGH': 9000.0}[segment]
            for index in range(n):
                rows.append({
                    'ID_NUMBER': next_id, 'current_tariff': current, 'arpu_segment': segment,
                    'data_segment': ('LITE', 'HEAVY', 'NON_USER')[index % 3],
                    'call_segment': ('LOW', 'MEDIUM', 'HIGH')[index % 3],
                    'predicted_arpu': value + index,
                })
                next_id += 1
    if scenario == 'small_and_empty':
        for _ in range(5):
            rows.append({'ID_NUMBER': next_id, 'current_tariff': None, 'arpu_segment': None,
                         'data_segment': None, 'call_segment': 'LOW', 'predicted_arpu': 1000.0})
            next_id += 1
    return pd.DataFrame(rows)


def _true_change(scenario: str, current: str, segment: str, target: str) -> float:
    if scenario == 'all_bad':
        return -0.20
    if scenario == 'mixed':
        return 0.35 if target == 'tariff_10' and segment in {'MID', 'HIGH'} else -0.12
    if scenario == 'rare_good':
        return 1.20 if (current, segment, target) == ('tariff_1', 'LOW', 'tariff_10') else -0.15
    if scenario == 'unlucky_noisy_pilots':
        return 0.08 if target == 'tariff_10' else -0.05
    if scenario == 'small_and_empty':
        return 0.30 if (current, segment, target) == ('tariff_10', 'HIGH', 'tariff_8') else -0.10
    raise ValueError(f'unknown scenario {scenario!r}')


def _impact_model(scenario: str) -> pd.DataFrame:
    return pd.DataFrame([
        {'tariff_plan_code_from': current, 'tariff_plan_code_to': target, 'arpu_segment': segment,
         'arpu_change_pct': _true_change(scenario, current, segment, target), 'conversion_rate': 1.0}
        for current in TARIFFS for target in TARIFFS if current != target for segment in SEGMENTS
    ])


def _fallback(current, target, segment, tariffs, fallback_conversion):
    return -0.1, fallback_conversion


def run_scenario(agent_class, scenario: str, seed: int) -> dict:
    profile, tariffs, model = _profile(scenario), _tariffs(), _impact_model(scenario)
    env, internals = make_environment(profile, model, tariffs, CHANNELS, TOTAL_BUDGET,
                                      MAX_TOTAL_CONTACTS, _fallback, seed=seed)
    started = time.perf_counter()
    exception = None
    try:
        final = agent_class().act(env) or []
    except Exception as exc:
        final = []
        exception = f'{type(exc).__name__}: {exc}'
    final = sanitize_campaigns(final, tariffs)[:10]
    preflight = validate_plan(final, profile, tariffs, CHANNELS,
                              env.remaining_budget, env.remaining_contacts)
    pilots = internals.executed_pilot_campaigns()
    all_campaigns = pd.DataFrame(pilots + final)
    if all_campaigns.empty:
        result = None
    else:
        for column in ('filter_arpu_segment', 'filter_data_segment', 'filter_call_segment',
                       'filter_current_tariff', 'explicit_ids'):
            if column not in all_campaigns:
                all_campaigns[column] = None
        result = score_campaigns(all_campaigns, profile, model, tariffs,
                                 float(profile['predicted_arpu'].sum()), _fallback, team_id='stress')
    return {
        'scenario': scenario, 'seed': seed, 'status': 'ok' if result is not None and exception is None else 'error',
        'net_arpu_gain': result['net_arpu_gain'] if result else None,
        'total_cost': result['total_cost'] if result else None,
        'total_contacts': result['total_contacts'] if result else None,
        'n_final_campaigns': len(final), 'n_pilots': len(pilots),
        'negative_observed_pilots': sum(item['observed_lift_ratio'] < 0 for item in env.pilot_history),
        'preflight_valid': preflight['valid'], 'preflight_errors': preflight['errors'],
        'exception': exception, 'seconds': round(time.perf_counter() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', default='agent:Agent')
    parser.add_argument('--seeds', type=parse_seeds, default=parse_seeds('0:5'))
    parser.add_argument('--out', type=Path, default=ROOT / 'reports' / 'benchmark_stress.json')
    args = parser.parse_args()
    module_name, separator, class_name = args.agent.partition(':')
    if not separator:
        parser.error("--agent must use 'module:Class' format")
    agent_class = getattr(__import__(module_name, fromlist=[class_name]), class_name)
    scenarios = ('all_bad', 'mixed', 'rare_good', 'unlucky_noisy_pilots', 'small_and_empty')
    rows = []
    for scenario in scenarios:
        for seed in args.seeds:
            row = run_scenario(agent_class, scenario, seed)
            rows.append(row)
            print(f"{scenario:<22} seed={seed} net={row['net_arpu_gain']} "
                  f"pilots={row['n_pilots']} negative={row['negative_observed_pilots']} valid={row['preflight_valid']}")
    by_scenario = {}
    for scenario in scenarios:
        subset = [row for row in rows if row['scenario'] == scenario]
        nets = [row['net_arpu_gain'] for row in subset if row['net_arpu_gain'] is not None]
        by_scenario[scenario] = {
            'runs': len(subset), 'scored_runs': len(nets), 'positive_runs': sum(net > 0 for net in nets),
            'median_net': statistics.median(nets) if nets else None,
            'min_net': min(nets) if nets else None, 'max_net': max(nets) if nets else None,
            'invalid_preflight_runs': sum(not row['preflight_valid'] for row in subset),
            'runs_with_negative_observed_pilots': sum(row['negative_observed_pilots'] > 0 for row in subset),
        }
    report = {
        'agent': args.agent,
        'disclaimer': 'Labelled synthetic stress scenarios; not organizer hidden tests or judging predictions.',
        'git': _git_metadata(),
        'versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': np.__version__},
        'seeds': args.seeds, 'summary': by_scenario, 'runs': rows,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with out.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(by_scenario, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
