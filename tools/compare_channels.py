"""Compare channel allocation with identical pilot decisions and observations.

Both agents use exactly the same current exploration policy. Only final channel
allocation differs. The historical shadow-price planner remains the reference.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import Agent
from strategy.portfolio import shadow_portfolio
from tools.benchmark import _git_metadata, parse_seeds


class ReferenceAgent(Agent):
    def plan_portfolio(self, options, budget, contacts):
        return shadow_portfolio(options, budget, contacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=parse_seeds, default=list(range(10)))
    parser.add_argument('--out', type=Path, default=ROOT / 'reports' / 'channel_allocation.json')
    args = parser.parse_args()
    output = args.out.resolve()
    os.chdir(ROOT)
    from local_eval import evaluate_agent
    provenance = _git_metadata()
    rows = []
    for seed in args.seeds:
        agents = [ReferenceAgent(), Agent()]
        results, seconds = [], []
        for agent in agents:
            started = time.perf_counter()
            result = evaluate_agent(agent, seed=seed, verbose=False)
            seconds.append(time.perf_counter() - started)
            if result is None or not agent.last_trace.get('preflight', {}).get('valid'):
                raise RuntimeError(f'No valid complete plan at seed {seed}; comparison aborted')
            results.append(result)
        if agents[0].last_trace['pilots'] != agents[1].last_trace['pilots']:
            raise RuntimeError(f'Pilot decisions differ at seed {seed}; comparison is not isolated')
        estimated = [sum(c['risk_adjusted_gain'] for c in a.last_trace['final']['campaigns']) for a in agents]
        if estimated[1] < estimated[0] - 1e-7:
            raise RuntimeError(f'Planner objective decreased at seed {seed}')
        row = {
            'seed': seed, 'identical_pilots': True,
            'reference_net': results[0]['net_arpu_gain'], 'current_net': results[1]['net_arpu_gain'],
            'net_difference': results[1]['net_arpu_gain'] - results[0]['net_arpu_gain'],
            'reference_objective': estimated[0], 'current_objective': estimated[1],
            'objective_difference': estimated[1] - estimated[0],
            'reference_cost': results[0]['total_cost'], 'current_cost': results[1]['total_cost'],
            'reference_contacts': results[0]['total_contacts'], 'current_contacts': results[1]['total_contacts'],
            'reference_seconds': seconds[0], 'current_seconds': seconds[1],
            'preflight_valid': True,
        }
        rows.append(row)
        print(f"seed={seed:2} net delta={row['net_difference']:,.2f} objective delta={row['objective_difference']:,.2f}", flush=True)
    nets = [r['current_net'] for r in rows]
    deltas = [r['net_difference'] for r in rows]
    report = {
        'git': provenance,
        'environment': 'Official mock; same pilots; improvement in estimates is not a hidden-score guarantee',
        'summary': {'runs': len(rows), 'positive_runs': sum(n > 0 for n in nets),
                    'net_improved_runs': sum(d > 1e-7 for d in deltas),
                    'net_regressed_runs': sum(d < -1e-7 for d in deltas),
                    'median_net': statistics.median(nets), 'min_net': min(nets),
                    'mean_net_difference': statistics.mean(deltas), 'min_net_difference': min(deltas),
                    'objective_improved_runs': sum(r['objective_difference'] > 1e-7 for r in rows),
                    'max_seconds': max(r['current_seconds'] for r in rows)},
        'runs': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
