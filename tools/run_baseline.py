"""Measure the unchanged organizer template; never overwrite agent.py."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    os.chdir(ROOT)  # The official kit reads CSV paths relative to cwd.

    import numpy as np
    import pandas as pd
    from agent_template import Agent
    from local_eval import evaluate_agent
    from make_submission import build_submission

    results = []
    for seed in range(args.runs):
        started = time.perf_counter()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = evaluate_agent(Agent(), seed=seed, verbose=False)
        if result is None:
            raise RuntimeError(f'Official template produced no score for seed {seed}')
        row = {
            'seed': seed,
            'net_arpu_gain': result['net_arpu_gain'],
            'total_cost': result['total_cost'],
            'total_contacts': result['total_contacts'],
            'unique_customers': result['unique_customers_targeted'],
            'n_pilots': result['n_pilots'],
            'n_final_campaigns': result['n_campaigns'] - result['n_pilots'],
            'seconds': time.perf_counter() - started,
            'warnings': captured.getvalue().strip(),
        }
        results.append(row)
        print(f"seed={seed:2d} net={row['net_arpu_gain']:12,.2f} "
              f"final={row['n_final_campaigns']} pilots={row['n_pilots']}")

    first = build_submission(Agent(), seed=42)
    second = build_submission(Agent(), seed=42)
    if not first.equals(second):
        raise RuntimeError('Template submission differs between repeated seed=42 runs')

    net = [r['net_arpu_gain'] for r in results]
    report = {
        'agent': 'Unmodified organizer agent_template.Agent',
        'environment': 'Official MOCK environment; not a prediction of judging results',
        'versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': np.__version__},
        'summary': {
            'runs': len(results), 'positive_runs': sum(x > 0 for x in net),
            'median_net': statistics.median(net), 'min_net': min(net), 'max_net': max(net),
            'deterministic_submission_seed42': True,
        },
        'runs': results,
    }
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    (out / 'baseline.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    first.to_csv(out / 'baseline_submission.csv', index=False)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
