"""Benchmark an Agent class with the official public evaluator.

    python tools/benchmark.py --agent agent:Agent --seeds 0:10 --out reports/benchmark.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_seeds(value: str) -> list[int]:
    """Parse either comma-separated seeds or a stop-exclusive start:stop range."""
    try:
        if ':' in value:
            parts = value.split(':')
            if len(parts) not in (2, 3):
                raise ValueError
            args = [int(part) for part in parts]
            seeds = list(range(*args))
        else:
            seeds = [int(part.strip()) for part in value.split(',') if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'invalid seeds {value!r}') from exc
    if not seeds:
        raise argparse.ArgumentTypeError('seeds must not be empty')
    return seeds


def _load_agent(spec: str):
    module_name, separator, class_name = spec.partition(':')
    if not separator or not module_name or not class_name:
        raise ValueError("agent must use 'module:Class' format")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _worker(queue, action: str, agent_spec: str, seed: int) -> None:
    started = time.perf_counter()
    captured = io.StringIO()
    try:
        agent_class = _load_agent(agent_spec)
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            if action == 'evaluate':
                import local_eval
                original_sanitize = local_eval.sanitize_campaigns
                discard = {'count': 0}

                def tracking_sanitize(campaigns, tariffs):
                    valid = original_sanitize(campaigns, tariffs)
                    raw_count = len(campaigns) if isinstance(campaigns, (list, tuple)) else 0
                    discard['count'] = max(raw_count - len(valid), 0) + max(len(valid) - 10, 0)
                    return valid

                local_eval.sanitize_campaigns = tracking_sanitize
                result = local_eval.evaluate_agent(agent_class(), seed=seed, verbose=False)
                payload = {'result': result, 'dropped_campaigns': discard['count']}
            elif action == 'submission':
                from make_submission import build_submission
                payload = build_submission(agent_class(), seed=seed).to_csv(index=False, lineterminator='\n')
            else:
                raise ValueError(f'unknown worker action {action!r}')
        queue.put({'ok': True, 'payload': payload, 'output': captured.getvalue(),
                   'seconds': time.perf_counter() - started})
    except BaseException as exc:
        queue.put({'ok': False, 'exception': f'{type(exc).__name__}: {exc}',
                   'output': captured.getvalue(), 'seconds': time.perf_counter() - started})


def _run_worker(action: str, agent_spec: str, seed: int, timeout: float) -> dict:
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=_worker, args=(queue, action, agent_spec, seed))
    started = time.perf_counter()
    process.start()
    process.join(timeout)
    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        queue.close()
        return {'ok': False, 'timeout': True, 'exception': f'timeout after {timeout:g}s',
                'output': '', 'seconds': elapsed}
    try:
        message = queue.get(timeout=1)
    except Exception:
        message = {'ok': False, 'exception': f'worker exited with code {process.exitcode} without a result',
                   'output': '', 'seconds': elapsed}
    finally:
        queue.close()
    message['seconds'] = elapsed
    return message


def _git_metadata() -> dict:
    def command(*args: str) -> str:
        return subprocess.run(['git', *args], cwd=ROOT, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
    try:
        status = command('status', '--porcelain')
        return {'sha': command('rev-parse', 'HEAD'), 'dirty': bool(status),
                'status_porcelain': status.splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'sha': None, 'dirty': None, 'error': f'{type(exc).__name__}: {exc}'}


def _run_row(agent_spec: str, seed: int, timeout: float) -> dict:
    message = _run_worker('evaluate', agent_spec, seed, timeout)
    output = message.get('output', '').strip()
    caught = re.search(r'\[!\] Агент упал: ([^\r\n]+)', output)
    detected_drops = len(re.findall(r'Кампания #\d+ отброшена', output))
    row: dict[str, Any] = {
        'seed': seed, 'status': 'ok', 'seconds': round(message['seconds'], 6),
        'net_arpu_gain': None, 'total_cost': None, 'total_contacts': None,
        'unique_customers': None, 'n_final_campaigns': None, 'n_pilots': None,
        'dropped_campaigns': detected_drops, 'exception': message.get('exception'), 'output': output,
    }
    if message.get('timeout'):
        row['status'] = 'timeout'
        return row
    if not message.get('ok'):
        row['status'] = 'error'
        return row
    payload = message.get('payload') or {}
    result = payload.get('result')
    row['dropped_campaigns'] = payload.get('dropped_campaigns', detected_drops)
    if result is None:
        row['status'] = 'no_score'
        row['exception'] = row['exception'] or 'official evaluator returned no score'
        return row
    if caught:
        row['status'] = 'agent_error'
        row['exception'] = caught.group(1)
    row.update({
        'net_arpu_gain': result['net_arpu_gain'],
        'total_cost': result['total_cost'],
        'total_contacts': result['total_contacts'],
        'unique_customers': result['unique_customers_targeted'],
        'n_final_campaigns': result['n_campaigns'] - result['n_pilots'],
        'n_pilots': result['n_pilots'],
    })
    return row


def _summary(rows: list[dict], deterministic: bool | None) -> dict:
    scored = [row for row in rows if row['net_arpu_gain'] is not None]
    nets = [row['net_arpu_gain'] for row in scored]
    return {
        'runs': len(rows), 'scored_runs': len(scored),
        'successful_runs': sum(row['status'] == 'ok' for row in rows),
        'failed_runs': sum(row['status'] != 'ok' for row in rows),
        'timeouts': sum(row['status'] == 'timeout' for row in rows),
        'positive_runs': sum(value > 0 for value in nets),
        'positive_rate_all_runs': sum(value > 0 for value in nets) / len(rows),
        'median_net': statistics.median(nets) if nets else None,
        'min_net': min(nets) if nets else None,
        'max_net': max(nets) if nets else None,
        'deterministic_submission_seed42': deterministic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', required=True, help="import path in 'module:Class' format")
    parser.add_argument('--seeds', type=parse_seeds, default=parse_seeds('0:10'))
    parser.add_argument('--out', type=Path, default=ROOT / 'reports' / 'benchmark.json')
    parser.add_argument('--timeout', type=float, default=300.0, help='seconds allowed per evaluation')
    args = parser.parse_args()
    if not math_is_positive_finite(args.timeout):
        parser.error('--timeout must be a finite positive number')

    try:
        _load_agent(args.agent)
    except (ImportError, AttributeError, ValueError) as exc:
        parser.error(f'cannot load {args.agent!r}: {type(exc).__name__}: {exc}')

    git = _git_metadata()
    rows = []
    for seed in args.seeds:
        row = _run_row(args.agent, seed, args.timeout)
        rows.append(row)
        net = 'NA' if row['net_arpu_gain'] is None else f"{row['net_arpu_gain']:,.2f}"
        print(f"seed={seed:4d} status={row['status']:<11} net={net:>14} "
              f"final={row['n_final_campaigns']} pilots={row['n_pilots']} seconds={row['seconds']:.2f}")

    first = _run_worker('submission', args.agent, 42, args.timeout)
    second = _run_worker('submission', args.agent, 42, args.timeout)
    deterministic = first.get('ok') and second.get('ok') and first['payload'] == second['payload']
    reproducibility = {
        'seed': 42, 'deterministic': bool(deterministic),
        'first_exception': first.get('exception'), 'second_exception': second.get('exception'),
        'first_seconds': round(first['seconds'], 6), 'second_seconds': round(second['seconds'], 6),
    }

    import numpy as np
    import pandas as pd
    report = {
        'agent': args.agent,
        'environment': 'Official public MOCK environment; not a prediction of judging results',
        'git': git,
        'versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': np.__version__},
        'seeds': args.seeds,
        'timeout_seconds': args.timeout,
        'summary': _summary(rows, bool(deterministic)),
        'submission_reproducibility': reproducibility,
        'runs': rows,
    }
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    csv_path = out.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    print(f'wrote {out} and {csv_path}')


def math_is_positive_finite(value: float) -> bool:
    try:
        return math.isfinite(value) and value > 0
    except TypeError:
        return False


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
