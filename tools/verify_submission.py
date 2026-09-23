"""Read-only acceptance check of the committed submission and its evidence.

Regenerates artifacts in a temporary directory using a fresh Python process.
Exit 0 means all listed checks passed; exit 1 means the report explains a failure.
It does not submit anything to a platform or replace the committed CSV/trace.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(timeout=300):
    started = time.perf_counter()
    checks = []
    report = {'environment': 'Official mock; validates reproducibility, not hidden judging performance', 'checks': checks}

    def record(name, passed, detail):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    try:
        csv_path = ROOT / 'submission.csv'
        trace_path = ROOT / 'reports' / 'decision_trace.json'
        csv_before = csv_path.read_bytes()
        trace_before = trace_path.read_bytes()
        trace = json.loads(trace_before)
        record('seed', trace.get('submission_seed') == 42, 'Committed evidence must use official seed 42')
        record('repeated_export', trace.get('deterministic_submission') is True, 'Exporter recorded two identical generations')
        canonical_csv = csv_before.replace(b'\r\n', b'\n')
        record('csv_hash', hashlib.sha256(canonical_csv).hexdigest() == trace.get('submission_sha256'),
               'Committed CSV matches the trace hash after LF normalization')
        for field in ('source_sha256', 'input_sha256'):
            manifest = trace.get(field)
            if not isinstance(manifest, dict) or not manifest:
                record(field, False, 'Missing nonempty checksum manifest')
                continue
            for name, expected in sorted(manifest.items()):
                path = (ROOT / name).resolve()
                safe = path.is_relative_to(ROOT)
                valid = safe and path.is_file() and sha256(path) == expected
                record(f'{field}:{name}', valid, 'Current file must match exported evidence')

        with tempfile.TemporaryDirectory(prefix='tandem-verify-') as directory:
            temporary = Path(directory).resolve()
            generated_csv, generated_trace = temporary / 'submission.csv', temporary / 'trace.json'
            process = subprocess.run(
                [sys.executable, str(ROOT / 'tools' / 'export_strategy.py'),
                 '--out', str(generated_csv), '--trace', str(generated_trace)],
                cwd=temporary, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
            )
            record('fresh_process_export', process.returncode == 0,
                   'Export from outside the project directory' if process.returncode == 0
                   else (process.stderr or process.stdout)[-4000:])
            if process.returncode == 0:
                generated = json.loads(generated_trace.read_text(encoding='utf-8'))
                record('csv_reproduced', generated_csv.read_bytes() == canonical_csv,
                       'Fresh seed-42 export equals committed CSV exactly')
                for field in ('source_sha256', 'input_sha256'):
                    record(f'{field}:complete', generated.get(field) == trace.get(field),
                           'Committed manifest includes every current export dependency')
                preflight = generated.get('preflight', {})
                final = generated.get('final', {})
                record('preflight', preflight.get('valid') is True, preflight.get('errors', []))
                record('disjoint_audiences', preflight.get('unique_customers') == preflight.get('total_contacts'),
                       'No repeated IDs between final campaign audiences')
                count = len(preflight.get('campaigns', []))
                record('campaign_count', 1 <= count <= 10, f'{count} final campaigns')
                record('total_budget', 0 <= final.get('total_cost_including_pilots', float('inf')) <= 100000,
                       final.get('total_cost_including_pilots'))
                record('total_contacts', 0 <= final.get('total_contacts_including_pilots', float('inf')) <= 15000,
                       final.get('total_contacts_including_pilots'))
                pilot_count = len(generated.get('pilots', []))
                record('pilots', 1 <= pilot_count <= 20, f'{pilot_count} recorded pilots')
                record('runtime', time.perf_counter() - started < timeout, 'Two seed-42 runs within the configured total time allowance')
                report['result'] = {'campaigns': count, 'pilots': pilot_count,
                                    'total_contacts': final.get('total_contacts_including_pilots'),
                                    'total_cost': final.get('total_cost_including_pilots'),
                                    'versions': generated.get('versions')}
        record('committed_artifacts_unchanged', csv_path.read_bytes() == csv_before and trace_path.read_bytes() == trace_before,
               'Verification did not replace committed submission or evidence')
    except Exception as exc:
        record('execution', False, f'{type(exc).__name__}: {exc}')
    report['passed'] = bool(checks) and all(c['passed'] for c in checks)
    report['seconds'] = time.perf_counter() - started
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    report = verify(args.timeout)
    for check in report['checks']:
        print(f"{'OK' if check['passed'] else 'FAIL'} {check['name']}")
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2, ensure_ascii=False))
    if not report['passed']:
        print(json.dumps([c for c in report['checks'] if not c['passed']], indent=2, ensure_ascii=False))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
