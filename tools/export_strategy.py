"""Generate the official seed-42 submission and an explanatory decision trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def replace_file(path, content):
    """Replace one artifact atomically; failure never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--trace", type=Path, default=ROOT / "reports" / "decision_trace.json")
    args = parser.parse_args()
    output, trace_path = args.out.resolve(), args.trace.resolve()
    if output == trace_path:
        parser.error("Submission and trace must be different files")
    for path, suffix, default in ((output, '.csv', ROOT / 'submission.csv'),
                                  (trace_path, '.json', ROOT / 'reports' / 'decision_trace.json')):
        if path.suffix.lower() != suffix:
            parser.error(f'Output must have {suffix} extension')
        # Custom artifacts may live outside the repository or inside reports/.
        # A resolved path handles ../ and symlinks before checking containment.
        if path.is_relative_to(ROOT) and path != default and not path.is_relative_to(ROOT / 'reports'):
            parser.error('Custom repository outputs must be inside reports/; source/input locations are protected')
    os.chdir(ROOT)
    from agent import Agent
    from make_submission import SUBMISSION_SEED, build_submission

    agent = Agent()
    submission = build_submission(agent, seed=SUBMISSION_SEED)
    csv_bytes = submission.to_csv(index=False, lineterminator='\n').encode('utf-8')
    second = build_submission(Agent(), seed=SUBMISSION_SEED)
    if csv_bytes != second.to_csv(index=False, lineterminator='\n').encode('utf-8'):
        raise RuntimeError('Submission differs between two seed-42 executions; existing artifacts preserved')
    if not agent.last_trace.get('preflight', {}).get('valid'):
        raise RuntimeError('Submission has no successful preflight; existing artifacts preserved')
    agent.last_trace["submission_seed"] = SUBMISSION_SEED
    agent.last_trace['hash_format'] = 'sha256-text-lf-v1'
    agent.last_trace["deterministic_submission"] = True
    agent.last_trace["submission_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    agent.last_trace["environment"] = "Organizer mock; observations do not predict judging effects"
    source_files = [ROOT / "agent.py", *sorted((ROOT / "strategy").glob("*.py")),
                    *sorted((ROOT / "validation").glob("*.py")), Path(__file__).resolve(),
                    ROOT / 'environment.py', ROOT / 'mock_environment.py', ROOT / 'scoring_core.py',
                    ROOT / 'make_submission.py', ROOT / 'requirements.txt']
    data_files = [ROOT / 'customer_profile.csv', ROOT / 'tariff_dictionary.csv',
                  ROOT / 'data' / 'dict_tariff.csv', ROOT / 'data' / 'change_tariff.csv']
    if output in source_files + data_files or trace_path in source_files + data_files:
        parser.error('Output must not overwrite source code or input data')
    agent.last_trace["source_sha256"] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for path in source_files
    }
    agent.last_trace['input_sha256'] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for path in data_files
    }
    import numpy as np
    import pandas as pd
    agent.last_trace['versions'] = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__}
    trace_bytes = (json.dumps(agent.last_trace, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    # Both payloads are fully generated and validated before touching output.
    # Replacements are individually atomic; the trace hash detects a mismatched pair.
    replace_file(trace_path, trace_bytes)
    replace_file(output, csv_bytes)
    print(f"Exported {len(submission)} campaigns to {output.name}")
    print(f"Decision trace: {trace_path}")


if __name__ == "__main__":
    main()
