"""Generate the official seed-42 submission and an explanatory decision trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--trace", type=Path, default=ROOT / "reports" / "decision_trace.json")
    args = parser.parse_args()
    output, trace_path = args.out.resolve(), args.trace.resolve()
    if output == trace_path:
        parser.error("Submission and trace must be different files")
    os.chdir(ROOT)
    from agent import Agent
    from make_submission import SUBMISSION_SEED, build_submission

    agent = Agent()
    submission = build_submission(agent, seed=SUBMISSION_SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    agent.last_trace["submission_seed"] = SUBMISSION_SEED
    agent.last_trace["environment"] = "Organizer mock; observations do not predict judging effects"
    source_files = [ROOT / "agent.py", *sorted((ROOT / "strategy").glob("*.py"))]
    agent.last_trace["source_sha256"] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    trace_path.write_text(json.dumps(agent.last_trace, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(submission)} campaigns to {output.name}")
    print(f"Decision trace: {trace_path}")


if __name__ == "__main__":
    main()
