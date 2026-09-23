"""Run unchanged public harnesses and detailed diagnostics without overwriting reports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--prefix", required=True, help="e.g. holdout_baseline")
    args = parser.parse_args()
    if not args.prefix.replace("_", "").replace("-", "").isalnum():
        parser.error("prefix must be a simple filename prefix")
    output = ROOT / "reports/pilot_research"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / (args.prefix + "_execution.json")
    if manifest.exists():
        parser.error("execution manifest exists; choose a fresh prefix")
    jobs = [("mock", "tools/benchmark.py"), ("stress", "tools/stress_benchmark.py"),
            ("diagnostics", "experiments/pilot_research/measure.py")]
    for tag, _ in jobs:
        if (output / (args.prefix + "_" + tag + ".json")).exists():
            parser.error("output exists; choose a fresh prefix")
    records = []
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [ROOT / "agent.py", ROOT / "experiments/pilot_research/agent.py",
                        ROOT / "experiments/pilot_research/measure.py"]}
    for tag, script in jobs:
        target = output / (args.prefix + "_" + tag + ".json")
        command = [sys.executable, "-u", script, "--agent", args.agent, "--seeds", args.seeds,
                   "--out", str(target.relative_to(ROOT))]
        print("running", tag, args.agent, args.seeds, flush=True)
        error = None
        with target.with_suffix(".txt").open("x", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=1800, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code, error = None, "suite subprocess exceeded 1800 seconds"
        records.append({"command": command[2:], "returncode": code, "error": error})
        manifest.write_text(json.dumps({"agent": args.agent, "seeds": args.seeds,
                                       "source_sha256": hashes, "commands": records}, indent=2) + "\n",
                            encoding="utf-8")
        if code != 0:
            raise SystemExit(f"{tag} failed; preserved log and manifest")


if __name__ == "__main__":
    main()
