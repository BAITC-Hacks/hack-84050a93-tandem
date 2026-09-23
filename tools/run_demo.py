"""Run the agent and build a fresh, self-contained decision report.

Every invocation creates its own directory. Existing submission/report artifacts
are neither read nor replaced. A status file and process output explain failures.
This runs the official seed-42 mock, not a marketing campaign or hidden evaluator.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if __package__:
    from .render_decision_report import expected_manifest_names
else:
    from render_decision_report import expected_manifest_names


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "demo_runs"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_status(directory, status):
    """Publish a complete status document, never a partly written JSON file."""
    payload = (json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = directory / ".run_status.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, directory / "run_status.json")


def input_snapshot():
    names = expected_manifest_names("source_sha256") | expected_manifest_names("input_sha256")
    names |= {"tools/run_demo.py", "tools/render_decision_report.py",
              "reporting/decision_report.template.html"}
    return {name: sha256(ROOT / name) for name in sorted(names)}


def run_pipeline(output_root=DEFAULT_OUTPUT, timeout=300):
    """Return (new run directory, status); a failed run is never marked complete."""
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 300:
        raise ValueError("Timeout must be greater than zero and at most 300 seconds")
    output_root = Path(output_root).resolve()
    if output_root.is_relative_to(ROOT) and not output_root.is_relative_to(ROOT / "reports"):
        raise ValueError("Repository demo output must be inside reports/")
    missing = [name for name in ("numpy", "pandas") if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError("Missing dependencies: " + ", ".join(missing)
                         + ". Install requirements.txt with the same Python interpreter.")
    for path in (ROOT / "tools/export_strategy.py", ROOT / "tools/render_decision_report.py",
                 ROOT / "reporting/decision_report.template.html"):
        if not path.is_file():
            raise ValueError(f"Required project file is missing: {path.relative_to(ROOT)}")

    output_root.mkdir(parents=True, exist_ok=True)
    # mkdtemp uses exclusive creation; even concurrent runs cannot share a folder.
    prefix = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_")
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=output_root)).resolve()
    started = time.monotonic()
    deadline = started + timeout
    status = {
        "status": "running", "started_at_utc": utc_now(),
        "environment": "Official organizer mock; does not predict hidden judging results",
        "submission_seed": 42, "timeout_seconds": timeout,
        "python": sys.version.split()[0], "steps": [],
    }
    csv_path = directory / "submission.csv"
    trace_path = directory / "decision_trace.json"
    html_path = directory / "decision_report.html"
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    steps = [
        ("export", "Run Agent twice and check identical seed-42 submissions", [
            sys.executable, str(ROOT / "tools/export_strategy.py"),
            "--out", str(csv_path), "--trace", str(trace_path),
        ]),
        ("report", "Check saved evidence and create the interactive report", [
            sys.executable, str(ROOT / "tools/render_decision_report.py"),
            "--submission", str(csv_path), "--trace", str(trace_path), "--out", str(html_path),
        ]),
    ]
    try:
        write_status(directory, status)
        status["input_sha256_before"] = input_snapshot()
        for name, label, command in steps:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Total time allowance expired before the next step")
            step = {"name": name, "status": "running", "stdout": f"{name}_stdout.txt",
                    "stderr": f"{name}_stderr.txt"}
            status["steps"].append(step)
            write_status(directory, status)
            print(f"[{len(status['steps'])}/{len(steps)}] {label}", flush=True)
            step_started = time.monotonic()
            try:
                with (directory / step["stdout"]).open("wb") as stdout, (directory / step["stderr"]).open("wb") as stderr:
                    process = subprocess.run(command, cwd=directory, env=environment,
                                             stdout=stdout, stderr=stderr, timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                step.update(status="timeout", seconds=time.monotonic() - step_started)
                raise TimeoutError(f"{name} exceeded the remaining total time allowance") from exc
            step.update(returncode=process.returncode, seconds=time.monotonic() - step_started,
                        status="complete" if process.returncode == 0 else "failed")
            if process.returncode != 0:
                raise RuntimeError(f"{name} exited with code {process.returncode}; see {name}_stderr.txt")
            write_status(directory, status)

        if time.monotonic() > deadline:
            raise TimeoutError("Total time allowance expired while finalizing the report")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        after = input_snapshot()
        if status["input_sha256_before"] != after:
            raise RuntimeError("Project inputs changed during the run; repeat with stable source files")
        for field in ("source_sha256", "input_sha256"):
            if any(value != after.get(name) for name, value in trace[field].items()):
                raise RuntimeError("Exported manifest differs from the captured project inputs")
        status["inputs_unchanged_during_run"] = True
        status["result"] = {
            "campaigns": len(trace["final"]["campaigns"]), "pilots": len(trace["pilots"]),
            "total_contacts": trace["final"]["total_contacts_including_pilots"],
            "total_cost": trace["final"]["total_cost_including_pilots"],
            "preflight_valid": trace["preflight"]["valid"],
            "deterministic_submission": trace["deterministic_submission"],
        }
        status["artifact_sha256"] = {path.name: sha256(path) for path in (csv_path, trace_path, html_path)}
        status["presentation_source_sha256"] = {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (Path(__file__).resolve(), ROOT / "tools/render_decision_report.py",
                         ROOT / "reporting/decision_report.template.html")
        }
        status["hash_format"] = "sha256-text-lf-v1"
        if time.monotonic() > deadline:
            raise TimeoutError("Total time allowance expired while recording evidence")
        status["status"] = "complete"
    except KeyboardInterrupt:
        status["status"] = "cancelled"
        status["error"] = "Interrupted by the user"
        for step in status["steps"]:
            if step["status"] == "running":
                step["status"] = "cancelled"
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        for step in status["steps"]:
            if step["status"] == "running":
                step["status"] = "failed"
    status["finished_at_utc"] = utc_now()
    status["seconds"] = time.monotonic() - started
    write_status(directory, status)
    return directory, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT,
                        help="Parent directory for a new unique run folder (default: reports/demo_runs)")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Total seconds for export and rendering; maximum 300 (default: 300)")
    args = parser.parse_args()
    try:
        directory, status = run_pipeline(args.output_root, args.timeout)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"\nStatus: {status['status']}")
    print(f"Run directory: {directory}")
    if status["status"] != "complete":
        print(status["error"], file=sys.stderr)
        print(f"Details: {directory / 'run_status.json'}", file=sys.stderr)
        raise SystemExit(130 if status["status"] == "cancelled" else 1)
    result = status["result"]
    print(f"{result['campaigns']} campaigns | {result['pilots']} pilots | "
          f"{result['total_contacts']} total contacts | {result['total_cost']:g} total cost")
    print(f"Open in a browser: {directory / 'decision_report.html'}")
    print("Official mock only. Existing submission artifacts were not replaced.")


if __name__ == "__main__":
    main()
