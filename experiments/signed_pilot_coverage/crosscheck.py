"""Bounded checks through the existing official benchmark/stress/export runners."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import platform
from queue import Empty
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FREEZE = ROOT / "reports/signed_pilot_coverage/freeze.json"
POLICIES = {
    "baseline": "agent:Agent",
    "candidate": "experiments.signed_pilot_coverage.agent:Agent",
}
SCENARIOS = ("all_bad", "mixed", "rare_good", "unlucky_noisy_pilots", "small_and_empty")
TIMEOUT = 60.0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def append_attempt(path, attempt):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check_freeze(freeze):
    expected = freeze["hashes_normalized_lf"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("freeze.hashes_normalized_lf must be a nonempty path-to-hash mapping")
    checks = {}
    for name, frozen_hash in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Frozen source outside repository")
        actual = digest(path) if path.is_file() else None
        checks[name] = {"expected": frozen_hash, "actual": actual, "unchanged": actual == frozen_hash}
    return {"passed": all(row["unchanged"] for row in checks.values()), "files": checks}


def stress_worker(queue, spec, scenario):
    output, errors = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    try:
        os.chdir(ROOT)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            from tools.benchmark import _load_agent
            from tools.stress_benchmark import run_scenario
            row = run_scenario(_load_agent(spec), scenario, 60)
        result = {"ok": True, "payload": row}
    except BaseException as exc:
        result = {"ok": False, "exception": f"{type(exc).__name__}: {exc}"}
    result.update(stdout=output.getvalue(), stderr=errors.getvalue(),
                  seconds=time.perf_counter() - started)
    queue.put(result)


def run_stress(spec, scenario):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=stress_worker, args=(queue, spec, scenario))
    started = time.perf_counter()
    process.start()
    try:
        while True:
            remaining = TIMEOUT - (time.perf_counter() - started)
            if remaining <= 0:
                result = {"ok": False, "timeout": True, "exception": "timeout after 60s"}
                break
            try:
                result = queue.get(timeout=min(0.1, remaining))
                break
            except Empty:
                if not process.is_alive():
                    result = {"ok": False, "exception": f"worker exited {process.exitcode} without result"}
                    break
        process.join(max(0.0, TIMEOUT - (time.perf_counter() - started)) if result.get("ok") else 0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
        queue.close()
    result["wall_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/signed_pilot_coverage/crosscheck"))
    args = parser.parse_args()
    out = (ROOT / args.out_dir).resolve()
    allowed = ROOT / "reports/signed_pilot_coverage"
    if not out.is_relative_to(allowed) or out == allowed:
        parser.error("Output must be a subdirectory of reports/signed_pilot_coverage")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("Output directory already exists and is nonempty")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    initial_guard = check_freeze(freeze)
    out.mkdir(parents=True, exist_ok=True)
    from tools.benchmark import _git_metadata, _run_row, _run_worker
    import numpy as np
    import pandas as pd
    base_sha = subprocess.run(["git", "rev-parse", "75716aa^{commit}"], cwd=ROOT, check=True,
                              text=True, capture_output=True).stdout.strip()
    root_hash = digest(ROOT / "submission.csv")
    report = {
        "git": _git_metadata(), "base_sha": base_sha,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "platform": platform.platform()},
        "freeze_file": FREEZE.relative_to(ROOT).as_posix(), "freeze_sha256_lf": digest(FREEZE),
        "freeze": freeze, "initial_guard": initial_guard,
        "policies": POLICIES, "seed": 60, "submission_seed": 42, "timeout_seconds": TIMEOUT,
        "root_submission_sha256_lf_before": root_hash,
        "limitations": ["Known synthetic models only; not hidden judging.",
                         "Existing stress runner does not report dropped/capped counts; those are not checked here.",
                         "Official benchmark worker combines stdout/stderr in its output field."],
        "attempts": [], "submission_comparison": {},
    }
    write_json(out / "report.json", report)
    if not initial_guard["passed"]:
        raise SystemExit("Frozen source mismatch: no runs executed")

    def preserve(attempt):
        append_attempt(out / "attempts.jsonl", attempt)
        report["attempts"].append(attempt)
        write_json(out / "report.json", report)
        print(attempt["kind"], attempt["policy"], attempt.get("scenario", ""), flush=True)

    for policy, spec in POLICIES.items():
        started = time.perf_counter()
        try:
            row = _run_row(spec, 60, TIMEOUT)
        except BaseException as exc:
            row = {"status": "error", "exception": f"{type(exc).__name__}: {exc}",
                   "seconds": time.perf_counter() - started}
        preserve({"kind": "official_mock", "policy": policy, "agent": spec, "seed": 60, "result": row})
        for scenario in SCENARIOS:
            try:
                result = run_stress(spec, scenario)
            except BaseException as exc:
                result = {"ok": False, "exception": f"{type(exc).__name__}: {exc}"}
            preserve({"kind": "existing_stress", "policy": policy, "agent": spec,
                      "scenario": scenario, "seed": 60, "result": result,
                      "dropped_campaigns": "NOT_REPORTED_BY_EXISTING_RUNNER",
                      "capped_campaigns": "NOT_REPORTED_BY_EXISTING_RUNNER"})
        exports = []
        for repeat in (1, 2):
            try:
                result = _run_worker("submission", spec, 42, TIMEOUT)
            except BaseException as exc:
                result = {"ok": False, "exception": f"{type(exc).__name__}: {exc}"}
            attempt = {"kind": "submission", "policy": policy, "agent": spec,
                       "seed": 42, "repeat": repeat, "result": result}
            if result.get("ok") and isinstance(result.get("payload"), str):
                payload = result["payload"].replace("\r\n", "\n").encode("utf-8")
                filename = f"{policy}_seed42_repeat{repeat}.csv"
                (out / filename).write_bytes(payload)
                attempt.update(csv_file=filename, sha256_lf=hashlib.sha256(payload).hexdigest())
                exports.append(payload)
            preserve(attempt)
        report["submission_comparison"][policy] = {
            "successful_exports": len(exports),
            "duplicates_equal_lf": len(exports) == 2 and exports[0] == exports[1],
            "equals_root_submission_lf": [hashlib.sha256(payload).hexdigest() == root_hash for payload in exports],
            "sha256_lf": [hashlib.sha256(payload).hexdigest() for payload in exports],
        }
        write_json(out / "report.json", report)
    report["final_guard"] = check_freeze(freeze)
    report["root_submission_sha256_lf_after"] = digest(ROOT / "submission.csv")
    report["root_submission_unchanged"] = report["root_submission_sha256_lf_after"] == root_hash
    write_json(out / "report.json", report)
    print(json.dumps(report["submission_comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
