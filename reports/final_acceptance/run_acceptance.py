"""Record unchanged official commands in a freshly created Python 3.12 venv."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/final_acceptance"
PYTHON = ROOT / ".venv/Scripts/python.exe"


def hashes(data):
    return dict(raw=hashlib.sha256(data).hexdigest(), lf=hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest())


def clean(text):
    for path, label in ((ROOT, "<repo>"), (Path.home(), "<user-home>")):
        text = text.replace(str(path), label).replace(path.as_posix(), label)
    return text


def run(name, args):
    started = time.perf_counter()
    env = dict(os.environ, PYTHONUTF8="1")
    result = subprocess.run([str(PYTHON), *args], cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - started
    for stream in ("stdout", "stderr"):
        (OUT / f"{name}_{stream}.txt").write_text(clean(getattr(result, stream)), encoding="utf-8")
    record = dict(command=[".venv/Scripts/python.exe", *args], exit_code=result.returncode,
                  seconds=seconds, stdout=f"{name}_stdout.txt", stderr=f"{name}_stderr.txt")
    if name.startswith("local_eval"):
        record["no_swallowed_error_or_drop_or_nan"] = not re.search(
            r"Агент упал|отброшена|\bnan\b", result.stdout + result.stderr, re.I)
    print(f"{name}: exit {result.returncode}, {seconds:.3f}s", flush=True)
    return record, result.stdout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path, trace_path = ROOT / "submission.csv", ROOT / "reports/decision_trace.json"
    original_csv, original_trace = csv_path.read_bytes(), trace_path.read_bytes()
    report = dict(base_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        started_at_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        os=dict(system=platform.system(), release=platform.release(), version=platform.version(), machine=platform.machine()),
        isolation="New network clone and new venv; no old workspace files or environments copied. Wheel cache may be reused by pip.",
        another_physical_computer_verified=False,
        original_artifacts={"submission.csv": hashes(original_csv), "reports/decision_trace.json": hashes(original_trace)}, commands={})

    def save():
        (OUT / "automatic_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    commands = [
        ("install_runtime", ["-m", "pip", "install", "-r", "requirements.txt"]),
        ("runtime_versions", ["-m", "pip", "freeze"]),
        ("local_eval_seed42", ["local_eval.py"]),
        ("local_eval_ten", ["local_eval.py", "--runs", "10"]),
        ("verify_help", ["tools/verify_submission.py", "--help"]),
        ("verify_before", ["tools/verify_submission.py"]),
        ("run_demo", ["tools/run_demo.py", "--output-root", "reports/final_acceptance/demo_runs"]),
        ("official_submission", ["make_submission.py"]),
        ("verify_after", ["tools/verify_submission.py"]),
        ("install_dev", ["-m", "pip", "install", "-r", "requirements-dev.txt"]),
        ("all_tests", ["-m", "pytest", "-q"]),
        ("dev_versions", ["-m", "pip", "freeze"]),
    ]
    for name, args in commands:
        before_csv, before_trace = csv_path.read_bytes(), trace_path.read_bytes()
        record, stdout = run(name, args)
        report["commands"][name] = record
        if name in ("verify_before", "verify_after", "run_demo"):
            record["root_artifacts_byte_unchanged"] = before_csv == csv_path.read_bytes() and before_trace == trace_path.read_bytes()
        if name == "official_submission":
            produced = csv_path.read_bytes()
            (OUT / "official_submission.csv").write_bytes(produced)
            report["official_csv"] = dict(produced_hashes=hashes(produced),
                identical_after_crlf_to_lf=produced.replace(b"\r\n", b"\n") == original_csv.replace(b"\r\n", b"\n"),
                raw_identical=produced == original_csv)
        if name == "run_demo" and record["exit_code"] == 0:
            html = Path(next(line.split(": ", 1)[1] for line in stdout.splitlines() if line.startswith("Open in a browser: ")))
            report["fresh_html"] = html.relative_to(ROOT).as_posix()
            status = json.loads((html.parent / "run_status.json").read_text(encoding="utf-8"))
            trace = json.loads((html.parent / "decision_trace.json").read_text(encoding="utf-8"))
            report["demo_result"] = status["result"]
            report["demo_sha256_lf"] = status["artifact_sha256"]
            report["must_have_seed42"] = {
                "act_without_error": status["status"] == "complete",
                "campaigns_1_to_10": 1 <= len(trace["final"]["campaigns"]) <= 10,
                "real_pilots_observed": bool(trace["pilots"]) and all(math.isfinite(p["observed_lift_ratio"]) for p in trace["pilots"]),
                "pilot_counts_sizes": len(trace["pilots"]) <= 20 and all(10 <= p["n_customers"] <= 200 for p in trace["pilots"]),
                "final_size_limit": all(c["contacts"] <= 5000 for c in trace["final"]["campaigns"]),
                "total_contacts_limit": trace["final"]["total_contacts_including_pilots"] <= 15000,
                "total_budget_limit": trace["final"]["total_cost_including_pilots"] <= 100000,
                "independent_preflight": trace["preflight"]["valid"],
                "deterministic": trace["deterministic_submission"],
                "expected_csv_hash": status["artifact_sha256"]["submission.csv"] == "58b9328d66fa2532a8df2bb13cfbe5d15ea295cb49fbbe1c88c40333d209abf8",
            }
            for path in html.parent.glob("*_std*.txt"):
                path.write_text(clean(path.read_text(encoding="utf-8")), encoding="utf-8")
        save()
        if name == "install_runtime" and record["exit_code"] != 0:
            break
    report["trace_unchanged"] = trace_path.read_bytes() == original_trace
    report["root_csv_after_commands"] = hashes(csv_path.read_bytes())
    # The regenerated CSV is preserved above; restore original bytes only if equivalent.
    if csv_path.read_bytes().replace(b"\r\n", b"\n") == original_csv.replace(b"\r\n", b"\n"):
        csv_path.write_bytes(original_csv)
        report["root_csv_original_bytes_restored"] = True
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("Fresh HTML:", report.get("fresh_html"), flush=True)


if __name__ == "__main__":
    main()
