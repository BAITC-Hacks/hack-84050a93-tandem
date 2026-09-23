"""Frozen paired signed-coverage experiment; reuse the existing official scorer harness."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pilot_research.measure import bounded_run, fixture, SCENARIOS, summarize
from scoring_core import apply_filters, sanitize_campaigns

BASE_SHA = "75716aa8b4dec4bcb9eeb471709c3caa19529211"
AGENTS = {"baseline": "agent:Agent", "candidate": "experiments.signed_pilot_coverage.agent:Agent"}
EPSILON = 1e-6


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def normalized_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_frozen():
    path = ROOT / "reports/signed_pilot_coverage/freeze.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = data["hashes_normalized_lf"]
    if not expected:
        raise ValueError("Freeze manifest has no source hashes")
    mismatches = []
    for relative, checksum in expected.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or normalized_hash(source) != checksum:
            mismatches.append(relative)
    if mismatches:
        raise ValueError("Frozen sources changed: " + ", ".join(mismatches))
    freeze_sha = git("log", "-1", "--format=%H", "--", "reports/signed_pilot_coverage/freeze.json")
    if not freeze_sha:
        raise ValueError("Candidate/protocol must be committed before measurement")
    committed = subprocess.check_output(["git", "show", freeze_sha + ":reports/signed_pilot_coverage/freeze.json"], cwd=ROOT)
    if committed.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError("Uncommitted freeze manifest changes")
    return {"freeze_commit": freeze_sha, "freeze_sha256_lf": normalized_hash(path),
            "verified_source_count": len(expected), "hashes_normalized_lf": expected}


def enrich(row, env):
    """Add public-output diagnostics; preserve every original harness field."""
    trace = row.get("agent_trace", {})
    row["preflight"] = trace.get("preflight")
    if "raw_final" not in row:
        row.update(dropped_campaigns=None, n_final=None, final_sizes=None)
        return row
    final = row["raw_final"]
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        sanitized = sanitize_campaigns(final, env.tariffs)
    row["sanitize_output"] = output.getvalue()
    row["dropped_campaigns"] = len(final) - len(sanitized)
    row["n_final"] = len(final)
    row["final_sizes"] = [len(apply_filters(env.customer_profile, campaign)) for campaign in final]
    if row["dropped_campaigns"] and row["status"] == "ok":
        row["status"] = "invalid_plan"
    return row


def pair_result(baseline, candidate):
    checks = {}
    for field in ("pilot_diagnostics", "pilot_net", "pilot_cost", "pilot_contacts", "n_pilots"):
        checks[field] = field in baseline and field in candidate and baseline[field] == candidate[field]
    for field in ("pilots", "prior", "reference_channel"):
        a, b = baseline.get("agent_trace", {}), candidate.get("agent_trace", {})
        checks["trace_" + field] = field in a and field in b and a[field] == b[field]
    same_model = bool(baseline.get("model_sha256")) and baseline.get("model_sha256") == candidate.get("model_sha256")
    comparable = baseline["status"] == candidate["status"] == "ok" and same_model
    return {"seed": baseline["seed"], "same_model": same_model,
            "pilot_equality": checks, "identical_pilots": all(checks.values()),
            "baseline_status": baseline["status"], "candidate_status": candidate["status"],
            "baseline_net": baseline.get("net"), "candidate_net": candidate.get("net"),
            "comparable": comparable,
            "candidate_minus_baseline": candidate["net"] - baseline["net"] if comparable else None}


def paired_summary(rows):
    summaries = {policy: summarize([r for r in rows if r["policy"] == policy]) for policy in AGENTS}
    result = {}
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        policies = {}
        for policy in AGENTS:
            stats = dict(summaries[policy].get(scenario, {}))
            part = [r for r in subset if r["policy"] == policy]
            stats["valid_positive_rate_all_attempts"] = stats.get("valid_positive", 0) / len(part) if part else None
            for field in ("dropped_campaigns", "capped_campaigns"):
                stats["total_" + field] = sum(r.get(field) or 0 for r in part)
                stats[field + "_checked_attempts"] = sum(r.get(field) is not None for r in part)
            for field in ("n_final",):
                values = [r[field] for r in part if r.get(field) is not None]
                stats["mean_" + field] = float(np.mean(values)) if values else None
            policies[policy] = stats
        pairs = []
        for seed in sorted({r["seed"] for r in subset}):
            pair = {r["policy"]: r for r in subset if r["seed"] == seed}
            if set(pair) == set(AGENTS):
                pairs.append(pair_result(pair["baseline"], pair["candidate"]))
        deltas = [p["candidate_minus_baseline"] for p in pairs if p["comparable"]]
        result[scenario] = {"policies": policies, "pairs": pairs, "paired": {
            "complete_valid_pairs": len(deltas), "identical_pilot_pairs": sum(p["identical_pilots"] for p in pairs),
            "same_model_pairs": sum(p["same_model"] for p in pairs),
            "median_candidate_minus_baseline": float(np.median(deltas)) if deltas else None,
            "q10_candidate_minus_baseline": float(np.quantile(deltas, .1)) if deltas else None,
            "min_candidate_minus_baseline": min(deltas) if deltas else None,
            "max_candidate_minus_baseline": max(deltas) if deltas else None,
            "candidate_wins": sum(x > EPSILON for x in deltas),
            "baseline_wins": sum(x < -EPSILON for x in deltas), "ties": sum(abs(x) <= EPSILON for x in deltas)},
            "worst_pairs": sorted([p for p in pairs if p["comparable"]], key=lambda p: p["candidate_minus_baseline"])[:3]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "holdout"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "reports/signed_pilot_coverage").resolve()) or out.suffix != ".json":
        parser.error("--out must be a .json file inside reports/signed_pilot_coverage")
    if out.exists() or out.with_suffix(".jsonl").exists():
        parser.error("refusing to overwrite earlier attempts")
    freeze = verify_frozen()
    seeds = list(range(60, 65)) if args.phase == "development" else list(range(70, 80))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    metadata = {"base_sha": BASE_SHA, "git_sha": git("rev-parse", "HEAD"), "freeze": freeze,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                             "os": platform.platform()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.with_suffix(".jsonl").open("x", encoding="utf-8", newline="\n") as journal, ThreadPoolExecutor(max_workers=2) as pool:
        for scenario in SCENARIOS:
            for seed in seeds:
                env, _, _, _ = fixture(scenario, seed)
                order = tuple(AGENTS) if seed % 2 == 0 else tuple(reversed(AGENTS))
                futures = {pool.submit(bounded_run, AGENTS[policy], scenario, seed, args.timeout): policy for policy in order}
                for future in as_completed(futures):
                    policy = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {"scenario": scenario, "seed": seed, "status": "error", "net": None,
                               "exception": f"runner {type(exc).__name__}: {exc}"}
                    row["policy"] = policy
                    row["agent"] = AGENTS[policy]
                    try:
                        enrich(row, env)
                    except Exception as exc:
                        row["diagnostic_exception"] = f"{type(exc).__name__}: {exc}"
                        row["status"] = "error"
                    rows.append(row)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    print(scenario, seed, policy, row["status"], row.get("net"), flush=True)
    report = {**metadata, "phase": args.phase, "seeds": seeds, "scenarios": list(SCENARIOS), "agents": AGENTS,
              "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "wall_seconds": time.perf_counter() - started, "attempts": len(rows), "paired_epsilon": EPSILON,
              "summary": paired_summary(rows), "journal": out.with_suffix(".jsonl").name,
              "disclaimer": "Known synthetic models and new noise seeds; not new models, hidden judging, or real revenue. No cross-model money aggregation."}
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
