"""Paired evaluation of zero-cost additions using the unchanged official scorer.

Truth and executed pilot IDs are available ONLY in this evaluator. Neither is
passed to the candidate. Every attempt, including a failure, is retained.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from experiments.pilot_research.measure import SCENARIOS, fixture as known_fixture, scored
from experiments.push_overlay.scenarios import CUSTOM_SCENARIOS, NEW_DEVELOPMENT, NEW_HOLDOUT, fixture as new_fixture
from scoring_core import apply_filters, sanitize_campaigns
from tools.benchmark import _git_metadata, parse_seeds
from validation.plan import validate_plan

POLICIES = {"baseline": "agent:Agent", "candidate": "experiments.push_overlay.agent:PushOverlayAgent"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      default=lambda x: x.item() if hasattr(x, "item") else str(x)).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def source_snapshot():
    trace = json.loads((ROOT / "reports/decision_trace.json").read_text(encoding="utf-8"))
    names = set(trace["source_sha256"]) | set(trace["input_sha256"])
    names |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments/push_overlay").glob("*.py")}
    names |= {"experiments/push_overlay/protocol.md", "experiments/pilot_research/measure.py",
              "tools/stress_benchmark.py", "tools/benchmark.py"}
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in sorted(names)}


def evaluate(spec, scenario, seed):
    env, internals, model, fallback = (new_fixture if scenario in CUSTOM_SCENARIOS else known_fixture)(scenario, seed)
    module, name = spec.split(":")
    agent = getattr(importlib.import_module(module), name)()
    started = time.perf_counter()
    raw = agent.act(env)
    elapsed = time.perf_counter() - started
    validation = validate_plan(raw, env.customer_profile, env.tariffs, env.channels,
                               env.remaining_budget, env.remaining_contacts)
    final = sanitize_campaigns(raw, env.tariffs)
    dropped = len(raw) - len(final) + max(0, len(final) - 10)
    final = final[:10]
    pilots = internals.executed_pilot_campaigns()
    result = scored(pilots + final, env, model, fallback)
    capped = sum(any(row.get(key, False) for key in (
        "capped_at_campaign_limit", "capped_at_reach_budget", "capped_at_money_budget"))
        for row in result["campaigns_detail"])
    net = float(result["net_arpu_gain"])
    if not math.isfinite(net):
        raise ValueError("Non-finite measured net")
    audiences = [set(apply_filters(env.customer_profile, row).ID_NUMBER.tolist()) for row in final]
    details = dict(status="ok" if validation["valid"] and not dropped and not capped else "invalid_plan",
                   net=net, total_contacts=result["total_contacts"], total_cost=result["total_cost"],
                   unique_customers=result["unique_customers_targeted"], campaigns=len(final),
                   pilots=len(pilots), dropped=dropped, capped=capped, preflight=validation,
                   agent_seconds=elapsed, raw_final=raw, trace=agent.last_trace,
                   pilot_history_sha256=fingerprint(env.pilot_history),
                   executed_pilots_sha256=fingerprint(pilots),
                   model_sha256=hashlib.sha256(model.to_csv(index=False).encode()).hexdigest(),
                   profile_sha256=hashlib.sha256(env.customer_profile.to_csv(index=False).encode()).hexdigest())
    return details, audiences, env.channels


def pair(scenario, seed):
    details = {}
    audiences = {}
    channels = {}
    for policy, spec in POLICIES.items():
        try:
            details[policy], audiences[policy], channels[policy] = evaluate(spec, scenario, seed)
        except Exception as exc:
            details[policy] = {"status": "error", "exception": f"{type(exc).__name__}: {exc}"}
    row = dict(scenario=scenario, seed=seed, policies=details, status="error", assertions={})
    if any(details[key]["status"] != "ok" for key in POLICIES):
        return row
    base, candidate = details["baseline"], details["candidate"]
    prefix_length = len(base["raw_final"])
    additions = candidate["raw_final"][prefix_length:]
    base_audience = set().union(*audiences["baseline"])
    tests = {
        "same_model": base["model_sha256"] == candidate["model_sha256"],
        "same_profile": base["profile_sha256"] == candidate["profile_sha256"],
        "same_pilot_observations": base["pilot_history_sha256"] == candidate["pilot_history_sha256"],
        "same_executed_pilots": base["executed_pilots_sha256"] == candidate["executed_pilots_sha256"],
        "base_prefix_unchanged": candidate["raw_final"][:prefix_length] == base["raw_final"],
        "additions_zero_cost": all(channels["candidate"][c["channel"]]["cost_per_contact"] == 0 for c in additions),
        "additions_subset_of_base": all(ids <= base_audience for ids in audiences["candidate"][prefix_length:]),
        "same_total_cost": math.isclose(candidate["total_cost"], base["total_cost"], rel_tol=0, abs_tol=1e-8),
        "same_unique_audience": candidate["unique_customers"] == base["unique_customers"],
        "within_contacts": candidate["total_contacts"] <= 15000,
        "within_campaigns": 1 <= candidate["campaigns"] <= 10,
        "net_nondecreasing": candidate["net"] >= base["net"] - 1e-6,
    }
    row.update(status="ok" if all(tests.values()) else "invariant_failure", assertions=tests,
               delta=candidate["net"] - base["net"], additions=len(additions),
               extra_contacts=candidate["total_contacts"] - base["total_contacts"])
    return row


def worker(queue, scenario, seed):
    os.chdir(ROOT)
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            row = pair(scenario, seed)
    except BaseException as exc:
        row = dict(scenario=scenario, seed=seed, status="error", exception=f"{type(exc).__name__}: {exc}")
    row["output"] = output.getvalue()
    queue.put(row)


def bounded_pair(scenario, seed, timeout):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=worker, args=(queue, scenario, seed))
    started = time.perf_counter()
    process.start()
    deadline = started + timeout
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                row = dict(scenario=scenario, seed=seed, status="timeout", exception=f"timeout after {timeout}s")
                break
            try:
                row = queue.get(timeout=min(0.1, remaining))
                break
            except Empty:
                if not process.is_alive():
                    row = dict(scenario=scenario, seed=seed, status="error", exception=f"worker exit {process.exitcode}")
                    break
        process.join(max(0, deadline - time.perf_counter()) if row["status"] == "ok" else 0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
        queue.close()
    row["wall_seconds"] = time.perf_counter() - started
    return row


def summarize(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    result = dict(attempts=len(rows), valid_pairs=len(valid), failures=len(rows)-len(valid))
    if not valid:
        return result
    for policy in POLICIES:
        values = [r["policies"][policy]["net"] for r in valid]
        result[policy] = dict(median=float(np.median(values)), q10=float(np.quantile(values, 0.1)),
                              minimum=min(values), positive=sum(v > 0 for v in values),
                              mean_contacts=float(np.mean([r["policies"][policy]["total_contacts"] for r in valid])))
    deltas = [r["delta"] for r in valid]
    result["paired"] = dict(median=float(np.median(deltas)), q10=float(np.quantile(deltas, 0.1)),
                            minimum=min(deltas), maximum=max(deltas),
                            wins=sum(v > 1e-6 for v in deltas), losses=sum(v < -1e-6 for v in deltas),
                            ties=sum(abs(v) <= 1e-6 for v in deltas),
                            mean_additions=float(np.mean([r["additions"] for r in valid])),
                            mean_extra_contacts=float(np.mean([r["extra_contacts"] for r in valid])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "holdout"), required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--scenarios", help="Optional comma-separated subset for a mechanism check")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chosen = list(SCENARIOS) + list(NEW_DEVELOPMENT if args.phase == "development" else NEW_HOLDOUT)
    if args.scenarios:
        chosen = args.scenarios.split(",")
        if not set(chosen) <= set(SCENARIOS) | set(CUSTOM_SCENARIOS):
            parser.error("Unknown scenario")
    out = args.out.resolve()
    raw = out.with_suffix(".jsonl")
    if out.suffix != ".json" or not out.is_relative_to(ROOT / "reports/push_overlay"):
        parser.error("Output must be a new .json inside reports/push_overlay/")
    if out.exists() or raw.exists():
        parser.error("Refusing to overwrite an existing measurement")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be finite and positive")
    out.parent.mkdir(parents=True, exist_ok=True)
    before = source_snapshot()
    report = dict(phase=args.phase, source=_git_metadata(), versions={"python":sys.version.split()[0],
                  "numpy":np.__version__, "pandas":pd.__version__}, policies=POLICIES,
                  scenarios=chosen, seeds=args.seeds, source_sha256=before,
                  scope="Synthetic public models; no claim about hidden judging or real-world effects")
    started = time.perf_counter()
    rows = []
    with raw.open("x", encoding="utf-8", newline="\n") as stream:
        for scenario in chosen:
            for seed in args.seeds:
                row = bounded_pair(scenario, seed, args.timeout)
                rows.append(row)
                stream.write(encoded(row).decode("utf-8") + "\n")
                stream.flush()
            group = [r for r in rows if r["scenario"] == scenario]
            print(scenario, json.dumps(summarize(group)), flush=True)
    after = source_snapshot()
    report.update(source_unchanged=before == after, seconds=time.perf_counter()-started,
                  summary={scenario:summarize([r for r in rows if r["scenario"]==scenario]) for scenario in chosen},
                  attempts=len(rows), failed_pairs=sum(r["status"]!="ok" for r in rows),
                  all_observed_deltas_nondecreasing=all(r.get("assertions", {}).get("net_nondecreasing",False) for r in rows))
    with out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("Saved", out.relative_to(ROOT), flush=True)
    if not report["source_unchanged"] or report["failed_pairs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
