"""Frozen paired comparison. Model truth is confined to this evaluator."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pilot_research.measure import fixture, scored
from scoring_core import sanitize_campaigns, MAX_CAMPAIGNS
from tools.benchmark import _git_metadata
from validation.plan import validate_plan

AGENTS = {"adaptive": "agent:Agent", "fixed": "experiments.adaptivity_ablation.fixed_agent:FixedSurveyAgent"}
SCENARIOS = ("mock", "mixed_1", "negative_1", "rare_1", "mixed_2", "negative_2", "rare_2")


def evaluate(policy, scenario, seed):
    env, internals, model, fallback = fixture(scenario, seed)
    module, cls = AGENTS[policy].split(":")
    agent = getattr(importlib.import_module(module), cls)()
    started = time.perf_counter()
    exception = None
    try:
        final = agent.act(env)
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
        final = []
    seconds = time.perf_counter() - started
    preflight = validate_plan(final, env.customer_profile, env.tariffs, env.channels,
                              env.remaining_budget, env.remaining_contacts)
    sanitized = sanitize_campaigns(final, env.tariffs)[:MAX_CAMPAIGNS]
    dropped = len(final) - len(sanitized)
    pilots = internals.executed_pilot_campaigns()
    result = scored(pilots + sanitized, env, model, fallback)
    pilot_result = scored(pilots, env, model, fallback)
    positive_keys = {(r.tariff_plan_code_from, r.arpu_segment, r.tariff_plan_code_to)
                     for r in model.itertuples() if r.arpu_change_pct * r.conversion_rate > 0}

    def positive(c):
        return (c.get("filter_current_tariff"), c.get("filter_arpu_segment"), c.get("target_tariff")) in positive_keys

    hits = [positive(c) for c in pilots]
    capped = sum(any(c.get(k, False) for k in ("capped_at_campaign_limit", "capped_at_reach_budget",
                 "capped_at_money_budget")) for c in (result or {}).get("campaigns_detail", []))
    status = "error" if exception or result is None else "invalid_plan" if not preflight["valid"] or capped or dropped else "ok"
    return dict(policy=policy, scenario=scenario, seed=seed, status=status, exception=exception,
                net=result["net_arpu_gain"] if result else None,
                pilot_net=pilot_result["net_arpu_gain"] if pilot_result else 0.0,
                pilot_cost=sum(p["cost"] for p in env.pilot_history),
                pilot_contacts=sum(p["n_customers"] for p in env.pilot_history), n_pilots=len(pilots),
                total_cost=result["total_cost"] if result else None,
                total_contacts=result["total_contacts"] if result else None,
                preflight_valid=preflight["valid"], preflight=preflight,
                dropped_campaigns=dropped, capped_campaigns=capped, agent_seconds=seconds,
                positive_discovered=any(hits), positive_deployed=any(positive(c) for c in sanitized),
                positive_observation=any(hit and p["observed_lift_ratio"] > 0 for hit, p in zip(hits, env.pilot_history)),
                first_positive_pilot=next((i + 1 for i, hit in enumerate(hits) if hit), None),
                pilot_diagnostics=[{**p, "current": c.get("filter_current_tariff"),
                                    "segment": c.get("filter_arpu_segment"), "target": c["target_tariff"],
                                    "true_positive_harness_only": hit} for c, p, hit in zip(pilots, env.pilot_history, hits)],
                raw_final=final, agent_trace=agent.last_trace,
                model_sha256=hashlib.sha256(model.to_csv(index=False).encode()).hexdigest())


def worker(queue, policy, scenario, seed):
    os.chdir(ROOT)
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            row = evaluate(policy, scenario, seed)
    except BaseException as exc:
        row = dict(policy=policy, scenario=scenario, seed=seed, status="error", net=None,
                   exception=f"{type(exc).__name__}: {exc}")
    row["output"] = output.getvalue()
    queue.put(row)


def bounded_run(policy, scenario, seed, timeout=45):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=worker, args=(queue, policy, scenario, seed))
    started = time.perf_counter()
    process.start()
    deadline = started + timeout
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                row = dict(policy=policy, scenario=scenario, seed=seed, status="timeout", net=None,
                           exception=f"timeout after {timeout}s")
                break
            try:
                row = queue.get(timeout=min(0.1, remaining))
                break
            except Empty:
                if not process.is_alive():
                    row = dict(policy=policy, scenario=scenario, seed=seed, status="error", net=None,
                               exception=f"worker exited {process.exitcode}")
                    break
        process.join(max(0, deadline - time.perf_counter()) if row["status"] != "timeout" else 0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
        queue.close()
    row["wall_seconds"] = time.perf_counter() - started
    return row


def summarize(rows):
    report = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        subset = [r for r in rows if r["scenario"] == scenario]
        entry = {"policies": {}, "pairs": []}
        for policy in sorted({r["policy"] for r in subset}):
            part = [r for r in subset if r["policy"] == policy]
            nets = [r["net"] for r in part if r.get("net") is not None]
            stats = dict(attempts=len(part), scored=len(nets),
                         median_net=float(np.median(nets)) if nets else None,
                         q10_net=float(np.quantile(nets, .1)) if nets else None,
                         min_net=min(nets) if nets else None,
                         valid_positive=sum(r["status"] == "ok" and (r.get("net") or 0) > 0 for r in part),
                         positive_discovery=sum(r.get("positive_discovered", False) for r in part),
                         positive_deployment=sum(r.get("positive_deployed", False) for r in part),
                         positive_observation=sum(r.get("positive_observation", False) for r in part),
                         statuses={s: sum(r["status"] == s for r in part) for s in ("ok", "error", "invalid_plan", "timeout")})
            stats["valid_positive_rate_all_attempts"] = stats["valid_positive"] / len(part)
            for name in ("pilot_cost", "pilot_contacts", "total_cost", "total_contacts", "n_pilots", "agent_seconds", "wall_seconds", "dropped_campaigns", "capped_campaigns"):
                values = [r[name] for r in part if r.get(name) is not None]
                stats["mean_" + name] = float(np.mean(values)) if values else None
            entry["policies"][policy] = stats
        for seed in sorted({r["seed"] for r in subset}):
            pair = {r["policy"]: r for r in subset if r["seed"] == seed}
            a, f = pair.get("adaptive"), pair.get("fixed")
            if not a or not f:
                continue
            comparable = a["status"] == f["status"] == "ok"
            entry["pairs"].append(dict(seed=seed, comparable=comparable,
                                      adaptive_net=a.get("net"), fixed_net=f.get("net"),
                                      adaptive_minus_fixed=a["net"] - f["net"] if comparable else None,
                                      same_model=a.get("model_sha256") == f.get("model_sha256")))
        deltas = [p["adaptive_minus_fixed"] for p in entry["pairs"] if p["comparable"]]
        entry["paired"] = dict(complete_valid_pairs=len(deltas),
                               median_adaptive_minus_fixed=float(np.median(deltas)) if deltas else None,
                               q10_adaptive_minus_fixed=float(np.quantile(deltas, .1)) if deltas else None,
                               min_adaptive_minus_fixed=min(deltas) if deltas else None,
                               max_adaptive_minus_fixed=max(deltas) if deltas else None,
                               adaptive_wins=sum(x > 1e-8 for x in deltas), fixed_wins=sum(x < -1e-8 for x in deltas),
                               ties=sum(abs(x) <= 1e-8 for x in deltas))
        report[scenario] = entry
    return report


def reproduction_checks(rows):
    saved = json.loads((ROOT / "reports/benchmark_review_agent.json").read_text())["runs"]
    old = {}
    for name in ("dev_baseline_diagnostics.jsonl", "dev_baseline_remaining.jsonl"):
        for line in (ROOT / "reports/pilot_research" / name).read_text().splitlines():
            row = json.loads(line)
            if row["scenario"] == "mock":
                old[row["seed"]] = row
    checks = []
    for row in rows:
        official = next(r for r in saved if r["seed"] == row["seed"])
        checks.append(dict(seed=row["seed"], checks={
            name: abs(row[name] - official[other]) < 1e-6
            for name, other in (("net", "net_arpu_gain"), ("total_cost", "total_cost"),
                                ("total_contacts", "total_contacts"), ("n_pilots", "n_pilots"))}
            | {"preflight": row["preflight_valid"] == official["preflight_valid"],
               "status": row["status"] == official["status"],
               "pilot_cost": row["pilot_cost"] == old[row["seed"]]["pilot_cost"],
               "pilot_contacts": row["pilot_contacts"] == old[row["seed"]]["pilot_contacts"]}))
    return dict(passed=all(all(c["checks"].values()) for c in checks), checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reproduce", "compare"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "reports/adaptivity_ablation").resolve()):
        parser.error("output must be under reports/adaptivity_ablation")
    if out.exists() or out.with_suffix(".jsonl").exists():
        parser.error("refusing to overwrite attempts")
    if args.mode == "compare":
        from experiments.adaptivity_ablation.verify import verify_frozen
        verify_frozen()
    seeds = list(range(5)) if args.mode == "reproduce" else list(range(50, 60))
    scenarios = ("mock",) if args.mode == "reproduce" else SCENARIOS
    policies = ("adaptive",) if args.mode == "reproduce" else ("adaptive", "fixed")
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = _git_metadata()
    rows = []
    with out.with_suffix(".jsonl").open("x", encoding="utf-8") as journal:
        for scenario in scenarios:
            for seed in seeds:
                # Alternate execution order to reduce systematic timing-order effects.
                for policy in policies if seed % 2 == 0 else tuple(reversed(policies)):
                    row = bounded_run(policy, scenario, seed)
                    rows.append(row)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    journal.flush()
                    print(scenario, seed, policy, row["status"], row.get("net"), flush=True)
    report = dict(git=metadata, seeds=seeds, scenarios=scenarios, agents=AGENTS,
                  versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__),
                  summary=summarize(rows), journal=out.with_suffix(".jsonl").name,
                  disclaimer="Known synthetic models with new noise seeds, not hidden judging or real revenue. Same seed does not imply identical client samples.")
    if args.mode == "reproduce":
        report["reproduction"] = reproduction_checks(rows)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    mp.freeze_support()
    main()
