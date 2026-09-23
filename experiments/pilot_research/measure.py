"""Additional synthetic scenarios and action diagnostics; truth stays in harness.

Run from the repository root with --agent module:Class --seeds 0:10 --out ...
Existing official benchmark and stress_benchmark remain the primary checks.
"""
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
from environment import make_environment
from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET
from scoring_core import score_campaigns
from tools.benchmark import _git_metadata, parse_seeds
from validation.plan import validate_plan

EXTRA = tuple(f"{kind}_{variant}" for kind in ("negative", "mixed", "rare") for variant in (1, 2))
SCENARIOS = ("mock",) + tuple("devin_" + s for s in (
    "all_bad", "mixed", "rare_good", "unlucky_noisy_pilots", "small_and_empty")) + EXTRA


def synthetic(scenario):
    """Fixed designs: two placements/magnitudes per effect family, no seed fitting."""
    kind, suffix = scenario.rsplit("_", 1)
    variant = int(suffix)
    rng = np.random.default_rng(20260923 + 100 * variant + {"negative": 1, "mixed": 2, "rare": 3}[kind])
    codes = ("tariff_1", "tariff_4", "tariff_8", "tariff_10")
    segments = ("LOW", "MID", "HIGH")
    tariffs = pd.read_csv(ROOT / "tariff_dictionary.csv")
    tariffs = tariffs[tariffs.tariff_plan_code.isin(codes)].copy()
    triples = [(c, s, t) for c in codes for s in segments for t in codes if c != t]
    order = rng.permutation(len(triples))
    good_count = {"negative": 2, "mixed": 12, "rare": 1}[kind]
    good = {triples[i] for i in order[:good_count]}
    good_effect, bad_effect = {
        ("negative", 1): (0.12, -0.22), ("negative", 2): (0.25, -0.12),
        ("mixed", 1): (0.35, -0.12), ("mixed", 2): (0.22, -0.20),
        ("rare", 1): (1.0, -0.16), ("rare", 2): (0.7, -0.08),
    }[(kind, variant)]
    model = pd.DataFrame([
        {"tariff_plan_code_from": c, "arpu_segment": s, "tariff_plan_code_to": t,
         "arpu_change_pct": good_effect if (c, s, t) in good else bad_effect,
         "conversion_rate": 0.8 if variant == 1 else 0.6}
        for c, s, t in triples
    ])
    rows = []
    for c in codes:
        for s in segments:
            n = int(rng.choice([60, 120, 240, 400]))
            if kind == "rare" and any(x[:2] == (c, s) for x in good):
                n = 18 if variant == 1 else 36
            for i in range(n):
                rows.append({"ID_NUMBER": len(rows) + 1, "current_tariff": c, "arpu_segment": s,
                             "data_segment": ("LITE", "HEAVY", "NON_USER")[i % 3],
                             "call_segment": ("LOW", "MEDIUM", "HIGH")[i % 3],
                             "predicted_arpu": {"LOW": 700, "MID": 3500, "HIGH": 9000}[s]
                             * float(rng.uniform(0.95, 1.05))})
    return pd.DataFrame(rows), tariffs, model


def fallback(current, target, segment, tariffs, conversion):
    return -0.1, conversion


def fixture(scenario, seed):
    if scenario == "mock":
        from mock_environment import make_mock_env, _mock_impact_model, _mock_fallback
        env, internals = make_mock_env(seed=seed)
        model = _mock_impact_model(pd.read_csv(ROOT / "data/change_tariff.csv"))
        return env, internals, model, _mock_fallback
    if scenario.startswith("devin_"):
        from tools.stress_benchmark import _profile, _tariffs, _impact_model, _fallback
        name = scenario.removeprefix("devin_")
        profile, tariffs, model = _profile(name), _tariffs(), _impact_model(name)
        fb = _fallback
    else:
        profile, tariffs, model = synthetic(scenario)
        fb = fallback
    env, internals = make_environment(profile, model, tariffs, CHANNELS, TOTAL_BUDGET,
                                      MAX_TOTAL_CONTACTS, fb, seed=seed)
    return env, internals, model, fb


def scored(campaigns, env, model, fb):
    if not campaigns:
        return None
    frame = pd.DataFrame(campaigns)
    for column in ("filter_arpu_segment", "filter_data_segment", "filter_call_segment",
                   "filter_current_tariff", "explicit_ids"):
        if column not in frame:
            frame[column] = None
    return score_campaigns(frame, env.customer_profile, model, env.tariffs,
                           float(env.customer_profile.predicted_arpu.sum()), fb)


def evaluate(spec, scenario, seed):
    env, internals, model, fb = fixture(scenario, seed)
    module, cls = spec.split(":")
    agent = getattr(importlib.import_module(module), cls)()
    started = time.perf_counter()
    exception = None
    try:
        final = agent.act(env)
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
        final = []
    agent_seconds = time.perf_counter() - started
    preflight = validate_plan(final, env.customer_profile, env.tariffs, env.channels,
                              env.remaining_budget, env.remaining_contacts)
    pilots = internals.executed_pilot_campaigns()
    # Invalid raw output is retained in diagnostics; scoring never hides a failure.
    final_valid = final if preflight["valid"] else []
    result = scored(pilots + final_valid, env, model, fb)
    pilot_result = scored(pilots, env, model, fb)
    positive_keys = set()
    for row in model.itertuples():
        if row.arpu_change_pct * row.conversion_rate > 0:
            positive_keys.add((row.tariff_plan_code_from, row.arpu_segment, row.tariff_plan_code_to))

    def key(c):
        return c.get("filter_current_tariff"), c.get("filter_arpu_segment"), c.get("target_tariff")

    hits = [key(p) in positive_keys for p in pilots]
    trace = agent.last_trace if hasattr(agent, "last_trace") else {}
    capped = sum(any(c.get(k, False) for k in ("capped_at_campaign_limit", "capped_at_reach_budget",
                    "capped_at_money_budget")) for c in (result or {}).get("campaigns_detail", []))
    status = "error" if exception or not result else "invalid_plan" if not preflight["valid"] or capped else "ok"
    return {
        "scenario": scenario, "seed": seed, "status": status, "exception": exception,
        "net": result["net_arpu_gain"] if result else None,
        "pilot_net": pilot_result["net_arpu_gain"] if pilot_result else 0.0,
        "pilot_cost": sum(p["cost"] for p in env.pilot_history),
        "pilot_contacts": sum(p["n_customers"] for p in env.pilot_history),
        "n_pilots": len(pilots), "total_cost": result["total_cost"] if result else None,
        "total_contacts": result["total_contacts"] if result else None,
        "agent_seconds": agent_seconds, "preflight_valid": preflight["valid"],
        "preflight_errors": preflight["errors"], "capped_campaigns": capped,
        "true_positive_pilot_actions": sum(hits),
        "true_positive_observed_positive": sum(hit and p["observed_lift_ratio"] > 0
                                              for hit, p in zip(hits, env.pilot_history)),
        "true_positive_final_campaigns": sum(key(c) in positive_keys for c in final_valid),
        "first_true_positive_pilot": next((i + 1 for i, hit in enumerate(hits) if hit), None),
        "pilot_diagnostics": [{"current": p.get("filter_current_tariff"),
                               "segment": p.get("filter_arpu_segment"),
                               "target": p["target_tariff"], "true_positive_harness_only": hit,
                               **observation} for p, hit, observation in zip(pilots, hits, env.pilot_history)],
        "agent_trace": trace, "raw_final": final,
        "model_sha256": hashlib.sha256(model.to_csv(index=False).encode()).hexdigest(),
    }


def worker(queue, spec, scenario, seed):
    os.chdir(ROOT)
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            row = evaluate(spec, scenario, seed)
        row["output"] = output.getvalue()
    except BaseException as exc:
        row = {"scenario": scenario, "seed": seed, "status": "error", "net": None,
               "exception": f"{type(exc).__name__}: {exc}", "output": output.getvalue()}
    queue.put(row)


def bounded_run(spec, scenario, seed, timeout):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=worker, args=(queue, spec, scenario, seed))
    start = time.perf_counter()
    process.start()
    deadline = start + timeout
    try:
        while True:
            if time.perf_counter() >= deadline:
                row = {"scenario": scenario, "seed": seed, "status": "timeout", "net": None,
                       "exception": f"timeout after {timeout}s"}
                break
            try:
                row = queue.get(timeout=min(0.1, max(0.001, deadline - time.perf_counter())))
                break
            except Empty:
                if not process.is_alive():
                    row = {"scenario": scenario, "seed": seed, "status": "error", "net": None,
                           "exception": f"worker exited {process.exitcode}"}
                    break
        process.join(max(0, deadline - time.perf_counter()) if row["status"] != "timeout" else 0)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(2)
        queue.close()
    row["wall_seconds"] = time.perf_counter() - start
    return row


def summarize(rows):
    result = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        subset = [r for r in rows if r["scenario"] == scenario]
        nets = [r["net"] for r in subset if r.get("net") is not None]
        entry = {"runs": len(subset), "scored": len(nets),
                 "valid_positive": sum(r["status"] == "ok" and (r.get("net") or 0) > 0 for r in subset),
                 "errors": sum(r["status"] == "error" for r in subset),
                 "timeouts": sum(r["status"] == "timeout" for r in subset),
                 "invalid": sum(r["status"] == "invalid_plan" for r in subset),
                 "median_net": float(np.median(nets)) if nets else None,
                 "min_net": min(nets) if nets else None,
                 "q10_net": float(np.quantile(nets, 0.1)) if nets else None,
                 "mean_loss": float(np.mean([max(-n, 0) for n in nets])) if nets else None,
                 "positive_discovery_runs": sum(r.get("true_positive_pilot_actions", 0) > 0 for r in subset),
                 "positive_deployment_runs": sum(r.get("true_positive_final_campaigns", 0) > 0 for r in subset)}
        for name in ("pilot_net", "pilot_cost", "pilot_contacts", "n_pilots", "total_cost",
                     "total_contacts", "agent_seconds", "wall_seconds"):
            values = [r[name] for r in subset if r.get(name) is not None]
            entry["mean_" + name] = float(np.mean(values)) if values else None
        result[scenario] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=list(range(10)))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    allowed = (ROOT / "reports/pilot_research").resolve()
    if not out.is_relative_to(allowed):
        parser.error("output must be inside reports/pilot_research")
    if out.exists() or out.with_suffix(".jsonl").exists():
        parser.error("refusing to overwrite earlier attempts")
    scenarios = args.scenarios.split(",")
    if set(scenarios) - set(SCENARIOS):
        parser.error("unknown scenario")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.with_suffix(".jsonl").open("x", encoding="utf-8") as journal:
        for scenario in scenarios:
            for seed in args.seeds:
                row = bounded_run(args.agent, scenario, seed, args.timeout)
                rows.append(row)
                journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                print(scenario, seed, row["status"], row.get("net"), flush=True)
    report = {"agent": args.agent, "git": _git_metadata(), "seeds": args.seeds,
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
              "summary": summarize(rows), "journal": out.with_suffix(".jsonl").name,
              "disclaimer": "Synthetic effects are harness-only; same seeds do not imply identical pilot samples across policies."}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    mp.freeze_support()
    main()
