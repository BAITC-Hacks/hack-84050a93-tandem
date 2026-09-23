"""Verify frozen source and independent harness agreement before reporting."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.pilot_research.summarize import load

OUT = ROOT / "reports/pilot_research"


def main():
    checks = []
    total_diagnostics = 0
    total_primary = 0
    status_counts = {}
    model_by_scenario = {}
    for phase, tags in [("dev", ("baseline", "sized", "stop")), ("holdout", ("baseline", "stop"))]:
        for tag in tags:
            prefix = phase + "_" + tag
            rows = load(prefix, development=phase == "dev")
            total_diagnostics += len(rows)
            for row in rows:
                status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
                if "model_sha256" in row:
                    model_by_scenario.setdefault(row["scenario"], set()).add(row["model_sha256"])
            indexed = {(r["scenario"], r["seed"]): r for r in rows}
            for suite in ("mock", "stress"):
                report = json.loads((OUT / (prefix + "_" + suite + ".json")).read_text(encoding="utf-8"))
                total_primary += len(report["runs"])
                for row in report["runs"]:
                    scenario = "mock" if suite == "mock" else "devin_" + row["scenario"]
                    diagnostic = indexed[(scenario, row["seed"])]
                    ok = row["status"] == diagnostic["status"] == "ok"
                    ok = ok and abs(row["net_arpu_gain"] - diagnostic["net"]) < 1e-7
                    ok = ok and all(row[k] == diagnostic[k] for k in (
                        "total_cost", "total_contacts", "n_pilots", "preflight_valid"))
                    checks.append({"prefix": prefix, "scenario": scenario, "seed": row["seed"], "matches": ok})
                if suite == "mock":
                    checks.append({"prefix": prefix, "submission_seed42":
                                   report["submission_reproducibility"]["deterministic"],
                                   "matches": report["submission_reproducibility"]["deterministic"]})
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    source = ROOT / "experiments/pilot_research/agent.py"
    frozen_matches = hashlib.sha256(source.read_bytes()).hexdigest() == selection["source_sha256"]
    changed = subprocess.check_output(["git", "diff", selection["base_sha"], "--name-only"], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    outside_scope = [p for p in changed + untracked if not (p.startswith("experiments/pilot_research/")
                      or p.startswith("reports/pilot_research/") or p == "docs/pilot_research.md")]
    report = {"frozen_policy_matches": frozen_matches, "outside_scope": outside_scope,
              "diagnostic_attempts": total_diagnostics, "primary_harness_attempts": total_primary,
              "diagnostic_status_counts": status_counts,
              "model_stable_across_agents_and_seeds": {k: len(v) == 1 for k, v in model_by_scenario.items()},
              "checks": checks, "all_checks_pass": all(c["matches"] for c in checks)
              and frozen_matches and not outside_scope and all(len(v) == 1 for v in model_by_scenario.values())}
    (OUT / "integrity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
