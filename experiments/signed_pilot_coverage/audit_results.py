"""Read-only audit of finished journals; added after freeze, never used by agents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.signed_pilot_coverage.measure import paired_summary, verify_frozen
from mock_environment import TOTAL_BUDGET, MAX_TOTAL_CONTACTS

OUT = ROOT / "reports/signed_pilot_coverage"
EPS = 1e-6
PROTECTED = ("mock", "devin_mixed", "mixed_1", "mixed_2")
NEGATIVE = ("devin_all_bad", "negative_1", "negative_2")


def main():
    reports = {p: json.loads((OUT / (p + ".json")).read_text(encoding="utf-8")) for p in ("development", "holdout")}
    rows = {p: [json.loads(line) for line in (OUT / (p + ".jsonl")).read_text(encoding="utf-8").splitlines()] for p in reports}
    checks, issues, losses, regressions = {}, [], [], []
    for phase, records in rows.items():
        expected = 120 if phase == "development" else 240
        checks[phase + "_attempts"] = len(records) == expected
        checks[phase + "_summary_recomputed_from_journal"] = paired_summary(records) == reports[phase]["summary"]
        for r in records:
            pre = r.get("preflight") or {}
            safe = (r["status"] == "ok" and r.get("exception") is None and r.get("preflight_valid") is True
                    and pre.get("valid") is True and r.get("dropped_campaigns") == r.get("capped_campaigns") == 0
                    and 1 <= r.get("n_final", 0) <= 10 and all(0 < n <= 5000 for n in r.get("final_sizes", []))
                    and r["total_contacts"] == r["pilot_contacts"] + pre["total_contacts"] <= MAX_TOTAL_CONTACTS
                    and abs(r["total_cost"] - r["pilot_cost"] - pre["total_cost"]) < EPS
                    and r["total_cost"] <= TOTAL_BUDGET)
            if not safe:
                issues.append({"phase": phase, "scenario": r["scenario"], "seed": r["seed"], "policy": r["policy"]})
        checks[phase + "_history_loaded"] = all(r.get("agent_trace", {}).get("prior") == "bounded historical prior from 14822 valid transitions" for r in records)
        positive_models = []
        for scenario, entry in reports[phase]["summary"].items():
            b, c = (entry["policies"][p] for p in ("baseline", "candidate"))
            for field in ("median_net", "q10_net", "min_net", "valid_positive"):
                if c[field] < b[field] - EPS:
                    regressions.append({"phase": phase, "scenario": scenario, "statistic": field,
                                        "baseline": b[field], "candidate": c[field], "delta": c[field] - b[field]})
            for pair in entry["pairs"]:
                if pair["candidate_minus_baseline"] is not None and pair["candidate_minus_baseline"] < -EPS:
                    losses.append({"phase": phase, "scenario": scenario, **pair})
                a, z = ({r["policy"]: r for r in records if r["scenario"] == scenario and r["seed"] == pair["seed"]}[p] for p in ("baseline", "candidate"))
                for field in ("remaining_budget_after_pilots", "remaining_contacts_after_pilots"):
                    checks[f"{phase}/{scenario}/{pair['seed']}/{field}"] = a["agent_trace"]["final"][field] == z["agent_trace"]["final"][field]
            checks[phase + "/" + scenario + "/pilots"] = all(p["identical_pilots"] and p["same_model"] and p["comparable"] for p in entry["pairs"])
            if scenario != "mock" and entry["paired"]["median_candidate_minus_baseline"] > EPS:
                positive_models.append(scenario)
        checks[phase + "_positive_two_stress_including_negative"] = len(positive_models) >= 2 and bool(set(positive_models) & set(NEGATIVE))
        checks[phase + "_protected_distributions"] = not any(x["phase"] == phase and x["scenario"] in PROTECTED and x["statistic"] != "min_net" for x in regressions)

    cross = json.loads((OUT / "crosscheck/report.json").read_text(encoding="utf-8"))
    cross_matches = []
    for attempt in cross["attempts"]:
        if attempt["kind"] == "submission":
            continue
        scenario = "mock" if attempt["kind"] == "official_mock" else "devin_" + attempt["scenario"]
        result = attempt["result"] if scenario == "mock" else attempt["result"].get("payload", {})
        row = next(r for r in rows["development"] if r["scenario"] == scenario and r["seed"] == 60 and r["policy"] == attempt["policy"])
        cross_matches.append({"scenario": scenario, "policy": attempt["policy"],
                              "status_ok": result.get("status") == "ok",
                              "net_exact": result.get("net_arpu_gain") == row["net"],
                              "cost_exact": result.get("total_cost") == row["total_cost"],
                              "contacts_exact": result.get("total_contacts") == row["total_contacts"]})
    checks["crosscheck_12_exact"] = len(cross_matches) == 12 and all(all(v for k, v in x.items() if k.endswith("exact") or k == "status_ok") for x in cross_matches)
    checks["exports_four_equal_root"] = all(x["successful_exports"] == 2 and x["duplicates_equal_lf"] and all(x["equals_root_submission_lf"]) for x in cross["submission_comparison"].values())
    frozen = verify_frozen()
    manifest = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    raw = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest["root_artifacts_raw_sha256"].items()}
    checks["root_artifacts_byte_identical"] = all(raw.values())
    changed = subprocess.check_output(["git", "diff", "--name-only", manifest["base_sha"]], cwd=ROOT, text=True).splitlines()
    changed += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    checks["only_allowed_paths"] = all(p.startswith(("experiments/signed_pilot_coverage/", "reports/signed_pilot_coverage/")) or p == "docs/signed_pilot_coverage.md" for p in changed)
    checks["safety_all_attempts"] = not issues
    decision = "KEEP_BASELINE" if regressions or not all(checks.values()) else "REVIEW_FOR_TRANSFER"
    audit = {"generated_utc": datetime.now(timezone.utc).isoformat(), "base_sha": manifest["base_sha"],
             "candidate_freeze_commit": frozen["freeze_commit"], "verified_frozen_files": frozen["verified_source_count"],
             "decision": decision, "checks": checks, "safety_issues": issues, "regressions": regressions,
             "losing_pairs": sorted(losses, key=lambda x: x["candidate_minus_baseline"]),
             "crosscheck_matches": cross_matches, "root_artifacts_raw_unchanged": raw,
             "reason": "Conservative engineering review: disclose all distribution regressions and retain baseline for mixed results. No significance or hidden-judge claim."}
    path = OUT / "audit.json"
    # This is a derived audit, never an attempt journal or measurement report.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"decision": decision, "failed_checks": [k for k, v in checks.items() if not v], "regressions": regressions, "losing_pairs": len(losses)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
