"""Check downloaded bytes, CSV preflight, and posterior arithmetic; no new runs."""
import hashlib
import json
from math import sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scoring_core import CHANNELS
from strategy.beliefs import HistoricalPrior
from validation.plan import validate_plan

OUT = ROOT / "reports/final_acceptance"
run = json.loads((OUT / "automatic_checks.json").read_text(encoding="utf-8"))
directory = (ROOT / run["fresh_html"]).parent
trace = json.loads((directory / "decision_trace.json").read_text(encoding="utf-8"))


def normalized_hash(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


downloads = {}
for name in ("submission.csv", "decision_trace.json"):
    path = Path.home() / "Downloads" / name
    data = path.read_bytes()
    saved = directory / name
    downloads[name] = dict(download_sha256_lf=normalized_hash(data),
                           fresh_artifact_sha256_lf=normalized_hash(saved.read_bytes()),
                           exact_raw_match=data == saved.read_bytes(),
                           normalized_match=data.replace(b"\r\n", b"\n") == saved.read_bytes().replace(b"\r\n", b"\n"))
    (OUT / ("browser_" + name)).write_bytes(data)

tariffs = pd.read_csv(ROOT / "data/dict_tariff.csv")
profile = pd.read_csv(ROOT / "customer_profile.csv")
frame = pd.read_csv(OUT / "official_submission.csv")
campaigns = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
pilot_contacts = sum(p["n_customers"] for p in trace["pilots"])
pilot_cost = sum(p["n_customers"] * CHANNELS[p["channel"]]["cost_per_contact"] for p in trace["pilots"])
preflight = validate_plan(campaigns, profile, tariffs, CHANNELS, 100000 - pilot_cost, 15000 - pilot_contacts)

reference = CHANNELS[trace["reference_channel"]]["conversion_multiplier"]
prior = HistoricalPrior(tariffs, ROOT / "data/change_tariff.csv", reference)
state, posterior_checks = {}, []
for index, p in enumerate(trace["pilots"], 1):
    key = p["current_tariff"], p["arpu_segment"], p["target_tariff"]
    if key not in state:
        b = prior.belief(*key)
        state[key] = [1 / b.prior_std ** 2, b.prior_mean / b.prior_std ** 2]
    precision, weighted = state[key]
    scale = reference / CHANNELS[p["channel"]]["conversion_multiplier"]
    observation_variance = 0.804 ** 2 / p["n_customers"] * scale ** 2
    precision += 1 / observation_variance
    weighted += p["observed_lift_ratio"] * scale / observation_variance
    mean, std = weighted / precision, sqrt(1 / precision)
    state[key] = [precision, weighted]
    error_mean = abs(mean - p["posterior_reference_mean"])
    error_std = abs(std - p["posterior_reference_std"])
    posterior_checks.append(dict(pilot=index, mean_error=error_mean, std_error=error_std,
                                 passed=error_mean < 1e-12 and error_std < 1e-12))

report = dict(base_sha=run["base_sha"], downloads=downloads,
              fresh_run=directory.relative_to(ROOT).as_posix(),
              official_csv_independent_preflight=preflight,
              pilot_contacts=pilot_contacts, pilot_cost=pilot_cost,
              posterior_checks=posterior_checks,
              posterior_note="Scalar Gaussian update reconstructed from logged real replies and original priors, without calling Belief.observe/posterior or running Agent again.",
              passed=preflight["valid"] and all(p["passed"] for p in posterior_checks)
                     and all(d["normalized_match"] for d in downloads.values()))
(OUT / "evidence_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"passed": report["passed"], "downloads": downloads,
                  "preflight_valid": preflight["valid"], "posterior_checks": sum(p["passed"] for p in posterior_checks)}, indent=2))
