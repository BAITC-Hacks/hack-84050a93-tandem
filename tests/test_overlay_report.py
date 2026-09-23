"""Report validation over public fixtures, without running an agent or scorer."""
from copy import deepcopy

import pandas as pd
import pytest

from tools import render_decision_report as renderer
from validation.overlay import validate_overlay_extension
from validation.plan import validate_plan


@pytest.fixture
def report_case(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    profile = pd.DataFrame([
        {"ID_NUMBER": offset + i, "current_tariff": current, "arpu_segment": segment,
         "data_segment": "LITE", "call_segment": "LOW", "predicted_arpu": 1000.0}
        for offset, current, segment in ((0, "tariff_1", "HIGH"), (10, "tariff_4", "LOW"),
                                        (20, "tariff_8", "HIGH")) for i in range(1, 11)
    ])
    tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_4", "tariff_8", "tariff_9"]})
    profile.to_csv(tmp_path / "customer_profile.csv", index=False)
    tariffs.to_csv(tmp_path / "data/dict_tariff.csv", index=False)
    base_raw = [{"campaign_name": "tandem_01", "filter_arpu_segment": "HIGH",
                 "filter_data_segment": None, "filter_call_segment": None,
                 "filter_current_tariff": "tariff_1", "target_tariff": "tariff_9", "channel": "sms"}]
    pilots = [{"current_tariff": current, "arpu_segment": segment, "target_tariff": "tariff_9",
               "channel": "sms", "n_customers": 10, "observed_lift_ratio": 0.1,
               "posterior_reference_mean": 0.1, "posterior_reference_std": 0.01,
               "estimated_information_value": 1.0, "confirmation": False, "stratified_segment": segment}
              for current, segment in (("tariff_1", "HIGH"), ("tariff_4", "LOW"))]
    channels = renderer.PUBLIC_CHANNELS
    remaining_budget, remaining_contacts = 99920.0, 14980
    base_preflight = validate_plan(base_raw, profile, tariffs, channels, remaining_budget, remaining_contacts)
    base = {"version": "adaptive-portfolio-v1.2", "warnings": [], "prior": "fixture",
            "reference_channel": "sms", "pilots": pilots, "preflight": base_preflight,
            "final": {"campaigns": [{**base_raw[0], "contacts": 10, "cost": 40.0,
                        "estimated_mean_gain": 900.0, "risk_adjusted_gain": 700.0,
                        "reference_scaled_mean": 0.1, "reference_scaled_std": 0.01}],
                      "contacts": 10, "cost": 40.0, "remaining_contacts_after_pilots": remaining_contacts,
                      "remaining_budget_after_pilots": remaining_budget,
                      "total_contacts_including_pilots": 30, "total_cost_including_pilots": 120.0,
                      "fallback": False}}
    full = base_raw + [{**base_raw[0], "campaign_name": "overlay_final", "channel": "push"},
                       {**base_raw[0], "campaign_name": "overlay_pilot", "channel": "push",
                        "filter_current_tariff": "tariff_4", "filter_arpu_segment": "LOW"}]
    checked = validate_overlay_extension(base_raw, full, pilots, profile, tariffs, channels,
                                         remaining_budget, remaining_contacts, pilot_contacts=20)
    assert checked["valid"]
    trace = {"version": "adaptive-portfolio-v1.3", "trace_schema": "nonadditive_full_coverage_overlay_v1",
             "warnings": [], "prior": "fixture", "reference_channel": "sms", "pilots": deepcopy(pilots),
             "base_trace": base, "baseline_prefix": deepcopy(base_raw), "overlay_validation": checked,
             "preflight": deepcopy(checked["preflight"]), "submission_seed": 42,
             "deterministic_submission": True, "environment": "public fixture",
             "versions": {"python": "3.12", "numpy": "fixture", "pandas": "fixture"},
             "hash_format": "sha256-text-lf-v1", "source_sha256": {}, "input_sha256": {},
             "final": {"campaigns": [{**c, "contacts": 10, "cost": 40.0 if i == 0 else 0.0,
                                        "role": "baseline" if i == 0 else "overlay"}
                                       for i, c in enumerate(full)],
                       "contacts": 30, "unique_customers": 20, "cost": 40.0,
                       "remaining_contacts_after_pilots": remaining_contacts,
                       "remaining_budget_after_pilots": remaining_budget,
                       "total_contacts_including_pilots": 50, "total_cost_including_pilots": 120.0,
                       "fallback": False}}
    return trace


def csv_for(trace):
    return renderer.campaign_csv(trace["final"]["campaigns"])


def test_overlay_report_reconstructs_both_coverage_kinds_without_additive_gains(report_case):
    original = deepcopy(report_case)
    base, overlays = renderer.validate_overlay_run(report_case, csv_for(report_case))
    assert report_case == original
    assert [entry["proof"]["kind"] for entry in overlays] == ["baseline", "pilot"]
    assert base["version"] == "adaptive-portfolio-v1.2"
    assert base["source_strategy_version"] == "adaptive-portfolio-v1.3"
    assert all("risk_adjusted_gain" not in entry["campaign"] for entry in overlays)
    assert report_case["final"]["unique_customers"] > base["preflight"]["unique_customers"]


def test_legacy_disjoint_validation_still_accepts_derived_base(report_case):
    base, _ = renderer.validate_overlay_run(report_case, csv_for(report_case))
    renderer.validate_run(base, renderer.campaign_csv(report_case["baseline_prefix"]))


def test_new_schema_can_report_zero_additions(report_case):
    trace = report_case
    trace["final"]["campaigns"] = trace["final"]["campaigns"][:1]
    trace["final"].update(contacts=10, unique_customers=10, total_contacts_including_pilots=30)
    checked = validate_overlay_extension(
        trace["baseline_prefix"], trace["baseline_prefix"], trace["pilots"],
        pd.read_csv(renderer.ROOT / "customer_profile.csv"),
        pd.read_csv(renderer.ROOT / "data/dict_tariff.csv"), renderer.PUBLIC_CHANNELS,
        99920.0, 14980, pilot_contacts=20)
    trace["preflight"] = checked["preflight"]
    trace["overlay_validation"] = checked
    _, additions = renderer.validate_overlay_run(trace, csv_for(trace))
    assert additions == []


@pytest.mark.parametrize("field", ["baseline_prefix", "full_prefix"])
def test_report_rejects_changed_base_prefix(report_case, field):
    row = report_case["baseline_prefix"][0] if field == "baseline_prefix" else report_case["final"]["campaigns"][0]
    row["target_tariff"] = "tariff_4"
    with pytest.raises(ValueError, match="prefix"):
        renderer.validate_overlay_run(report_case, csv_for(report_case))


def test_report_rejects_uncovered_new_audience(report_case):
    campaign = report_case["final"]["campaigns"][-1]
    campaign.update(filter_current_tariff="tariff_8", filter_arpu_segment="HIGH")
    with pytest.raises(ValueError, match="coverage validation failed"):
        renderer.validate_overlay_run(report_case, csv_for(report_case))


@pytest.mark.parametrize("location", ["row", "total"])
def test_report_rejects_invented_additive_gain(report_case, location):
    owner = report_case["final"]["campaigns"][-1] if location == "row" else report_case["final"]
    owner["risk_adjusted_gain"] = 1000.0
    with pytest.raises(ValueError, match="additive gain"):
        renderer.validate_overlay_run(report_case, csv_for(report_case))


@pytest.mark.parametrize("change", ["missing", "tampered"])
def test_report_requires_reproduced_coverage_proofs(report_case, change):
    if change == "missing":
        del report_case["overlay_validation"]
    else:
        report_case["overlay_validation"]["proofs"][0]["matched_size"] = 9
    with pytest.raises(ValueError, match="overlay_validation"):
        renderer.validate_overlay_run(report_case, csv_for(report_case))


@pytest.mark.parametrize("value", [None, 30])
def test_report_requires_actual_unique_final_audience(report_case, value):
    report_case["final"]["unique_customers"] = value
    with pytest.raises(ValueError, match="unique_customers"):
        renderer.validate_overlay_run(report_case, csv_for(report_case))


def test_report_rejects_csv_and_saved_resource_disagreement(report_case):
    original_csv = csv_for(report_case)
    with pytest.raises(ValueError, match="CSV differs"):
        renderer.validate_overlay_run(report_case, original_csv.replace("overlay_final", "other_name"))
    report_case["final"]["campaigns"][-1]["contacts"] = 9
    with pytest.raises(ValueError, match="audience differs"):
        renderer.validate_overlay_run(report_case, original_csv)


def test_report_rejects_changed_observations(report_case):
    report_case["pilots"][0]["observed_lift_ratio"] = 2.0
    with pytest.raises(ValueError, match="pilot observations"):
        renderer.validate_overlay_run(report_case, csv_for(report_case))
