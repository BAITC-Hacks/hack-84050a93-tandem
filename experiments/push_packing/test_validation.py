from copy import deepcopy

import pandas as pd
import pytest

from experiments.push_packing.validation import validate_packing


@pytest.fixture
def case():
    profile = pd.DataFrame({"ID_NUMBER": range(60), "current_tariff": ["t1"] * 20 + ["t2"] * 20 + ["t3"] * 20,
                            "arpu_segment": ["LOW"] * 60, "data_segment": ["LITE", "HEAVY"] * 30,
                            "call_segment": ["LOW", "HIGH"] * 30, "predicted_arpu": [0.] + [100.] * 59})
    tariffs = pd.DataFrame({"tariff_plan_code": ["t1", "t2", "t3", "t4"]})
    channels = {"push": {"cost_per_contact": 0, "conversion_multiplier": .5},
                "sms": {"cost_per_contact": 4, "conversion_multiplier": .65}}
    def campaign(current, channel="sms", target="t4", name=None):
        return {"campaign_name": name or current + channel, "filter_current_tariff": current,
                "filter_arpu_segment": "LOW", "filter_data_segment": None, "filter_call_segment": None,
                "channel": channel, "target_tariff": target}
    base = [campaign("t1"), campaign("t2")]
    incumbent = base + [campaign("t1", "push"), campaign("t2", "push")]
    pilots = [{"current_tariff": t, "arpu_segment": "LOW", "target_tariff": "t4", "channel": "sms",
               "n_customers": 10, "observed_lift_ratio": .1,
               "posterior_reference_mean": .1, "posterior_reference_std": .05} for t in ("t1", "t2", "t3")]
    trace = {"baseline_prefix": deepcopy(base), "pilots": pilots}
    packed = deepcopy(base) + [campaign("t1;t2", "push", name="merged")]
    return {"incumbent_campaigns": incumbent, "candidate_campaigns": packed, "incumbent_trace": trace,
            "profile": profile, "tariffs": tariffs, "channels": channels,
            "remaining_budget": 100000, "remaining_contacts": 14970, "pilot_contacts": 30}


def test_compression_preserves_entire_incumbent_and_composite_proofs(case):
    result = validate_packing(**case)
    assert result["valid"], result["errors"]
    assert result["preserved_incumbent_effect_rows"] and result["unchanged_cost"]
    assert result["missing_incumbent_effect_rows"] == 0
    assert result["new_effect_rows"] == 0
    assert len(result["proofs"][0]["portions"]) == 2
    assert {p["target_pilot_index"] for p in result["proofs"][0]["portions"]} == {1, 2}
    assert result["warnings"]


def test_identity_plan_is_valid(case):
    case["candidate_campaigns"] = deepcopy(case["incumbent_campaigns"])
    assert validate_packing(**case)["valid"]


def test_losing_incumbent_overlay_is_rejected_even_with_base_preserved(case):
    case["candidate_campaigns"][-1]["filter_current_tariff"] = "t1"
    result = validate_packing(**case)
    assert not result["valid"]
    assert result["missing_incumbent_effect_rows"] == 20


def test_changed_target_drops_incumbent_effect_rows(case):
    case["candidate_campaigns"][-1]["target_tariff"] = "t3"
    result = validate_packing(**case)
    assert not result["valid"]
    assert result["missing_incumbent_effect_rows"] == 40


def test_new_audience_cannot_be_forged_by_trace_anchor_claims(case):
    case["candidate_campaigns"][-1]["filter_current_tariff"] = "t1;t2;t3"
    case["incumbent_trace"]["overlay"] = {"anchors": [{"source": "full_single_pilot", "audience_size": 20}]}
    result = validate_packing(**case)
    assert not result["valid"]
    assert any("unproven customers" in e for e in result["errors"])


@pytest.mark.parametrize("missing", [pd.NA, pd.NaT, float("nan")])
def test_missing_filter_cannot_hide_a_new_audience(case, missing):
    case["candidate_campaigns"][-1]["filter_current_tariff"] = "t1;t2;t3"
    case["candidate_campaigns"][-1]["filter_data_segment"] = missing
    result = validate_packing(**case)
    assert not result["valid"]
    assert any("unproven customers" in e for e in result["errors"])


def test_single_complete_pilot_can_prove_new_portion(case):
    case["incumbent_trace"]["pilots"][-1]["n_customers"] = 20
    case["pilot_contacts"] = 40
    case["candidate_campaigns"][-1]["filter_current_tariff"] = "t1;t2;t3"
    result = validate_packing(**case)
    assert result["valid"], result["errors"]
    assert result["new_effect_rows"] == 20
    assert result["proofs"][0]["portions"][-1]["coverage_sources"] == [
        {"kind": "pilot", "source_index": 3, "matched_contacts": 20}]


def test_partial_pilots_cannot_be_summed(case):
    case["incumbent_trace"]["pilots"].append(deepcopy(case["incumbent_trace"]["pilots"][-1]))
    case["pilot_contacts"] = 40
    case["candidate_campaigns"][-1]["filter_current_tariff"] = "t1;t2;t3"
    assert not validate_packing(**case)["valid"]


@pytest.mark.parametrize("field,value", [("campaign_name", "changed"), ("target_tariff", "t3"), ("channel", "push")])
def test_original_base_prefix_must_be_exact(case, field, value):
    case["candidate_campaigns"][0][field] = value
    result = validate_packing(**case)
    assert not result["valid"]
    assert any("exact original" in e for e in result["errors"])


def test_cost_increase_and_nonpush_suffix_rejected(case):
    case["candidate_campaigns"][-1]["channel"] = "sms"
    assert not validate_packing(**case)["valid"]


@pytest.mark.parametrize("field,value", [("remaining_contacts", 79), ("remaining_budget", 159), ("pilot_contacts", 14921)])
def test_quotas(case, field, value):
    case[field] = value
    assert not validate_packing(**case)["valid"]


def test_campaign_count_limit(case):
    case["candidate_campaigns"] += [deepcopy(case["candidate_campaigns"][-1])] * 8
    assert not validate_packing(**case)["valid"]


def test_merged_campaign_must_not_exceed_5000(case):
    profile = pd.concat([case["profile"].iloc[:40]] * 126, ignore_index=True).assign(ID_NUMBER=range(5040))
    case["profile"] = profile
    result = validate_packing(**case)
    assert not result["valid"]
    assert any("maximum is 5000" in e for e in result["errors"])


def test_unknown_target_rejected(case):
    case["candidate_campaigns"][-1]["target_tariff"] = "unknown"
    assert not validate_packing(**case)["valid"]


def test_nonfinite_pilot_evidence_rejected(case):
    case["incumbent_trace"]["pilots"][0]["posterior_reference_mean"] = float("nan")
    result = validate_packing(**case)
    assert not result["valid"]
    assert any("finite pilot evidence" in e for e in result["errors"])


def test_zero_baseline_row_still_needs_coverage(case):
    case["profile"].loc[40, "predicted_arpu"] = 0
    case["incumbent_trace"]["pilots"][-1]["n_customers"] = 19
    case["pilot_contacts"] = 39
    case["candidate_campaigns"][-1]["filter_current_tariff"] = "t1;t2;t3"
    assert not validate_packing(**case)["valid"]


def test_no_mutation(case):
    before = deepcopy(case)
    assert validate_packing(**case)["valid"]
    for key in case:
        if isinstance(case[key], pd.DataFrame):
            pd.testing.assert_frame_equal(case[key], before[key])
        else:
            assert case[key] == before[key]
