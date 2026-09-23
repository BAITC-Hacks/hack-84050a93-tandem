from copy import deepcopy

import pandas as pd
import pytest

from validation.overlay import validate_overlay_extension


@pytest.fixture
def inputs():
    profile = pd.DataFrame({
        "ID_NUMBER": range(40), "current_tariff": ["tariff_1"] * 20 + ["tariff_2"] * 20,
        "arpu_segment": ["LOW"] * 40,
        "data_segment": ["LITE", "HEAVY"] * 20,
        "call_segment": ["LOW", "HIGH"] * 20,
        "predicted_arpu": [0.0] + [100.0] * 39,
    })
    tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_2", "tariff_3"]})
    channels = {"push": {"cost_per_contact": 0, "conversion_multiplier": .5},
                "sms": {"cost_per_contact": 4, "conversion_multiplier": .65}}
    return profile, tariffs, channels


def campaign(current="tariff_1", **updates):
    value = {"campaign_name": "base", "filter_current_tariff": current, "filter_arpu_segment": "LOW",
             "filter_data_segment": None, "filter_call_segment": None,
             "target_tariff": "tariff_2", "channel": "sms"}
    value.update(updates)
    return value


def pilot(current="tariff_1", n=10, target="tariff_3", **updates):
    value = {"current_tariff": current, "arpu_segment": "LOW", "target_tariff": target,
             "channel": "sms", "n_customers": n, "observed_lift_ratio": -.1}
    value.update(updates)
    return value


def overlay(base, **updates):
    value = {**base, "campaign_name": "overlay", "channel": "push", "target_tariff": "tariff_3"}
    value.update(updates)
    return value


def check(inputs, base=None, full=None, pilots=None, budget=100000, contacts=15000, **kwargs):
    base = [campaign()] if base is None else base
    full = base + [overlay(base[0])] if full is None else full
    pilots = [pilot()] if pilots is None else pilots
    return validate_overlay_extension(base, full, pilots, *inputs, budget, contacts, **kwargs)


def test_valid_exact_baseline_subset_and_warnings(inputs):
    base = [campaign(filter_data_segment="LITE")]
    result = check(inputs, base=base)
    assert result["valid"], result["errors"]
    assert result["preflight"]["total_contacts"] == 20
    assert result["preflight"]["unique_customers"] == 10
    assert result["warnings"] and result["preflight"]["warnings"]
    assert result["proofs"][0] == {
        "campaign_index": 1, "campaign_name": "overlay", "kind": "baseline", "source_index": 0,
        "target_pilot_index": 1, "contacts": 10, "matched_size": 10,
    }


def test_valid_single_complete_pilot_outside_base(inputs):
    base = [campaign(current="tariff_2")]
    result = check(inputs, base=base, full=base + [overlay(campaign())], pilots=[pilot(n=20)])
    assert result["valid"], result["errors"]
    assert result["proofs"][0]["kind"] == "pilot"
    assert result["proofs"][0]["source_index"] == 1
    assert result["proofs"][0]["matched_size"] == 20
    assert result["total_contacts_including_pilots"] == 60


@pytest.mark.parametrize("change", [
    {"campaign_name": "renamed"}, {"target_tariff": "tariff_3"}, {"filter_data_segment": float("nan")},
])
def test_changed_raw_prefix_rejected(inputs, change):
    base = [campaign()]
    result = check(inputs, base=base, full=[{**base[0], **change}, overlay(base[0])])
    assert not result["valid"]
    assert any("exact unchanged baseline prefix" in error for error in result["errors"])


def test_missing_prefix_rejected(inputs):
    result = check(inputs, full=[])
    assert not result["valid"]
    assert any("prefix" in error for error in result["errors"])


def test_new_audience_and_sum_of_partial_pilots_are_rejected(inputs):
    base = [campaign(current="tariff_2")]
    full = base + [overlay(campaign())]
    for pilots in ([pilot(n=10)], [pilot(n=10), pilot(n=10)]):
        result = check(inputs, base=base, full=full, pilots=pilots)
        assert not result["valid"]
        assert any("single complete-pilot coverage proof" in error for error in result["errors"])


def test_subset_of_complete_pilot_is_not_whole_cell_proof(inputs):
    base = [campaign(current="tariff_2")]
    full = base + [overlay(campaign(filter_data_segment="LITE"))]
    result = check(inputs, base=base, full=full, pilots=[pilot(n=20)])
    assert not result["valid"]
    assert result["proofs"][0]["kind"] is None


@pytest.mark.parametrize("pilots", [[], [pilot(target="tariff_2")], [pilot(current="tariff_2")]])
def test_unpiloted_target_or_wrong_cell_is_rejected(inputs, pilots):
    result = check(inputs, pilots=pilots)
    assert not result["valid"]
    assert any("target was not piloted" in error for error in result["errors"])


@pytest.mark.parametrize("cost", [4, -1, float("nan"), float("inf"), True, "0"])
def test_push_must_cost_finite_numeric_zero(inputs, cost):
    inputs[2]["push"]["cost_per_contact"] = cost
    result = check(inputs)
    assert not result["valid"]
    assert any("finite cost_per_contact exactly zero" in error for error in result["errors"])


def test_paid_overlay_rejected(inputs):
    base = [campaign()]
    result = check(inputs, base=base, full=base + [overlay(base[0], channel="sms")])
    assert not result["valid"]
    assert any("must use push" in error for error in result["errors"])


def test_contact_money_campaign_and_global_limits(inputs):
    assert not check(inputs, contacts=39)["valid"]
    assert not check(inputs, budget=79)["valid"]
    base = [campaign()]
    assert not check(inputs, base=base, full=base + [overlay(base[0])] * 10)["valid"]
    result = check(inputs, pilot_contacts=14961)
    assert not result["valid"]
    assert any("maximum is 15000" in error for error in result["errors"])
    assert check(inputs, pilot_contacts=14960)["valid"]


def test_campaign_size_cap(inputs):
    profile, tariffs, channels = inputs
    large = pd.concat([profile.iloc[[0]]] * 5001, ignore_index=True).assign(ID_NUMBER=range(5001))
    result = check((large, tariffs, channels))
    assert not result["valid"]
    assert any("maximum is 5000" in error for error in result["errors"])


def test_zero_baseline_rows_are_included_in_full_pilot_size(inputs):
    base = [campaign(current="tariff_2")]
    full = base + [overlay(campaign())]
    partial = check(inputs, base=base, full=full, pilots=[pilot(n=19)])
    complete = check(inputs, base=base, full=full, pilots=[pilot(n=20)])
    assert not partial["valid"]
    assert complete["valid"] and complete["proofs"][0]["matched_size"] == 20


@pytest.mark.parametrize("bad_ids", [[0] * 40, [None] + list(range(1, 40))])
def test_unique_present_profile_ids_required(inputs, bad_ids):
    inputs[0]["ID_NUMBER"] = bad_ids
    result = check(inputs)
    assert not result["valid"]
    assert any("present and unique" in error for error in result["errors"])


def test_baseline_must_itself_be_strict_and_disjoint(inputs):
    base = [campaign(), campaign(campaign_name="second")]
    result = check(inputs, base=base, full=base)
    assert not result["valid"]
    assert any("baseline final audiences must be disjoint" in error for error in result["errors"])


@pytest.mark.parametrize("value", [9, -1, 10.5, float("nan"), True])
def test_pilot_contact_override_cannot_undercount_or_be_invalid(inputs, value):
    result = check(inputs, pilot_contacts=value)
    assert not result["valid"]
    assert any("at least the recorded" in error for error in result["errors"])


def test_override_accounts_for_unrecorded_nonfinite_response(inputs):
    result = check(inputs, pilot_contacts=30)
    assert result["valid"]
    assert result["recorded_pilot_contacts"] == 10
    assert result["total_contacts_including_pilots"] == 70
    assert any("absent from the recorded decision trace" in warning for warning in result["warnings"])


def test_raw_env_history_and_filtered_pilot_are_rejected(inputs):
    raw = {"target_tariff": "tariff_3", "channel": "sms", "n_customers": 20}
    assert not check(inputs, pilots=[raw])["valid"]
    assert not check(inputs, pilots=[pilot(n=20, filter_data_segment="LITE")])["valid"]
    assert not check(inputs, pilots=[pilot(n=20, filter_current_tariff=["tariff_1"])])["valid"]


def test_invalid_profile_and_nonlist_inputs_fail_closed(inputs):
    base = [campaign()]
    assert not check(inputs, base=base, full=tuple(base))["valid"]
    invalid_profile = (pd.DataFrame({"ID_NUMBER": [1]}), inputs[1], inputs[2])
    result = check(invalid_profile)
    assert not result["valid"]
    assert result["proofs"][0]["kind"] is None


def test_full_pilot_coverage_can_have_a_different_target(inputs):
    base = [campaign(current="tariff_2")]
    result = check(inputs, base=base, full=base + [overlay(campaign())],
                   pilots=[pilot(n=20, target="tariff_2"), pilot(n=10, target="tariff_3")])
    assert result["valid"], result["errors"]
    assert result["proofs"][0]["source_index"] == 1
    assert result["proofs"][0]["target_pilot_index"] == 2


def test_inputs_unchanged_and_noop_valid(inputs):
    base, pilots = [campaign()], [pilot()]
    full = base + [overlay(base[0])]
    old_base, old_full, old_pilots = deepcopy(base), deepcopy(full), deepcopy(pilots)
    old_profile, old_tariffs, old_channels = inputs[0].copy(deep=True), inputs[1].copy(deep=True), deepcopy(inputs[2])
    assert check(inputs, base=base, full=full, pilots=pilots)["valid"]
    assert check(inputs, base=base, full=base, pilots=[])["valid"]
    assert base == old_base and full == old_full and pilots == old_pilots
    pd.testing.assert_frame_equal(inputs[0], old_profile)
    pd.testing.assert_frame_equal(inputs[1], old_tariffs)
    assert inputs[2] == old_channels
