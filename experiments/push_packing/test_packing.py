"""Focused effect-preserving packing checks; no model scoring."""
from copy import deepcopy

import pandas as pd
import pytest

from agent import Agent as IncumbentAgent
from experiments.push_packing.agent import PackedPushAgent, atomic_keys, pack_and_fill
from strategy.full_coverage_push import selected_audience


@pytest.fixture
def case():
    currents = list("ABCDEFGHI")
    rows = [{"ID_NUMBER": i, "current_tariff": "A", "arpu_segment": "LOW", "data_segment": "LITE",
             "call_segment": "LOW", "predicted_arpu": 700.0} for i in range(3)]
    for current in currents:
        for _ in range(4):
            rows.append({"ID_NUMBER": len(rows), "current_tariff": current, "arpu_segment": "HIGH",
                         "data_segment": "LITE", "call_segment": "LOW", "predicted_arpu": 9000.0})
    def campaign(current, segment, target="T", channel="push", name="source"):
        return dict(campaign_name=name, filter_current_tariff=current, filter_arpu_segment=segment,
                    filter_data_segment=None, filter_call_segment=None, target_tariff=target, channel=channel)
    prefix = [campaign("A", "LOW", name="base")]
    suffix = [campaign(current, "HIGH", name=f"old_{current}") for current in currents]
    anchors = [{"source": "baseline_final", "campaign": deepcopy(prefix[0])}] + [
        {"source": "full_single_pilot", "campaign": campaign(current, "HIGH", channel="sms")}
        for current in currents]
    pilots = [dict(current_tariff=current, arpu_segment="HIGH", target_tariff=target,
                   channel="sms", n_customers=4) for current in currents for target in ("T", "U")]
    pilots.append(dict(current_tariff="A", arpu_segment="LOW", target_tariff="T", channel="sms", n_customers=3))
    for pilot in pilots:
        pilot.update(posterior_reference_mean=0.1, posterior_reference_std=0.05, observed_lift_ratio=0.1)
    candidates = [dict(anchor_index=i, target_tariff="T", selected=True) for i in range(1, 10)] + [
        dict(anchor_index=i, target_tariff="U", selected=False) for i in (1, 2)]
    return dict(incumbent_campaigns=prefix + suffix,
                incumbent_trace={"baseline_prefix": prefix, "pilots": pilots,
                                 "overlay": {"anchors": anchors, "candidates": candidates}},
                profile=pd.DataFrame(rows), tariffs=pd.DataFrame({"tariff_plan_code": currents + ["T", "U"]}),
                channels={"push": {"cost_per_contact": 0, "conversion_multiplier": .5},
                          "sms": {"cost_per_contact": 4, "conversion_multiplier": .65}},
                remaining_budget=1000, remaining_contacts=1000, pilot_contacts=75)


def effects(profile, plan):
    return {(i, campaign["target_tariff"], campaign["channel"]) for campaign in plan
            for i in selected_audience(profile, campaign).ID_NUMBER}


def test_union_exact_prefix_and_incumbent_impacts_preserved(case):
    original = deepcopy(case["incumbent_campaigns"])
    plan, preflight, details = pack_and_fill(**case)
    assert case["incumbent_campaigns"] == original
    assert plan[0] == original[0]
    assert plan[1]["campaign_name"] == "old_A"
    assert plan[1]["filter_current_tariff"] == "A;B;C;D;E;F;G;H;I"
    assert effects(case["profile"], original) == effects(case["profile"], plan[:2])
    assert effects(case["profile"], original) <= effects(case["profile"], plan)
    assert preflight["valid"] and preflight["total_contacts"] == 47
    assert preflight["total_cost"] == 0 and len(plan) == 4
    assert len(details["additions"]) == 2


def test_no_new_offer_returns_identical_plan_object(case):
    case["incumbent_trace"]["overlay"]["candidates"] = []
    plan, _, details = pack_and_fill(**case)
    assert plan is case["incumbent_campaigns"]
    assert not details["changed"] and not details["merges"]


def test_semantic_duplicate_after_union_is_skipped(case):
    case["incumbent_trace"]["overlay"]["candidates"].insert(0, dict(anchor_index=1, target_tariff="T", selected=False))
    plan, _, details = pack_and_fill(**case)
    assert len(plan) == 4 and len(details["additions"]) == 2
    assert ("A", "T", "push", "HIGH", None, None) in atomic_keys(plan[1])


@pytest.mark.parametrize("mode", ["contacts", "global", "paid", "group_cap"])
def test_resource_and_group_caps_return_incumbent(case, mode):
    if mode == "contacts": case["remaining_contacts"] = 39
    elif mode == "global": case["pilot_contacts"] = 15000 - 39
    elif mode == "paid": case["channels"]["push"]["cost_per_contact"] = 1
    else:
        # Every original campaign fits, while merging the whole group does not.
        profile = pd.concat([case["profile"]] * 150, ignore_index=True)
        profile["ID_NUMBER"] = range(len(profile))
        case["profile"] = profile
        case["remaining_contacts"] = 14000
    plan, _, details = pack_and_fill(**case)
    assert plan is case["incumbent_campaigns"] and not details["changed"]


def test_empty_rows_are_rejected_not_silently_dropped(case):
    case["incumbent_campaigns"][1]["filter_data_segment"] = "HEAVY"
    with pytest.raises(ValueError, match="empty segment"):
        pack_and_fill(**case)


def test_noop_preserves_trace_and_calls_incumbent_once(monkeypatch, case):
    calls = []
    case["incumbent_trace"]["overlay"]["candidates"] = []
    def original(self, env):
        calls.append(env)
        self.last_trace = case["incumbent_trace"]
        return case["incumbent_campaigns"]
    monkeypatch.setattr(IncumbentAgent, "act", original)
    class PublicEnv:
        customer_profile = case["profile"]
        tariffs = case["tariffs"]
        channels = case["channels"]
        remaining_budget = case["remaining_budget"]
        remaining_contacts = case["remaining_contacts"]
        pilot_history = [{"n_customers": 75, "cost": 300}]
    agent, env = PackedPushAgent(), PublicEnv()
    assert agent.act(env) is case["incumbent_campaigns"]
    assert agent.last_trace is case["incumbent_trace"]
    assert calls == [env]


def test_mock42_exact_noop_without_net_scoring():
    from mock_environment import make_mock_env
    env1, internal1 = make_mock_env(seed=42)
    env2, internal2 = make_mock_env(seed=42)
    incumbent, packed = IncumbentAgent(), PackedPushAgent()
    assert packed.act(env2) == incumbent.act(env1)
    assert packed.last_trace == incumbent.last_trace
    assert env1.pilot_history == env2.pilot_history
    assert internal1.executed_pilot_campaigns() == internal2.executed_pilot_campaigns()


def test_changed_wrapper_requires_independent_preservation_check(monkeypatch, case):
    def original(self, env):
        self.last_trace = case["incumbent_trace"]
        return case["incumbent_campaigns"]
    monkeypatch.setattr(IncumbentAgent, "act", original)
    class PublicEnv:
        customer_profile = case["profile"]
        tariffs = case["tariffs"]
        channels = case["channels"]
        remaining_budget = case["remaining_budget"]
        remaining_contacts = case["remaining_contacts"]
        pilot_history = [{"n_customers": 75, "cost": 300}]
    agent = PackedPushAgent()
    plan = agent.act(PublicEnv())
    assert len(plan) == 4
    assert agent.last_trace["packing_validation"]["valid"]
    assert agent.last_trace["final"]["total_contacts_including_pilots"] == 122
    assert agent.last_trace["incumbent_trace"] == case["incumbent_trace"]
