"""Focused checks for the free-Push suffix and its public-input boundary."""

import ast
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from agent import Agent as ProductionAgent
from experiments.push_overlay.agent import (
    PushOverlayAgent, _ranking_gain, add_push_overlays, selected_audience,
)
from strategy.beliefs import positive_normal


@pytest.fixture
def public_case():
    profile = pd.DataFrame([
        {"ID_NUMBER": i + 1, "current_tariff": "tariff_1",
         "arpu_segment": "HIGH" if i < 12 else "LOW",
         "data_segment": "LITE" if i % 2 == 0 else "HEAVY",
         "call_segment": "LOW" if i % 3 == 0 else "HIGH",
         "predicted_arpu": 9000.0 if i < 12 else 700.0}
        for i in range(18)
    ])
    tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_8", "tariff_10"],
                            "price_tariff": [1000.0, 3000.0, 5000.0]})
    channels = {"sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
                "push": {"cost_per_contact": 0, "conversion_multiplier": 0.5}}
    base = [{"campaign_name": "base_high", "filter_current_tariff": "tariff_1",
             "filter_arpu_segment": "HIGH", "filter_data_segment": "LITE",
             "filter_call_segment": "LOW", "target_tariff": "tariff_8", "channel": "sms"},
            {"campaign_name": "base_low", "filter_current_tariff": "tariff_1",
             "filter_arpu_segment": "LOW", "filter_data_segment": None,
             "filter_call_segment": None, "target_tariff": "tariff_8", "channel": "push"}]
    trace = {"reference_channel": "sms", "pilots": [
        {"current_tariff": "tariff_1", "arpu_segment": segment,
         "target_tariff": target, "n_customers": 20,
         "posterior_reference_mean": mean, "posterior_reference_std": 0.04}
        for segment in ("HIGH", "LOW")
        for target, mean in (("tariff_8", 0.1), ("tariff_10", 0.3))
    ]}
    return {"base_campaigns": base, "base_trace": trace, "profile": profile,
            "tariffs": tariffs, "channels": channels, "remaining_budget": 1000,
            "remaining_contacts": 100, "pilot_contacts": 80}


def test_exact_prefix_audiences_piloted_targets_and_no_input_mutation(public_case):
    original_plan = deepcopy(public_case["base_campaigns"])
    original_trace = deepcopy(public_case["base_trace"])
    original_profile = public_case["profile"].copy(deep=True)
    plan, preflight, details = add_push_overlays(**public_case)
    assert plan[:len(original_plan)] == original_plan
    assert public_case["base_campaigns"] == original_plan
    assert public_case["base_trace"] == original_trace
    pd.testing.assert_frame_equal(public_case["profile"], original_profile)
    assert len(details["overlays"]) == 3
    original_covered = set().union(*(set(selected_audience(original_profile, c).ID_NUMBER)
                                     for c in original_plan))
    for overlay in details["overlays"]:
        campaign = overlay["campaign"]
        source = original_plan[overlay["base_index"]]
        assert {k: v for k, v in campaign.items() if k.startswith("filter_")} == {
            k: v for k, v in source.items() if k.startswith("filter_")}
        audience = set(selected_audience(original_profile, campaign).ID_NUMBER)
        assert audience == set(selected_audience(original_profile, source).ID_NUMBER)
        assert audience <= original_covered
        assert campaign["channel"] == "push"
        assert any(p["target_tariff"] == campaign["target_tariff"]
                   and p["current_tariff"] == campaign["filter_current_tariff"]
                   and p["arpu_segment"] == campaign["filter_arpu_segment"]
                   for p in original_trace["pilots"])
    assert preflight["valid"]
    assert preflight["total_cost"] == 8
    assert preflight["unique_customers"] == 8
    assert preflight["total_contacts"] == 18


def test_duplicate_existing_push_is_skipped(public_case):
    plan, _, _ = add_push_overlays(**public_case)
    low_original = public_case["base_campaigns"][1]
    matching = [c for c in plan if c["channel"] == "push" and c["target_tariff"] == "tariff_8"
                and all(c.get(k) == low_original.get(k) for k in low_original if k.startswith("filter_"))]
    assert matching == [low_original]


@pytest.mark.parametrize("cost", [0.01, float("nan"), None, True])
def test_no_finite_exactly_free_push_returns_base(public_case, cost):
    public_case["base_campaigns"] = public_case["base_campaigns"][:1]
    public_case["channels"]["push"]["cost_per_contact"] = cost
    plan, preflight, details = add_push_overlays(**public_case)
    assert plan == public_case["base_campaigns"]
    assert preflight["valid"]
    assert details["skip_reason"] == "no_valid_free_push"


def test_missing_push_returns_base(public_case):
    public_case["base_campaigns"] = public_case["base_campaigns"][:1]
    del public_case["channels"]["push"]
    assert add_push_overlays(**public_case)[0] == public_case["base_campaigns"]


@pytest.mark.parametrize("mode", ["remaining_contacts", "total_contact_cap", "campaign_slots"])
def test_exhausted_capacity_returns_exact_base(public_case, mode):
    if mode == "remaining_contacts":
        public_case["remaining_contacts"] = 8
    elif mode == "total_contact_cap":
        public_case["pilot_contacts"] = 14992
    else:
        original = public_case["base_campaigns"][0]
        public_case["base_campaigns"] = [{**original, "campaign_name": f"base_{i}"} for i in range(10)]
    plan, preflight, details = add_push_overlays(**public_case)
    assert plan == public_case["base_campaigns"]
    assert details["skip_reason"] == "no_final_capacity"
    assert preflight["valid"]


def test_contact_limit_skips_large_candidate_and_appends_smaller(public_case):
    public_case["remaining_contacts"] = 10
    plan, preflight, details = add_push_overlays(**public_case)
    assert len(plan) == 3
    assert preflight["total_contacts"] == 10
    assert details["overlays"][0]["contacts"] == 2


def test_no_piloted_target_and_campaign_cap_guard(public_case):
    public_case["base_trace"]["pilots"] = []
    assert add_push_overlays(**public_case)[0] == public_case["base_campaigns"]
    original = public_case["profile"].iloc[[0]]
    public_case["profile"] = pd.concat([original] * 5001, ignore_index=True)
    public_case["profile"]["ID_NUMBER"] = range(5001)
    public_case["base_campaigns"] = public_case["base_campaigns"][:1]
    public_case["remaining_contacts"] = 15000
    public_case["remaining_budget"] = 100000
    with pytest.raises(ValueError, match="maximum is 5000"):
        add_push_overlays(**public_case)


def test_same_target_uses_shared_uncertainty_and_ranking_is_bounded():
    posterior = (0.1, 0.2)
    value = _ranking_gain(posterior, posterior, 1.0, 0.5, True)
    assert value == pytest.approx(positive_normal(-0.05, 0.1))
    assert _ranking_gain(posterior, posterior, 1.0, 1.0, True) == 0.0
    assert _ranking_gain((-100, 0), (100, 0), 1.0, 1.0, False) == 2.0


def test_repeatability_and_trace_are_actual(monkeypatch, public_case):
    class PublicOnlyEnvironment:
        def __init__(self):
            self.customer_profile = public_case["profile"]
            self.tariffs = public_case["tariffs"]
            self.channels = public_case["channels"]
            self.remaining_contacts = public_case["remaining_contacts"]
            self.remaining_budget = public_case["remaining_budget"]
            self.pilot_history = [{"n_customers": 80, "cost": 320.0}]

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected environment access: {name}")

    calls = []

    def fake_baseline(self, env):
        calls.append(id(env))
        self.last_trace = deepcopy(public_case["base_trace"])
        return deepcopy(public_case["base_campaigns"])

    monkeypatch.setattr(ProductionAgent, "act", fake_baseline)
    first, second = PushOverlayAgent(), PushOverlayAgent()
    env = PublicOnlyEnvironment()
    first_plan, second_plan = first.act(env), second.act(env)
    assert calls == [id(env), id(env)]
    assert first_plan == second_plan
    assert first.last_trace == second.last_trace
    trace = first.last_trace
    assert trace["base_trace"] == public_case["base_trace"]
    assert trace["baseline_prefix"] == public_case["base_campaigns"]
    assert trace["pilots"] == public_case["base_trace"]["pilots"]
    assert trace["final"]["contacts"] == 18
    assert trace["final"]["total_contacts_including_pilots"] == 98
    assert trace["final"]["cost"] == 8
    assert trace["final"]["total_cost_including_pilots"] == 328
    assert trace["preflight"]["total_contacts"] == trace["final"]["contacts"]
    assert not any("gain" in key for key in trace["final"])
    trace["base_trace"]["pilots"].clear()
    assert public_case["base_trace"]["pilots"]
    assert trace["pilots"]


def test_real_baseline_prefix_pilots_and_resources_match():
    from mock_environment import make_mock_env

    base_env, base_internals = make_mock_env(seed=42)
    candidate_env, candidate_internals = make_mock_env(seed=42)
    baseline, candidate = ProductionAgent(), PushOverlayAgent()
    base_plan = baseline.act(base_env)
    plan = candidate.act(candidate_env)
    assert json.dumps(plan[:len(base_plan)], ensure_ascii=False) == json.dumps(base_plan, ensure_ascii=False)
    assert candidate_env.pilot_history == base_env.pilot_history
    assert candidate_internals.executed_pilot_campaigns() == base_internals.executed_pilot_campaigns()
    assert candidate_env.remaining_contacts == base_env.remaining_contacts
    assert candidate_env.remaining_budget == base_env.remaining_budget
    assert candidate.last_trace["preflight"]["valid"]
    assert candidate.last_trace["final"]["total_contacts_including_pilots"] <= 15000
    assert candidate.last_trace["preflight"]["total_cost"] == baseline.last_trace["preflight"]["total_cost"]
    assert all(c["contacts"] <= 5000 for c in candidate.last_trace["final"]["campaigns"])


def test_policy_imports_and_attributes_exclude_hidden_access():
    source = Path(__file__).with_name("agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports <= {"copy", "math", "numbers", "numpy", "pandas", "agent",
                       "strategy.beliefs", "validation.plan"}
    forbidden = {"__closure__", "__globals__", "__dict__", "_model", "_rng", "_Internals"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(tree))
