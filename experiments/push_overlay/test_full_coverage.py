"""Coverage proof checks; no effect-model outcomes are measured here."""

import ast
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from agent import Agent as ProductionAgent
from experiments.push_overlay import agent as version_one
from experiments.push_overlay import full_coverage_agent as candidate_module
from experiments.push_overlay.agent import selected_audience
from experiments.push_overlay.full_coverage_agent import (
    FullCoveragePushAgent, add_full_coverage_push, coverage_anchors,
)
from experiments.push_overlay.test_overlay import public_case


@pytest.fixture
def full_case(public_case):
    for pilot in public_case["base_trace"]["pilots"]:
        pilot["channel"] = "sms"
        pilot["n_customers"] = 6 if pilot["arpu_segment"] == "HIGH" else 3
    # Keep only the LOW final audience; HIGH is covered by one full pilot.
    public_case["base_campaigns"] = public_case["base_campaigns"][1:]
    public_case["base_trace"]["pilots"][0]["n_customers"] = 12
    return public_case


def test_one_full_pilot_allows_exact_cell_outside_final_audiences(full_case):
    original = deepcopy(full_case["base_campaigns"])
    trace_before = deepcopy(full_case["base_trace"])
    plan, preflight, details = add_full_coverage_push(**full_case)
    assert plan[:len(original)] == original
    assert full_case["base_campaigns"] == original
    assert full_case["base_trace"] == trace_before
    pilot_anchors = [a for a in details["anchors"] if a["source"] == "full_single_pilot"]
    assert len(pilot_anchors) == 1
    anchor = pilot_anchors[0]
    assert anchor["source_pilot_step"] == 1
    assert anchor["pilot_n_customers"] == anchor["audience_size"] == 12
    assert anchor["campaign"]["target_tariff"] == "tariff_8"
    assert anchor["campaign"]["channel"] == "sms"
    assert anchor["campaign"]["filter_data_segment"] is None
    assert anchor["campaign"]["filter_call_segment"] is None
    high = [c for c in plan if c["filter_arpu_segment"] == "HIGH"]
    assert {c["target_tariff"] for c in high} == {"tariff_8", "tariff_10"}
    assert all(set(selected_audience(full_case["profile"], c).ID_NUMBER) == set(range(1, 13)) for c in high)
    assert preflight["valid"] and preflight["total_cost"] == 0
    assert preflight["total_contacts"] == 36
    assert preflight["unique_customers"] == 18


def test_partial_pilot_counts_cannot_prove_union_coverage(full_case):
    full_case["base_trace"]["pilots"][0]["n_customers"] = 6
    high_pilots = [p for p in full_case["base_trace"]["pilots"] if p["arpu_segment"] == "HIGH"]
    assert sum(p["n_customers"] for p in high_pilots) == 12
    plan, _, details = add_full_coverage_push(**full_case)
    assert all(a["source"] == "baseline_final" for a in details["anchors"])
    assert all(c["filter_arpu_segment"] == "LOW" for c in plan)


@pytest.mark.parametrize("sampled", [11, 13, 0, True, float("nan")])
def test_only_exact_positive_finite_sample_equality_is_coverage(full_case, sampled):
    full_case["base_trace"]["pilots"][0]["n_customers"] = sampled
    anchors = coverage_anchors(full_case["base_campaigns"], full_case["base_trace"], full_case["profile"])
    assert all(a["source"] == "baseline_final" for a in anchors)


def test_explicit_pilot_slice_and_oversized_cell_rejected(full_case):
    pilot = full_case["base_trace"]["pilots"][0]
    pilot["filter_data_segment"] = "LITE"
    assert all(a["source"] == "baseline_final" for a in coverage_anchors(
        full_case["base_campaigns"], full_case["base_trace"], full_case["profile"]))
    pilot["filter_data_segment"] = None
    huge = pd.concat([full_case["profile"].iloc[[0]]] * 5001, ignore_index=True)
    huge["ID_NUMBER"] = range(5001)
    pilot["n_customers"] = 5001
    assert coverage_anchors([], full_case["base_trace"], huge) == []


def test_duplicate_full_pilots_do_not_duplicate_push_suffix(full_case):
    full_case["base_trace"]["pilots"].append(deepcopy(full_case["base_trace"]["pilots"][0]))
    plan, _, details = add_full_coverage_push(**full_case)
    assert len([a for a in details["anchors"] if a["source"] == "full_single_pilot"]) == 2
    signatures = [version_one._signature(c) for c in plan if c["channel"] == "push"]
    assert len(signatures) == len(set(signatures))
    assert len(plan) == 4


@pytest.mark.parametrize("mode", ["paid_push", "nan_push", "contacts", "global_contacts", "slots"])
def test_free_channel_and_capacity_guards(full_case, mode):
    if mode == "paid_push":
        full_case["channels"]["push"]["cost_per_contact"] = 0.01
    elif mode == "nan_push":
        # Make the preserved baseline use a valid channel while rejecting bad Push.
        full_case["base_campaigns"][0]["channel"] = "sms"
        full_case["channels"]["push"]["cost_per_contact"] = float("nan")
    elif mode == "contacts":
        full_case["remaining_contacts"] = 6
    elif mode == "global_contacts":
        full_case["pilot_contacts"] = 14994
    else:
        original = full_case["base_campaigns"][0]
        full_case["base_campaigns"] = [{**original, "campaign_name": f"base_{i}"} for i in range(10)]
    plan, preflight, details = add_full_coverage_push(**full_case)
    assert plan == full_case["base_campaigns"]
    assert preflight["valid"]
    assert not details["overlays"]


def test_rank_helpers_unchanged_and_public_scores_are_nonadditive(full_case):
    assert candidate_module._ranking_gain is version_one._ranking_gain
    assert candidate_module.selected_audience is version_one.selected_audience
    assert candidate_module._signature is version_one._signature
    _, _, details = add_full_coverage_push(**full_case)
    assert "nonadditive" in details["ranking"]
    assert all(0 <= c["ranking_ratio_proxy"] <= 2 for c in details["candidates"])
    for overlay in details["overlays"]:
        anchor = details["anchors"][overlay["anchor_index"]]
        assert anchor["source"] == overlay["source"]
        assert {k: v for k, v in overlay["campaign"].items() if k.startswith("filter_")} == {
            k: v for k, v in anchor["campaign"].items() if k.startswith("filter_")}


def test_unpiloted_target_is_never_added(full_case):
    full_case["base_trace"]["pilots"] = [p for p in full_case["base_trace"]["pilots"]
                                          if p["target_tariff"] != "tariff_10"]
    plan, _, _ = add_full_coverage_push(**full_case)
    assert all(c["target_tariff"] == "tariff_8" for c in plan)


def test_single_super_call_repeatability_trace_totals_and_public_access(monkeypatch, full_case):
    class PublicOnlyEnvironment:
        def __init__(self):
            self.customer_profile = full_case["profile"]
            self.tariffs = full_case["tariffs"]
            self.channels = full_case["channels"]
            self.remaining_contacts = full_case["remaining_contacts"]
            self.remaining_budget = full_case["remaining_budget"]
            self.pilot_history = [{"n_customers": 80, "cost": 320.0}]

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected environment access: {name}")

    calls = []

    def fake_base(self, env):
        calls.append(id(env))
        self.last_trace = deepcopy(full_case["base_trace"])
        return deepcopy(full_case["base_campaigns"])

    monkeypatch.setattr(ProductionAgent, "act", fake_base)
    first, second, env = FullCoveragePushAgent(), FullCoveragePushAgent(), PublicOnlyEnvironment()
    assert first.act(env) == second.act(env)
    assert calls == [id(env), id(env)]
    assert first.last_trace == second.last_trace
    trace = first.last_trace
    assert trace["base_trace"] == full_case["base_trace"]
    assert trace["pilots"] == full_case["base_trace"]["pilots"]
    assert trace["baseline_prefix"] == full_case["base_campaigns"]
    assert trace["final"]["contacts"] == trace["preflight"]["total_contacts"] == 36
    assert trace["final"]["total_contacts_including_pilots"] == 116
    assert trace["final"]["total_cost_including_pilots"] == 320
    assert not any("gain" in key for key in trace["final"])
    trace["base_trace"]["pilots"].clear()
    assert trace["pilots"] and full_case["base_trace"]["pilots"]


def test_unique_names_and_reference_channel_are_preserved(full_case):
    full_case["base_campaigns"][0]["campaign_name"] = "full_push_overlay_01"
    plan, _, details = add_full_coverage_push(**full_case)
    assert plan[0] == full_case["base_campaigns"][0]
    assert len({c["campaign_name"] for c in plan}) == len(plan)
    pilot_anchor = next(a for a in details["anchors"] if a["source"] == "full_single_pilot")
    assert pilot_anchor["campaign"]["channel"] == full_case["base_trace"]["pilots"][0]["channel"]


def test_policy_source_has_no_scorer_model_or_hidden_access():
    tree = ast.parse(Path(__file__).with_name("full_coverage_agent.py").read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports <= {"copy", "math", "numpy", "pandas", "agent",
                       "experiments.push_overlay.agent", "validation.plan"}
    forbidden = {"__closure__", "__globals__", "__dict__", "_model", "_rng", "_Internals"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(tree))


def test_real_seed42_prefix_pilots_and_actual_coverage_without_net_scoring():
    from mock_environment import make_mock_env

    base_env, base_internals = make_mock_env(seed=42)
    candidate_env, candidate_internals = make_mock_env(seed=42)
    baseline, candidate = ProductionAgent(), FullCoveragePushAgent()
    base_plan = baseline.act(base_env)
    plan = candidate.act(candidate_env)
    assert json.dumps(plan[:len(base_plan)], ensure_ascii=False) == json.dumps(base_plan, ensure_ascii=False)
    base_pilots = base_internals.executed_pilot_campaigns()
    assert candidate_internals.executed_pilot_campaigns() == base_pilots
    assert candidate_env.pilot_history == base_env.pilot_history
    assert candidate_env.remaining_contacts == base_env.remaining_contacts
    assert candidate_env.remaining_budget == base_env.remaining_budget
    covered = set().union(*(set(selected_audience(base_env.customer_profile, c).ID_NUMBER) for c in base_plan))
    for pilot in base_pilots:
        covered.update(pilot["explicit_ids"])
    for campaign in plan[len(base_plan):]:
        assert set(selected_audience(candidate_env.customer_profile, campaign).ID_NUMBER) <= covered
    trace = candidate.last_trace
    assert trace["preflight"]["valid"] and len(plan) <= 10
    assert trace["final"]["total_contacts_including_pilots"] <= 15000
    assert trace["preflight"]["total_cost"] == baseline.last_trace["preflight"]["total_cost"]
    assert all(c["contacts"] <= 5000 for c in trace["final"]["campaigns"])
