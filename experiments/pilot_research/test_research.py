"""Checks for the changed policy and independent research harness."""
import pandas as pd
import pytest

from experiments.pilot_research.agent import Agent, StopAgent
from experiments.pilot_research.measure import fixture, summarize, synthetic
from validation.plan import validate_plan


def test_first_probes_are_small_and_history_path_is_preserved():
    env, _, _, _ = fixture("mock", 0)
    agent = Agent()
    plan = agent.act(env)
    seen = set()
    for pilot in agent.last_trace["pilots"]:
        key = (pilot["current_tariff"], pilot["arpu_segment"], pilot["target_tariff"])
        if key not in seen:
            assert 10 <= pilot["n_customers"] <= 40
        seen.add(key)
    assert agent.last_trace["prior"].startswith("bounded historical prior")
    assert validate_plan(plan, env.customer_profile, env.tariffs, env.channels,
                         env.remaining_budget, env.remaining_contacts)["valid"]


def test_stop_requires_broad_evidence_and_returns_valid_plan():
    env, _, _, _ = fixture("devin_all_bad", 0)
    agent = StopAgent()
    plan = agent.act(env)
    stop = agent.last_trace["stop"]
    assert stop["step"] >= 6
    assert stop["pooled_upper"] < 0
    assert {p["arpu_segment"] for p in agent.last_trace["pilots"]} == {"LOW", "MID", "HIGH"}
    assert len(env.pilot_history) < 20
    assert validate_plan(plan, env.customer_profile, env.tariffs, env.channels,
                         env.remaining_budget, env.remaining_contacts)["valid"]


def test_repeat_call_does_not_reuse_evidence():
    agent = StopAgent()
    first, _, _, _ = fixture("devin_rare_good", 42)
    second, _, _, _ = fixture("devin_rare_good", 42)
    plan1 = agent.act(first)
    trace1 = agent.last_trace
    plan2 = agent.act(second)
    assert plan1 == plan2
    assert trace1 == agent.last_trace


@pytest.mark.parametrize("kind", ["negative", "mixed", "rare"])
def test_synthetic_design_is_reproducible_and_moves_positive_pairs(kind):
    a = synthetic(kind + "_1")
    b = synthetic(kind + "_1")
    for left, right in zip(a, b):
        pd.testing.assert_frame_equal(left, right)
    other = synthetic(kind + "_2")[2]
    cols = ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"]
    good1 = set(a[2].loc[a[2].arpu_change_pct > 0, cols].itertuples(index=False, name=None))
    good2 = set(other.loc[other.arpu_change_pct > 0, cols].itertuples(index=False, name=None))
    assert good1 != good2


def test_failures_remain_in_summary_denominator():
    rows = [{"scenario": "x", "seed": 0, "status": "ok", "net": 10},
            {"scenario": "x", "seed": 1, "status": "invalid_plan", "net": 20},
            {"scenario": "x", "seed": 2, "status": "timeout", "net": None}]
    summary = summarize(rows)["x"]
    assert summary["runs"] == 3
    assert summary["valid_positive"] == 1
    assert summary["timeouts"] == summary["invalid"] == 1


def test_paired_comparison_rejects_different_seed_sets():
    from experiments.pilot_research.summarize import paired
    with pytest.raises(ValueError, match="unpaired"):
        paired([{"scenario": "x", "seed": 0, "net": 1}],
               [{"scenario": "x", "seed": 1, "net": 2}])


def test_paired_comparison_retains_failed_attempts():
    from experiments.pilot_research.summarize import paired
    base = [{"scenario": "x", "seed": 0, "net": -10},
            {"scenario": "x", "seed": 1, "net": 20}]
    candidate = [{"scenario": "x", "seed": 0, "net": -5},
                 {"scenario": "x", "seed": 1, "net": None}]
    result = paired(base, candidate)["x"]
    assert result["runs"] == 2
    assert result["paired_scored"] == result["wins"] == 1
    assert result["mean_delta"] == 5


def test_policy_digest_accepts_line_endings_but_detects_source_changes():
    from experiments.pilot_research.audit import source_digest
    assert source_digest(b"a\r\nb\r\n") == source_digest(b"a\nb\n")
    assert source_digest(b"a\nb\n") != source_digest(b"a\nc\n")
