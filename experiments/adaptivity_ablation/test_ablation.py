"""Mechanism checks use public fixtures and controlled replies, not new-series scores."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from experiments.adaptivity_ablation import fixed_agent
from experiments.adaptivity_ablation.build_candidate import build
from scoring_core import CHANNELS
from strategy.beliefs import HistoricalPrior

ROOT = Path(__file__).resolve().parents[2]


class Replies:
    def __init__(self, reply, budget=100000, contacts=15000):
        self.tariffs = pd.read_csv(ROOT / "tariff_dictionary.csv")
        self.tariffs = self.tariffs[self.tariffs.tariff_plan_code.isin(["tariff_1", "tariff_4", "tariff_8", "tariff_10"])]
        self.customer_profile = pd.DataFrame([
            dict(ID_NUMBER=i + 240 * j, current_tariff=current, arpu_segment=segment,
                 predicted_arpu=value, data_segment="LITE", call_segment="LOW")
            for j, (current, segment, value) in enumerate(
                (c, s, v) for c in ("tariff_1", "tariff_4") for s, v in (("HIGH", 9000), ("MID", 3500), ("LOW", 700)))
            for i in range(240)])
        self.channels = deepcopy(CHANNELS)
        self.remaining_budget, self.remaining_contacts = budget, contacts
        self.pilots_left = 20
        self.reply = reply
        self.actions = []
        self.before = None

    def run_pilot(self, **action):
        if self.before:
            self.before()
        n = action["n_customers"]
        cost = n * self.channels[action["channel"]]["cost_per_contact"]
        assert 10 <= n <= 200 and n <= self.remaining_contacts and cost <= self.remaining_budget
        self.remaining_budget -= cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        self.actions.append(action)
        return {"n_customers": n, "observed_lift_ratio": self.reply}


def test_only_documented_transformation():
    source = (ROOT / "agent.py").read_text(encoding="utf-8")
    assert (ROOT / "experiments/adaptivity_ablation/fixed_agent.py").read_text(encoding="utf-8") == build(source)
    # Final option generation, portfolio choice, validation, and trace are byte-identical.
    marker = "        options = make_options("
    assert source[source.index(marker):] == build(source)[build(source).index(marker):]


def test_schedule_complete_before_first_reply_and_invariant():
    schedules, actions = [], []
    for reply in (-0.8, 0.8):
        env, agent = Replies(reply), fixed_agent.FixedSurveyAgent()
        snapshots = []
        env.before = lambda: snapshots.append(deepcopy(agent.last_trace["fixed_schedule"]))
        agent.act(env)
        schedule = agent.last_trace["fixed_schedule"]
        assert all(s == schedule for s in snapshots)
        assert env.actions == schedule
        assert {a["filter_arpu_segment"] for a in schedule} == {"HIGH", "MID", "LOW"}
        assert len({a["filter_current_tariff"] for a in schedule[:6]}) > 1
        schedules.append(schedule)
        actions.append(env.actions)
    assert schedules[0] == schedules[1] and actions[0] == actions[1]


def test_final_beliefs_contain_only_real_replies(monkeypatch):
    captured = []
    original = fixed_agent.make_options

    def capture(profile, beliefs, *args, **kwargs):
        captured.extend(deepcopy(beliefs))
        return original(profile, beliefs, *args, **kwargs)

    monkeypatch.setattr(fixed_agent, "make_options", capture)
    env, agent = Replies(-0.37), fixed_agent.FixedSurveyAgent()
    agent.act(env)
    reference = env.channels[agent.last_trace["reference_channel"]]["conversion_multiplier"]
    prior = HistoricalPrior(env.tariffs, ROOT / "data/change_tariff.csv", reference)
    expected = {(b.current, b.segment, b.target): prior.belief(b.current, b.segment, b.target) for b in captured}
    for action in env.actions:
        key = action["filter_current_tariff"], action["filter_arpu_segment"], action["target_tariff"]
        expected[key].observe(-0.37, action["n_customers"], env.channels[action["channel"]]["conversion_multiplier"], reference)
    for actual in captured:
        wanted = expected[actual.current, actual.segment, actual.target]
        assert actual.pilots == wanted.pilots
        assert actual.sampled == wanted.sampled
        assert actual.posterior() == pytest.approx(wanted.posterior())


def test_small_resource_limits():
    env, agent = Replies(0.1, budget=120, contacts=80), fixed_agent.FixedSurveyAgent()
    # Every selectable final cell is 240 people; 80 total contacts cannot fit it.
    # Preserve the existing planner's explicit failure, never exceed resources.
    with pytest.raises(RuntimeError, match="No feasible final campaign"):
        agent.act(env)
    assert env.remaining_budget >= 0 and env.remaining_contacts >= 1
    assert all(10 <= a["n_customers"] <= 200 for a in env.actions)
    from agent import Agent
    with pytest.raises(RuntimeError, match="No feasible final campaign"):
        Agent().act(Replies(0.1, budget=120, contacts=80))
