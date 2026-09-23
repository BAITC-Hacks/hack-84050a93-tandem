"""Independent subset-enumeration checks of the actual candidate formula."""

import ast
from itertools import product
from math import prod
from pathlib import Path

import pandas as pd
import pytest

from experiments.signed_pilot_coverage import portfolio


def candidate_increment(ratio, effects):
    # Execute the actual nested production-candidate function, not a duplicate
    # mathematical implementation. Its only closure value is pilot_effects.
    source = Path(portfolio.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "incremental_ratio")
    namespace = {"pilot_effects": sorted(effects, reverse=True)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<candidate formula>", "exec"), namespace)
    return namespace["incremental_ratio"](ratio)


def enumerate_contacts(ratio, effects):
    """Enumerate independent contact/no-contact subsets and signed scorer delta."""
    result = 0.0
    for subset in product((False, True), repeat=len(effects)):
        probability = prod(q if active else 1 - q for active, (_, q) in zip(subset, effects))
        contacted = [s for active, (s, _) in zip(subset, effects) if active]
        if contacted:
            prior = max(contacted)
            increment = max(prior, ratio) - prior
        else:
            increment = ratio
        result += probability * increment
    return result


@pytest.mark.parametrize("effects", [
    [], [(-0.2, 0.0)], [(0.3, 0.0)], [(-0.2, 1.0)], [(0.3, 1.0)],
    [(-0.2, 0.4)], [(0.3, 0.4)], [(-0.2, 0.4), (0.3, 0.6)],
    [(-0.2, 0.4), (-0.2, 0.6)], [(0.2, 0.3), (0.2, 0.7)],
    [(0.2, 0.2), (-0.1, 0.5), (-0.4, 0.8)],
    [(-0.1, 1.0), (0.2, 0.2), (-0.4, 0.6)],
])
@pytest.mark.parametrize("ratio", [-0.5, -0.2, -0.05, 0.0, 0.1, 0.3, 0.5])
def test_signed_formula_matches_exhaustive_subsets(effects, ratio):
    assert candidate_increment(ratio, effects) == pytest.approx(enumerate_contacts(ratio, effects), abs=1e-14)


@pytest.mark.parametrize("effects", [[], [(0.0, 1.0)], [(0.2, 0.4)],
    [(0.0, 0.2), (0.3, 0.5), (0.1, 0.8)], [(0.2, 0.2), (0.2, 0.5), (0.5, 1.0)]])
@pytest.mark.parametrize("ratio", [0.0, 0.05, 0.2, 0.8])
def test_nonnegative_case_equals_old_formula(effects, ratio):
    uncovered, sunk = 1.0, 0.0
    for signed_effect, probability in sorted(effects, reverse=True):
        sunk += uncovered * probability * min(max(0.0, ratio), max(0.0, signed_effect))
        uncovered *= 1 - probability
    assert candidate_increment(ratio, effects) == pytest.approx(ratio - sunk, abs=1e-14)


def test_recovery_can_be_positive_with_negative_final_effect():
    # 50% of clients already suffered -20%; final -5% restores 15% for them.
    # Others lose 5%, so incremental effect is +5%, not old formula's -5%.
    assert candidate_increment(-0.05, [(-0.2, 0.5)]) == pytest.approx(0.05)


class FixtureBelief:
    def __init__(self, target, mean, std, samples):
        self.target = target
        self.key = ("current", "LOW")
        self.mean, self.std = mean, std
        self.samples = samples
        self.pilots = len(samples)

    def posterior(self):
        return self.mean, self.std


def test_make_options_uses_all_targets_latest_signed_mean_scale_and_full_cost():
    profile = pd.DataFrame({"current_tariff": ["current"] * 10, "arpu_segment": ["LOW"] * 10,
                            "data_segment": ["LITE"] * 10, "call_segment": ["LOW"] * 10,
                            "_value": [100.0] * 10})
    beliefs = [FixtureBelief("winner", -0.05, 0.01, [(2, 1.0)]),
               FixtureBelief("other_target", -0.2, 0.01, [(4, 1.0), (3, 0.5)])]
    channels = {"sms": {"conversion_multiplier": 1.0, "cost_per_contact": 2.0}}
    options = portfolio.make_options(profile, beliefs, channels, 1.0, risk_weight=1.65)
    assert len(options) == 1
    option = options[0]
    effects = [(-0.05, 0.2), (-0.2, 0.4), (-0.1, 0.3)]
    assert option.campaign["target_tariff"] == "winner"
    assert option.cost == 20.0
    assert option.mean_gain == pytest.approx(1000 * enumerate_contacts(-0.05, effects) - 20.0)
    assert option.gain == pytest.approx(1000 * enumerate_contacts(-0.05 - 1.65 * 0.01, effects) - 20.0)


def test_isolated_agent_changes_only_import_and_root_history_path():
    root = Path(__file__).resolve().parents[2]
    original = (root / "agent.py").read_text(encoding="utf-8")
    candidate = (Path(__file__).parent / "agent.py").read_text(encoding="utf-8")
    expected = original.replace("from strategy.portfolio import choose_portfolio, make_options",
                                "from experiments.signed_pilot_coverage.portfolio import choose_portfolio, make_options")
    expected = expected.replace('Path(__file__).resolve().parent / "data" / "change_tariff.csv"',
                                'Path(__file__).resolve().parents[2] / "data" / "change_tariff.csv"')
    assert candidate == expected
    assert (root / "data" / "change_tariff.csv").is_file()
