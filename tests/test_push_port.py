"""Port equivalence; historical outcome measurements remain frozen at fbcd614.

The integrated experimental entry points import the explicit BaselineAgent.
Their algorithm bodies and the production helpers must retain frozen ASTs.
These checks compare campaigns/pilots, without remeasuring outcome scores.
"""

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from agent import Agent, BaselineAgent
from experiments.pilot_research.measure import fixture
from experiments.push_overlay import agent as first_candidate
from experiments.push_overlay import full_coverage_agent as full_candidate
from experiments.push_overlay.measure import POLICIES


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_BASELINE = "75716aa8b4dec4bcb9eeb471709c3caa19529211"
FROZEN_EXPERIMENT = "fbcd614bea7d9c4a5d63f3ef0f3b199aa7ab5af0"


def source_tree(path, revision=None):
    if revision is None:
        source = (ROOT / path).read_text(encoding="utf-8")
    else:
        try:
            source = subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT,
                text=True, encoding="utf-8", stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            pytest.skip(f"Historical AST audit requires local Git revision {revision}: {exc}")
    return ast.parse(source)


def named_node(tree, name, kind):
    return next(node for node in tree.body if isinstance(node, kind) and node.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def test_explicit_baseline_bindings_preserve_experimental_meaning():
    assert first_candidate.ProductionAgent is BaselineAgent
    assert full_candidate.ProductionAgent is BaselineAgent
    assert POLICIES["baseline"] == "agent:BaselineAgent"
    assert first_candidate.PushOverlayAgent.__bases__ == (BaselineAgent,)
    assert full_candidate.FullCoveragePushAgent.__bases__ == (BaselineAgent,)


def test_original_baseline_ast_unchanged_except_class_name():
    frozen = named_node(source_tree("agent.py", ORIGINAL_BASELINE), "Agent", ast.ClassDef)
    integrated = deepcopy(named_node(source_tree("agent.py"), "BaselineAgent", ast.ClassDef))
    integrated.name = "Agent"
    assert dump(integrated) == dump(frozen)


def test_frozen_helper_asts_and_experimental_algorithm_bodies_unchanged():
    production = source_tree("strategy/full_coverage_push.py")
    groups = {
        "experiments/push_overlay/agent.py": (
            "_finite_number", "_normalized", "_signature", "selected_audience", "_ranking_gain"),
        "experiments/push_overlay/full_coverage_agent.py": ("coverage_anchors", "add_full_coverage_push"),
    }
    for path, functions in groups.items():
        frozen = source_tree(path, FROZEN_EXPERIMENT)
        adapted = source_tree(path)
        for name in functions:
            original = named_node(frozen, name, ast.FunctionDef)
            assert dump(named_node(production, name, ast.FunctionDef)) == dump(original), name
        for node in frozen.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert dump(named_node(adapted, node.name, type(node))) == dump(node), node.name
    old_helpers = source_tree("experiments/push_overlay/agent.py", FROZEN_EXPERIMENT)
    for name in ("MAX_TOTAL_CONTACTS", "FILTER_COLUMNS", "FILTER_KEYS"):
        def assignment(tree):
            return next(node for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert dump(assignment(production)) == dump(assignment(old_helpers)), name


@pytest.mark.parametrize("scenario,seed", [
    ("mock", 42), ("devin_all_bad", 5), ("mixed_2", 3), ("negative_2", 0), ("rare_1", 1),
])
def test_production_exactly_matches_frozen_candidate_actions(scenario, seed):
    integrated_env, integrated_internals, _, _ = fixture(scenario, seed)
    frozen_env, frozen_internals, _, _ = fixture(scenario, seed)
    integrated, frozen = Agent(), full_candidate.FullCoveragePushAgent()
    integrated_plan, frozen_plan = integrated.act(integrated_env), frozen.act(frozen_env)
    assert json.dumps(integrated_plan, ensure_ascii=False) == json.dumps(frozen_plan, ensure_ascii=False)
    assert integrated_env.pilot_history == frozen_env.pilot_history
    assert integrated_internals.executed_pilot_campaigns() == frozen_internals.executed_pilot_campaigns()
    assert integrated_env.remaining_contacts == frozen_env.remaining_contacts
    assert integrated_env.remaining_budget == frozen_env.remaining_budget
    assert integrated.last_trace["baseline_prefix"] == frozen.last_trace["baseline_prefix"]
    assert integrated.last_trace["base_trace"] == frozen.last_trace["base_trace"]
    assert integrated.last_trace["preflight"] == frozen.last_trace["preflight"]
    assert integrated.last_trace["overlay"] == frozen.last_trace["overlay"]
    assert integrated.last_trace["overlay_validation"]["valid"]
