"""Prespecified synthetic fixtures for the push-overlay experiment.

Harness-only module: never import it from an agent implementation. These eight
models are new synthetic effects, not estimates of the hidden judging model.
Development and holdout use disjoint model seeds. A fixture's profile and model
do not change with its pilot-noise seed. Positive offers are sampled uniformly
over tariff/ARPU/target triples by a separate generator, without observing cell
sizes, baseline values, historical transitions, or an agent's decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from environment import make_environment
from mock_environment import CHANNELS, MAX_TOTAL_CONTACTS, TOTAL_BUDGET


ROOT = Path(__file__).resolve().parents[2]
SEGMENTS = ("LOW", "MID", "HIGH")
DEVELOPMENT_TARIFFS = ("tariff_2", "tariff_4", "tariff_8", "tariff_11", "tariff_17")
HOLDOUT_TARIFFS = ("tariff_3", "tariff_5", "tariff_9", "tariff_12", "tariff_16", "tariff_21")
NEW_DEVELOPMENT = (
    "overlay_dev_mixed", "overlay_dev_negative", "overlay_dev_rare", "overlay_dev_high_conversion",
)
NEW_HOLDOUT = (
    "overlay_holdout_mixed", "overlay_holdout_negative", "overlay_holdout_rare", "overlay_holdout_high_conversion",
)
CUSTOM_SCENARIOS = NEW_DEVELOPMENT + NEW_HOLDOUT


@dataclass(frozen=True)
class Scenario:
    split: str
    family: str
    model_seed: int
    tariffs: tuple[str, ...]
    customers: int
    positive_count: int
    positive_effect_range: tuple[float, float]
    negative_effect_range: tuple[float, float]
    conversion_range: tuple[float, float]


# One declaration made before evaluation; no candidate outcomes select these
# configurations. 5 tariffs yield 60 nonidentity triples; 6 yield 90 triples.
CONFIGS = {
    "overlay_dev_mixed": Scenario(
        "development", "mixed", 2026092301, DEVELOPMENT_TARIFFS, 2160, 21,
        (0.06, 0.70), (-0.35, -0.015), (0.25, 0.80)),
    "overlay_dev_negative": Scenario(
        "development", "negative", 2026092302, DEVELOPMENT_TARIFFS, 2160, 0,
        (0.06, 0.70), (-0.40, -0.015), (0.25, 0.85)),
    "overlay_dev_rare": Scenario(
        "development", "rare", 2026092303, DEVELOPMENT_TARIFFS, 2160, 1,
        (0.60, 1.20), (-0.24, -0.025), (0.40, 0.75)),
    "overlay_dev_high_conversion": Scenario(
        "development", "high_conversion", 2026092304, DEVELOPMENT_TARIFFS, 2160, 30,
        (0.04, 0.45), (-0.24, -0.020), (0.88, 0.995)),
    "overlay_holdout_mixed": Scenario(
        "holdout", "mixed", 2026092401, HOLDOUT_TARIFFS, 2700, 36,
        (0.04, 0.55), (-0.30, -0.010), (0.20, 0.90)),
    "overlay_holdout_negative": Scenario(
        "holdout", "negative", 2026092402, HOLDOUT_TARIFFS, 2700, 0,
        (0.04, 0.55), (-0.30, -0.010), (0.30, 0.90)),
    "overlay_holdout_rare": Scenario(
        "holdout", "rare", 2026092403, HOLDOUT_TARIFFS, 2700, 1,
        (0.45, 0.95), (-0.20, -0.015), (0.35, 0.85)),
    "overlay_holdout_high_conversion": Scenario(
        "holdout", "high_conversion", 2026092404, HOLDOUT_TARIFFS, 2700, 36,
        (0.03, 0.60), (-0.30, -0.010), (0.90, 0.999)),
}


def _config(name: str) -> Scenario:
    if name not in CONFIGS:
        raise ValueError(f"Unknown push-overlay scenario: {name!r}")
    return CONFIGS[name]


def synthetic(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return profile, public tariff subset, and harness-only effect model."""
    config = _config(name)
    profile_seed, effect_seed, placement_seed = np.random.SeedSequence(config.model_seed).spawn(3)
    profile_rng = np.random.default_rng(profile_seed)
    effect_rng = np.random.default_rng(effect_seed)
    placement_rng = np.random.default_rng(placement_seed)

    public_tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    indexed = public_tariffs.set_index("tariff_plan_code", drop=False)
    missing = set(config.tariffs) - set(indexed.index)
    if missing:
        raise ValueError(f"Official tariff dictionary is missing {sorted(missing)}")
    tariffs = indexed.loc[list(config.tariffs)].reset_index(drop=True).copy()

    cells = [(current, segment) for current in config.tariffs for segment in SEGMENTS]
    triples = [(current, segment, target) for current, segment in cells
               for target in config.tariffs if target != current]
    positive_indices = set(int(i) for i in placement_rng.choice(
        len(triples), size=config.positive_count, replace=False))
    model_rows = []
    for index, (current, segment, target) in enumerate(triples):
        effect_range = (config.positive_effect_range if index in positive_indices
                        else config.negative_effect_range)
        model_rows.append({
            "tariff_plan_code_from": current,
            "arpu_segment": segment,
            "tariff_plan_code_to": target,
            "arpu_change_pct": float(effect_rng.uniform(*effect_range)),
            "conversion_rate": float(effect_rng.uniform(*config.conversion_range)),
        })
    model = pd.DataFrame(model_rows)

    # Allocate rows independently of all effects. Every cell supports a legal
    # pilot; unequal cell sizes are intentional but never used to place a winner.
    minimum_cell = 40
    weights = profile_rng.dirichlet(np.full(len(cells), 2.0))
    counts = minimum_cell + profile_rng.multinomial(
        config.customers - minimum_cell * len(cells), weights)
    baseline_ranges = {"LOW": (150.0, 950.0), "MID": (1200.0, 4800.0), "HIGH": (6000.0, 15000.0)}
    rows = []
    for (current, segment), count in zip(cells, counts):
        lower, upper = baseline_ranges[segment]
        values = np.exp(profile_rng.uniform(np.log(lower), np.log(upper), size=int(count)))
        data = profile_rng.choice(("NON_USER", "LITE", "HEAVY"), size=int(count))
        calls = profile_rng.choice(("LOW", "MEDIUM", "HIGH"), size=int(count))
        for value, data_segment, call_segment in zip(values, data, calls):
            rows.append({
                "ID_NUMBER": len(rows) + 1,
                "current_tariff": current,
                "arpu_segment": segment,
                "data_segment": str(data_segment),
                "call_segment": str(call_segment),
                "predicted_arpu": float(value),
            })
    return pd.DataFrame(rows), tariffs, model


def fallback(current, target, segment, tariffs, conversion):
    """Used only by the harness for unlisted offers; no hidden-model lookup."""
    if current == target:
        return 0.0, 0.0
    return -0.10, float(conversion)


def fixture(name: str, seed: int):
    """Mirror pilot_research.measure.fixture; seed controls pilot noise only."""
    profile, tariffs, model = synthetic(name)
    env, internals = make_environment(
        profile, model, tariffs, CHANNELS, TOTAL_BUDGET,
        MAX_TOTAL_CONTACTS, fallback, seed=seed,
    )
    return env, internals, model, fallback


def _frame_hash(frame: pd.DataFrame) -> str:
    data = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def metadata(name: str) -> dict:
    """Serializable prespecified design and generated input hashes for reports.

    Does not run an agent or a pilot. Hash format is UTF-8 CSV with LF, no index,
    and %.17g floating-point fields; noise-seed choices are deliberately absent.
    """
    config = asdict(_config(name))
    profile, tariffs, model = synthetic(name)
    positive = model.loc[model.arpu_change_pct * model.conversion_rate > 0]
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cell_sizes = profile.groupby(["current_tariff", "arpu_segment"], sort=True, observed=True).size()
    return {
        "name": name,
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "profile_sha256": _frame_hash(profile),
        "tariffs_sha256": _frame_hash(tariffs),
        "model_sha256": _frame_hash(model),
        "hash_format": "sha256-utf8-csv-lf-float17g-v1",
        "customers": int(len(profile)),
        "cells": int(len(cell_sizes)),
        "smallest_cell": int(cell_sizes.min()),
        "largest_cell": int(cell_sizes.max()),
        "modeled_offers": int(len(model)),
        "positive_offers": int(len(positive)),
        "call_saturation_offers": int((model.conversion_rate * CHANNELS["call"]["conversion_multiplier"] > 1.0).sum()),
        "positive_triples": positive[["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"]].to_dict("records"),
        "rng_streams": {"profile": [0], "effects": [1], "positive_placement": [2]},
        "limitations": [
            "Synthetic fixed models, not a prediction of hidden judging performance.",
            "Repeating pilot-noise seeds does not create additional effect models.",
            "Positive placements do not depend on cell sizes or agent behavior.",
            "High conversion exercises saturation in the scorer; it does not force an agent to use calls.",
        ],
    }
