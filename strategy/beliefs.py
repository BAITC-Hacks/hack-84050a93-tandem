"""Weak historical priors and Gaussian updates from paid, noisy pilots.

History describes customers who changed tariff, not everyone who received an
offer. Its transition frequency is deliberately NOT used as conversion rate.
No environment implementation or hidden effect model is imported here.
"""

from dataclasses import dataclass, field
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


NOISE_STD = 0.804  # Documented public pilot measurement noise.
ARPU_SEGMENTS = ("LOW", "MID", "HIGH")


def positive_normal(mean: float, std: float) -> float:
    """E[max(X, 0)] for a normal variable, including zero-variance limit."""
    if std <= 1e-12:
        return max(mean, 0.0)
    z = mean / std
    return mean * (0.5 + 0.5 * erf(z / sqrt(2))) + std * exp(-z * z / 2) / sqrt(2 * pi)


@dataclass
class Belief:
    current: str
    segment: str
    target: str
    prior_mean: float
    prior_std: float
    history_n: int = 0
    precision: float = 0.0
    weighted_sum: float = 0.0
    sampled: int = 0
    pilots: int = 0
    samples: list = field(default_factory=list)

    @property
    def key(self):
        return self.current, self.segment

    def posterior(self, shift=0.0):
        prior_precision = 1.0 / self.prior_std**2
        variance = 1.0 / (prior_precision + self.precision)
        mean = variance * ((self.prior_mean + shift) * prior_precision + self.weighted_sum)
        return float(mean), sqrt(variance)

    def observe(self, ratio, n, multiplier, reference_multiplier):
        scale = reference_multiplier / multiplier
        variance = NOISE_STD**2 / n * scale**2
        self.precision += 1.0 / variance
        self.weighted_sum += ratio * scale / variance
        self.sampled += int(n)
        self.pilots += 1
        self.samples.append((int(n), float(multiplier / reference_multiplier)))


class HistoricalPrior:
    """Bounded, partially pooled historical changes; modest assumed conversion.

The conversion assumption is only a weak ranking prior. Even the largest
historical group retains domain-shift uncertainty, so one full pilot has
substantially more weight than history in a deployed decision.
"""

    def __init__(self, tariffs, path: Path, reference_multiplier: float):
        self.reference_multiplier = reference_multiplier
        self.prices = {
            str(row.tariff_plan_code): max(float(row.price_tariff), 0.0)
            for row in tariffs.itertuples()
            if pd.notna(row.price_tariff) and np.isfinite(float(row.price_tariff))
        }
        self.cells = {}
        self.pairs = {}
        self.target_segments = {}
        self.segments = {}
        self.status = "history unavailable; tariff-feature prior"
        if not path.is_file():
            return
        columns = ["AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M", "tariff_plan_code_from", "tariff_plan_code_to"]
        try:
            history = pd.read_csv(path, usecols=columns)
        except (OSError, ValueError, pd.errors.ParserError):
            return
        before = pd.to_numeric(history[columns[0]], errors="coerce")
        after = pd.to_numeric(history[columns[1]], errors="coerce")
        valid = np.isfinite(before) & np.isfinite(after) & (before >= 0) & (after >= 0)
        history = history.loc[valid].copy()
        before, after = before.loc[valid], after.loc[valid]
        history["segment"] = np.where(before < 1000, "LOW", np.where(before <= 5000, "MID", "HIGH"))
        # Stabilize near-zero denominators and winsorize BEFORE aggregation.
        history["change"] = ((after - before) / before.clip(lower=100)).clip(-1.0, 2.0)
        pair_keys = ["tariff_plan_code_from", "tariff_plan_code_to"]
        for key, group in history.groupby(pair_keys, sort=True, observed=True):
            self.pairs[key] = (float(group.change.mean()), len(group))
        for key, group in history.groupby(pair_keys + ["segment"], sort=True, observed=True):
            self.cells[key] = (float(group.change.mean()), len(group))
        for key, group in history.groupby(["tariff_plan_code_to", "segment"], sort=True, observed=True):
            self.target_segments[key] = (float(group.change.mean()), len(group))
        for key, group in history.groupby("segment", sort=True, observed=True):
            self.segments[key] = float(group.change.mean())
        self.status = f"bounded historical prior from {len(history)} valid transitions"

    def belief(self, current, segment, target):
        before = self.prices.get(current, 0.0)
        after = self.prices.get(target, before)
        # Price is an uncertain feature, never an assumed causal revenue effect.
        feature = float(np.clip((after - before) / max(before, 2000.0), -0.6, 1.0)) * 0.35
        pair, pair_n = self.pairs.get((current, target), (feature, 0))
        pair_weight = pair_n / (pair_n + 60.0)
        pair_prior = pair_weight * pair + (1.0 - pair_weight) * feature
        target_mean, target_n = self.target_segments.get((target, segment), (self.segments.get(segment, 0.0), 0))
        target_weight = target_n / (target_n + 40.0)
        segment_prior = target_weight * target_mean + (1.0 - target_weight) * self.segments.get(segment, 0.0)
        # Pool predominantly WITHIN ARPU segment. Transitions by near-zero-ARPU
        # customers must not make the same offer look profitable for HIGH users.
        pooled = 0.65 * segment_prior + 0.15 * pair_prior + 0.20 * feature
        cell, n = self.cells.get((current, target, segment), (pooled, 0))
        weight = n / (n + 25.0)
        conditional_change = weight * cell + (1.0 - weight) * pooled
        mean = 0.30 * self.reference_multiplier * conditional_change
        std = 0.20 + 0.06 / sqrt(1.0 + n / 20.0)
        return Belief(current, segment, target, float(mean), std, int(n))
