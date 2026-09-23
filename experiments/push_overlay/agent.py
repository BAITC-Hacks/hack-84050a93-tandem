"""Append free Push offers to already covered audiences after one baseline act.

This experimental trace has its own schema; production exporters must not infer
additive gains from it. No effect model, evaluator, or environment internals are
read. The ranking is a disclosed posterior approximation, not measured revenue.
"""

from copy import deepcopy
from math import hypot, isfinite
from numbers import Real

import numpy as np
import pandas as pd

from agent import Agent as ProductionAgent
from strategy.beliefs import positive_normal
from validation.plan import MAX_CAMPAIGNS, MAX_CUSTOMERS_PER_CAMPAIGN, validate_plan


MAX_TOTAL_CONTACTS = 15000
FILTER_COLUMNS = (
    ("filter_arpu_segment", "arpu_segment"),
    ("filter_data_segment", "data_segment"),
    ("filter_call_segment", "call_segment"),
)
FILTER_KEYS = tuple(key for key, _ in FILTER_COLUMNS) + ("filter_current_tariff",)


def _finite_number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(float(value))


def _normalized(value):
    return None if value is None or pd.isna(value) else value


def _signature(campaign):
    return (campaign.get("target_tariff"), campaign.get("channel"),
            *(_normalized(campaign.get(key)) for key in FILTER_KEYS))


def selected_audience(profile, campaign):
    """Evaluate the exact supported public filters; never broaden an audience."""
    part = profile
    for key, column in FILTER_COLUMNS:
        value = _normalized(campaign.get(key))
        if value is not None:
            part = part[part[column] == value]
    current = _normalized(campaign.get("filter_current_tariff"))
    if current is not None:
        wanted = [item.strip() for item in str(current).split(";") if item.strip()]
        part = part[part.current_tariff.isin(wanted)]
    return part


def _ranking_gain(base_posterior, target_posterior, base_scale, push_scale, same_target):
    """Bounded Gaussian positive-part proxy, with shared same-target error.

Different targets use an independence approximation solely for ordering. Channel
scales ignore saturation and prior pilot coverage. No sum is a portfolio gain.
"""
    base_mean, base_std = base_posterior
    target_mean, target_std = target_posterior
    mean_delta = push_scale * target_mean - base_scale * base_mean
    if same_target:
        std_delta = abs(push_scale - base_scale) * base_std
    else:
        std_delta = hypot(push_scale * target_std, base_scale * base_std)
    return min(2.0, max(0.0, positive_normal(mean_delta, std_delta)))


def add_push_overlays(base_campaigns, base_trace, profile, tariffs, channels,
                      remaining_budget, remaining_contacts, pilot_contacts=0):
    """Pure planning over explicit public inputs; no environment object required."""
    plan = deepcopy(base_campaigns)
    final_contact_limit = min(int(remaining_contacts), MAX_TOTAL_CONTACTS - int(pilot_contacts))
    preflight = validate_plan(plan, profile, tariffs, channels, remaining_budget, final_contact_limit)
    if not preflight["valid"]:
        raise ValueError("Invalid baseline for overlay: " + "; ".join(preflight["errors"]))
    details = {"baseline_prefix_length": len(plan), "overlays": [], "candidates": [],
               "ranking": "mass_times_clipped_gaussian_positive_part; nonadditive heuristic",
               "final_contact_limit": final_contact_limit, "skip_reason": None}
    push = channels.get("push")
    if (not isinstance(push, dict) or not _finite_number(push.get("cost_per_contact"))
            or float(push["cost_per_contact"]) != 0.0
            or not _finite_number(push.get("conversion_multiplier"))
            or not 0.0 < float(push["conversion_multiplier"]) <= 1.2):
        details["skip_reason"] = "no_valid_free_push"
        return plan, preflight, details
    if len(plan) >= MAX_CAMPAIGNS or preflight["total_contacts"] >= final_contact_limit:
        details["skip_reason"] = "no_final_capacity"
        return plan, preflight, details

    reference = channels.get(base_trace.get("reference_channel"), {})
    if (not _finite_number(reference.get("conversion_multiplier"))
            or not 0.0 < float(reference["conversion_multiplier"]) <= 1.0):
        details["skip_reason"] = "no_valid_reference_estimate"
        return plan, preflight, details
    reference_multiplier = float(reference["conversion_multiplier"])
    posteriors = {}
    for pilot in base_trace.get("pilots", []):
        mean, std = pilot.get("posterior_reference_mean"), pilot.get("posterior_reference_std")
        if (not _finite_number(mean) or not _finite_number(std) or float(std) < 0
                or not _finite_number(pilot.get("n_customers")) or pilot["n_customers"] <= 0):
            continue
        key = (pilot.get("current_tariff"), pilot.get("arpu_segment"), pilot.get("target_tariff"))
        posteriors[key] = (float(mean), float(std))

    push_scale = float(push["conversion_multiplier"]) / reference_multiplier
    candidates = []
    known = {_signature(campaign) for campaign in plan}
    for base_index, original in enumerate(base_campaigns):
        current = _normalized(original.get("filter_current_tariff"))
        segment = _normalized(original.get("filter_arpu_segment"))
        base_target = original["target_tariff"]
        base_key = (current, segment, base_target)
        if base_key not in posteriors:
            continue
        base_multiplier = channels[original["channel"]].get("conversion_multiplier")
        if not _finite_number(base_multiplier) or not 0.0 < float(base_multiplier) <= 1.2:
            continue
        audience = selected_audience(profile, original)
        contacts = len(audience)
        if not 0 < contacts <= MAX_CUSTOMERS_PER_CAMPAIGN:
            continue
        values = pd.to_numeric(audience.predicted_arpu, errors="coerce")
        mass = float(values.where(np.isfinite(values), 0.0).clip(lower=0.0).fillna(0.0).sum())
        targets = sorted(target for c, s, target in posteriors if c == current and s == segment)
        for target in targets:
            overlay = deepcopy(original)
            overlay["target_tariff"] = target
            overlay["channel"] = "push"
            if _signature(overlay) in known:
                continue
            ratio_proxy = _ranking_gain(
                posteriors[base_key], posteriors[(current, segment, target)],
                float(base_multiplier) / reference_multiplier, push_scale, target == base_target)
            score = mass * ratio_proxy
            if not isfinite(score):
                continue
            candidates.append({"base_index": base_index, "target_tariff": target,
                               "contacts": contacts, "ranking_proxy": float(score),
                               "ranking_ratio_proxy": float(ratio_proxy), "campaign": overlay})

    candidates.sort(key=lambda c: (-c["ranking_proxy"], c["base_index"], c["target_tariff"]))
    names = {campaign.get("campaign_name") for campaign in plan}
    next_name = 1
    for candidate in candidates:
        record = {key: value for key, value in candidate.items() if key != "campaign"}
        details["candidates"].append({**record, "selected": False})
        if len(plan) >= MAX_CAMPAIGNS:
            continue
        if candidate["contacts"] + preflight["total_contacts"] > final_contact_limit:
            continue
        overlay = candidate["campaign"]
        if _signature(overlay) in known:
            continue
        while f"push_overlay_{next_name:02d}" in names:
            next_name += 1
        overlay["campaign_name"] = f"push_overlay_{next_name:02d}"
        trial = validate_plan(plan + [overlay], profile, tariffs, channels,
                              remaining_budget, final_contact_limit)
        if not trial["valid"]:
            continue
        plan.append(overlay)
        preflight = trial
        known.add(_signature(overlay))
        names.add(overlay["campaign_name"])
        details["overlays"].append({**record, "campaign": deepcopy(overlay)})
        details["candidates"][-1]["selected"] = True
    if not details["overlays"]:
        details["skip_reason"] = "no_feasible_new_overlay"
    return plan, preflight, details


class PushOverlayAgent(ProductionAgent):
    """The production policy runs once; only the returned final suffix changes."""

    def act(self, env):
        base_campaigns = super().act(env)
        base_trace = deepcopy(self.last_trace)
        pilot_contacts = sum(int(pilot["n_customers"]) for pilot in env.pilot_history)
        pilot_cost = sum(float(pilot["cost"]) for pilot in env.pilot_history)
        plan, preflight, details = add_push_overlays(
            base_campaigns, base_trace, env.customer_profile, env.tariffs, env.channels,
            env.remaining_budget, env.remaining_contacts, pilot_contacts)
        self.last_trace = {
            "version": "push-overlay-experiment-v1",
            "trace_schema": "experimental_nonadditive_overlay_v1",
            "base_trace": base_trace,
            "baseline_prefix": deepcopy(base_campaigns),
            "pilots": deepcopy(base_trace.get("pilots", [])),
            "reference_channel": base_trace.get("reference_channel"),
            "preflight": preflight,
            "overlay": details,
            "final": {
                "campaigns": [{**deepcopy(campaign), "contacts": item["segment_size"],
                               "cost": item["cost"],
                               "role": "baseline" if index < len(base_campaigns) else "overlay"}
                              for index, (campaign, item) in enumerate(zip(plan, preflight["campaigns"]))],
                "contacts": preflight["total_contacts"], "cost": preflight["total_cost"],
                "unique_customers": preflight["unique_customers"],
                "remaining_contacts_after_pilots": int(env.remaining_contacts),
                "remaining_budget_after_pilots": float(env.remaining_budget),
                "total_contacts_including_pilots": pilot_contacts + preflight["total_contacts"],
                "total_cost_including_pilots": pilot_cost + preflight["total_cost"],
                "note": "Overlays intentionally overlap the baseline; ranking proxies are nonadditive and are not judged profit.",
            },
        }
        return plan


Agent = PushOverlayAgent
