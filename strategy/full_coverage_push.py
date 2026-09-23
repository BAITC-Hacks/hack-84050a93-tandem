"""Free Push additions on audiences already reached by the base plan or a full pilot.

Policy functions are identical to the frozen FullCoveragePushAgent experiment.
They use only public inputs and observed pilot summaries. Ranking is a heuristic;
conditional non-decrease follows from the published scorer and validated coverage.
"""
from copy import deepcopy
from math import hypot, isfinite
from numbers import Real

import numpy as np
import pandas as pd

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



def coverage_anchors(base_campaigns, base_trace, profile):
    """Evidence for each allowed audience, without reading individual pilot IDs."""
    anchors = []
    for index, campaign in enumerate(base_campaigns):
        anchors.append({"anchor_index": len(anchors), "source": "baseline_final",
                        "source_final_index": index,
                        "audience_size": len(selected_audience(profile, campaign)),
                        "coverage_evidence": "exact_original_final_audience",
                        "campaign": deepcopy(campaign)})
    for step, pilot in enumerate(base_trace.get("pilots", []), start=1):
        current, segment = pilot.get("current_tariff"), pilot.get("arpu_segment")
        if not isinstance(current, str) or segment not in {"LOW", "MID", "HIGH"}:
            continue
        # ProductionAgent uses neither data nor call filters in its pilots.
        # Reject an explicit slice rather than pretending it covered a whole cell.
        if any(_normalized(pilot.get(key)) is not None
               for key in ("filter_data_segment", "filter_call_segment")):
            continue
        campaign = {"filter_current_tariff": current, "filter_arpu_segment": segment,
                    "filter_data_segment": None, "filter_call_segment": None,
                    "target_tariff": pilot.get("target_tariff"), "channel": pilot.get("channel")}
        size = len(selected_audience(profile, campaign))
        sampled = pilot.get("n_customers")
        if (not 0 < size <= MAX_CUSTOMERS_PER_CAMPAIGN or not _finite_number(sampled)
                or float(sampled) != size):
            continue
        anchors.append({"anchor_index": len(anchors), "source": "full_single_pilot",
                        "source_pilot_step": step, "pilot_n_customers": int(sampled),
                        "audience_size": size,
                        "coverage_evidence": "one_actual_whole_cell_pilot_n_equals_public_cell_size",
                        "campaign": campaign})
    return anchors


def add_full_coverage_push(base_campaigns, base_trace, profile, tariffs, channels,
                           remaining_budget, remaining_contacts, pilot_contacts=0):
    """Keep the baseline prefix, rank v1-style offers over all proven anchors."""
    plan = deepcopy(base_campaigns)
    limit = min(int(remaining_contacts), MAX_TOTAL_CONTACTS - int(pilot_contacts))
    preflight = validate_plan(plan, profile, tariffs, channels, remaining_budget, limit)
    if not preflight["valid"]:
        raise ValueError("Invalid baseline for full-coverage overlay: " + "; ".join(preflight["errors"]))
    anchors = coverage_anchors(base_campaigns, base_trace, profile)
    details = {"baseline_prefix_length": len(plan), "anchors": anchors,
               "overlays": [], "candidates": [], "final_contact_limit": limit,
               "ranking": "v1_mass_times_clipped_gaussian_positive_part; nonadditive heuristic",
               "skip_reason": None}
    push = channels.get("push")
    if (not isinstance(push, dict) or not _finite_number(push.get("cost_per_contact"))
            or float(push["cost_per_contact"]) != 0.0
            or not _finite_number(push.get("conversion_multiplier"))
            or not 0.0 < float(push["conversion_multiplier"]) <= 1.2):
        details["skip_reason"] = "no_valid_free_push"
        return plan, preflight, details
    if len(plan) >= MAX_CAMPAIGNS or preflight["total_contacts"] >= limit:
        details["skip_reason"] = "no_final_capacity"
        return plan, preflight, details
    reference = channels.get(base_trace.get("reference_channel"), {})
    if (not _finite_number(reference.get("conversion_multiplier"))
            or not 0.0 < float(reference["conversion_multiplier"]) <= 1.0):
        details["skip_reason"] = "no_valid_reference_estimate"
        return plan, preflight, details
    reference_multiplier = float(reference["conversion_multiplier"])
    push_scale = float(push["conversion_multiplier"]) / reference_multiplier

    posteriors = {}
    for pilot in base_trace.get("pilots", []):
        mean, std = pilot.get("posterior_reference_mean"), pilot.get("posterior_reference_std")
        if (not _finite_number(mean) or not _finite_number(std) or float(std) < 0
                or not _finite_number(pilot.get("n_customers")) or pilot["n_customers"] <= 0):
            continue
        key = (pilot.get("current_tariff"), pilot.get("arpu_segment"), pilot.get("target_tariff"))
        posteriors[key] = (float(mean), float(std))

    known = {_signature(campaign) for campaign in plan}
    candidates = []
    for anchor in anchors:
        original = anchor["campaign"]
        current = _normalized(original.get("filter_current_tariff"))
        segment = _normalized(original.get("filter_arpu_segment"))
        anchor_target = original["target_tariff"]
        anchor_key = (current, segment, anchor_target)
        if anchor_key not in posteriors:
            continue
        channel = channels.get(original["channel"], {})
        anchor_multiplier = channel.get("conversion_multiplier")
        if not _finite_number(anchor_multiplier) or not 0.0 < float(anchor_multiplier) <= 1.2:
            continue
        audience = selected_audience(profile, original)
        if not 0 < len(audience) <= MAX_CUSTOMERS_PER_CAMPAIGN:
            continue
        values = pd.to_numeric(audience.predicted_arpu, errors="coerce")
        mass = float(values.where(np.isfinite(values), 0.0).clip(lower=0.0).fillna(0.0).sum())
        targets = sorted(target for c, s, target in posteriors if c == current and s == segment)
        for target in targets:
            campaign = deepcopy(original)
            campaign["target_tariff"] = target
            campaign["channel"] = "push"
            if _signature(campaign) in known:
                continue
            ratio = _ranking_gain(posteriors[anchor_key], posteriors[(current, segment, target)],
                                  float(anchor_multiplier) / reference_multiplier,
                                  push_scale, target == anchor_target)
            score = mass * ratio
            if not isfinite(score):
                continue
            candidates.append({"anchor_index": anchor["anchor_index"], "source": anchor["source"],
                               "target_tariff": target, "contacts": len(audience),
                               "ranking_proxy": float(score), "ranking_ratio_proxy": float(ratio),
                               "campaign": campaign})
    candidates.sort(key=lambda c: (-c["ranking_proxy"], c["anchor_index"], c["target_tariff"]))
    names = {campaign.get("campaign_name") for campaign in plan}
    next_name = 1
    for candidate in candidates:
        record = {key: value for key, value in candidate.items() if key != "campaign"}
        details["candidates"].append({**record, "selected": False})
        if len(plan) >= MAX_CAMPAIGNS or candidate["contacts"] + preflight["total_contacts"] > limit:
            continue
        campaign = candidate["campaign"]
        if _signature(campaign) in known:
            continue
        while f"full_push_overlay_{next_name:02d}" in names:
            next_name += 1
        campaign["campaign_name"] = f"full_push_overlay_{next_name:02d}"
        trial = validate_plan(plan + [campaign], profile, tariffs, channels, remaining_budget, limit)
        if not trial["valid"]:
            continue
        plan.append(campaign)
        preflight = trial
        names.add(campaign["campaign_name"])
        known.add(_signature(campaign))
        details["overlays"].append({**record, "campaign": deepcopy(campaign)})
        details["candidates"][-1]["selected"] = True
    if not details["overlays"]:
        details["skip_reason"] = "no_feasible_new_overlay"
    return plan, preflight, details


