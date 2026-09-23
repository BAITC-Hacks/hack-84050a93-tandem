"""Free Push over final audiences or a cell covered by one complete pilot.

The public baseline trace is produced by ProductionAgent, whose pilots select a
whole current-tariff/ARPU cell before sampling. Equality of one actual sample size
and that cell's public size proves coverage; sums of partial samples do not.
"""

from copy import deepcopy
from math import isfinite

import numpy as np
import pandas as pd

from agent import Agent as ProductionAgent
from experiments.push_overlay.agent import (
    MAX_TOTAL_CONTACTS, _finite_number, _normalized, _ranking_gain, _signature,
    selected_audience,
)
from validation.plan import MAX_CAMPAIGNS, MAX_CUSTOMERS_PER_CAMPAIGN, validate_plan


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


class FullCoveragePushAgent(ProductionAgent):
    def act(self, env):
        base_campaigns = super().act(env)
        base_trace = deepcopy(self.last_trace)
        pilot_contacts = sum(int(pilot["n_customers"]) for pilot in env.pilot_history)
        pilot_cost = sum(float(pilot["cost"]) for pilot in env.pilot_history)
        plan, preflight, details = add_full_coverage_push(
            base_campaigns, base_trace, env.customer_profile, env.tariffs, env.channels,
            env.remaining_budget, env.remaining_contacts, pilot_contacts)
        self.last_trace = {
            "version": "full-coverage-push-experiment-v1",
            "trace_schema": "experimental_nonadditive_full_coverage_overlay_v1",
            "base_trace": base_trace, "baseline_prefix": deepcopy(base_campaigns),
            "pilots": deepcopy(base_trace.get("pilots", [])),
            "reference_channel": base_trace.get("reference_channel"),
            "preflight": preflight, "overlay": details,
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
                "note": "Each overlay repeats a final audience or one completely sampled pilot cell; ranking proxies are nonadditive, not judged profit.",
            },
        }
        return plan


Agent = FullCoveragePushAgent
