"""Preserve incumbent effects, compact free Push, then fill freed slots once."""

from collections import defaultdict
from copy import deepcopy

from agent import Agent as IncumbentAgent
from experiments.push_packing.validation import validate_packing
from strategy.full_coverage_push import _finite_number, _normalized, selected_audience
from validation.plan import validate_plan


FILTERS = ("filter_arpu_segment", "filter_data_segment", "filter_call_segment")


def atomic_keys(campaign):
    """A union-current campaign has the same offer for each atomic current."""
    current = campaign.get("filter_current_tariff")
    if not isinstance(current, str):
        return set()
    return {(part.strip(), campaign.get("target_tariff"), campaign.get("channel"),
             *(_normalized(campaign.get(key)) for key in FILTERS))
            for part in current.split(";") if part.strip()}


def pack_and_fill(incumbent_campaigns, incumbent_trace, profile, tariffs, channels,
                  remaining_budget, remaining_contacts, pilot_contacts=0):
    limit = min(int(remaining_contacts), 15000 - int(pilot_contacts))
    original_check = validate_plan(incumbent_campaigns, profile, tariffs, channels, remaining_budget, limit)
    if not original_check["valid"]:
        raise ValueError("Invalid incumbent: " + "; ".join(original_check["errors"]))
    prefix = incumbent_trace["baseline_prefix"]
    prefix_length = len(prefix)
    if incumbent_campaigns[:prefix_length] != prefix:
        raise ValueError("Incumbent baseline prefix does not match its trace")
    details = {"baseline_prefix_length": prefix_length, "merges": [], "additions": [], "changed": False}
    push = channels.get("push", {})
    if not _finite_number(push.get("cost_per_contact")) or float(push["cost_per_contact"]) != 0.0:
        return incumbent_campaigns, original_check, details
    groups = defaultdict(list)
    for index, campaign in enumerate(incumbent_campaigns[prefix_length:], start=prefix_length):
        current = campaign.get("filter_current_tariff")
        if campaign.get("channel") != "push" or not isinstance(current, str) or ";" in current or not current:
            continue
        key = (campaign["target_tariff"], *(_normalized(campaign.get(f)) for f in FILTERS))
        groups[key].append(index)
    replacements, removed = {}, set()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        currents = [incumbent_campaigns[i]["filter_current_tariff"] for i in indices]
        if len(set(currents)) != len(currents):
            continue
        sizes = [original_check["campaigns"][i]["segment_size"] for i in indices]
        if not all(n > 0 for n in sizes) or sum(sizes) > 5000:
            continue
        merged = deepcopy(incumbent_campaigns[indices[0]])
        merged["filter_current_tariff"] = ";".join(sorted(currents))
        expected = set().union(*(set(selected_audience(profile, incumbent_campaigns[i]).ID_NUMBER) for i in indices))
        actual = set(selected_audience(profile, merged).ID_NUMBER)
        if actual != expected or len(actual) != sum(sizes):
            continue
        replacements[indices[0]] = merged
        removed.update(indices[1:])
        details["merges"].append({"indices": indices, "current_tariffs": sorted(currents),
                                   "contacts": len(actual), "campaign": deepcopy(merged)})
    plan = [deepcopy(replacements.get(i, campaign)) for i, campaign in enumerate(incumbent_campaigns) if i not in removed]
    preflight = validate_plan(plan, profile, tariffs, channels, remaining_budget, limit)
    if not preflight["valid"] or preflight["total_contacts"] != original_check["total_contacts"]:
        raise ValueError("Packing changed incumbent contacts or validity")
    known = set().union(*(atomic_keys(campaign) for campaign in plan))
    names = {campaign.get("campaign_name") for campaign in plan}
    overlay = incumbent_trace["overlay"]
    next_name = 1
    for candidate_index, candidate in enumerate(overlay["candidates"]):
        if candidate["selected"] or len(plan) >= 10:
            continue
        anchor_index = candidate["anchor_index"]
        anchor = overlay["anchors"][anchor_index]
        campaign = deepcopy(anchor["campaign"])
        campaign["channel"] = "push"
        campaign["target_tariff"] = candidate["target_tariff"]
        keys = atomic_keys(campaign)
        if not keys or keys <= known:
            continue
        audience = selected_audience(profile, campaign)
        count = len(audience)
        if not 0 < count <= 5000 or count + preflight["total_contacts"] > limit:
            continue
        current, segment = campaign.get("filter_current_tariff"), campaign.get("filter_arpu_segment")
        if not any(p.get("current_tariff") == current and p.get("arpu_segment") == segment
                   and p.get("target_tariff") == campaign["target_tariff"] and p.get("n_customers", 0) > 0
                   for p in incumbent_trace["pilots"]):
            continue
        while f"packing_push_{next_name:02d}" in names:
            next_name += 1
        campaign["campaign_name"] = f"packing_push_{next_name:02d}"
        checked = validate_plan(plan + [campaign], profile, tariffs, channels, remaining_budget, limit)
        if not checked["valid"]:
            continue
        plan.append(campaign)
        known.update(keys)
        names.add(campaign["campaign_name"])
        preflight = checked
        details["additions"].append({"candidate_index": candidate_index, "anchor_index": anchor_index,
                                     "contacts": count, "campaign": deepcopy(campaign)})
    if not details["additions"]:
        details["merges"] = []
        return incumbent_campaigns, original_check, details
    details["changed"] = True
    return plan, preflight, details


class PackedPushAgent(IncumbentAgent):
    def act(self, env):
        incumbent = super().act(env)
        incumbent_trace = self.last_trace
        pilot_contacts = sum(int(p["n_customers"]) for p in env.pilot_history)
        plan, preflight, details = pack_and_fill(
            incumbent, incumbent_trace, env.customer_profile, env.tariffs, env.channels,
            env.remaining_budget, env.remaining_contacts, pilot_contacts)
        if not details["changed"]:
            return incumbent
        preservation = validate_packing(
            incumbent, plan, incumbent_trace, env.customer_profile, env.tariffs, env.channels,
            env.remaining_budget, env.remaining_contacts, pilot_contacts)
        if not preservation["valid"]:
            raise RuntimeError("Invalid incumbent-preserving packing: " + "; ".join(preservation["errors"]))
        self.last_trace = {
            "version": "packed-push-experiment-v1", "trace_schema": "experimental_packing_v1",
            "incumbent_trace": deepcopy(incumbent_trace), "incumbent_campaigns": deepcopy(incumbent),
            "baseline_prefix": deepcopy(incumbent_trace["baseline_prefix"]),
            "pilots": deepcopy(incumbent_trace["pilots"]), "packing": details, "preflight": preflight,
            "packing_validation": preservation,
            "final": {"campaigns": deepcopy(plan), "contacts": preflight["total_contacts"],
                      "cost": preflight["total_cost"], "unique_customers": preflight["unique_customers"],
                      "total_contacts_including_pilots": pilot_contacts + preflight["total_contacts"],
                      "total_cost_including_pilots": sum(float(p["cost"]) for p in env.pilot_history) + preflight["total_cost"],
                      "note": "Incumbent effects retained; free offers appended. No additive gain estimate."},
        }
        return plan


Agent = PackedPushAgent
