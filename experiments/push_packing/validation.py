"""Independent packing certificate against the entire accepted v1.3 incumbent.

Only public profile IDs are used. Pilot records must be authentic tandem v1.3
trace entries (whole current-tariff/ARPU pilots, actual sample sizes). This API
does not authenticate an invented trace and never reads actual hidden pilot IDs.
No model, scorer, candidate implementation, or revenue estimate is imported.
"""
from __future__ import annotations

from numbers import Real
import math

import numpy as np
import pandas as pd

from validation.overlay import validate_overlay_extension, _same_raw
from validation.plan import validate_plan


FILTERS = (("filter_arpu_segment", "arpu_segment"), ("filter_data_segment", "data_segment"),
           ("filter_call_segment", "call_segment"))


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _present(value):
    if value is None:
        return False
    missing = pd.isna(value)
    return not (isinstance(missing, (bool, np.bool_)) and missing)


def _audience(profile, campaign):
    part = profile
    for key, column in FILTERS:
        if _present(campaign.get(key)):
            part = part[part[column] == campaign[key]]
    if _present(campaign.get("filter_current_tariff")):
        codes = [s.strip() for s in str(campaign["filter_current_tariff"]).split(";") if s.strip()]
        part = part[part.current_tariff.isin(codes)]
    return part


def validate_packing(incumbent_campaigns, candidate_campaigns, incumbent_trace, profile,
                     tariffs, channels, remaining_budget, remaining_contacts, pilot_contacts):
    """Certify unchanged old offers and covered, free new effects, with no caps.

    Preservation is by (public ID, target, channel), not campaign name or count.
    Composite proofs split each suffix audience into current-tariff/ARPU portions.
    Each portion cites base-final and/or single-full-pilot coverage plus a pilot
    of its offered target. Partial pilot sample counts are never summed.
    """
    errors, warnings, proofs = [], [], []
    result = {"valid": False, "errors": errors, "warnings": warnings, "proofs": proofs,
              "incumbent_validation": None, "preflight": None,
              "preserved_incumbent_effect_rows": False, "unchanged_cost": False}
    if not isinstance(incumbent_trace, dict):
        errors.append("incumbent_trace must be an authentic v1.3 trace dict")
        return result
    base = incumbent_trace.get("baseline_prefix")
    pilots = incumbent_trace.get("pilots")
    if not isinstance(base, list) or not isinstance(pilots, list):
        errors.append("incumbent trace requires baseline_prefix and pilots lists")
        return result
    try:
        old = validate_overlay_extension(base, incumbent_campaigns, pilots, profile, tariffs, channels,
                                         remaining_budget, remaining_contacts, pilot_contacts=pilot_contacts)
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid incumbent input: {type(exc).__name__}: {exc}")
        return result
    result["incumbent_validation"] = old
    errors.extend("incumbent: " + e for e in old["errors"])
    warnings.extend("incumbent: " + w for w in old["warnings"])
    if not isinstance(candidate_campaigns, list):
        errors.append("candidate campaigns must be a list")
        return result
    try:
        new = validate_plan(candidate_campaigns, profile, tariffs, channels, remaining_budget, remaining_contacts)
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid candidate input: {type(exc).__name__}: {exc}")
        return result
    result["preflight"] = new
    errors.extend("candidate: " + e for e in new["errors"])
    warnings.extend("candidate: " + w for w in new["warnings"])
    if not _same_raw(candidate_campaigns[:len(base)], base):
        errors.append("candidate must preserve the exact original BaselineAgent prefix")
    if not _finite(pilot_contacts) or not float(pilot_contacts).is_integer() or pilot_contacts < 0:
        errors.append("pilot_contacts must be a finite non-negative integer")
    elif pilot_contacts + new["total_contacts"] > 15000:
        errors.append("pilot plus candidate contacts exceed 15000")
    result["total_contacts_including_pilots"] = (int(pilot_contacts) + new["total_contacts"]
                                               if _finite(pilot_contacts) else None)
    result["unchanged_cost"] = new["total_cost"] == old["preflight"]["total_cost"]
    if not result["unchanged_cost"]:
        errors.append("candidate final cost must equal the entire incumbent final cost")
    if errors:
        return result

    def offers(plan):
        return {(customer, campaign["target_tariff"], campaign["channel"])
                for campaign in plan for customer in _audience(profile, campaign).ID_NUMBER}

    incumbent_rows, candidate_rows = offers(incumbent_campaigns), offers(candidate_campaigns)
    missing = incumbent_rows - candidate_rows
    result["incumbent_effect_rows"] = len(incumbent_rows)
    result["candidate_effect_rows"] = len(candidate_rows)
    result["new_effect_rows"] = len(candidate_rows - incumbent_rows)
    result["missing_incumbent_effect_rows"] = len(missing)
    result["preserved_incumbent_effect_rows"] = not missing
    if missing:
        errors.append(f"candidate drops {len(missing)} incumbent (ID,target,channel) effect rows")

    # Coverage sources are reconstructed from public inputs, never candidate claims.
    sources = []
    for index, campaign in enumerate(base):
        sources.append({"kind": "baseline", "source_index": index,
                        "ids": set(_audience(profile, campaign).ID_NUMBER)})
    target_evidence = {}
    for index, pilot in enumerate(pilots, start=1):
        key = (pilot["current_tariff"], pilot["arpu_segment"], pilot["target_tariff"])
        mean, std, observed = (pilot.get(k) for k in
                               ("posterior_reference_mean", "posterior_reference_std", "observed_lift_ratio"))
        if _finite(mean) and _finite(std) and std >= 0 and _finite(observed):
            target_evidence.setdefault(key, index)
        cell = profile[(profile.current_tariff == key[0]) & (profile.arpu_segment == key[1])]
        if len(cell) > 0 and pilot["n_customers"] == len(cell):
            sources.append({"kind": "pilot", "source_index": index, "ids": set(cell.ID_NUMBER)})
    covered = set().union(*(source["ids"] for source in sources)) if sources else set()
    if candidate_campaigns[len(base):]:
        cost = channels.get("push", {}).get("cost_per_contact")
        if not _finite(cost) or cost != 0:
            errors.append("suffix requires finite zero-cost Push")
    for index, campaign in enumerate(candidate_campaigns[len(base):], start=len(base)):
        if campaign["channel"] != "push":
            errors.append(f"suffix campaign #{index + 1} must use free Push")
        audience = _audience(profile, campaign)
        proof = {"campaign_index": index, "campaign_name": campaign.get("campaign_name"),
                 "contacts": len(audience), "kind": "composite_covered_push", "portions": []}
        proofs.append(proof)
        for (current, segment), part in audience.groupby(["current_tariff", "arpu_segment"], dropna=False, observed=True):
            ids = set(part.ID_NUMBER)
            used = [{"kind": source["kind"], "source_index": source["source_index"],
                     "matched_contacts": len(ids & source["ids"])}
                    for source in sources if ids & source["ids"]]
            target_index = target_evidence.get((current, segment, campaign["target_tariff"]))
            proof["portions"].append({"current_tariff": current, "arpu_segment": segment,
                                     "contacts": len(ids), "covered_contacts": len(ids & covered),
                                     "target_pilot_index": target_index, "coverage_sources": used})
            if ids - covered:
                errors.append(f"suffix campaign #{index + 1} includes {len(ids - covered)} unproven customers")
            if target_index is None:
                errors.append(f"suffix campaign #{index + 1} target lacks finite pilot evidence in {current}/{segment}")
    result["valid"] = not errors
    return result
