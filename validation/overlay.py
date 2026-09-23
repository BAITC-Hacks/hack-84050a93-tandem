"""Independent public-input validation of a covered-audience Push extension.

This API trusts ``pilots`` to be the decision trace emitted by tandem's current
Agent: each record represents one real pilot filtered ONLY by current_tariff
and arpu_segment, with its actual n_customers. Raw env.pilot_history lacks those
filters and is not accepted. This function cannot authenticate arbitrary caller
records. It never reads hidden pilot IDs and never estimates revenue.

Pass pilot_contacts from the full public env.pilot_history when the decision
trace omitted responses (for example non-finite observations). The default is
the sum of the supplied records. Pilot proof indices are one-based positions in
that supplied trace, not necessarily original environment pilot numbers.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd

from validation.plan import FILTERS, validate_plan


MAX_TOTAL_CONTACTS = 15000
FILTER_KEYS = ("filter_current_tariff", "filter_arpu_segment", "filter_data_segment", "filter_call_segment")


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def _missing(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return value is pd.NA


def _same_raw(left, right):
    """Preserve prefix keys and values; do not normalize its representation."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_raw(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_raw(a, b) for a, b in zip(left, right))
    if _missing(left) and _missing(right):
        return True
    try:
        result = left == right
        return isinstance(result, (bool, np.bool_)) and bool(result)
    except (TypeError, ValueError):
        return False


def _signature(campaign):
    result = []
    for key in FILTER_KEYS:
        value = campaign.get(key)
        if _missing(value):
            result.append(None)
        elif isinstance(value, str):
            result.append(value)
        else:
            return None
    return tuple(result)


def _preflight(campaigns, profile, tariffs, channels, budget, contacts):
    try:
        return validate_plan(campaigns, profile, tariffs, channels, budget, contacts)
    except (TypeError, ValueError, KeyError) as exc:
        return {"valid": False, "errors": [f"Invalid preflight input: {type(exc).__name__}: {exc}"],
                "warnings": [], "campaigns": [], "total_contacts": 0, "total_cost": 0.0,
                "unique_customers": 0}


def validate_overlay_extension(base_campaigns, full_campaigns, pilots, profile, tariffs,
                               channels, remaining_budget, remaining_contacts, *, pilot_contacts=None):
    """Check an exact baseline prefix plus free, already-covered Push audiences.

    Each suffix audience must match all filters of one baseline campaign, or
    exactly one whole current-tariff/ARPU cell covered by a SINGLE recorded
    pilot. Summing partial pilots never establishes complete coverage. Every
    suffix target separately needs a real pilot in that same cell.
    """
    errors = []
    proofs = []
    base_preflight = _preflight(base_campaigns, profile, tariffs, channels, remaining_budget, remaining_contacts)
    preflight = _preflight(full_campaigns, profile, tariffs, channels, remaining_budget, remaining_contacts)
    errors.extend(f"baseline: {error}" for error in base_preflight["errors"])
    errors.extend(f"full plan: {error}" for error in preflight["errors"])
    if base_preflight["total_contacts"] != base_preflight["unique_customers"]:
        errors.append("baseline final audiences must be disjoint")
    warnings = ([f"baseline: {warning}" for warning in base_preflight["warnings"]]
                + [f"full plan: {warning}" for warning in preflight["warnings"]])

    base_is_list = isinstance(base_campaigns, list)
    full_is_list = isinstance(full_campaigns, list)
    if not base_is_list or not full_is_list:
        errors.append("baseline and full campaigns must be raw lists")
    base = base_campaigns if base_is_list else []
    full = full_campaigns if full_is_list else []
    if len(full) < len(base) or not _same_raw(full[:len(base)], base):
        errors.append("full plan must begin with the exact unchanged baseline prefix")

    known = (set(tariffs["tariff_plan_code"].dropna())
             if isinstance(tariffs, pd.DataFrame) and "tariff_plan_code" in tariffs else set())
    profile_usable = (isinstance(profile, pd.DataFrame)
                      and {"ID_NUMBER", "current_tariff", "arpu_segment"}.issubset(profile.columns))
    if profile_usable:
        if profile.ID_NUMBER.isna().any() or profile.ID_NUMBER.duplicated().any():
            errors.append("profile ID_NUMBER values must be present and unique")
            profile_usable = False
    else:
        errors.append("profile requires ID_NUMBER, current_tariff and arpu_segment")

    if not isinstance(pilots, list):
        errors.append("pilots must be a list of trusted tandem Agent trace records")
        pilots = []
    if len(pilots) > 20:
        errors.append("recorded pilot count exceeds 20")
    recorded_contacts = 0
    records = []
    for index, pilot in enumerate(pilots, start=1):
        label = f"pilot record #{index}"
        if not isinstance(pilot, dict):
            errors.append(f"{label} must be a dict")
            continue
        count = pilot.get("n_customers")
        count_valid = _finite(count) and float(count).is_integer() and 0 < count <= 200
        if count_valid:
            recorded_contacts += int(count)
        else:
            errors.append(f"{label} actual n_customers must be an integer in 1..200")
        current, segment, target = (pilot.get(key) for key in ("current_tariff", "arpu_segment", "target_tariff"))
        fields_valid = (isinstance(current, str) and current in known
                        and isinstance(segment, str) and segment in FILTERS["filter_arpu_segment"][1]
                        and isinstance(target, str) and target in known
                        and isinstance(pilot.get("channel"), str) and isinstance(channels, dict)
                        and pilot["channel"] in channels)
        if not fields_valid:
            errors.append(f"{label} lacks valid tandem current_tariff/ARPU/target/channel fields")
        # A caller cannot label a filtered pilot as a complete whole-cell pilot.
        extra_filter = any(not _missing(pilot.get(key)) for key in (
            "filter_data_segment", "filter_call_segment", "data_segment", "call_segment", "explicit_ids"))
        mismatched_filter = any(key in pilot and not _missing(pilot[key]) and not _same_raw(pilot[key], expected)
                                for key, expected in (("filter_current_tariff", current), ("filter_arpu_segment", segment)))
        if extra_filter or mismatched_filter:
            errors.append(f"{label} is not an unfiltered whole-cell tandem pilot trace")
        if count_valid and fields_valid and not extra_filter and not mismatched_filter and profile_usable:
            size = int(((profile.current_tariff == current) & (profile.arpu_segment == segment)).sum())
            if count > size:
                errors.append(f"{label} actual size exceeds its full public cell")
                continue
            records.append({"index": index, "current": current, "segment": segment,
                            "target": target, "count": int(count), "size": size})

    effective_pilot_contacts = recorded_contacts
    if pilot_contacts is not None:
        if not _finite(pilot_contacts) or not float(pilot_contacts).is_integer() or pilot_contacts < recorded_contacts:
            errors.append("pilot_contacts must be an integer at least the recorded pilot contact sum")
        else:
            effective_pilot_contacts = int(pilot_contacts)
            if effective_pilot_contacts > recorded_contacts:
                warnings.append("pilot_contacts includes contacts absent from the recorded decision trace")
    total_contacts = effective_pilot_contacts + preflight["total_contacts"]
    if total_contacts > MAX_TOTAL_CONTACTS:
        errors.append(f"pilots plus full plan require {total_contacts} contacts; maximum is {MAX_TOTAL_CONTACTS}")

    suffix = full[len(base):]
    push = channels.get("push") if isinstance(channels, dict) else None
    if suffix and (not isinstance(push, dict) or not _finite(push.get("cost_per_contact"))
                   or float(push["cost_per_contact"]) != 0.0):
        errors.append("overlay Push must have finite cost_per_contact exactly zero")
    for full_index, campaign in enumerate(suffix, start=len(base)):
        label = f"overlay campaign #{full_index + 1}"
        detail = preflight["campaigns"][full_index] if full_index < len(preflight["campaigns"]) else {}
        proof = {"campaign_index": full_index, "campaign_name": None, "kind": None,
                 "source_index": None, "target_pilot_index": None,
                 "contacts": detail.get("segment_size", 0), "matched_size": None}
        proofs.append(proof)
        if not isinstance(campaign, dict):
            errors.append(f"{label} must be a dict")
            continue
        proof["campaign_name"] = campaign.get("campaign_name")
        if campaign.get("channel") != "push":
            errors.append(f"{label} must use push")
        signature = _signature(campaign)
        if signature is None:
            errors.append(f"{label} has unsupported audience filter values")
            continue
        current, segment, _, _ = signature
        matching = [record for record in records if record["current"] == current and record["segment"] == segment]
        target_pilots = [record for record in matching if record["target"] == campaign.get("target_tariff")]
        if not target_pilots:
            errors.append(f"{label} target was not piloted in the same current-tariff/ARPU cell")
        else:
            proof["target_pilot_index"] = target_pilots[0]["index"]
        for index, original in enumerate(base):
            if (base_preflight["valid"] and index < len(base_preflight["campaigns"])
                    and isinstance(original, dict) and signature == _signature(original)):
                proof.update(kind="baseline", source_index=index,
                             matched_size=base_preflight["campaigns"][index]["segment_size"])
                break
        if proof["kind"] is None and signature[2:] == (None, None):
            complete = [record for record in matching if record["count"] == record["size"] and record["size"] > 0]
            if complete:
                record = complete[0]
                proof.update(kind="pilot", source_index=record["index"], matched_size=record["size"])
        if proof["kind"] is None:
            errors.append(f"{label} audience lacks an exact baseline or single complete-pilot coverage proof")
        elif proof["contacts"] != proof["matched_size"]:
            errors.append(f"{label} audience size does not match its coverage proof")

    return {
        "valid": not errors, "errors": errors, "warnings": warnings,
        "base_preflight": base_preflight, "preflight": preflight, "proofs": proofs,
        "baseline_prefix_length": len(base), "recorded_pilot_contacts": recorded_contacts,
        "pilot_contacts": effective_pilot_contacts, "total_contacts_including_pilots": total_contacts,
        "assumption": "Pilots are authentic current tandem Agent whole-cell trace records; no hidden IDs are inspected.",
    }
