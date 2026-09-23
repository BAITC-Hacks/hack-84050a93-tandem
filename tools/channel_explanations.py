"""Reconstruct channel alternatives from a validated saved decision trace.

Each alternative changes one campaign's channel, keeping its audience, target,
pilots, and all other campaigns fixed. The formulas mirror make_options in
strategy/portfolio.py; they estimate incremental final-campaign gain after an
approximate allowance for prior pilot coverage. They are not measured profit,
causal estimates, a rerun of the agent, or a proof of global optimality.

Only public profile data and recorded posterior summaries are read. The caller
must first validate the trace and its source manifest. This module additionally
checks the profile hash and reproduces every selected campaign's saved metrics;
any disagreement prevents presentation of alternatives. No customer IDs leave
this helper and no agent/environment module is imported.
"""

import csv
import hashlib
import io
import math
from pathlib import Path


# Public case constants, matching mock_environment.CHANNELS. The selected-row
# consistency check fails closed if reconstruction disagrees with the trace.
CHANNELS = {
    "push": (0.0, 0.50),
    "sms": (4.0, 0.65),
    "digital_ads": (22.0, 0.85),
    "call": (160.0, 1.20),
}
RISK_WEIGHT = 1.65
TOTAL_BUDGET = 100_000.0
TOTAL_CONTACTS = 15_000
PROFILE_FIELDS = (
    "ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu",
    "data_segment", "call_segment",
)
# Canonical sha256-text-lf-v1 hashes of the implementation this adapter mirrors.
# Updating an unselected channel or the coverage formula must trigger an explicit
# adapter review even if the selected-row consistency check still happens to pass.
SUPPORTED_SOURCES = {
    "agent.py": "0ee4d0a4d448b40229887da4983bbf67a4c00a6dc8e907ed28f80b63adc098f2",
    "strategy/portfolio.py": "417b3b27da080f22e92fb16d964a8849592a9c464936763672cfd922f2e6bf40",
    "mock_environment.py": "e1e4a480126d2896e89a868194b3ed7f7ee48d7c5fc2c8a2491970d8e231f506",
}


def _require(condition, message):
    if not condition:
        raise ValueError("Channel explanation: " + message)


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value),
             f"{label} must be a finite number")
    return float(value)


def _same(actual, recorded, label):
    recorded = _number(recorded, label)
    _require(math.isfinite(actual) and math.isclose(
        actual, recorded, rel_tol=1e-10, abs_tol=1e-6),
        f"{label} does not reproduce the saved decision")


def _profile_rows(trace, path):
    content = Path(path).read_bytes().replace(b"\r\n", b"\n")
    expected = trace.get("input_sha256", {}).get("customer_profile.csv")
    _require(hashlib.sha256(content).hexdigest() == expected,
             "customer_profile.csv hash does not match the saved trace")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    _require(reader.fieldnames is not None and set(PROFILE_FIELDS) <= set(reader.fieldnames),
             "customer profile is missing required fields")
    _require(len(reader.fieldnames) == len(set(reader.fieldnames)),
             "customer profile has duplicate columns")
    rows, ids = [], set()
    for row in reader:
        _require(None not in row and all(row.get(field) is not None for field in PROFILE_FIELDS),
                 "customer profile has a malformed row")
        identifier = row["ID_NUMBER"]
        _require(identifier and identifier not in ids,
                 "customer profile IDs must be present and unique")
        ids.add(identifier)
        # Agent.act coerces invalid/non-finite values to zero and clips negatives.
        try:
            value = float(row["predicted_arpu"])
        except (TypeError, ValueError):
            value = 0.0
        value = max(value, 0.0) if math.isfinite(value) else 0.0
        rows.append({field: row[field] for field in PROFILE_FIELDS if field != "predicted_arpu"})
        rows[-1]["_value"] = value
    return rows


def build_channel_explanations(trace, profile_path):
    """Return campaign_name -> four replacement rows, or raise ValueError.

    ``replacement_feasible`` checks the unchanged portfolio's total contacts and
    the new total spend, including pilots. Deltas compare this one campaign with
    its selected channel; they do not include new pilots or reoptimized campaigns.
    """
    _require(trace.get("version") == "adaptive-portfolio-v1.2",
             "unsupported strategy version")
    _require(trace.get("hash_format") == "sha256-text-lf-v1",
             "unsupported source hash format")
    manifest = trace.get("source_sha256", {})
    for source, expected in SUPPORTED_SOURCES.items():
        _require(manifest.get(source) == expected,
                 f"unsupported implementation of {source}; review the explanation adapter before rendering")
    reference = trace.get("reference_channel")
    _require(reference in CHANNELS and CHANNELS[reference][1] <= 1.0,
             "unsupported reference channel")
    reference_multiplier = CHANNELS[reference][1]
    profile = _profile_rows(trace, profile_path)
    cells = {}
    for row in profile:
        cells.setdefault((row["current_tariff"], row["arpu_segment"]), []).append(row)

    # Every sample uses the offer's FINAL posterior, including samples from
    # other targets in this cell. This mirrors the production coverage estimate.
    latest, samples = {}, {}
    pilot_cost, pilot_contacts = 0.0, 0
    for index, pilot in enumerate(trace["pilots"], start=1):
        channel = pilot["channel"]
        _require(channel in CHANNELS, f"unknown channel for pilot {index}")
        n = pilot["n_customers"]
        _require(type(n) is int and 10 <= n <= 200,
                 f"invalid sample size for pilot {index}")
        key = (pilot["current_tariff"], pilot["arpu_segment"], pilot["target_tariff"])
        mean = _number(pilot["posterior_reference_mean"], f"pilot {index} mean")
        std = _number(pilot["posterior_reference_std"], f"pilot {index} std")
        _require(std >= 0, f"negative uncertainty for pilot {index}")
        latest[key] = mean, std
        samples.setdefault(key, []).append((n, CHANNELS[channel][1] / reference_multiplier))
        pilot_cost += n * CHANNELS[channel][0]
        pilot_contacts += n

    final = trace["final"]
    campaigns = final["campaigns"]
    _require(1 <= len(campaigns) <= 10, "invalid final campaign count")
    final_cost = sum(_number(c["cost"], "campaign cost") for c in campaigns)
    final_contacts = sum(c["contacts"] for c in campaigns)
    total_cost = pilot_cost + final_cost
    total_contacts = pilot_contacts + final_contacts
    _same(final_cost, final["cost"], "final cost")
    _same(final_contacts, final["contacts"], "final contacts")
    _same(total_cost, final["total_cost_including_pilots"], "total cost")
    _same(total_contacts, final["total_contacts_including_pilots"], "total contacts")
    _same(TOTAL_BUDGET - pilot_cost, final["remaining_budget_after_pilots"], "remaining budget")
    _same(TOTAL_CONTACTS - pilot_contacts, final["remaining_contacts_after_pilots"], "remaining contacts")
    _require(total_contacts <= TOTAL_CONTACTS and total_cost <= TOTAL_BUDGET + 1e-8,
             "selected portfolio exceeds the public limits")

    result, occupied = {}, set()
    for campaign in campaigns:
        name = campaign["campaign_name"]
        _require(name not in result, "duplicate campaign name")
        cell_key = campaign["filter_current_tariff"], campaign["filter_arpu_segment"]
        cell = cells.get(cell_key, [])
        _require(bool(cell), f"{name} has no audience cell")
        offer_key = (*cell_key, campaign["target_tariff"])
        _require(offer_key in latest, f"{name} has no direct pilot evidence")
        mean, std = latest[offer_key]
        conservative = mean - RISK_WEIGHT * std
        selected_channel = campaign["channel"]
        _require(selected_channel in CHANNELS, f"{name} has an unknown selected channel")

        part = [row for row in cell if all(
            campaign.get("filter_" + field) is None
            or row[field] == campaign["filter_" + field]
            for field in ("data_segment", "call_segment")
        )]
        contacts = len(part)
        _require(type(campaign["contacts"]) is int and contacts == campaign["contacts"]
                 and 0 < contacts <= 5000, f"{name} audience count does not reproduce")
        audience_ids = {row["ID_NUMBER"] for row in part}
        _require(not audience_ids & occupied, "final campaign audiences overlap")
        occupied.update(audience_ids)
        mass = math.fsum(row["_value"] for row in part)
        pilot_effects = []
        for key, observations in samples.items():
            if key[:2] == cell_key:
                pilot_mean = latest[key][0]
                for n, scale in observations:
                    pilot_effects.append((max(0.0, pilot_mean * scale), min(1.0, n / len(cell))))
        pilot_effects.sort(reverse=True)

        def incremental_ratio(ratio):
            uncovered, sunk = 1.0, 0.0
            for pilot_ratio, probability in pilot_effects:
                sunk += uncovered * probability * min(max(0.0, ratio), pilot_ratio)
                uncovered *= 1.0 - probability
            return ratio - sunk

        alternatives = []
        for channel in CHANNELS:
            unit_cost, multiplier = CHANNELS[channel]
            cost = contacts * unit_cost
            scale = (min(multiplier, 1.0) if conservative >= 0 else multiplier) / reference_multiplier
            mean_ratio = mean * min(multiplier, 1.0) / reference_multiplier
            risk_gain = mass * incremental_ratio(conservative * scale) - cost
            mean_gain = mass * incremental_ratio(mean_ratio) - cost
            replacement_total = total_cost - campaign["cost"] + cost
            alternatives.append({
                "channel": channel,
                "contacts": contacts,
                "cost": cost,
                "cost_delta": cost - campaign["cost"],
                "estimated_mean_gain": mean_gain,
                "risk_adjusted_gain": risk_gain,
                "reference_scaled_mean": mean_ratio,
                "reference_scaled_std": std * scale,
                "delta_estimated_mean_gain": mean_gain - campaign["estimated_mean_gain"],
                "delta_risk_adjusted_gain": risk_gain - campaign["risk_adjusted_gain"],
                "total_cost_including_pilots": replacement_total,
                "replacement_feasible": replacement_total <= TOTAL_BUDGET + 1e-8
                                        and total_contacts <= TOTAL_CONTACTS,
                "selected": channel == selected_channel,
            })
        selected = next(row for row in alternatives if row["selected"])
        for metric in ("cost", "estimated_mean_gain", "risk_adjusted_gain",
                       "reference_scaled_mean", "reference_scaled_std"):
            _same(selected[metric], campaign[metric], f"{name}.{metric}")
        # Eliminate meaningless rounding deltas on the selected row after its
        # independently reconstructed metrics have passed the consistency check.
        selected["cost_delta"] = 0.0
        selected["delta_estimated_mean_gain"] = 0.0
        selected["delta_risk_adjusted_gain"] = 0.0
        result[name] = alternatives
    return result
