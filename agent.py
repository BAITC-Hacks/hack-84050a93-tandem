"""tandem: adaptive pilot allocation and disjoint revenue-aware campaigns.

Public entry point: Agent().act(env). No network, hidden-model access, mutable
module state, random tie breaks, or dependencies outside numpy and pandas.
The optional last_trace attribute explains the most recent decision.
"""

from copy import deepcopy
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.beliefs import ARPU_SEGMENTS, NOISE_STD, HistoricalPrior, positive_normal
from strategy.portfolio import choose_portfolio, make_options
from validation.plan import validate_plan
from validation.overlay import validate_overlay_extension
from strategy.full_coverage_push import add_full_coverage_push


class BaselineAgent:
    def __init__(self):
        self.last_trace = {}

    def plan_portfolio(self, options, budget, contacts):
        """Separate planning from exploration for controlled algorithm comparisons."""
        return choose_portfolio(options, budget, contacts)

    def act(self, env):
        self.last_trace = {"pilots": [], "warnings": [], "version": "adaptive-portfolio-v1.2"}
        profile = env.customer_profile.copy().reset_index(drop=True)
        required = {"ID_NUMBER", "current_tariff", "arpu_segment", "predicted_arpu", "data_segment", "call_segment"}
        if required - set(profile.columns):
            raise ValueError(f"Missing public profile fields: {sorted(required - set(profile.columns))}")
        if profile.ID_NUMBER.duplicated().any():
            raise ValueError("Audience IDs must be unique")
        value = pd.to_numeric(profile.predicted_arpu, errors="coerce")
        profile["_value"] = value.where(np.isfinite(value), 0.0).clip(lower=0).fillna(0.0)
        if not np.isfinite(value).all() or (value < 0).any():
            self.last_trace["warnings"].append("Invalid baseline values have zero planning value; rows still consume contacts")

        channels = {
            name: dict(info) for name, info in sorted(env.channels.items())
            if isfinite(float(info["conversion_multiplier"])) and 0 < float(info["conversion_multiplier"]) <= 1.2
            and isfinite(float(info["cost_per_contact"])) and float(info["cost_per_contact"]) >= 0
        }
        if not channels:
            raise ValueError("No valid public communication channels")
        reference_channels = [c for c in channels if channels[c]["conversion_multiplier"] <= 1]
        if not reference_channels:
            raise ValueError("Need an unsaturated reference channel with multiplier <= 1")
        reference_channel = "sms" if "sms" in reference_channels else min(reference_channels, key=lambda c: (channels[c]["cost_per_contact"], c))
        reference_multiplier = float(channels[reference_channel]["conversion_multiplier"])
        initial_budget = float(env.remaining_budget)
        initial_contacts = int(env.remaining_contacts)
        if not isfinite(initial_budget) or initial_budget < 0 or initial_contacts < 1:
            raise ValueError("Invalid available budget or contact limit")

        tariffs = sorted(str(t) for t in env.tariffs.tariff_plan_code.dropna().unique())
        prior = HistoricalPrior(env.tariffs, Path(__file__).resolve().parent / "data" / "change_tariff.csv", reference_multiplier)
        self.last_trace["prior"] = prior.status
        self.last_trace["reference_channel"] = reference_channel
        cells = {}
        for key, cell in profile.groupby(["current_tariff", "arpu_segment"], sort=True, observed=True):
            if key[0] in tariffs and key[1] in ARPU_SEGMENTS and len(cell) >= 10 and float(cell["_value"].sum()) > 0:
                cells[key] = {"n": len(cell), "mass": float(cell["_value"].sum())}
        if not cells:
            raise ValueError("No nonempty valid audience cell eligible for a pilot")
        beliefs = [prior.belief(current, segment, target)
                   for current, segment in sorted(cells) for target in tariffs if target != current]
        if not beliefs:
            raise ValueError("No alternative tariff available")

        # Economically identical offers without historical evidence provide no
        # useful initial distinction. Keep them available only when observed
        # historical transitions actually support investigating the target.
        feature_columns = [c for c in ("price_tariff", "Data_in_PKG", "Min_another_operator_in_PKG",
                           "Min_another_operator_and_city_in_PKG") if c in env.tariffs]
        if feature_columns:
            descriptors = {str(row["tariff_plan_code"]): tuple(row[c] for c in feature_columns)
                           for row in env.tariffs.to_dict("records")}
            distinctive = [b for b in beliefs if b.history_n > 0 or descriptors[b.current] != descriptors[b.target]]
            beliefs = distinctive or beliefs

        pilot_limit = min(20, max(0, int(env.pilots_left)))
        initial_segments = [s for s in ("HIGH", "MID", "LOW") if any(key[1] == s for key in cells)]
        reserve_contacts = min(2500, max(1, initial_contacts // 3))
        pilot_budget = min(initial_budget * 0.18, 18000.0)
        for step in range(pilot_limit):
            if env.pilots_left <= 0 or env.remaining_contacts <= reserve_contacts:
                break
            spent = initial_budget - float(env.remaining_budget)
            pilot_channel = reference_channel
            unit_cost = float(channels[pilot_channel]["cost_per_contact"])
            if unit_cost > 0 and min(pilot_budget - spent, env.remaining_budget) < 10 * unit_cost:
                free = [c for c in channels if channels[c]["cost_per_contact"] == 0 and channels[c]["conversion_multiplier"] <= 1]
                if not free:
                    break
                pilot_channel = max(free, key=lambda c: (channels[c]["conversion_multiplier"], c))
                unit_cost = 0.0
            multiplier = float(channels[pilot_channel]["conversion_multiplier"])
            available = int(env.remaining_contacts) - reserve_contacts
            if unit_cost > 0:
                available = min(available, int(min(env.remaining_budget, pilot_budget - spent) // unit_cost))
            if available < 10:
                break

            best_by_cell = {}
            probes_by_cell = {}
            residuals = {}
            segment_evidence = {s: [0, 0] for s in initial_segments}
            for belief in beliefs:
                if belief.pilots:
                    mean, std = belief.posterior()
                    best_by_cell[belief.key] = max(best_by_cell.get(belief.key, 0.0), mean - 0.7 * std)
                    probes_by_cell[belief.key] = probes_by_cell.get(belief.key, 0) + belief.pilots
                    residuals.setdefault(belief.key, []).append(mean - belief.prior_mean)
                    segment_evidence[belief.segment][0] += 1
                    segment_evidence[belief.segment][1] += int(mean > 1.65 * std)

            # Later measurements focus on promising winners to limit selection
            # bias from choosing the largest among many noisy first observations.
            confirm = []
            if step >= max(1, pilot_limit - 5):
                for belief in beliefs:
                    if 0 < belief.pilots < 3:
                        mean, std = belief.posterior()
                        if mean - 0.7 * std > 0:
                            confirm.append(belief)
            coverage_segment = None
            if step < min(2 * len(initial_segments), max(1, pilot_limit // 2)):
                coverage_segment = initial_segments[step % len(initial_segments)]
            pool = confirm or beliefs
            if coverage_segment is not None:
                pool = [b for b in beliefs if b.segment == coverage_segment and not b.pilots]
                # Second stratified round covers another current tariff when possible.
                fresh_cells = [b for b in pool if probes_by_cell.get(b.key, 0) == 0]
                if fresh_cells:
                    pool = fresh_cells
                if not pool:
                    # A stratum with only one available offer can exhaust its
                    # warm-up candidates. Continue learning in the other strata.
                    coverage_segment = None
                    pool = confirm or beliefs
            best_action, best_value = None, -float("inf")
            for belief in pool:
                cell = cells[belief.key]
                shift = 0.0
                if not belief.pilots and len(residuals.get(belief.key, ())) >= 2:
                    shift = float(np.clip(np.median(residuals[belief.key]) * 0.4, -0.12, 0.12))
                mean, std = belief.posterior(shift)
                threshold = best_by_cell.get(belief.key, 0.0)
                for requested in (100, 200):
                    n = min(requested, cell["n"], available)
                    if n < 10:
                        continue
                    obs_var = NOISE_STD**2 / n * (reference_multiplier / multiplier)**2
                    next_var = 1.0 / (1.0 / std**2 + 1.0 / obs_var)
                    future_mean_std = sqrt(max(std**2 - next_var, 0.0))
                    # An unpiloted offer is not deployable: its current decision
                    # value is zero, even when its historical prior is promising.
                    current_value = max(mean - threshold, 0.0) if belief.pilots else 0.0
                    knowledge_gain = max(0.0, positive_normal(mean - threshold, future_mean_std) - current_value)
                    if coverage_segment is not None:
                        # In stratified warm-up, favour historical upside over
                        # unsupported variance; later rounds use full information gain.
                        knowledge_gain = max(mean - threshold, 0.0) + 0.25 * knowledge_gain
                    confirmation_gain = max(0.0, std - sqrt(next_var)) * (0.65 if confirm else 0.25)
                    if mean < threshold or not belief.pilots:
                        confirmation_gain = 0.0
                    breadth = 1.0 / (1.0 + 0.45 * probes_by_cell.get(belief.key, 0))
                    attempts, promising = segment_evidence[belief.segment]
                    # Smoothed stratum-level evidence prevents endless probing
                    # of high-value but consistently unresponsive customers.
                    # A floor preserves exploration after unlucky early pilots.
                    stratum_weight = max(0.15, (promising + 1.0) / (attempts + 2.0))
                    if coverage_segment is not None or confirm:
                        stratum_weight = 1.0
                    value_of_info = cell["mass"] * (knowledge_gain * breadth * stratum_weight + confirmation_gain)
                    # Contacts used for learning cannot also be deployed. This
                    # opportunity cost allows early stopping when learning is weak.
                    deployment_cost = n * max(threshold, 0.0) * cell["mass"] / cell["n"]
                    pilot_value = n * cell["mass"] / cell["n"] * mean * multiplier / reference_multiplier
                    score = value_of_info + min(pilot_value, 0.0) - n * unit_cost - 0.35 * deployment_cost
                    if score > best_value:
                        best_value = score
                        best_action = belief, int(n), bool(confirm), float(value_of_info)
            if best_action is None or (best_value <= 0 and step > 0):
                break
            belief, n, is_confirmation, information_value = best_action
            result = env.run_pilot(
                target_tariff=belief.target, channel=pilot_channel, n_customers=n,
                filter_current_tariff=belief.current, filter_arpu_segment=belief.segment,
            )
            actual_n = int(result["n_customers"])
            observed = float(result["observed_lift_ratio"])
            if actual_n < 1 or not isfinite(observed):
                self.last_trace["warnings"].append("Non-finite or empty pilot response; no update from this observation")
                continue
            belief.observe(observed, actual_n, multiplier, reference_multiplier)
            mean, std = belief.posterior()
            self.last_trace["pilots"].append({
                "current_tariff": belief.current, "arpu_segment": belief.segment,
                "target_tariff": belief.target, "channel": pilot_channel,
                "n_customers": actual_n, "observed_lift_ratio": observed,
                "posterior_reference_mean": mean, "posterior_reference_std": std,
                "confirmation": is_confirmation, "estimated_information_value": information_value,
                "stratified_segment": coverage_segment,
            })

        options = make_options(profile, beliefs, channels, reference_multiplier, risk_weight=1.65)
        chosen = self.plan_portfolio(options, float(env.remaining_budget), int(env.remaining_contacts))
        if not chosen:
            raise RuntimeError("No feasible final campaign after pilots")
        campaigns = []
        for index, option in enumerate(chosen, start=1):
            campaigns.append({"campaign_name": f"tandem_{index:02d}", **option.campaign})
        preflight = validate_plan(campaigns, env.customer_profile, env.tariffs,
                                  env.channels, env.remaining_budget, env.remaining_contacts)
        self.last_trace["preflight"] = preflight
        if not preflight["valid"]:
            raise RuntimeError("Final plan rejected: " + "; ".join(preflight["errors"]))
        if preflight["unique_customers"] != preflight["total_contacts"]:
            raise RuntimeError("Final plan contains overlapping audiences")
        self.last_trace["final"] = {
            "campaigns": [{**campaign, "contacts": option.contacts, "cost": option.cost,
                           "estimated_mean_gain": option.mean_gain, "risk_adjusted_gain": option.gain,
                           "reference_scaled_mean": option.mean_ratio, "reference_scaled_std": option.ratio_std}
                          for campaign, option in zip(campaigns, chosen)],
            "contacts": sum(o.contacts for o in chosen), "cost": sum(o.cost for o in chosen),
            "remaining_contacts_after_pilots": int(env.remaining_contacts),
            "remaining_budget_after_pilots": float(env.remaining_budget),
            "total_contacts_including_pilots": initial_contacts - int(env.remaining_contacts) + preflight["total_contacts"],
            "total_cost_including_pilots": initial_budget - float(env.remaining_budget) + preflight["total_cost"],
            "fallback": not any(o.gain > 0 for o in chosen),
            "note": "Planning estimates, not measured judging profit; final audiences are disjoint.",
        }
        return campaigns


class Agent(BaselineAgent):
    def act(self, env):
        base_campaigns = super().act(env)
        base_trace = deepcopy(self.last_trace)
        pilot_contacts = sum(int(pilot["n_customers"]) for pilot in env.pilot_history)
        pilot_cost = sum(float(pilot["cost"]) for pilot in env.pilot_history)
        plan, preflight, details = add_full_coverage_push(
            base_campaigns, base_trace, env.customer_profile, env.tariffs, env.channels,
            env.remaining_budget, env.remaining_contacts, pilot_contacts)
        extension = validate_overlay_extension(
            base_campaigns, plan, base_trace.get("pilots", []),
            env.customer_profile, env.tariffs, env.channels,
            env.remaining_budget, env.remaining_contacts, pilot_contacts=pilot_contacts)
        if not extension["valid"]:
            raise RuntimeError("Invalid covered-audience extension: " + "; ".join(extension["errors"]))
        self.last_trace = {
            "version": "adaptive-portfolio-v1.3",
            "trace_schema": "nonadditive_full_coverage_overlay_v1",
            "base_trace": base_trace, "baseline_prefix": deepcopy(base_campaigns),
            "pilots": deepcopy(base_trace.get("pilots", [])),
            "reference_channel": base_trace.get("reference_channel"),
            "preflight": preflight, "overlay": details,
            "overlay_validation": extension,
            "warnings": deepcopy(base_trace.get("warnings", [])),
            "prior": base_trace.get("prior"),
            "final": {
                "campaigns": [{**deepcopy(campaign), "contacts": item["segment_size"],
                               "cost": item["cost"],
                               "role": "baseline" if index < len(base_campaigns) else "overlay"}
                              for index, (campaign, item) in enumerate(zip(plan, preflight["campaigns"]))],
                "contacts": preflight["total_contacts"], "cost": preflight["total_cost"],
                "unique_customers": preflight["unique_customers"],
                "fallback": base_trace["final"]["fallback"],
                "remaining_contacts_after_pilots": int(env.remaining_contacts),
                "remaining_budget_after_pilots": float(env.remaining_budget),
                "total_contacts_including_pilots": pilot_contacts + preflight["total_contacts"],
                "total_cost_including_pilots": pilot_cost + preflight["total_cost"],
                "note": "Each overlay repeats a final audience or one completely sampled pilot cell; ranking proxies are nonadditive, not judged profit.",
            },
        }
        return plan
