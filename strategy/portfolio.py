"""Deterministic disjoint campaign packing under all three final limits.

Searches several budget/contact shadow prices and repairs the best portfolio.
This is a bounded heuristic, not a claim of globally optimal allocation.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class Option:
    campaign: dict
    mask: int
    contacts: int
    cost: float
    gain: float
    mean_gain: float
    mean_ratio: float
    ratio_std: float


def audience_parts(cell):
    """Whole cell and valid data/call slices, retaining every matched row."""
    seen = set()
    for data, calls in product((None, "NON_USER", "LITE", "HEAVY"), (None, "LOW", "MEDIUM", "HIGH")):
        part = cell
        if data is not None:
            part = part[part.data_segment == data]
        if calls is not None:
            part = part[part.call_segment == calls]
        if not 0 < len(part) <= 5000:
            continue
        indices = tuple(int(i) for i in part.index)
        if indices in seen:
            continue
        seen.add(indices)
        mask = 0
        for index in indices:
            mask |= 1 << index
        yield part, data, calls, mask


def make_options(profile, beliefs, channels, reference_multiplier, risk_weight=0.85):
    """Only a PILOTED target can enter the normal final portfolio."""
    winners = {}
    for belief in beliefs:
        if not belief.pilots:
            continue
        mean, std = belief.posterior()
        conservative = mean - risk_weight * std
        old = winners.get(belief.key)
        rank = (conservative, mean, belief.target)
        if old is None or rank > old[0]:
            winners[belief.key] = (rank, belief, mean, std)

    options = []
    for key in sorted(winners):
        _, belief, mean, std = winners[key]
        cell = profile[(profile.current_tariff == key[0]) & (profile.arpu_segment == key[1])]
        conservative = mean - risk_weight * std
        pilot_effects = []
        for sampled in beliefs:
            if sampled.key == key and sampled.pilots:
                pilot_mean, _ = sampled.posterior()
                for n, scale in sampled.samples:
                    pilot_effects.append((max(0.0, pilot_mean * scale), min(1.0, n / len(cell))))
        pilot_effects.sort(reverse=True)

        def incremental_ratio(ratio):
            # Individual pilot IDs are deliberately unknown. Under the public
            # uniform sampling rule, estimate prior coverage probabilistically.
            # Never count a pilot's gain again as incremental final-campaign gain.
            uncovered, sunk = 1.0, 0.0
            for pilot_ratio, probability in pilot_effects:
                sunk += uncovered * probability * min(max(0.0, ratio), pilot_ratio)
                uncovered *= 1.0 - probability
            return ratio - sunk

        for part, data, calls, mask in audience_parts(cell):
            mass = float(part["_value"].sum())
            for channel in sorted(channels):
                info = channels[channel]
                multiplier = float(info["conversion_multiplier"])
                cost = len(part) * float(info["cost_per_contact"])
                # For m>1 saturation is possible. Use the lower positive-effect
                # scale (m=1), and the larger negative-effect scale for downside.
                scale = (min(multiplier, 1.0) if conservative >= 0 else multiplier) / reference_multiplier
                ratio = conservative * scale
                options.append(Option(
                    campaign={
                        "filter_arpu_segment": key[1], "filter_data_segment": data,
                        "filter_call_segment": calls, "filter_current_tariff": key[0],
                        "target_tariff": belief.target, "channel": channel,
                    },
                    mask=mask, contacts=len(part), cost=float(cost),
                    gain=float(mass * incremental_ratio(ratio) - cost),
                    mean_gain=float(mass * incremental_ratio(mean * min(multiplier, 1.0) / reference_multiplier) - cost),
                    mean_ratio=float(mean * min(multiplier, 1.0) / reference_multiplier),
                    ratio_std=float(std * scale),
                ))
    return options


def _greedy(options, budget, contacts, money_shadow, contact_shadow, fixed=()):
    chosen = list(fixed)
    occupied = 0
    for option in chosen:
        occupied |= option.mask
        budget -= option.cost
        contacts -= option.contacts
    order = sorted(range(len(options)), key=lambda i: (
        -(options[i].gain - money_shadow * options[i].cost - contact_shadow * options[i].contacts), i
    ))
    # Second pass fills feasible unused capacity after shadow-price selection.
    orders = [order, sorted(range(len(options)), key=lambda i: (-options[i].gain, i))]
    for pass_index, indices in enumerate(orders):
        for i in indices:
            option = options[i]
            if len(chosen) >= 10:
                break
            if option.gain <= 0 or option.mask & occupied:
                continue
            if option.cost > budget + 1e-8 or option.contacts > contacts:
                continue
            if pass_index == 0 and option.gain - money_shadow * option.cost - contact_shadow * option.contacts <= 0:
                continue
            chosen.append(option)
            occupied |= option.mask
            budget -= option.cost
            contacts -= option.contacts
    return chosen


def choose_portfolio(options, budget, contacts):
    feasible = [o for o in options if o.cost <= budget + 1e-8 and o.contacts <= contacts]
    positive = [o for o in feasible if o.gain > 0]
    if not positive:
        # The case requires >=1 campaign even if all evidence is unfavourable.
        # Select the least conservative TOTAL loss, not a large unproven offer.
        return [max(feasible, key=lambda o: (o.gain, -o.contacts, -o.cost))] if feasible else []
    paid = [o.gain / o.cost for o in positive if o.cost > 0]
    density = [o.gain / o.contacts for o in positive]
    money_grid = [0.0] + (list(np.quantile(paid, [0.1, 0.3, 0.5, 0.7, 0.9])) if paid else [])
    contact_grid = [0.0] + list(np.quantile(density, [0.2, 0.5, 0.8]))
    best, best_score = [], -float("inf")
    best_prices = (0.0, 0.0)
    for money_shadow, contact_shadow in product(money_grid, contact_grid):
        plan = _greedy(positive, budget, contacts, money_shadow, contact_shadow)
        score = sum(o.gain for o in plan)
        if score > best_score + 1e-8:
            best, best_score, best_prices = plan, score, (money_shadow, contact_shadow)
    # Replace one coarse campaign by finer disjoint pieces where beneficial.
    for _ in range(2):
        improved = False
        for remove in range(len(best)):
            fixed = best[:remove] + best[remove + 1:]
            for money_shadow, contact_shadow in (best_prices, (0.0, 0.0)):
                plan = _greedy(positive, budget, contacts, money_shadow, contact_shadow, fixed)
                score = sum(o.gain for o in plan)
                if score > best_score + 1e-8:
                    best, best_score = plan, score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return sorted(best, key=lambda o: (-o.gain, o.campaign["filter_current_tariff"], o.campaign["channel"]))
