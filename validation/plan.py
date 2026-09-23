"""Preflight validation for final campaign plans."""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

MAX_CAMPAIGNS = 10
MAX_CUSTOMERS_PER_CAMPAIGN = 5000
FILTERS = {
    'filter_arpu_segment': ('arpu_segment', {'LOW', 'MID', 'HIGH'}),
    'filter_data_segment': ('data_segment', {'NON_USER', 'LITE', 'HEAVY'}),
    'filter_call_segment': ('call_segment', {'LOW', 'MEDIUM', 'HIGH'}),
}


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
        return not (isinstance(missing, (bool, np.bool_)) and missing)
    except (TypeError, ValueError):
        return True


def _resource(value: Any, name: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
        errors.append(f'{name} must be a finite non-negative number')
        return None
    return float(value)


def _known_tariffs(tariffs: pd.DataFrame, errors: list[str]) -> set[Any]:
    if not isinstance(tariffs, pd.DataFrame) or 'tariff_plan_code' not in tariffs.columns:
        errors.append("tariffs must contain column 'tariff_plan_code'")
        return set()
    return set(tariffs['tariff_plan_code'].dropna())


def validate_plan(campaigns, profile, tariffs, channels, remaining_budget, remaining_contacts) -> dict:
    """Validate a final plan against public constraints without mutating inputs."""
    errors: list[str] = []
    warnings: list[str] = []
    details: list[dict] = []
    budget = _resource(remaining_budget, 'remaining_budget', errors)
    contacts = _resource(remaining_contacts, 'remaining_contacts', errors)
    known_tariffs = _known_tariffs(tariffs, errors)

    if not isinstance(profile, pd.DataFrame):
        errors.append('profile must be a pandas DataFrame')
        profile = pd.DataFrame()
    if not isinstance(channels, dict):
        errors.append('channels must be a dict')
        channels = {}
    if campaigns is None:
        campaigns = []
    if not isinstance(campaigns, (list, tuple)):
        errors.append('campaigns must be a list')
        campaigns = []

    if not campaigns:
        errors.append('plan must contain at least one campaign')
    if len(campaigns) > MAX_CAMPAIGNS:
        errors.append(f'plan has {len(campaigns)} campaigns; maximum is {MAX_CAMPAIGNS}')

    total_contacts = 0
    total_cost = 0.0
    selected_ids: list[set[Any]] = []
    has_ids = 'ID_NUMBER' in profile.columns
    if not has_ids:
        errors.append("profile must contain column 'ID_NUMBER'")

    for index, original in enumerate(campaigns):
        prefix = f'campaign #{index + 1}'
        if not isinstance(original, dict):
            errors.append(f'{prefix} must be a dict')
            details.append({'index': index, 'segment_size': 0, 'cost': 0.0})
            selected_ids.append(set())
            continue
        campaign = deepcopy(original)
        segment = profile
        usable = True
        target = campaign.get('target_tariff')
        channel = campaign.get('channel')
        if target not in known_tariffs:
            errors.append(f'{prefix} has unknown target_tariff {target!r}')
            usable = False
        if channel not in channels:
            errors.append(f'{prefix} has unknown channel {channel!r}')
            usable = False

        for field, (column, allowed) in FILTERS.items():
            value = campaign.get(field)
            if _present(value):
                if value not in allowed:
                    errors.append(f'{prefix} has invalid {field} {value!r}')
                    usable = False
                elif column not in profile.columns:
                    errors.append(f"profile must contain column {column!r}")
                    usable = False
                else:
                    segment = segment[segment[column] == value]

        tariff_filter = campaign.get('filter_current_tariff')
        if _present(tariff_filter):
            wanted = [value.strip() for value in str(tariff_filter).split(';') if value.strip()]
            unknown = sorted(set(wanted) - known_tariffs)
            if not wanted:
                errors.append(f'{prefix} has an empty filter_current_tariff')
                usable = False
            if unknown:
                errors.append(f'{prefix} has unknown filter_current_tariff values {unknown}')
                usable = False
            if 'current_tariff' not in profile.columns:
                errors.append("profile must contain column 'current_tariff'")
                usable = False
            elif wanted:
                segment = segment[segment['current_tariff'].isin(wanted)]

        size = int(len(segment)) if usable else 0
        if usable and size == 0:
            errors.append(f'{prefix} selects an empty segment')
        if size > MAX_CUSTOMERS_PER_CAMPAIGN:
            errors.append(f'{prefix} selects {size} customers; maximum is {MAX_CUSTOMERS_PER_CAMPAIGN}')

        unit_cost = 0.0
        if channel in channels:
            raw_cost = channels[channel].get('cost_per_contact') if isinstance(channels[channel], dict) else None
            if isinstance(raw_cost, bool) or not isinstance(raw_cost, Real) or not math.isfinite(float(raw_cost)) or raw_cost < 0:
                errors.append(f'{prefix} channel {channel!r} has invalid cost_per_contact')
                usable = False
                size = 0
            else:
                unit_cost = float(raw_cost)
        cost = size * unit_cost
        total_contacts += size
        total_cost += cost
        ids = set(segment['ID_NUMBER']) if usable and has_ids else set()
        overlap = len(ids & set().union(*selected_ids)) if selected_ids else 0
        if overlap:
            warnings.append(f'{prefix} overlaps previous campaigns by {overlap} customers')
        selected_ids.append(ids)
        details.append({
            'index': index,
            'campaign_name': campaign.get('campaign_name', f'campaign_{index + 1}'),
            'segment_size': size,
            'cost_per_contact': unit_cost,
            'cost': cost,
            'overlap_with_previous': overlap,
        })

    all_ids = set().union(*selected_ids) if selected_ids else set()
    unique_customers = len(all_ids)
    if total_contacts > unique_customers:
        warnings.append(f'{total_contacts - unique_customers} contacts are duplicates across campaigns')
    if contacts is not None and total_contacts > contacts:
        errors.append(f'plan requires {total_contacts} contacts; only {contacts:g} remain')
    if budget is not None and total_cost > budget:
        errors.append(f'plan costs {total_cost:g}; only {budget:g} remain')

    return {
        'valid': not errors,
        'errors': errors,
        'warnings': warnings,
        'total_contacts': total_contacts,
        'total_cost': total_cost,
        'unique_customers': unique_customers,
        'campaigns': details,
    }
