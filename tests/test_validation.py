from copy import deepcopy

import pandas as pd
import pytest

from validation.plan import validate_plan


@pytest.fixture
def profile():
    return pd.DataFrame({
        'ID_NUMBER': [1, 2, 3, 4, 5, 6],
        'current_tariff': ['tariff_1', 'tariff_1', 'tariff_2', 'tariff_2', 'tariff_3', None],
        'arpu_segment': ['LOW', 'LOW', 'MID', 'MID', 'HIGH', 'HIGH'],
        'data_segment': ['LITE', 'HEAVY', 'LITE', 'HEAVY', 'NON_USER', None],
        'call_segment': ['LOW', 'MEDIUM', 'LOW', 'HIGH', 'MEDIUM', 'HIGH'],
        'predicted_arpu': [100, 200, 300, 400, 500, 600],
    })


@pytest.fixture
def tariffs():
    return pd.DataFrame({'tariff_plan_code': ['tariff_1', 'tariff_2', 'tariff_3']})


@pytest.fixture
def channels():
    return {
        'push': {'cost_per_contact': 0, 'conversion_multiplier': .5},
        'sms': {'cost_per_contact': 4, 'conversion_multiplier': .65},
        'call': {'cost_per_contact': 160, 'conversion_multiplier': 1.2},
    }


def check(campaigns, profile, tariffs, channels, budget=100_000, contacts=15_000):
    return validate_plan(campaigns, profile, tariffs, channels, budget, contacts)


def campaign(**changes):
    value = {'campaign_name': 'test', 'filter_arpu_segment': 'LOW',
             'target_tariff': 'tariff_3', 'channel': 'sms'}
    value.update(changes)
    return value


def test_valid_campaign_and_free_push(profile, tariffs, channels):
    result = check([campaign(channel='push')], profile, tariffs, channels)
    assert result['valid']
    assert result['total_contacts'] == result['unique_customers'] == 2
    assert result['total_cost'] == 0
    assert result['campaigns'][0]['segment_size'] == 2


@pytest.mark.parametrize('field,value,message', [
    ('target_tariff', 'tariff_99', 'unknown target_tariff'),
    ('channel', 'email', 'unknown channel'),
    ('filter_arpu_segment', 'VIP', 'invalid filter_arpu_segment'),
    ('filter_data_segment', 'MEDIUM', 'invalid filter_data_segment'),
    ('filter_call_segment', 'MID', 'invalid filter_call_segment'),
    ('filter_current_tariff', 'tariff_99', 'unknown filter_current_tariff'),
])
def test_unknown_values_are_errors(profile, tariffs, channels, field, value, message):
    result = check([campaign(**{field: value})], profile, tariffs, channels)
    assert not result['valid']
    assert any(message in error for error in result['errors'])


def test_empty_and_too_many_campaigns(profile, tariffs, channels):
    empty = check([], profile, tariffs, channels)
    many = check([campaign()] * 11, profile, tariffs, channels)
    assert not empty['valid'] and 'at least one' in empty['errors'][0]
    assert not many['valid'] and any('maximum is 10' in error for error in many['errors'])


def test_empty_and_oversize_segments(tariffs, channels):
    small = pd.DataFrame({'ID_NUMBER': [1], 'current_tariff': ['tariff_1'],
                          'arpu_segment': ['LOW'], 'data_segment': ['LITE'], 'call_segment': ['LOW']})
    empty = check([campaign(filter_arpu_segment='HIGH')], small, tariffs, channels)
    large = pd.concat([small] * 5001, ignore_index=True).assign(ID_NUMBER=range(5001))
    oversize = check([campaign()], large, tariffs, channels)
    assert any('empty segment' in error for error in empty['errors'])
    assert any('maximum is 5000' in error for error in oversize['errors'])


def test_budget_contacts_and_overlap(profile, tariffs, channels):
    campaigns = [campaign(campaign_name='a'), campaign(campaign_name='b', channel='call')]
    result = check(campaigns, profile, tariffs, channels, budget=100, contacts=3)
    assert not result['valid']
    assert result['total_contacts'] == 4
    assert result['unique_customers'] == 2
    assert result['total_cost'] == 328
    assert any('contacts' in error for error in result['errors'])
    assert any('costs' in error for error in result['errors'])
    assert any('overlaps' in warning for warning in result['warnings'])


@pytest.mark.parametrize('budget,contacts', [(float('nan'), 10), (float('inf'), 10), ('100', 10), (100, -1)])
def test_invalid_resources(profile, tariffs, channels, budget, contacts):
    result = check([campaign()], profile, tariffs, channels, budget, contacts)
    assert not result['valid']
    assert any('finite non-negative' in error for error in result['errors'])


def test_inputs_are_not_mutated(profile, tariffs, channels):
    campaigns = [campaign(filter_current_tariff='tariff_1; tariff_2')]
    original_campaigns, original_channels = deepcopy(campaigns), deepcopy(channels)
    original_profile, original_tariffs = profile.copy(deep=True), tariffs.copy(deep=True)
    check(campaigns, profile, tariffs, channels)
    assert campaigns == original_campaigns
    assert channels == original_channels
    pd.testing.assert_frame_equal(profile, original_profile)
    pd.testing.assert_frame_equal(tariffs, original_tariffs)


@pytest.mark.parametrize('field,value', [
    ('target_tariff', []), ('channel', {}), ('filter_arpu_segment', ['LOW']),
    ('explicit_ids', [1, 2]), ('filter_typo', 'LOW'),
])
def test_malformed_and_unsupported_fields(profile, tariffs, channels, field, value):
    result = check([campaign(**{field: value})], profile, tariffs, channels)
    assert not result['valid']


def test_fractional_contacts_and_duplicate_ids(profile, tariffs, channels):
    assert not check([campaign()], profile, tariffs, channels, contacts=2.5)['valid']
    profile.loc[1, 'ID_NUMBER'] = profile.loc[0, 'ID_NUMBER']
    assert not check([campaign()], profile, tariffs, channels)['valid']
