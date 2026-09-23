"""Reproducible read-only audit of the organizer dataset.

    python tools/analyze_data.py

Writes reports/data_audit.json and reports/candidate_priors.csv.
Source CSV files are only read; every transformation happens in memory.
History in data/change_tariff.csv is a different, selected sample: numbers
derived from it are priors for pilots, not the judging audience model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    'customer_profile': 'customer_profile.csv',
    'arpu_monthly': 'data/arpu_monthly.csv',
    'change_tariff': 'data/change_tariff.csv',
    'dict_tariff': 'data/dict_tariff.csv',
    'traffic': 'data/traffic.csv',
    'tariff_dictionary': 'tariff_dictionary.csv',
}
KEYS = {
    'customer_profile': ['ID_NUMBER'],
    'arpu_monthly': ['ID_NUMBER', 'TIME_KEY'],
    'change_tariff': ['ID_NUMBER', 'TIME_KEY'],
    'dict_tariff': ['tariff_plan_code'],
    'traffic': ['ID_NUMBER', 'time_key'],
    'tariff_dictionary': ['tariff_plan_code'],
}
TIME_COLUMNS = {'arpu_monthly': 'TIME_KEY', 'change_tariff': 'TIME_KEY', 'traffic': 'time_key'}
SEGMENTS = {
    'arpu_segment': ['LOW', 'MID', 'HIGH'],
    'data_segment': ['NON_USER', 'LITE', 'HEAVY'],
    'call_segment': ['LOW', 'MEDIUM', 'HIGH'],
}
# Same ARPU bins and exclusions as the public mock uses to derive its stub effects.
ARPU_BINS = [-np.inf, 1000, 5000, np.inf]
MIN_PREV_ARPU = 100
LIFT_CLIP = (-1.0, 3.0)
MIN_RELIABLE_N = 30
MAX_CUSTOMERS_PER_CAMPAIGN = 5000
MISSING = '<missing>'


def clean(value):
    """Convert numpy/pandas scalars to JSON-safe values; NaN/inf become None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors='coerce')
    finite = s[np.isfinite(s)]
    return {
        'count': int(s.notna().sum()), 'nan': int(s.isna().sum()),
        'inf': int(np.isinf(s).sum()), 'negative': int((s < 0).sum()), 'zero': int((s == 0).sum()),
        'min': finite.min(), 'p01': finite.quantile(0.01), 'p50': finite.median(),
        'p99': finite.quantile(0.99), 'max': finite.max(), 'mean': finite.mean(), 'sum': finite.sum(),
    }


def file_summary(name: str, df: pd.DataFrame, path: Path) -> dict:
    key = KEYS[name]
    summary = {
        'path': FILES[name], 'sha256': sha256(path), 'rows': len(df), 'columns': len(df.columns),
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
        'missing': {c: int(n) for c, n in df.isna().sum().items() if n},
        'duplicate_rows': int(df.duplicated().sum()),
        'key': key, 'duplicate_keys': int(df.duplicated(key).sum()),
    }
    if name in TIME_COLUMNS:
        t = pd.to_datetime(df[TIME_COLUMNS[name]], errors='coerce')
        summary['time'] = {
            'column': TIME_COLUMNS[name], 'unparsed': int(t.isna().sum()),
            'min': str(t.min().date()), 'max': str(t.max().date()),
            'rows_per_period': {str(k.date()): int(v) for k, v in t.value_counts().sort_index().items()},
        }
    if 'ID_NUMBER' in df.columns:
        summary['unique_ids'] = int(df['ID_NUMBER'].nunique())
    return summary


def segment_bins(arpu: pd.Series) -> pd.Series:
    return pd.cut(arpu, bins=ARPU_BINS, labels=SEGMENTS['arpu_segment']).astype('string')


def history_lift(change: pd.DataFrame) -> pd.DataFrame:
    df = change[change['AVG_ARPU_PREV_3M'] >= MIN_PREV_ARPU].copy()
    df['arpu_segment'] = segment_bins(df['AVG_ARPU_PREV_3M'])
    df['lift'] = (df['AVG_ARPU_NEXT_3M'] - df['AVG_ARPU_PREV_3M']) / df['AVG_ARPU_PREV_3M']
    df['lift_clipped'] = df['lift'].clip(*LIFT_CLIP)
    return df


def audit(data: dict[str, pd.DataFrame]) -> dict:
    profile, change, arpu, traffic = (data[k] for k in ('customer_profile', 'change_tariff', 'arpu_monthly', 'traffic'))
    known = set(data['dict_tariff']['tariff_plan_code'])
    report: dict = {'files': {}}
    for name, df in data.items():
        report['files'][name] = file_summary(name, df, ROOT / FILES[name])

    dt, td = data['dict_tariff'], data['tariff_dictionary']
    shared = [c for c in dt.columns if c in td.columns]
    merged = dt[shared].merge(td[shared], how='outer', indicator=True)
    report['tariff_dictionaries_consistent'] = bool((merged['_merge'] == 'both').all())

    domain = {
        'unknown_current_tariff': sorted(set(profile['current_tariff'].dropna()) - known),
        'unknown_history_from': sorted(set(change['tariff_plan_code_from'].dropna()) - known),
        'unknown_history_to': sorted(set(change['tariff_plan_code_to'].dropna()) - known),
        'unknown_traffic_tariff': sorted(set(traffic['tariff_plan_code'].dropna()) - known),
        'tariffs_without_customers': sorted(known - set(profile['current_tariff'].dropna())),
        'history_same_from_to_rows': int((change['tariff_plan_code_from'] == change['tariff_plan_code_to']).sum()),
    }
    for col, allowed in SEGMENTS.items():
        values = profile[col].value_counts(dropna=False)
        domain[col] = {
            'counts': {(MISSING if pd.isna(k) else k): int(v) for k, v in values.items()},
            'invalid_values': sorted(set(profile[col].dropna()) - set(allowed)),
        }
    report['domain'] = domain

    # Rows with a missing filter column can never be selected by that filter.
    p = profile.assign(**{c: profile[c].fillna(MISSING) for c in ['current_tariff', *SEGMENTS]})
    grid = (p.groupby(['current_tariff', 'arpu_segment'])['predicted_arpu']
            .agg(n='size', arpu_sum='sum', arpu_mean='mean').reset_index()
            .sort_values(['current_tariff', 'arpu_segment']))
    report['audience_grid'] = {
        'cells': len(grid), 'cells_with_n_ge_200': int((grid['n'] >= 200).sum()),
        'cells_with_n_lt_10': int((grid['n'] < 10).sum()),
        'rows': grid.to_dict('records'),
    }
    one_dim = {}
    for col in ['current_tariff', *SEGMENTS]:
        g = p.groupby(col)['predicted_arpu'].agg(n='size', arpu_sum='sum', arpu_mean='mean')
        one_dim[col] = g.sort_values('n', ascending=False).reset_index().to_dict('records')
    report['audience_by_filter'] = one_dim
    report['segments_over_campaign_cap'] = {
        col: sorted(k for k, v in p[col].value_counts().items() if v > MAX_CUSTOMERS_PER_CAMPAIGN)
        for col in ['current_tariff', *SEGMENTS]
    }

    report['predicted_arpu'] = describe(profile['predicted_arpu'])
    report['predicted_arpu']['share_top_10pct_customers'] = float(
        profile['predicted_arpu'].nlargest(len(profile) // 10).sum() / profile['predicted_arpu'].sum())
    seg_check = {}
    for col in ['ARPU_current', 'ARPU_3m_avg']:
        derived = segment_bins(profile[col])
        known_seg = profile['arpu_segment'].notna()
        seg_check[col] = float((derived[known_seg] == profile.loc[known_seg, 'arpu_segment']).mean())
    report['arpu_segment_matches_mock_bins_share'] = seg_check
    report['arpu_segment_ranges'] = {
        k: {'ARPU_3m_avg_min': g['ARPU_3m_avg'].min(), 'ARPU_3m_avg_max': g['ARPU_3m_avg'].max(),
            'predicted_arpu_median': g['predicted_arpu'].median()}
        for k, g in profile.groupby('arpu_segment')
    }

    ids = set(profile['ID_NUMBER'])
    hist_ids = set(change['ID_NUMBER'])
    report['joins'] = {
        'profile_ids_in_arpu_monthly': len(ids & set(arpu['ID_NUMBER'])),
        'profile_ids_in_traffic': len(ids & set(traffic['ID_NUMBER'])),
        'profile_ids_in_change_tariff': len(ids & hist_ids),
        'change_tariff_ids_not_in_profile': len(hist_ids - ids),
        'change_tariff_customers_with_multiple_switches': int((change['ID_NUMBER'].value_counts() > 1).sum()),
        'traffic_ids_not_in_profile': len(set(traffic['ID_NUMBER']) - ids),
        'arpu_ids_not_in_profile': len(set(arpu['ID_NUMBER']) - ids),
        'traffic_arpu_same_id_month_pairs': int(len(
            traffic.assign(m=pd.to_datetime(traffic['time_key']))[['ID_NUMBER', 'm']].drop_duplicates()
            .merge(arpu.assign(m=pd.to_datetime(arpu['TIME_KEY']))[['ID_NUMBER', 'm']].drop_duplicates()))),
    }
    last_traffic = (traffic.sort_values('time_key').groupby('ID_NUMBER')['tariff_plan_code'].last())
    both = profile.set_index('ID_NUMBER')['current_tariff'].dropna().to_frame().join(last_traffic, how='inner')
    report['joins']['profile_tariff_equals_last_traffic_tariff_share'] = (
        float((both['current_tariff'] == both['tariff_plan_code']).mean()) if len(both) else None)
    overlap = change[change['ID_NUMBER'].isin(ids)].sort_values('TIME_KEY').groupby('ID_NUMBER').last()
    ov = overlap.join(profile.set_index('ID_NUMBER')['current_tariff'], how='inner')
    report['joins']['overlap_current_equals_history_to_share'] = (
        float((ov['current_tariff'] == ov['tariff_plan_code_to']).mean()) if len(ov) else None)
    report['joins']['overlap_current_equals_history_from_share'] = (
        float((ov['current_tariff'] == ov['tariff_plan_code_from']).mean()) if len(ov) else None)

    h = history_lift(change)
    prev_all = change['AVG_ARPU_PREV_3M']
    report['history_bias'] = {
        'rows_total': len(change),
        'rows_prev_arpu_below_100_excluded': int((prev_all < MIN_PREV_ARPU).sum()),
        'rows_prev_arpu_nonpositive': int((prev_all <= 0).sum()),
        'lift_raw': describe(h['lift']),
        'rows_lift_clipped_high': int((h['lift'] > LIFT_CLIP[1]).sum()),
        'rows_lift_clipped_low': int((h['lift'] < LIFT_CLIP[0]).sum()),
        'share_positive_lift': float((h['lift'] > 0).mean()),
        'history_prev_arpu_median': float(prev_all.median()),
        'profile_arpu_3m_avg_median': float(profile['ARPU_3m_avg'].median()),
        'history_arpu_segment_share': {k: float(v) for k, v in h['arpu_segment'].value_counts(normalize=True).items()},
        'profile_arpu_segment_share': {k: float(v) for k, v in profile['arpu_segment'].value_counts(normalize=True).items()},
        'history_from_top10': {k: int(v) for k, v in change['tariff_plan_code_from'].value_counts().head(10).items()},
        'profile_current_top10': {k: int(v) for k, v in profile['current_tariff'].value_counts().head(10).items()},
        'change_time_after_profile_traffic': str(pd.to_datetime(change['TIME_KEY']).min().date())
        > str(pd.to_datetime(traffic['time_key']).max().date()),
    }
    return clean(report)


def candidate_priors(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    h = history_lift(data['change_tariff'])
    keys = ['tariff_plan_code_from', 'arpu_segment', 'tariff_plan_code_to']
    g = (h.groupby(keys)
         .agg(history_n=('lift', 'size'), lift_median=('lift', 'median'),
              lift_mean_clipped=('lift_clipped', 'mean'), lift_std=('lift_clipped', 'std'))
         .reset_index())
    totals = g.groupby(keys[:2])['history_n'].transform('sum')
    g['history_share'] = g['history_n'] / totals
    g['lift_se'] = g['lift_std'] / np.sqrt(g['history_n'])
    g['reliable'] = g['history_n'] >= MIN_RELIABLE_N
    g = g.rename(columns={'tariff_plan_code_from': 'current_tariff', 'tariff_plan_code_to': 'target_tariff'})

    aud = (data['customer_profile'].groupby(['current_tariff', 'arpu_segment'])['predicted_arpu']
           .agg(audience_n='size', audience_arpu_sum='sum').reset_index())
    out = g.merge(aud, on=['current_tariff', 'arpu_segment'], how='left')
    out['audience_n'] = out['audience_n'].fillna(0).astype(int)
    out['audience_arpu_sum'] = out['audience_arpu_sum'].fillna(0.0)
    cols = ['current_tariff', 'arpu_segment', 'target_tariff', 'history_n', 'lift_median',
            'lift_mean_clipped', 'lift_std', 'audience_n', 'audience_arpu_sum',
            'history_share', 'lift_se', 'reliable']
    out = out[cols].sort_values(['current_tariff', 'arpu_segment', 'target_tariff']).reset_index(drop=True)
    return out.round(6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', default=str(ROOT / 'reports'))
    args = parser.parse_args()
    data = {name: pd.read_csv(ROOT / path) for name, path in FILES.items()}
    report = audit(data)
    report['meta'] = {
        'tool': 'tools/analyze_data.py',
        'versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': np.__version__},
        'notes': ['History is a selected sample of switchers: a prior, not the judging model.',
                  'Missing segment values are reported as <missing>, never imputed.'],
    }
    priors = candidate_priors(data)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data_audit.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    priors.to_csv(out / 'candidate_priors.csv', index=False, lineterminator='\n')
    b = report['history_bias']
    print(f"profile rows={report['files']['customer_profile']['rows']} "
          f"history rows={b['rows_total']} excluded(prev<100)={b['rows_prev_arpu_below_100_excluded']}")
    print(f"priors rows={len(priors)} reliable={int(priors['reliable'].sum())} -> {out}")


if __name__ == '__main__':
    main()
