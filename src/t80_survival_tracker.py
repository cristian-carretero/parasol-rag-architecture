"""
Module: src/t80_survival_tracker.py
Description: Physical lifecycle tracking for outdoor perovskite devices.
Computes the initial performance peak (PCE and pFF) within the first 3 days,
establishes the T80 thresholds (80% of peak), and tracks the time-series
to identify the exact day of structural collapse (3 consecutive days below T80).
Outputs a clean metrics table used by downstream Machine Learning modules.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    DAYLIGHT_IRRADIANCE_MIN_W_M2,
    T80_CONFIRM_CONSECUTIVE_DAYS,
    T80_FRACTION,
    T80_INITIAL_PEAK_DAYS,
    T80_PCE_CONFIRM_IRRADIANCE_MIN_W_M2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("T80Tracker")

def extract_t80(group: pd.DataFrame, target_col: str, is_below_col: str, day_col: str, dt_col: str, n_consecutive_days: int = T80_CONFIRM_CONSECUTIVE_DAYS):
    """Identifies the exact day a device permanently crosses the T80 threshold."""
    group = group.sort_values('Date_Day').reset_index(drop=True)
    streak = group[is_below_col].rolling(window=n_consecutive_days, min_periods=1).sum()
    
    standard_collapse = streak == n_consecutive_days
    
    truncated_collapse = pd.Series(False, index=group.index)
    if len(group) > 0 and group[is_below_col].iloc[-1] == 1:
        truncated_collapse.iloc[-1] = True
        
    combined_failure = standard_collapse | truncated_collapse
    
    if combined_failure.any():
        idx_confirmation = int(np.flatnonzero(combined_failure.to_numpy())[0])
        idx_start = idx_confirmation

        while idx_start > 0 and group.loc[idx_start - 1, is_below_col] == 1:
            idx_start -= 1
        survival_day = float(pd.to_numeric(group[day_col], errors='coerce').iloc[idx_start])
        failure_date = group[dt_col].iloc[idx_start].to_pydatetime()

        return survival_day, failure_date
        
    return np.inf, pd.NaT

def generate_t80_metrics_table(df: pd.DataFrame, irradiance_threshold: float = DAYLIGHT_IRRADIANCE_MIN_W_M2) -> pd.DataFrame:
    """Generates the comprehensive physical health tracking table for the fleet."""
    df_proc = df.copy()
    if df_proc.index.name == "Timestamp":
        df_proc = df_proc.reset_index()

    df_proc['Datetime'] = pd.to_datetime(df_proc['Timestamp'], utc=True)
    df_proc['Date_Day'] = df_proc['Datetime'].dt.date
    df_proc['Day_Zero'] = df_proc.groupby('cell_name')['Datetime'].transform('min')
    df_proc['Exposure_Days'] = (df_proc['Datetime'] - df_proc['Day_Zero']).dt.total_seconds() / 86400.0

    df_daylight = df_proc[df_proc['POA_Irradiance_W_m2'] > irradiance_threshold].copy()

    # --- PCE TRACKING ---
    df_daily_pce = (
        df_daylight.groupby(['cell_name', 'Date_Day'])
        .agg(
            PCE_max=('PCE', 'max'), 
            POA_max=('POA_Irradiance_W_m2', 'max'),
            Exposure_Days_max=('Exposure_Days', 'max'),
            Datetime_max=('Datetime', 'max') 
        )
        .reset_index()
        .sort_values(by=['cell_name', 'Date_Day'])
    )

    first_3_days_pce = df_daily_pce.groupby('cell_name').head(T80_INITIAL_PEAK_DAYS)
    idx_max_initial = first_3_days_pce.groupby('cell_name')['PCE_max'].idxmax()
    initial_peak_pce = first_3_days_pce.loc[idx_max_initial, ['cell_name', 'Exposure_Days_max', 'PCE_max']].rename(
        columns={'PCE_max': 'PCE_initial', 'Exposure_Days_max': 'Peak_Day_PCE'}
    )
    df_daily_pce = df_daily_pce.merge(initial_peak_pce, on='cell_name', how='left')
    df_daily_pce['T80_threshold_PCE'] = df_daily_pce['PCE_initial'] * T80_FRACTION
    df_daily_pce['Is_Below_T80_PCE'] = (
        (df_daily_pce['Exposure_Days_max'] >= df_daily_pce['Peak_Day_PCE']) &
        (df_daily_pce['PCE_max'] < df_daily_pce['T80_threshold_PCE']) &
        (df_daily_pce['POA_max'] > T80_PCE_CONFIRM_IRRADIANCE_MIN_W_M2)
    ).astype(int)

    # --- pFF TRACKING ---
    df_daily_pff = (
        df_daylight.groupby(['cell_name', 'Date_Day'])
        .agg(
            pFF_max=('pFF', 'max'),
            Exposure_Days_max=('Exposure_Days', 'max'),
            Datetime_max=('Datetime', 'max')
        )
        .reset_index()
        .sort_values(by=['cell_name', 'Date_Day'])
    )

    first_3_days_pff = df_daily_pff.groupby('cell_name').head(T80_INITIAL_PEAK_DAYS)
    idx_max_initial_pff = first_3_days_pff.groupby('cell_name')['pFF_max'].idxmax()
    initial_peak_pff = first_3_days_pff.loc[idx_max_initial_pff, ['cell_name', 'Exposure_Days_max', 'pFF_max']].rename(
        columns={'pFF_max': 'pFF_initial', 'Exposure_Days_max': 'Peak_Day_pFF'}
    )
    df_daily_pff = df_daily_pff.merge(initial_peak_pff, on='cell_name', how='left')
    df_daily_pff['T80_threshold_pFF'] = df_daily_pff['pFF_initial'] * T80_FRACTION
    df_daily_pff['Is_Below_T80_pFF'] = (
        (df_daily_pff['Exposure_Days_max'] >= df_daily_pff['Peak_Day_pFF']) &
        (df_daily_pff['pFF_max'] < df_daily_pff['T80_threshold_pFF'])
    ).astype(int)

    # --- EXTRACT SURVIVAL DAYS ---
    results = []
    cells = df_daily_pce['cell_name'].unique()

    pce_initial_by_cell = (
        pd.to_numeric(
            initial_peak_pce.set_index('cell_name')['PCE_initial'],
            errors='coerce'
        )
        .to_dict()
    )

    pff_initial_by_cell = (
        pd.to_numeric(
            initial_peak_pff.set_index('cell_name')['pFF_initial'],
            errors='coerce'
        )
        .to_dict()
    )

    for cell in cells:
        group_pce = df_daily_pce[df_daily_pce['cell_name'] == cell]
        group_pff = df_daily_pff[df_daily_pff['cell_name'] == cell]
        
        s_days_pce, dt_pce = extract_t80(group_pce, 'PCE_max', 'Is_Below_T80_PCE', 'Exposure_Days_max', 'Datetime_max')
        s_days_pff, dt_pff = extract_t80(group_pff, 'pFF_max', 'Is_Below_T80_pFF', 'Exposure_Days_max', 'Datetime_max')
        
        comb_days = min(s_days_pce, s_days_pff)
        comb_dt = dt_pce if (s_days_pce <= s_days_pff and s_days_pce != np.inf) else (dt_pff if s_days_pff < s_days_pce else pd.NaT)
        
        pce_initial = float(pce_initial_by_cell[str(cell)])
        pff_initial = float(pff_initial_by_cell[str(cell)])

        results.append({
            'cell_name': cell,
            'PCE_initial': pce_initial,
            'T80_threshold_PCE': pce_initial * T80_FRACTION,
            'survival_days_pce': s_days_pce,
            't80_failure_date_pce': dt_pce,
            'pFF_initial': pff_initial,
            'T80_threshold_pFF': pff_initial * T80_FRACTION,
            'survival_days_pff': s_days_pff,
            't80_failure_date_pff': dt_pff,
            'combined_survival_days': comb_days,
            'combined_failure_date': comb_dt
        })

    return pd.DataFrame(results).set_index('cell_name')

if __name__ == "__main__":
    SURVIVAL_DIR = Path("data/survival/outdoor")
    survival_file = SURVIVAL_DIR / "survival_dataset.parquet"
    METRICS_OUT = SURVIVAL_DIR / "t80_metrics_table.parquet"
    
    if not survival_file.exists():
        logger.error(f"Input file not found: {survival_file}")
    else:
        logger.info("Tracking T80 physical lifecycle...")
        df_raw = pd.read_parquet(survival_file)
        t80_metrics = generate_t80_metrics_table(df_raw)
        
        t80_metrics.to_parquet(METRICS_OUT, engine='pyarrow', compression='snappy')
        logger.info(f"Exported T80 metrics table for {len(t80_metrics)} devices to: {METRICS_OUT}")
        print("\n--- Physical Health Overview ---")
        print(t80_metrics[['PCE_initial', 'combined_survival_days']].head())