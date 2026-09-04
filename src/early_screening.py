"""
Module: src/early_screening.py
Description: Infant mortality screening and short-term LOOCV gate.
Trains a Dual Digital Twin (PCE & pFF) on the mature phase (>14d) of healthy cells 
to establish a stable baseline. Projects predictions backward onto the burn-in 
phase (<=14d) to detect premature thermodynamic degradation and filter defective 
devices before long-term production modeling.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, export_text

# Professional MLOps logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EarlyScreening")

# Global configuration constants
DIGITAL_TWIN_ALERT_FLOOR_PCE = 0.5
DIGITAL_TWIN_ALERT_FLOOR_PFF = 0.15   
PCE_INITIAL_REF_FLOOR = 1e-3
FEATURES = [
    'POA_Irradiance_W_m2', 
    'ModuleTemp_C', 
    'AbsoluteHumidity_g_m3',
    'Delta_Temp_C_per_h',
    'Delta_Hum_g_m3_per_h'
]

# =====================================================================
# 1. DUAL DIGITAL TWIN: TRAINING & INFERENCE PIPELINE
# =====================================================================
def train_and_evaluate_censored_twin(
    df: pd.DataFrame,
    healthy_cells: list | None = None,
    irradiance_threshold: float = 100.0,
    burn_in_days: float = 14.0,
    min_pce_initial_ref: float = PCE_INITIAL_REF_FLOOR
) -> Tuple[
    pd.DataFrame,
    dict,
    dict,
    dict, dict, dict, dict, dict,
    pd.DataFrame,
    list,
    Dict[str, xgb.XGBRegressor]
]:
    
    print("--- Initializing Early Screening (Burn-in Gate) ---")

    df_proc = df.copy()
    if df_proc.index.name == "Timestamp":
        df_proc = df_proc.reset_index()

    df_proc['Datetime'] = pd.to_datetime(df_proc['Timestamp'], utc=True)
    df_proc['Date_Day'] = df_proc['Datetime'].dt.date
    df_proc['Day_Zero'] = df_proc.groupby('cell_name')['Datetime'].transform('min')
    df_proc['Exposure_Days'] = (df_proc['Datetime'] - df_proc['Day_Zero']).dt.total_seconds() / 86400.0

    df_daylight = df_proc[df_proc['POA_Irradiance_W_m2'] > irradiance_threshold].copy()

    # Calculate daily peaks for T80 evaluation
    df_daily = (
        df_daylight.groupby(['cell_name', 'Date_Day'])
        .agg(
            PCE_max=('PCE', 'max'), 
            Exposure_Days_max=('Exposure_Days', 'max'),
            Datetime_max=('Datetime', 'max') 
        )
        .reset_index()
        .sort_values(by=['cell_name', 'Date_Day'])
    )

    first_3_days = df_daily.groupby('cell_name').head(3)
    idx_max_initial = first_3_days.groupby('cell_name')['PCE_max'].idxmax()
    initial_peak = first_3_days.loc[idx_max_initial, ['cell_name', 'Exposure_Days_max', 'PCE_max']].rename(
        columns={'PCE_max': 'PCE_initial', 'Exposure_Days_max': 'Peak_Day'}
    )
    df_daily = df_daily.merge(initial_peak, on='cell_name', how='left')
    df_daily['T80_threshold'] = df_daily['PCE_initial'] * 0.80
    
    df_daily['Is_Below_T80'] = (
        (df_daily['Exposure_Days_max'] >= df_daily['Peak_Day']) &
        (df_daily['PCE_max'] < df_daily['T80_threshold'])  
    ).astype(int)

    # Independent T80 censoring for PCE and pFF.
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
    first_3_days_pff = df_daily_pff.groupby('cell_name').head(3)
    idx_max_initial_pff = first_3_days_pff.groupby('cell_name')['pFF_max'].idxmax()
    initial_peak_pff = first_3_days_pff.loc[
        idx_max_initial_pff, ['cell_name', 'Exposure_Days_max', 'pFF_max']
    ].rename(columns={
        'pFF_max': 'pFF_initial',
        'Exposure_Days_max': 'Peak_Day_pFF'
    })
    df_daily_pff = df_daily_pff.merge(initial_peak_pff, on='cell_name', how='left')
    df_daily_pff['T80_threshold_pFF'] = df_daily_pff['pFF_initial'] * 0.80
    df_daily_pff['Is_Below_T80_pFF'] = (
        (df_daily_pff['Exposure_Days_max'] >= df_daily_pff['Peak_Day_pFF']) &
        (df_daily_pff['pFF_max'] < df_daily_pff['T80_threshold_pFF'])
    ).astype(int)

    n_consecutive_days = 3
    survival_days_pce, t80_dates_pce = {}, {}
    survival_days_pff, t80_dates_pff = {}, {}

    def extract_t80(group, target_col, initial_col, is_below_col, day_col, dt_col):
        group = group.sort_values('Date_Day').reset_index(drop=True)
        streak = group[is_below_col].rolling(
            window=n_consecutive_days, min_periods=1
        ).sum()
        standard_collapse = streak == n_consecutive_days
        derivative = group[target_col].diff()
        catastrophic_collapse = (
            (derivative <= -(group[initial_col] * 0.20)) &
            (group[is_below_col] == 1)
        )
        truncated_collapse = pd.Series(False, index=group.index)
        if len(group) > 0 and group[is_below_col].iloc[-1] == 1:
            truncated_collapse.iloc[-1] = True
        combined_failure = standard_collapse | catastrophic_collapse | truncated_collapse
        if combined_failure.any():
            idx_confirmation = int(np.flatnonzero(combined_failure.to_numpy())[0])
            idx_start = idx_confirmation
            while idx_start > 0 and group.loc[idx_start - 1, is_below_col] == 1:
                idx_start -= 1
            return float(group.loc[idx_start, day_col]), group.loc[idx_start, dt_col]
        return np.inf, pd.NaT

    for cell, group in df_daily.groupby('cell_name'):
        survival_days_pce[cell], t80_dates_pce[cell] = extract_t80(
            group, 'PCE_max', 'PCE_initial', 'Is_Below_T80',
            'Exposure_Days_max', 'Datetime_max'
        )
    for cell, group in df_daily_pff.groupby('cell_name'):
        survival_days_pff[cell], t80_dates_pff[cell] = extract_t80(
            group, 'pFF_max', 'pFF_initial', 'Is_Below_T80_pFF',
            'Exposure_Days_max', 'Datetime_max'
        )

    survival_days_combined = {
        cell: min(survival_days_pce.get(cell, np.inf), survival_days_pff.get(cell, np.inf))
        for cell in survival_days_pce
    }
    t80_dates_combined = {}
    for cell in survival_days_combined:
        pce_days = survival_days_pce.get(cell, np.inf)
        pff_days = survival_days_pff.get(cell, np.inf)
        if pce_days <= pff_days and pce_days != np.inf:
            t80_dates_combined[cell] = t80_dates_pce[cell]
        elif pff_days < pce_days:
            t80_dates_combined[cell] = t80_dates_pff[cell]
        else:
            t80_dates_combined[cell] = pd.NaT

    if healthy_cells is None:
        healthy_cells = [str(cell) for cell, days in survival_days_combined.items() if days > burn_in_days]
        print(f"    Auto-detected healthy cohort (Survival_Days > {burn_in_days}): {healthy_cells}")

    df_daylight['Death_Day'] = df_daylight['cell_name'].map(survival_days_combined)
    df_censored = (
        df_daylight[df_daylight['Exposure_Days'] <= df_daylight['Death_Day']]
        .drop(columns=['Death_Day'])
        .sort_values(by=['cell_name', 'Timestamp'])
        .reset_index(drop=True)
    )

    # Validate essential targets without relying on an optimization-disabled assert
    target_pce = 'PCE'
    target_pff = 'pFF'
    for target in (target_pce, target_pff):
        try:
            df_censored[target] = pd.to_numeric(df_censored[target], errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Target column '{target}' contains non-numeric values.") from exc

    df_censored = df_censored.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [target_pce, target_pff]).reset_index(drop=True)
    if df_censored.empty:
        raise ValueError("No valid censored observations remain after target and feature validation.")

    # Inter-cell PCE normalization (Relative to baseline)
    pce_initial_map = df_daily.drop_duplicates('cell_name').set_index('cell_name')['PCE_initial']
    df_censored['PCE_Initial_Ref'] = pd.to_numeric(
        df_censored['cell_name'].map(pce_initial_map), errors='coerce'
    )
    invalid_refs = df_censored.loc[
        df_censored['PCE_Initial_Ref'].notna() &
        (df_censored['PCE_Initial_Ref'] <= min_pce_initial_ref),
        'cell_name'
    ].unique().tolist()
    if invalid_refs:
        raise ValueError(
            f"PCE_Initial_Ref must be greater than {min_pce_initial_ref:g}; "
            f"invalid cells: {invalid_refs}"
        )

    df_censored = df_censored.dropna(subset=['PCE_Initial_Ref'])
    if df_censored.empty:
        raise ValueError("No observations have a valid PCE_Initial_Ref for relative normalization.")
    df_censored['PCE_Relative'] = df_censored['PCE'] / df_censored['PCE_Initial_Ref']
    target_pce_rel = 'PCE_Relative'

    train_mask = (
        df_censored['cell_name'].isin(healthy_cells) &
        (df_censored['Exposure_Days'] > burn_in_days)
    )
    
    X_train = df_censored.loc[train_mask, FEATURES]
    y_train_pce = df_censored.loc[train_mask, target_pce_rel]
    y_train_pff = df_censored.loc[train_mask, target_pff]
    print(f"    Training rows (mature phase > {burn_in_days}d): {len(X_train)}")

    # Dual Digital Twin Training
    dt_model_pce = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42, n_jobs=-1)
    dt_model_pff = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, n_jobs=-1)
    
    dt_model_pce.fit(X_train, y_train_pce)
    dt_model_pff.fit(X_train, y_train_pff)

    # Predictions & Underperformance calculation
    df_censored['Twin_PCE_Pred_Relative'] = dt_model_pce.predict(df_censored[FEATURES])
    df_censored['Twin_PCE_Pred'] = df_censored['Twin_PCE_Pred_Relative'] * df_censored['PCE_Initial_Ref']
    df_censored['Underperformance_PCE'] = (df_censored['Twin_PCE_Pred_Relative'] - df_censored[target_pce_rel]) * df_censored['PCE_Initial_Ref']
    
    df_censored['Twin_pFF_Pred'] = dt_model_pff.predict(df_censored[FEATURES])
    df_censored['Underperformance_pFF'] = df_censored['Twin_pFF_Pred'] - df_censored[target_pff]

    # Calculate baselines on absolute scale
    y_train_abs = y_train_pce * df_censored.loc[train_mask, 'PCE_Initial_Ref']
    train_preds_abs = dt_model_pce.predict(X_train) * df_censored.loc[train_mask, 'PCE_Initial_Ref']
    
    base_mae_pce = float(mean_absolute_error(y_train_abs, train_preds_abs))
    base_mae_pff = float(mean_absolute_error(y_train_pff, dt_model_pff.predict(X_train)))
    
    alert_thresh_pce = max(float(base_mae_pce * 3.0), DIGITAL_TWIN_ALERT_FLOOR_PCE)
    alert_thresh_pff = max(float(base_mae_pff * 3.0), DIGITAL_TWIN_ALERT_FLOOR_PFF)

    # Dual Alert Logic (OR Gate): Power Drop OR Structural Mutation
    action_mask = df_censored['Exposure_Days'] <= burn_in_days

    # Register individual alerts while preserving the combined alert.
    df_censored['Alert_PCE'] = False
    df_censored['Alert_pFF'] = False
    df_censored.loc[action_mask, 'Alert_PCE'] = (
        df_censored.loc[action_mask, 'Underperformance_PCE'] > alert_thresh_pce
    )
    df_censored.loc[action_mask, 'Alert_pFF'] = (
        df_censored.loc[action_mask, 'Underperformance_pFF'] > alert_thresh_pff
    )
    df_censored['Digital_Twin_Alert'] = df_censored['Alert_PCE'] | df_censored['Alert_pFF']
    df_censored['In_Action_Window'] = action_mask

    df_censored['T80_Threshold'] = df_censored['cell_name'].map(
        df_daily.drop_duplicates('cell_name').set_index('cell_name')['T80_threshold']
    )

    thresholds = {'pce': alert_thresh_pce, 'pff': alert_thresh_pff}
    models = {'pce': dt_model_pce, 'pff': dt_model_pff}
    return (
        df_censored, thresholds, survival_days_combined, t80_dates_combined,
        survival_days_pce, t80_dates_pce, survival_days_pff, t80_dates_pff,
        df_daily, healthy_cells, models
    )

# =====================================================================
# 2. LOOCV (LEAVE-ONE-OUT CROSS-VALIDATION) FRAMEWORK
# =====================================================================
def execute_loocv_validation(
    df_censored: pd.DataFrame,
    healthy_cells: list,
    burn_in_days: float = 14.0,
    alert_freq_limit: float = 15.0
) -> pd.DataFrame:
    
    target_pce_rel = 'PCE_Relative'
    target_pff = 'pFF'
    loocv_results = []
    
    for holdout_cell in healthy_cells:
        train_cells = [c for c in healthy_cells if c != holdout_cell]
        
        train_mask = (df_censored['cell_name'].isin(train_cells)) & (df_censored['Exposure_Days'] > burn_in_days)
        test_mask = (df_censored['cell_name'] == holdout_cell) & (df_censored['Exposure_Days'] <= burn_in_days)
        
        X_train = df_censored.loc[train_mask, FEATURES]
        y_train_pce = df_censored.loc[train_mask, target_pce_rel]
        y_train_pff = df_censored.loc[train_mask, target_pff]
        
        X_test = df_censored.loc[test_mask, FEATURES]
        y_test_pce = df_censored.loc[test_mask, target_pce_rel]
        y_test_pff = df_censored.loc[test_mask, target_pff]
        
        if len(X_test) == 0:
            continue
            
        loocv_model_pce = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42, n_jobs=-1)
        loocv_model_pff = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, n_jobs=-1)
        
        loocv_model_pce.fit(X_train, y_train_pce)
        loocv_model_pff.fit(X_train, y_train_pff)

        # Scale reconstruction for PCE
        ref_train = np.asarray(pd.to_numeric(df_censored.loc[train_mask, 'PCE_Initial_Ref'], errors='coerce'), dtype=float)
        ref_test = np.asarray(pd.to_numeric(df_censored.loc[test_mask, 'PCE_Initial_Ref'], errors='coerce'), dtype=float)

        y_train_pce_abs = np.asarray(y_train_pce, dtype=float) * ref_train
        y_test_pce_abs = np.asarray(y_test_pce, dtype=float) * ref_test
        
        train_preds_pce_abs = loocv_model_pce.predict(X_train) * ref_train
        test_preds_pce_abs = loocv_model_pce.predict(X_test) * ref_test

        base_mae_pce = float(mean_absolute_error(y_train_pce_abs, train_preds_pce_abs))
        base_mae_pff = float(mean_absolute_error(y_train_pff, loocv_model_pff.predict(X_train)))

        alert_thresh_pce = max(float(base_mae_pce * 3.0), DIGITAL_TWIN_ALERT_FLOOR_PCE)
        alert_thresh_pff = max(float(base_mae_pff * 3.0), DIGITAL_TWIN_ALERT_FLOOR_PFF)

        underperformance_pce = test_preds_pce_abs - y_test_pce_abs
        underperformance_pff = loocv_model_pff.predict(X_test) - np.asarray(y_test_pff, dtype=float)
        
        alerts = (underperformance_pce > alert_thresh_pce) | (underperformance_pff > alert_thresh_pff)
        alert_pct = alerts.sum() / len(y_test_pce_abs) * 100.0

        loocv_results.append({
            'Holdout_Cell': holdout_cell,
            'Train_MAE_PCE': base_mae_pce,
            'Action_Window_Points': len(y_test_pce_abs),
            'Alert_Freq_Pct': float(alert_pct),
            'Validation_Status': 'PASS' if alert_pct <= alert_freq_limit else 'FAIL'
        })
        
    return pd.DataFrame(loocv_results).round(4)


# =====================================================================
# 3. DIAGNOSTIC RISK ASSESSMENT GENERATOR
# =====================================================================
def generate_diagnostic_summary(
    df_twin: pd.DataFrame,
    survival_days_comb: dict, t80_dates_comb: dict,
    survival_days_pce: dict, t80_dates_pce: dict,
    survival_days_pff: dict, t80_dates_pff: dict,
    burn_in_days: float = 14.0,
    min_points_required: int = 100,
    early_transient_days: float = 1.0,
    alert_freq_limit: float = 15.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    df_action = df_twin[df_twin['Exposure_Days'] <= burn_in_days].copy()
    df_action = df_action.sort_values(by=['cell_name', 'Datetime'])

    df_action['Cum_Points'] = df_action.groupby('cell_name').cumcount().add(1)

    # Accumulated combined alerts (OR gate).
    df_action['Alert_Int'] = df_action['Digital_Twin_Alert'].astype(int)
    df_action['Cum_Alert_Pct'] = (
        df_action.groupby('cell_name')['Alert_Int'].cumsum() / df_action['Cum_Points']
    ) * 100.0

    # Accumulated PCE alerts.
    df_action['Alert_PCE_Int'] = df_action['Alert_PCE'].astype(int)
    df_action['Cum_Alert_Pct_PCE'] = (
        df_action.groupby('cell_name')['Alert_PCE_Int'].cumsum() / df_action['Cum_Points']
    ) * 100.0

    # Accumulated pFF alerts.
    df_action['Alert_pFF_Int'] = df_action['Alert_pFF'].astype(int)
    df_action['Cum_Alert_Pct_pFF'] = (
        df_action.groupby('cell_name')['Alert_pFF_Int'].cumsum() / df_action['Cum_Points']
    ) * 100.0

    def get_crossings(column_name):
        valid = (
            (df_action[column_name] > alert_freq_limit) &
            (df_action['Cum_Points'] >= min_points_required) &
            (df_action['Exposure_Days'] > early_transient_days)
        )
        crossing_rows = df_action[valid]
        crossing_days = crossing_rows.groupby('cell_name')['Exposure_Days'].min()
        crossing_dates = crossing_rows.groupby('cell_name')['Datetime'].first()
        return crossing_days, crossing_dates

    time_15pct_day, time_15pct_date = get_crossings('Cum_Alert_Pct')
    time_15pct_day_pce, time_15pct_date_pce = get_crossings('Cum_Alert_Pct_PCE')
    time_15pct_day_pff, time_15pct_date_pff = get_crossings('Cum_Alert_Pct_pFF')

    summary = df_action.groupby('cell_name').agg(
        Alert_Count=('Digital_Twin_Alert', 'sum'),
        Alert_PCE_Count=('Alert_PCE', 'sum'),
        Alert_pFF_Count=('Alert_pFF', 'sum'),
        Data_Points=('PCE', 'count')
    )
    
    summary['alert_freq_pct'] = (summary['Alert_Count'] / summary['Data_Points']) * 100.0
    summary['alert_pce_pct'] = (summary['Alert_PCE_Count'] / summary['Data_Points']) * 100.0
    summary['alert_pff_pct'] = (summary['Alert_pFF_Count'] / summary['Data_Points']) * 100.0
    summary['survival_days'] = [survival_days_comb.get(str(cell), np.inf) for cell in summary.index]
    summary['t80_failure_date'] = [t80_dates_comb.get(str(cell), pd.NaT) for cell in summary.index]
    summary['survival_days_pce'] = [survival_days_pce.get(str(cell), np.inf) for cell in summary.index]
    summary['t80_failure_date_pce'] = [t80_dates_pce.get(str(cell), pd.NaT) for cell in summary.index]
    summary['survival_days_pff'] = [survival_days_pff.get(str(cell), np.inf) for cell in summary.index]
    summary['t80_failure_date_pff'] = [t80_dates_pff.get(str(cell), pd.NaT) for cell in summary.index]

    early_collapse_comb = summary['survival_days'] <= burn_in_days
    early_collapse_pce = summary['survival_days_pce'] <= burn_in_days
    early_collapse_pff = summary['survival_days_pff'] <= burn_in_days
    no_alert_dates = pd.Series(pd.NaT, index=summary.index, dtype='datetime64[ns]').to_numpy()

    # Final combined evaluation.
    summary['extrinsic_failure'] = early_collapse_comb | (summary['alert_freq_pct'] > alert_freq_limit)
    summary['threshold_15pct_day'] = np.where(summary['extrinsic_failure'], time_15pct_day.reindex(summary.index).to_numpy(), np.nan)
    summary['ml_alert_date'] = np.where(
        summary['extrinsic_failure'],
        time_15pct_date.reindex(summary.index).to_numpy(),
        no_alert_dates
    )

    # Final PCE-only evaluation.
    summary['extrinsic_failure_pce'] = early_collapse_pce | (summary['alert_pce_pct'] > alert_freq_limit)
    summary['threshold_15pct_day_pce'] = np.where(
        summary['extrinsic_failure_pce'],
        time_15pct_day_pce.reindex(summary.index).to_numpy(),
        np.nan
    )
    summary['ml_alert_date_pce'] = np.where(
        summary['extrinsic_failure_pce'],
        time_15pct_date_pce.reindex(summary.index).to_numpy(),
        no_alert_dates
    )

    # Final pFF-only evaluation.
    summary['extrinsic_failure_pff'] = early_collapse_pff | (summary['alert_pff_pct'] > alert_freq_limit)
    summary['threshold_15pct_day_pff'] = np.where(
        summary['extrinsic_failure_pff'],
        time_15pct_day_pff.reindex(summary.index).to_numpy(),
        np.nan
    )
    summary['ml_alert_date_pff'] = np.where(
        summary['extrinsic_failure_pff'],
        time_15pct_date_pff.reindex(summary.index).to_numpy(),
        no_alert_dates
    )

    summary = summary.drop(
        columns=['Alert_Count', 'Alert_PCE_Count', 'Alert_pFF_Count', 'Data_Points']
    )
    numeric_cols = summary.select_dtypes(include=['float64', 'float32']).columns
    summary[numeric_cols] = summary[numeric_cols].round(3)

    summary_combined = summary[[
        'extrinsic_failure', 'alert_freq_pct', 'alert_pce_pct', 'alert_pff_pct',
        'survival_days', 't80_failure_date', 'threshold_15pct_day', 'ml_alert_date'
    ]].sort_values(by='alert_freq_pct', ascending=False)

    summary_pce = summary[[
        'extrinsic_failure_pce', 'alert_pce_pct', 'survival_days_pce', 't80_failure_date_pce',
        'threshold_15pct_day_pce', 'ml_alert_date_pce'
    ]].rename(columns={
        'extrinsic_failure_pce': 'extrinsic_failure',
        'survival_days_pce': 'survival_days',
        't80_failure_date_pce': 't80_failure_date',
        'threshold_15pct_day_pce': 'threshold_15pct_day',
        'ml_alert_date_pce': 'ml_alert_date'
    }).sort_values(by='alert_pce_pct', ascending=False)

    summary_pff = summary[[
        'extrinsic_failure_pff', 'alert_pff_pct', 'survival_days_pff', 't80_failure_date_pff',
        'threshold_15pct_day_pff', 'ml_alert_date_pff'
    ]].rename(columns={
        'extrinsic_failure_pff': 'extrinsic_failure',
        'survival_days_pff': 'survival_days',
        't80_failure_date_pff': 't80_failure_date',
        'threshold_15pct_day_pff': 'threshold_15pct_day',
        'ml_alert_date_pff': 'ml_alert_date'
    }).sort_values(by='alert_pff_pct', ascending=False)

    return summary_combined, summary_pce, summary_pff


# =====================================================================
# 4. LOOCV GATE — Filters the training cohort
# =====================================================================
def apply_loocv_gate(
    loocv_table: pd.DataFrame,
    healthy_cells: list,
    alert_freq_limit: float = 15.0
) -> Tuple[list, list]:
    
    if loocv_table.empty:
        return list(healthy_cells), []

    failed = loocv_table.loc[
        loocv_table['Alert_Freq_Pct'] > alert_freq_limit, 'Holdout_Cell'
    ].tolist()
    gated_cohort = [c for c in healthy_cells if c not in failed]
    
    return gated_cohort, failed


# =====================================================================
# 5. PRODUCTION ORCHESTRATOR
# =====================================================================
def run_screening_pipeline(
    df: pd.DataFrame,
    burn_in_days: float = 14.0,
    irradiance_threshold: float = 100.0,
    alert_freq_limit: float = 15.0,
    verbose: bool = True
) -> dict:
    
    def log(msg=""):
        if verbose: print(msg)

    log("=" * 80)
    log(" INITIALIZING SELF-HEALING DUAL DIGITAL TWIN PIPELINE")
    log("=" * 80)

    _, _, s_days, t80_d, s_days_pce, t80_pce, s_days_pff, t80_pff, df_daily, screening_cohort, _ = train_and_evaluate_censored_twin(
        df=df, healthy_cells=None,
        irradiance_threshold=irradiance_threshold, burn_in_days=burn_in_days
    )
    
    current_cohort = screening_cohort.copy()
    iteration = 1
    max_iterations = 5
    
    df_twin_final = None
    df_twin_models_final = {}
    alert_threshold_final = {}
    loocv_final = pd.DataFrame()
    
    while iteration <= max_iterations:
        log(f"\n--- ITERATION {iteration} | Active Cohort: {current_cohort} ---")
        
        df_twin, alert_thr, s_days, t80_d, s_days_pce, t80_pce, s_days_pff, t80_pff, _, _, dt_models = train_and_evaluate_censored_twin(
            df=df, healthy_cells=current_cohort,
            irradiance_threshold=irradiance_threshold, burn_in_days=burn_in_days
        )
        
        df_twin_final = df_twin
        df_twin_models_final = dt_models
        alert_threshold_final = alert_thr
        
        if len(current_cohort) < 2:
            log(f"\n⚠ CRITICAL: Active cohort reduced to {len(current_cohort)} cell(s). Halting loop.")
            break

        loocv_table = execute_loocv_validation(
            df_twin, current_cohort, burn_in_days=burn_in_days, alert_freq_limit=alert_freq_limit
        )
        loocv_final = loocv_table
        log(loocv_table.to_string(index=False))
        
        _, failed_cells = apply_loocv_gate(loocv_table, current_cohort, alert_freq_limit)
        
        if not failed_cells:
            log(f"\n✅ CONVERGENCE REACHED AT ITERATION {iteration}. Baseline is pure.")
            break
            
        log(f"⚠ LOOCV Gate triggered. Evicting unstable cells: {failed_cells}")
        current_cohort = [c for c in current_cohort if c not in failed_cells]
        iteration += 1
        
    if iteration > max_iterations:
        log("\n⚠ Maximum iterations reached. Forcing termination with current stable state.")

    log("\n" + "=" * 80)
    log(" FINAL DIAGNOSTICS GENERATION")
    log("=" * 80)
    
    if df_twin_final is None:
        raise RuntimeError("Pipeline completed without a valid digital twin dataframe.")

    summary_table, summary_pce, summary_pff = generate_diagnostic_summary(
        df_twin_final, s_days, t80_d, s_days_pce, t80_pce, s_days_pff, t80_pff,
        burn_in_days=burn_in_days, alert_freq_limit=alert_freq_limit
    )

    validated_cells = summary_table[~summary_table['extrinsic_failure']].index.tolist()
    defective_cells = summary_table[summary_table['extrinsic_failure']].index.tolist()

    return {
        'df_twin': df_twin_final,
        'df_twin_models': df_twin_models_final,
        'alert_threshold': alert_threshold_final,
        'survival_days': s_days,
        'df_daily': df_daily,
        'screening_cohort': screening_cohort,
        'gated_out_cells': [c for c in screening_cohort if c not in current_cohort],
        'production_cohort': current_cohort,
        'loocv_production': loocv_final,
        'summary_table': summary_table,
        'summary_pce': summary_pce,
        'summary_pff': summary_pff,
        'validated_cells': validated_cells,
        'defective_cells': defective_cells,
    }


# =====================================================================
# 6. EXPLAINABILITY & ROOT CAUSE ANALYSIS (SURROGATE TREES)
# =====================================================================
def extract_surrogate_rules(
    df_twin: pd.DataFrame, cell_name: str, threshold_day: float, window_days: float = 2.0
) -> Tuple[Optional[DecisionTreeClassifier], Optional[List[str]], Optional[str]]:
    
    if pd.isna(threshold_day):
        return None, None, None

    window_data = df_twin[(df_twin['cell_name'] == cell_name) & (df_twin['Exposure_Days'] <= threshold_day + window_days)].copy()
    target = 'Digital_Twin_Alert'

    if window_data[target].nunique() < 2:
        logger.warning(f"[{cell_name}] Insufficient anomaly variance to train surrogate tree.")
        return None, None, None

    X, y = window_data[FEATURES], window_data[target].astype(int)
    surrogate_tree = DecisionTreeClassifier(
        max_depth=3, 
        min_samples_leaf=5, 
        class_weight='balanced', 
        random_state=42
    )
    surrogate_tree.fit(X, y)

    tree_rules = export_text(surrogate_tree, feature_names=FEATURES, decimals=1)
    
    print("\n" + "="*85)
    print(f" SURROGATE RULE EXTRACTION: [{cell_name}] (Critical Window up to Day {threshold_day + window_days:.1f})")
    print("-" * 85)
    print(tree_rules)
    print("="*85 + "\n")

    return surrogate_tree, FEATURES, tree_rules


# =====================================================================
# 7. EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    
    SURVIVAL_DIR = Path("data/survival/outdoor")
    survival_file = SURVIVAL_DIR / "survival_dataset.parquet"
    ANOMALY_DIR = Path("data/anomaly/outdoor")
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR = Path("data/anomaly/diagnostics/")
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not survival_file.exists():
        logger.error(f"Survival dataset artifact not found at: {survival_file}")
    else:
        logger.info(f"Loading survival dataset matrix from: {survival_file}")
        df_final = pd.read_parquet(survival_file)

        BURN_IN_DAYS = 14.0
        ALERT_FREQ_LIMIT = 15.0

        results = run_screening_pipeline(
            df=df_final,
            burn_in_days=BURN_IN_DAYS,
            irradiance_threshold=100.0,
            alert_freq_limit=ALERT_FREQ_LIMIT,
            verbose=True
        )

        summary_table = results['summary_table']
        df_twin_results = results['df_twin']
        alert_thresholds = results['alert_threshold']

        print("\n" + "=" * 80)
        print("                                PREMATURE FAILURE RISK ASSESSMENT                                ")
        print("=" * 80)
        cols_to_show = [
            'extrinsic_failure', 'alert_freq_pct', 'alert_pce_pct', 'alert_pff_pct',
            'threshold_15pct_day', 'survival_days'
        ]
        print(summary_table[cols_to_show].to_string())
        print("=" * 80)
        print(f"""
[DIAGNOSTIC METRICS LEGEND]
• Extrinsic_Failure   : True if cell suffers early T80 collapse (<={BURN_IN_DAYS:.0f} days) OR exhibits a
                        persistent dual anomaly (Alert_Freq_Pct > {ALERT_FREQ_LIMIT:.1f}%) during the 
                        action window (Exposure_Days <= {BURN_IN_DAYS:.0f}).
• Alert_Freq_Pct      : Percentage of daylight observations where PCE OR structural pFF 
                        underperformed significantly relative to the mature baseline model.
• Threshold_15pct_Day : Exposure day when the cumulative alert rate crossed {ALERT_FREQ_LIMIT:.0f}%.
• Survival_Days       : Exposure days to reach T80 structural collapse.

[PRODUCTION COHORT TRACE]
• Screening cohort (T80 only):        {results['screening_cohort']}
• Expelled by LOOCV gate:             {results['gated_out_cells'] if results['gated_out_cells'] else 'none'}
• Production cohort (trained model):  {results['production_cohort']}
""" + "=" * 80 + "\n")

        # ---------------------------------------------------------------------
        # DATA EXPORT 1: Enriched Parquet Dataset
        # ---------------------------------------------------------------------
        logger.info("Enriching the original dataset architecture with dynamic anomaly metrics...")

        # Keep the normalized PCE series on the full telemetry history. The
        # censored twin dataframe is reserved for T80-aware diagnostics.
        pce_initial_map = (
            results['df_daily']
            .drop_duplicates('cell_name')
            .set_index('cell_name')['PCE_initial']
        )
        df_enriched_full = df_final.copy()
        df_enriched_full['PCE_Initial_Ref'] = pd.to_numeric(
            df_enriched_full['cell_name'].map(pce_initial_map), errors='coerce'
        )
        df_enriched_full['PCE_Relative'] = (
            df_enriched_full['PCE'] / df_enriched_full['PCE_Initial_Ref']
        )
        
        df_enriched_full = df_enriched_full.merge(
            df_twin_results[[
                'Twin_PCE_Pred_Relative', 'Twin_PCE_Pred', 'Underperformance_PCE', 'Alert_PCE',
                'cell_name', 'Timestamp', 'Twin_pFF_Pred', 'Underperformance_pFF', 'Alert_pFF',
                'Digital_Twin_Alert', 'In_Action_Window', 'T80_Threshold', 'Exposure_Days', 'Date_Day'
            ]],
            on=['cell_name', 'Timestamp'], how='left'
        )

        df_enriched_full = df_enriched_full.merge(
            summary_table[['extrinsic_failure', 'threshold_15pct_day', 'survival_days']],
            left_on='cell_name', right_index=True, how='left'
        )
        df_enriched_full['Digital_Twin_Alert'] = df_enriched_full['Digital_Twin_Alert'].fillna(False)

        # ---------------------------------------------------------------------
        # GENERACIÓN DE LOS 3 ARCHIVOS PARQUET
        # ---------------------------------------------------------------------

        # 1. Dataset del Modelo PCE Norm (PCE_Relative)
        pce_cols = ['cell_name', 'Timestamp', 'Exposure_Days'] + FEATURES + \
               ['PCE_Initial_Ref', 'PCE_Relative', 'Twin_PCE_Pred_Relative', 'Underperformance_PCE', 'Alert_PCE']
        df_pce = df_enriched_full[pce_cols]
        pce_parquet_path = ANOMALY_DIR / "pce_norm_predictions.parquet"
        df_pce.to_parquet(pce_parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"Exported PCE Norm matrix ({len(df_pce)} rows) to: {pce_parquet_path.name}")

        # 2. Dataset del Modelo pFF
        pff_cols = ['cell_name', 'Timestamp', 'Exposure_Days'] + FEATURES + \
               ['pFF', 'Twin_pFF_Pred', 'Underperformance_pFF', 'Alert_pFF']
        df_pff = df_enriched_full[pff_cols]
        pff_parquet_path = ANOMALY_DIR / "pff_predictions.parquet"
        df_pff.to_parquet(pff_parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"Exported pFF matrix ({len(df_pff)} rows) to: {pff_parquet_path.name}")

        # 3. Dataset Final Completo
        enriched_parquet_path = ANOMALY_DIR / "anomaly_scored_dataset.parquet"
        df_enriched_full.to_parquet(enriched_parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"Exported final anomaly-scored dataset ({len(df_enriched_full)} rows) to: {enriched_parquet_path.name}")

        # ---------------------------------------------------------------------
        # XAI ROOT CAUSE DIAGNOSTICS (Rule Extraction Only)
        # ---------------------------------------------------------------------
        defective_cells = results['defective_cells']
        print(f"--> Extracting surrogate rules for identified defective devices: {defective_cells}\n")
        
        xai_diagnostics_dict = {}

        for cell_id in defective_cells:
            raw_t_day = summary_table.loc[cell_id, 'threshold_15pct_day']
            t_day = float(raw_t_day) if pd.notna(raw_t_day) else np.nan
            
            tree_model, feat_names, tree_rules = extract_surrogate_rules(df_twin_results, cell_id, t_day)
            
            if tree_model is not None and feat_names is not None:
                extracted_rules_list = tree_rules.strip().split('\n') if isinstance(tree_rules, str) else []
                xai_diagnostics_dict[cell_id] = {
                    "Threshold_15pct_Day": float(t_day) if pd.notna(t_day) else None,
                    "Features_Used": feat_names,
                    "Extracted_Rules": extracted_rules_list
                }

        # ---------------------------------------------------------------------
        # DATA EXPORT 2: XAI Surrogate Rules JSON
        # ---------------------------------------------------------------------
        if xai_diagnostics_dict:
            xai_json_path = DIAGNOSTICS_DIR / "xai_surrogate_rules.json"
            with open(xai_json_path, 'w', encoding='utf-8') as f:
                json.dump(xai_diagnostics_dict, f, indent=4)
            logger.info(f"Exported deterministic XAI surrogate rules to: {xai_json_path.name}")

        validated_cells = results['validated_cells']
        print(f"\nFinal validated cohort secured for climate modeling: {validated_cells}")

        # ---------------------------------------------------------------------
        # DATA EXPORT 3: Joblib Artifacts for Dashboard
        # ---------------------------------------------------------------------
        ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")
        ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            "summary_table": summary_table,
            "summary_pce": results['summary_pce'],
            "summary_pff": results['summary_pff'],
            "healthy_cohort": results['production_cohort'],
            "alert_thresholds": alert_thresholds,
            "model_pce": results['df_twin_models']['pce'], 
            "model_pff": results['df_twin_models']['pff']
        }
        
        joblib.dump(artifacts, ARTIFACTS_PATH)
        logger.info(f"Exported Dual Digital Twin artifacts to: {ARTIFACTS_PATH}")