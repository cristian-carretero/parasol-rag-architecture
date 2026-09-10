"""
Module: src/early_screening.py
Description: Infant mortality screening and LOOCV gate.
Phase 1: Executes an empirical Grid Search to determine the optimal burn-in window.
Phase 2: Uses the pre-calculated physical health (T80) to isolate the mature phase
of healthy cells and trains a Dual Digital Twin (PCE & pFF) for rapid anomaly detection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    ALERT_FREQUENCY_THRESHOLD_PCT,
    BURN_IN_DAYS,
    BURN_IN_GRID_WINDOWS,
    DAYLIGHT_IRRADIANCE_MIN_W_M2,
    FEATURES,
    MIN_PHYSICAL_MAE_PCE,
    MIN_PHYSICAL_MAE_PFF,
    PCE_INITIAL_REF_FLOOR,
    RESIDUAL_ALERT_QUANTILE,
    T80_FRACTION,
    XGB_PCE_PARAMS,
    XGB_PFF_PARAMS,
)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EarlyScreening")

# ==============================================================================
# CORE PIPELINE FUNCTIONS
# ==============================================================================
def preprocess_telemetry_data(df: pd.DataFrame, irradiance_threshold: float = DAYLIGHT_IRRADIANCE_MIN_W_M2) -> pd.DataFrame:
    """Standardize timestamps, calculate exposure days, and filter night-time data."""
    df_proc = df.reset_index() if df.index.name == "Timestamp" else df.copy()
    df_proc['Datetime'] = pd.to_datetime(df_proc['Timestamp'], utc=True)
    df_proc['Day_Zero'] = df_proc.groupby('cell_name')['Datetime'].transform('min')
    df_proc['Exposure_Days'] = (df_proc['Datetime'] - df_proc['Day_Zero']).dt.total_seconds() / 86400.0
    return df_proc[df_proc['POA_Irradiance_W_m2'] > irradiance_threshold].copy()


def train_and_evaluate_censored_twin(
    df: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    healthy_cells: Optional[List[str]] = None,
    irradiance_threshold: float = DAYLIGHT_IRRADIANCE_MIN_W_M2,
    burn_in_days: float = BURN_IN_DAYS
) -> Tuple[pd.DataFrame, Dict[str, float], List[str], Dict[str, Any]]:
    """Train the Dual Digital Twin (PCE & pFF) exclusively on the mature phase of healthy cells."""
    logger.info(f"Initializing Early Screening (Burn-in Gate: {burn_in_days} days)")

    df_daylight = preprocess_telemetry_data(df, irradiance_threshold)
    df_daylight = df_daylight.merge(
        t80_metrics[['PCE_initial', 'combined_survival_days']], 
        left_on='cell_name', right_index=True, how='inner'
    )

    if healthy_cells is None:
        healthy_cells = t80_metrics[t80_metrics['combined_survival_days'] > burn_in_days].index.tolist()
        logger.info(f"Auto-detected healthy cohort: {healthy_cells}")

    df_censored = df_daylight[df_daylight['Exposure_Days'] <= df_daylight['combined_survival_days']].copy()
    df_censored = df_censored.dropna(subset=FEATURES + ['PCE', 'pFF', 'PCE_initial']).reset_index(drop=True)
    df_censored['PCE_Relative'] = df_censored['PCE'] / df_censored['PCE_initial'].clip(lower=PCE_INITIAL_REF_FLOOR)

    train_mask = (df_censored['cell_name'].isin(healthy_cells)) & (df_censored['Exposure_Days'] > burn_in_days)
    X_train = df_censored.loc[train_mask, FEATURES]
    
    models = {
        'pce': xgb.XGBRegressor(**XGB_PCE_PARAMS).fit(X_train, df_censored.loc[train_mask, 'PCE_Relative']),
        'pff': xgb.XGBRegressor(**XGB_PFF_PARAMS).fit(X_train, df_censored.loc[train_mask, 'pFF'])
    }

    df_censored['Twin_PCE_Pred_Relative'] = models['pce'].predict(df_censored[FEATURES])
    df_censored['Twin_pFF_Pred'] = models['pff'].predict(df_censored[FEATURES])
    df_censored['Twin_PCE_Pred'] = df_censored['Twin_PCE_Pred_Relative'] * df_censored['PCE_initial']
    
    df_censored['Underperformance_PCE'] = df_censored['Twin_PCE_Pred_Relative'] - df_censored['PCE_Relative']
    df_censored['Underperformance_pFF'] = df_censored['Twin_pFF_Pred'] - df_censored['pFF']

    # Thresholds calculation In-Sample
    res_pce = np.abs(models['pce'].predict(X_train) - df_censored.loc[train_mask, 'PCE_Relative'])
    res_pff = np.abs(models['pff'].predict(X_train) - df_censored.loc[train_mask, 'pFF'])
    
    alert_percentile = RESIDUAL_ALERT_QUANTILE * 100
    thresholds = {
        'pce': max(float(np.percentile(res_pce, alert_percentile)) if len(res_pce) > 0 else MIN_PHYSICAL_MAE_PCE, MIN_PHYSICAL_MAE_PCE),
        'pff': max(float(np.percentile(res_pff, alert_percentile)) if len(res_pff) > 0 else MIN_PHYSICAL_MAE_PFF, MIN_PHYSICAL_MAE_PFF)
    }

    action_mask = df_censored['Exposure_Days'] <= burn_in_days
    df_censored['Alert_PCE'] = False
    df_censored['Alert_pFF'] = False
    df_censored.loc[action_mask, 'Alert_PCE'] = df_censored.loc[action_mask, 'Underperformance_PCE'] > thresholds['pce']
    df_censored.loc[action_mask, 'Alert_pFF'] = df_censored.loc[action_mask, 'Underperformance_pFF'] > thresholds['pff']
    df_censored['Digital_Twin_Alert'] = df_censored['Alert_PCE'] | df_censored['Alert_pFF']
    df_censored['In_Action_Window'] = action_mask

    return df_censored, thresholds, healthy_cells, models


def execute_loocv_validation(df_censored: pd.DataFrame, healthy_cells: list, burn_in_days: float = BURN_IN_DAYS) -> pd.DataFrame:
    """Perform Leave-One-Out Cross-Validation ensuring strict separation of Test Residuals."""
    loocv_results = []
    alert_percentile = RESIDUAL_ALERT_QUANTILE * 100

    for holdout_cell in healthy_cells:
        train_cells = [c for c in healthy_cells if c != holdout_cell]

        train_mask = (df_censored['cell_name'].isin(train_cells)) & (df_censored['Exposure_Days'] > burn_in_days)
        early_mask = (df_censored['cell_name'] == holdout_cell) & (df_censored['Exposure_Days'] <= burn_in_days)

        X_train = df_censored.loc[train_mask, FEATURES]
        y_train_pce = df_censored.loc[train_mask, 'PCE_Relative']
        y_train_pff = df_censored.loc[train_mask, 'pFF']

        X_early = df_censored.loc[early_mask, FEATURES]
        y_early_pce = df_censored.loc[early_mask, 'PCE_Relative']
        y_early_pff = df_censored.loc[early_mask, 'pFF']

        if len(X_train) == 0:
            continue

        model_pce = xgb.XGBRegressor(**XGB_PCE_PARAMS).fit(X_train, y_train_pce)
        model_pff = xgb.XGBRegressor(**XGB_PFF_PARAMS).fit(X_train, y_train_pff)

        # 1. THRESHOLD INTEGRITY: Calculated strictly on the IN-SAMPLE training data
        res_train_pce = np.abs(model_pce.predict(X_train) - y_train_pce)
        res_train_pff = np.abs(model_pff.predict(X_train) - y_train_pff)

        thr_pce = max(float(np.percentile(res_train_pce, alert_percentile)), MIN_PHYSICAL_MAE_PCE)
        thr_pff = max(float(np.percentile(res_train_pff, alert_percentile)), MIN_PHYSICAL_MAE_PFF)

        # 2. EVALUATION
        if len(X_early) > 0:
            underperf_pce = model_pce.predict(X_early) - np.asarray(y_early_pce)
            underperf_pff = model_pff.predict(X_early) - np.asarray(y_early_pff)
            
            alerts = (underperf_pce > thr_pce) | (underperf_pff > thr_pff)
            alert_pct = (alerts.sum() / len(alerts)) * 100.0
        else:
            alert_pct = 0.0

        loocv_results.append({
            'Holdout_Cell': holdout_cell,
            'Train_MAE_PCE': float(np.mean(res_train_pce)),
            'Train_MAE_pFF': float(np.mean(res_train_pff)),
            'Action_Window_Points': len(X_early),
            'Alert_Freq_Pct': float(alert_pct),
            'Validation_Status': 'PASS' if alert_pct <= ALERT_FREQUENCY_THRESHOLD_PCT else 'FAIL'
        })

    return pd.DataFrame(loocv_results).round(4)


# ==============================================================================
# DIAGNOSTICS & STATUS GENERATION
# ==============================================================================
def generate_diagnostic_summary(df_twin: pd.DataFrame, t80_metrics: pd.DataFrame, burn_in_days: float = BURN_IN_DAYS) -> pd.DataFrame:
    """Generate final status summary aggregating ML alerts and physical T80 limits."""
    df_action = df_twin[df_twin['Exposure_Days'] <= burn_in_days].copy().sort_values(by=['cell_name', 'Datetime'])
    df_action['Cum_Points'] = df_action.groupby('cell_name').cumcount().add(1)

    summary = df_action.groupby('cell_name').agg(
        Alert_Count=('Digital_Twin_Alert', 'sum'),
        Alert_PCE_Count=('Alert_PCE', 'sum'),
        Alert_pFF_Count=('Alert_pFF', 'sum'),
        Data_Points=('PCE', 'count')
    )
    
    summary['alert_freq_pct'] = (summary['Alert_Count'] / summary['Data_Points']) * 100.0
    summary['alert_pce_pct'] = (summary['Alert_PCE_Count'] / summary['Data_Points']) * 100.0
    summary['alert_pff_pct'] = (summary['Alert_pFF_Count'] / summary['Data_Points']) * 100.0

    summary = summary.merge(t80_metrics, left_index=True, right_index=True, how='left')

    summary['extrinsic_failure'] = (
        (summary['combined_survival_days'] <= burn_in_days) |
        (summary['alert_freq_pct'] > ALERT_FREQUENCY_THRESHOLD_PCT)
    )

    def get_diagnostic_status(row):
        if not row['extrinsic_failure']: 
            return "Healthy"
        
        t80_tags = [t for t, k in [("PCE", 'survival_days_pce'), ("pFF", 'survival_days_pff')] if row.get(k, float('inf')) <= burn_in_days]
        ml_tags = [t for t, k in [("PCE", 'alert_pce_pct'), ("pFF", 'alert_pff_pct')] if row.get(k, 0) > ALERT_FREQUENCY_THRESHOLD_PCT]

        parts = []
        if len(t80_tags) == 2: parts.append("T80(Dual)")
        elif t80_tags: parts.append(f"T80({t80_tags[0]})")

        if len(ml_tags) == 2: parts.append("ML(Dual)")
        elif ml_tags: parts.append(f"ML({ml_tags[0]})")
        elif row.get('alert_freq_pct', 0) > ALERT_FREQUENCY_THRESHOLD_PCT: parts.append("ML(Combined)")

        return " + ".join(parts) if parts else "Unknown"

    summary['Diagnostic_Status'] = summary.apply(get_diagnostic_status, axis=1)

    df_action['Cum_Alert_Pct'] = (df_action.groupby('cell_name')['Digital_Twin_Alert'].cumsum() / df_action['Cum_Points']) * 100.0
    valid_crossings = df_action[(df_action['Cum_Alert_Pct'] > ALERT_FREQUENCY_THRESHOLD_PCT) & (df_action['Cum_Points'] >= 100)]
    time_alert_day = valid_crossings.groupby('cell_name')['Exposure_Days'].min()
    summary['threshold_pct_day'] = np.where(summary['alert_freq_pct'] > ALERT_FREQUENCY_THRESHOLD_PCT, time_alert_day.reindex(summary.index).to_numpy(), np.nan)

    cols_to_keep = [
        'extrinsic_failure', 'Diagnostic_Status', 'alert_freq_pct', 'alert_pce_pct', 'alert_pff_pct',
        'combined_survival_days', 'threshold_pct_day',
        'survival_days_pce', 't80_failure_date_pce',
        'survival_days_pff', 't80_failure_date_pff',
        'combined_failure_date', 'PCE_initial', 'pFF_initial',
        'T80_threshold_PCE', 'T80_threshold_pFF'
    ]
    return summary[[c for c in cols_to_keep if c in summary.columns]].sort_values(by='alert_freq_pct', ascending=False)

# ==============================================================================
# PHASE 1: GRID SEARCH
# ==============================================================================
def run_burn_in_grid_search(
    df: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    windows: tuple[float, ...] = BURN_IN_GRID_WINDOWS,
) -> pd.DataFrame:
    """PHASE 1: Sensitivity analysis to empirically determine the end of the burn-in phase."""
    print("\n" + "=" * 80)
    print(" PHASE 1: BURN-IN GRID SEARCH (Sensitivity Analysis)")
    print("=" * 80)

    results = []
    for w in windows:
        # COHORTE DINÁMICA: Celdas que sobreviven más que la ventana candidata w
        cohort_w = t80_metrics[t80_metrics['combined_survival_days'] > w].index.tolist()
        
        if len(cohort_w) < 2:
            logger.warning(f"Window {w}d leaves too few healthy cells (N={len(cohort_w)}). Skipping.")
            continue

        print(f"\n>>> Evaluating candidate window: {w} days (Cohort N={len(cohort_w)}): {cohort_w}")
        try:
            df_twin, thresholds, _, _ = train_and_evaluate_censored_twin(df, t80_metrics, healthy_cells=cohort_w, burn_in_days=w)
            loocv_df = execute_loocv_validation(df_twin, cohort_w, burn_in_days=w)

            mean_mae_pce = loocv_df['Train_MAE_PCE'].mean() if 'Train_MAE_PCE' in loocv_df.columns else np.nan
            mean_mae_pff = loocv_df['Train_MAE_pFF'].mean() if 'Train_MAE_pFF' in loocv_df.columns else np.nan

            results.append({
                'Window_Days': w,
                'Healthy_Cells_N': len(cohort_w),
                'Q98_PCE_Raw': thresholds.get('pce', np.nan),
                'Q98_pFF_Raw': thresholds.get('pff', np.nan),
                'LOOCV_MAE_PCE': mean_mae_pce,
                'LOOCV_MAE_pFF': mean_mae_pff
            })
        except Exception as e:
            logger.error(f"Error evaluating window of {w} days: {e}")

    df_results = pd.DataFrame(results).round(4)
    print("\n[GRID SEARCH RESULTS - STABILIZATION (DYNAMIC COHORT)]")
    print(df_results.to_string(index=False))
    return df_results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    SURVIVAL_DIR = Path("data/survival/outdoor")
    ANOMALY_DIR = Path("data/anomaly/outdoor")
    DIAGNOSTICS_DIR = Path("data/anomaly/diagnostics/")
    
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df_final = pd.read_parquet(SURVIVAL_DIR / "survival_dataset.parquet")
        t80_metrics = pd.read_parquet(SURVIVAL_DIR / "t80_metrics_table.parquet")
    except FileNotFoundError:
        logger.error("Missing input files. Ensure t80_survival_tracker.py has been executed.")
        return

    # PHASE 1: GRID SEARCH
    grid_results = run_burn_in_grid_search(df_final, t80_metrics, windows=BURN_IN_GRID_WINDOWS)
    grid_parquet_path = DIAGNOSTICS_DIR / "burn_in_grid_search.parquet"
    grid_results.to_parquet(grid_parquet_path, engine='pyarrow')
    logger.info(f"Grid search results exported to: {grid_parquet_path.name}")

    # PHASE 2: PRODUCTION
    print(f"\nInitializing Phase 2 (Production) with optimal window: {BURN_IN_DAYS} days")

    # 1. Initial Screening Model
    df_twin_p1, _, screening_cohort, _ = train_and_evaluate_censored_twin(
        df_final, t80_metrics, burn_in_days=BURN_IN_DAYS
    )

    initial_summary = generate_diagnostic_summary(df_twin_p1, t80_metrics, burn_in_days=BURN_IN_DAYS)
    
    # 2. Extract cells that physically failed during burn-in
    failed_physical = initial_summary[initial_summary['combined_survival_days'] <= BURN_IN_DAYS].index.tolist()
    physical_survivors = [c for c in screening_cohort if c not in failed_physical]
    print(f"\n[PHYSICAL GATE] Evicted due to early T80 death: {failed_physical}")
    
    # 3. Leave-One-Out Cross-Validation on physical survivors to catch ML anomalies
    loocv_screening = execute_loocv_validation(df_twin_p1, physical_survivors, burn_in_days=BURN_IN_DAYS)
    failed_by_loocv = loocv_screening.loc[loocv_screening['Alert_Freq_Pct'] > ALERT_FREQUENCY_THRESHOLD_PCT, 'Holdout_Cell'].tolist()
    print(f"[LOOCV GATE] Evicted due to ML Anomalies: {failed_by_loocv}")
    
    # 4. Define final production cohort
    failed = list(set(failed_physical + failed_by_loocv))
    production_cohort = [c for c in screening_cohort if c not in failed]
    print(f"\n[FINAL GATE] Production cohort secured: {production_cohort}\n")

    # 5. Final Production Training
    df_twin_final, final_thresholds, _, dt_models_final = train_and_evaluate_censored_twin(
        df_final, t80_metrics, healthy_cells=production_cohort, burn_in_days=BURN_IN_DAYS
    )

    summary_table = generate_diagnostic_summary(df_twin_final, t80_metrics, burn_in_days=BURN_IN_DAYS)
    print("\n--- Final Diagnostic Summary ---")
    print(summary_table.to_string())

    # Save Artifacts for Dashboard integration
    joblib.dump({
        "summary_table": summary_table,
        "screening_cohort": screening_cohort,
        "gated_out_cells": failed,
        "healthy_cohort": production_cohort,
        "alert_thresholds": final_thresholds,
        "model_pce": dt_models_final['pce'],
        "model_pff": dt_models_final['pff']
    }, Path("data/anomaly/artifacts/early_failure_artifacts.joblib"))

    df_twin_final.to_parquet(ANOMALY_DIR / "anomaly_scored_dataset.parquet", engine='pyarrow')
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()