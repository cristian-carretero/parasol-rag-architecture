"""
Module: src/07_jv_mppt_early_screening.py
Description: Infant mortality screening and LOOCV gate.
Phase 1: Executes an empirical Grid Search to determine the optimal burn-in window.
Phase 2: Uses the pre-calculated physical health (T80) to isolate the mature phase
of healthy cells and trains a Dual Digital Twin (PCE & pFF) for rapid anomaly detection.

Threshold policy
----------------
Alert thresholds are derived from OUT-OF-FOLD (OOF) absolute residuals of the
trained models, never from in-sample predictions. In-sample thresholds are
systematically optimistic (typically 2-3x tighter than OOF) and inflate the
alert-frequency signal that the gate depends on. OOF thresholds are a
principled approximation to the residuals the model would produce on new data.

Each threshold is the maximum of three layers:
  - alert level      = RESIDUAL_ALERT_QUANTILE of the OOF residuals
  - dynamic floor    = RESIDUAL_FLOOR_QUANTILE of the OOF residuals
  - absolute floor   = MIN_PHYSICAL_MAE_*_ABSOLUTE (last-resort guardrail)

Downstream contract (v3)
------------------------
The screening artifact exposes the cells that failed the gate split by the
mechanism that excluded them, so downstream consumers can reason about the
two failure modes independently:

  - gated_out_by_physical: cells whose combined survival <= BURN_IN_DAYS.
    These never entered the screening cohort in the first place.
  - gated_out_by_loocv:    cells that survived the physical gate but whose
    behavioural alert frequency exceeded ALERT_FREQUENCY_THRESHOLD_PCT.
  - gated_out_cells:       union of the two lists above (legacy key, still
    populated for backward compatibility with pre-v3 consumers).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

from src.config import (
    ALERT_FREQUENCY_THRESHOLD_PCT,
    BURN_IN_DAYS,
    BURN_IN_GRID_WINDOWS,
    DAYLIGHT_IRRADIANCE_MIN_W_M2,
    FEATURES,
    INITIAL_REF_FLOOR,          
    MIN_PHYSICAL_MAE_PCE_ABSOLUTE,
    MIN_PHYSICAL_MAE_PFF_ABSOLUTE,
    RESIDUAL_ALERT_QUANTILE,
    RESIDUAL_FLOOR_QUANTILE,
    XGB_PCE_PARAMS,
    XGB_PFF_PARAMS,
)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EarlyScreening")

# Schema version of the serialized artifact. Bump when the contract changes.
#  - v1: legacy, no schema_version field
#  - v2: added t80_target_metric, t80_target_days, gate_metric
#  - v3: split gated_out_cells into physical vs LOOCV categories (additive)
SCREENING_SCHEMA_VERSION = 3


# ==============================================================================
# STATISTICAL HELPERS
# ==============================================================================
def oof_abs_residuals(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Out-of-fold absolute residuals.

    Splits are contiguous (shuffle=False) because X is ordered by
    (cell_name, Exposure_Days). A shuffled split would leak future
    information from later days of the same cell into the training fold.

    Returns |y - yhat_oof|. If n < 2, returns [nan] so callers can guard on
    np.isnan().
    """
    n = len(X)
    if n < 2:
        return np.array([np.nan])
    n_splits_eff = max(2, min(n_splits, n))
    oof = np.empty(n, dtype=float)
    kf = KFold(n_splits=n_splits_eff, shuffle=False)
    for tr, te in kf.split(X):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    return np.abs(oof - y.values)


def _threshold_components(
    residuals: np.ndarray,
    absolute_floor: float,
) -> Dict[str, float]:
    """
    Return the three layers of the alert threshold separately, plus the
    final layered value.

    The layered threshold is max(alert, dynamic_floor, absolute_floor).
    Exposing the three components allows diagnostic artefacts to record
    which layer is active, rather than a single opaque number.
    """
    if residuals.size == 0 or np.isnan(residuals).all():
        return {
            'alert': np.nan,
            'dynamic_floor': np.nan,
            'absolute_floor': absolute_floor,
            'layered': absolute_floor,
        }
    alert_level = float(np.percentile(residuals, RESIDUAL_ALERT_QUANTILE * 100))
    dynamic_floor = float(np.percentile(residuals, RESIDUAL_FLOOR_QUANTILE * 100))
    return {
        'alert': alert_level,
        'dynamic_floor': dynamic_floor,
        'absolute_floor': absolute_floor,
        'layered': max(alert_level, dynamic_floor, absolute_floor),
    }


def _threshold(residuals: np.ndarray, absolute_floor: float) -> float:
    """
    Compute the alert threshold from OOF residuals.

    Layered decision: max(alert-level, dynamic-floor, absolute-floor).
    The alert level is the primary criterion; the two floors are guardrails
    that only activate when the model is unusually precise.
    """
    return _threshold_components(residuals, absolute_floor)['layered']


def _find_cohort_plateau(
    t80_metrics: pd.DataFrame,
    windows: tuple,
) -> Tuple[List[float], set]:
    """
    Find the longest consecutive run of burn-in windows that produce the
    same survivor cohort.

    The plateau is a structural property of the dataset, independent of any
    model or metric: it identifies the range of burn-in values beyond which
    adding more days does not change which cells survive. Within that range
    the choice of burn-in is not critical; outside of it, the cohort is
    still evolving and the choice would be arbitrary.

    Returns (plateau_windows, cohort_set).
    """
    if not windows:
        return [], set()

    cohorts = {
        w: set(t80_metrics[t80_metrics['combined_survival_days'] > w].index)
        for w in windows
    }
    sorted_w = sorted(windows)
    best_run: List[float] = []
    best_cohort: set = set()
    current_run: List[float] = [sorted_w[0]]
    current_cohort = cohorts[sorted_w[0]]

    for w in sorted_w[1:]:
        if cohorts[w] == current_cohort:
            current_run.append(w)
        else:
            if len(current_run) > len(best_run):
                best_run, best_cohort = current_run, current_cohort
            current_run = [w]
            current_cohort = cohorts[w]

    if len(current_run) > len(best_run):
        best_run, best_cohort = current_run, current_cohort

    return best_run, best_cohort


def _plateau_center(plateau: List[float]) -> float:
    """
    Geometric midpoint of the plateau, snapped to the nearest grid value.

    The geometric mean is preferred over the arithmetic mean because burn-in
    is a multiplicative quantity: the relative difference between 10 and 14
    is comparable to the relative difference between 14 and 21. Snapping to
    an actual grid value avoids reporting a burn-in that was never evaluated.
    """
    if not plateau:
        return float("nan")
    if len(plateau) == 1:
        return plateau[0]
    lo, hi = min(plateau), max(plateau)
    center = math.sqrt(lo * hi) if lo > 0 else (lo + hi) / 2
    return min(plateau, key=lambda w: abs(w - center))


# ==============================================================================
# CORE PIPELINE FUNCTIONS
# ==============================================================================
def preprocess_telemetry_data(
    df: pd.DataFrame,
    irradiance_threshold: float = DAYLIGHT_IRRADIANCE_MIN_W_M2,
) -> pd.DataFrame:
    """Standardize timestamps, calculate exposure days, and filter night-time data."""
    df_proc = df.reset_index() if df.index.name == "Timestamp" else df.copy()
    df_proc['Datetime'] = pd.to_datetime(df_proc['Timestamp'], utc=True)
    df_proc['Day_Zero'] = df_proc.groupby('cell_name')['Datetime'].transform('min')
    df_proc['Exposure_Days'] = (
        (df_proc['Datetime'] - df_proc['Day_Zero']).dt.total_seconds() / 86400.0
    )
    return df_proc[df_proc['POA_Irradiance_W_m2'] > irradiance_threshold].copy()


def train_and_evaluate_censored_twin(
    df: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    healthy_cells: Optional[List[str]] = None,
    irradiance_threshold: float = DAYLIGHT_IRRADIANCE_MIN_W_M2,
    burn_in_days: float = BURN_IN_DAYS,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str], Dict[str, Any]]:
    """Train the Dual Digital Twin (PCE & pFF) exclusively on the mature phase of healthy cells."""
    logger.info(f"Initializing Early Screening (Burn-in Gate: {burn_in_days} days)")

    df_daylight = preprocess_telemetry_data(df, irradiance_threshold)
    df_daylight = df_daylight.merge(
        t80_metrics[['PCE_initial', 'pFF_initial', 'combined_survival_days']],
        left_on='cell_name', right_index=True, how='inner',
    )

    if healthy_cells is None:
        healthy_cells = t80_metrics[
            t80_metrics['combined_survival_days'] > burn_in_days
        ].index.tolist()
        logger.info(f"Auto-detected healthy cohort: {healthy_cells}")

    df_censored = df_daylight[
        df_daylight['Exposure_Days'] <= df_daylight['combined_survival_days']
    ].copy()
    df_censored = df_censored.dropna(
        subset=FEATURES + ['PCE', 'pFF', 'PCE_initial', 'pFF_initial']
    ).reset_index(drop=True)

    df_censored['PCE_Relative'] = (
        df_censored['PCE'] / df_censored['PCE_initial'].clip(lower=INITIAL_REF_FLOOR)
    )
    df_censored['pFF_Relative'] = (
        df_censored['pFF'] / df_censored['pFF_initial'].clip(lower=INITIAL_REF_FLOOR)
    )

    train_mask = (
        df_censored['cell_name'].isin(healthy_cells)
        & (df_censored['Exposure_Days'] > burn_in_days)
    )
    X_train = df_censored.loc[train_mask, FEATURES]
    y_train_pce = df_censored.loc[train_mask, 'PCE_Relative']
    y_train_pff = df_censored.loc[train_mask, 'pFF_Relative'] 

    models = {
        'pce': xgb.XGBRegressor(**XGB_PCE_PARAMS).fit(X_train, y_train_pce),
        'pff': xgb.XGBRegressor(**XGB_PFF_PARAMS).fit(X_train, y_train_pff),
    }

    df_censored['Twin_PCE_Pred_Relative'] = models['pce'].predict(df_censored[FEATURES])
    df_censored['Twin_pFF_Pred_Relative'] = models['pff'].predict(df_censored[FEATURES])
    df_censored['Twin_PCE_Pred'] = (df_censored['Twin_PCE_Pred_Relative'] * df_censored['PCE_initial'])
    df_censored['Twin_pFF_Pred'] = (df_censored['Twin_pFF_Pred_Relative'] * df_censored['pFF_initial'])

    df_censored['Underperformance_PCE'] = (
        df_censored['Twin_PCE_Pred_Relative'] - df_censored['PCE_Relative']
    )
    df_censored['Underperformance_pFF'] = (
        df_censored['Twin_pFF_Pred_Relative'] - df_censored['pFF_Relative']
    )

    # Thresholds from OOF residuals: honest estimate of generalization error.
    res_pce = oof_abs_residuals(
        lambda: xgb.XGBRegressor(**XGB_PCE_PARAMS), X_train, y_train_pce,
    )
    res_pff = oof_abs_residuals(
        lambda: xgb.XGBRegressor(**XGB_PFF_PARAMS), X_train, y_train_pff,
    )

    thresholds = {
        'pce': _threshold(res_pce, MIN_PHYSICAL_MAE_PCE_ABSOLUTE),
        'pff': _threshold(res_pff, MIN_PHYSICAL_MAE_PFF_ABSOLUTE),
    }

    action_mask = df_censored['Exposure_Days'] <= burn_in_days
    df_censored['Alert_PCE'] = False
    df_censored['Alert_pFF'] = False
    df_censored.loc[action_mask, 'Alert_PCE'] = (
        df_censored.loc[action_mask, 'Underperformance_PCE'] > thresholds['pce']
    )
    df_censored.loc[action_mask, 'Alert_pFF'] = (
        df_censored.loc[action_mask, 'Underperformance_pFF'] > thresholds['pff']
    )
    df_censored['Digital_Twin_Alert'] = (
        df_censored['Alert_PCE'] | df_censored['Alert_pFF']
    )
    df_censored['In_Action_Window'] = action_mask

    return df_censored, thresholds, healthy_cells, models


def execute_loocv_validation(
    df_censored: pd.DataFrame,
    healthy_cells: list,
    burn_in_days: float = BURN_IN_DAYS,
) -> pd.DataFrame:
    """Perform Leave-One-Out Cross-Validation ensuring strict separation of Test Residuals."""
    loocv_results = []

    for holdout_cell in healthy_cells:
        train_cells = [c for c in healthy_cells if c != holdout_cell]

        train_mask = (
            df_censored['cell_name'].isin(train_cells)
            & (df_censored['Exposure_Days'] > burn_in_days)
        )
        early_mask = (
            (df_censored['cell_name'] == holdout_cell)
            & (df_censored['Exposure_Days'] <= burn_in_days)
        )

        X_train = df_censored.loc[train_mask, FEATURES]
        y_train_pce = df_censored.loc[train_mask, 'PCE_Relative']
        y_train_pff = df_censored.loc[train_mask, 'pFF_Relative']

        X_early = df_censored.loc[early_mask, FEATURES]
        y_early_pce = df_censored.loc[early_mask, 'PCE_Relative']
        y_early_pff = df_censored.loc[early_mask, 'pFF_Relative']

        if len(X_train) == 0:
            continue

        model_pce = xgb.XGBRegressor(**XGB_PCE_PARAMS).fit(X_train, y_train_pce)
        model_pff = xgb.XGBRegressor(**XGB_PFF_PARAMS).fit(X_train, y_train_pff)

        # Thresholds derived from OOF residuals of the fold's training cells,
        # never from the holdout. See oof_abs_residuals() for the rationale.
        res_train_pce = oof_abs_residuals(
            lambda: xgb.XGBRegressor(**XGB_PCE_PARAMS), X_train, y_train_pce,
        )
        res_train_pff = oof_abs_residuals(
            lambda: xgb.XGBRegressor(**XGB_PFF_PARAMS), X_train, y_train_pff,
        )

        thr_pce = _threshold(res_train_pce, MIN_PHYSICAL_MAE_PCE_ABSOLUTE)
        thr_pff = _threshold(res_train_pff, MIN_PHYSICAL_MAE_PFF_ABSOLUTE)

        # 2. EVALUATION
        #
        # Three-state validation status:
        #   - PASS    : the holdout has data and its alert frequency is below
        #               ALERT_FREQUENCY_THRESHOLD_PCT.
        #   - FAIL    : the holdout has data and its alert frequency exceeds
        #               ALERT_FREQUENCY_THRESHOLD_PCT.
        #   - NO_DATA : the holdout has zero points in the action window. The
        #               gate cannot determine whether the cell is healthy or
        #               anomalous. The fold is kept in the raw record for
        #               traceability, but is excluded from aggregates and does
        #               NOT silently count as PASS. Downstream consumers must
        #               treat these as indeterminate and review manually.
        if len(X_early) > 0:
            underperf_pce = model_pce.predict(X_early) - np.asarray(y_early_pce)
            underperf_pff = model_pff.predict(X_early) - np.asarray(y_early_pff)

            alerts = (underperf_pce > thr_pce) | (underperf_pff > thr_pff)
            alert_pct = (alerts.sum() / len(alerts)) * 100.0

            # Holdout absolute errors, kept as lists so that the caller can
            # pool them across folds to obtain a single global MAE and SE.
            abs_errors_pce = np.abs(underperf_pce).tolist()
            abs_errors_pff = np.abs(underperf_pff).tolist()

            validation_status = (
                'PASS' if alert_pct <= ALERT_FREQUENCY_THRESHOLD_PCT else 'FAIL'
            )
        else:
            alert_pct = float('nan')
            abs_errors_pce = []
            abs_errors_pff = []
            validation_status = 'NO_DATA'

        loocv_results.append({
            'Holdout_Cell': holdout_cell,
            'OOF_MAE_PCE': float(np.mean(res_train_pce)),
            'OOF_MAE_pFF': float(np.mean(res_train_pff)),
            'Fold_MAE_PCE': float(np.mean(abs_errors_pce)) if abs_errors_pce else np.nan,
            'Fold_MAE_PFF': float(np.mean(abs_errors_pff)) if abs_errors_pff else np.nan,
            'Action_Window_Points': len(X_early),
            'Alert_Freq_Pct': float(alert_pct),
            'Validation_Status': validation_status,
            'Abs_Errors_PCE': abs_errors_pce,
            'Abs_Errors_PFF': abs_errors_pff,
        })

    df = pd.DataFrame(loocv_results)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


# ==============================================================================
# DIAGNOSTICS & STATUS GENERATION
# ==============================================================================
def generate_diagnostic_summary(
    df_twin: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    burn_in_days: float = BURN_IN_DAYS,
) -> pd.DataFrame:
    """Generate final status summary aggregating ML alerts and physical T80 limits."""
    df_action = df_twin[df_twin['Exposure_Days'] <= burn_in_days].copy().sort_values(
        by=['cell_name', 'Datetime']
    )
    df_action['Cum_Points'] = df_action.groupby('cell_name').cumcount().add(1)

    summary = df_action.groupby('cell_name').agg(
        Alert_Count=('Digital_Twin_Alert', 'sum'),
        Alert_PCE_Count=('Alert_PCE', 'sum'),
        Alert_pFF_Count=('Alert_pFF', 'sum'),
        Data_Points=('PCE', 'count'),
    )

    summary['alert_freq_pct'] = (summary['Alert_Count'] / summary['Data_Points']) * 100.0
    summary['alert_pce_pct'] = (summary['Alert_PCE_Count'] / summary['Data_Points']) * 100.0
    summary['alert_pff_pct'] = (summary['Alert_pFF_Count'] / summary['Data_Points']) * 100.0

    summary = summary.merge(t80_metrics, left_index=True, right_index=True, how='left')

    summary['extrinsic_failure'] = (
        (summary['combined_survival_days'] <= burn_in_days)
        | (summary['alert_freq_pct'] > ALERT_FREQUENCY_THRESHOLD_PCT)
    )

    def get_diagnostic_status(row):
        if not row['extrinsic_failure']:
            return "Healthy"

        t80_tags = [
            t for t, k in [("PCE", 'survival_days_pce'), ("pFF", 'survival_days_pff')]
            if row.get(k, float('inf')) <= burn_in_days
        ]
        ml_tags = [
            t for t, k in [("PCE", 'alert_pce_pct'), ("pFF", 'alert_pff_pct')]
            if row.get(k, 0) > ALERT_FREQUENCY_THRESHOLD_PCT
        ]

        parts = []
        if len(t80_tags) == 2:
            parts.append("T80(Dual)")
        elif t80_tags:
            parts.append(f"T80({t80_tags[0]})")

        if len(ml_tags) == 2:
            parts.append("ML(Dual)")
        elif ml_tags:
            parts.append(f"ML({ml_tags[0]})")
        elif row.get('alert_freq_pct', 0) > ALERT_FREQUENCY_THRESHOLD_PCT:
            parts.append("ML(Combined)")

        return " + ".join(parts) if parts else "Unknown"

    summary['Diagnostic_Status'] = summary.apply(get_diagnostic_status, axis=1)

    df_action['Cum_Alert_Pct'] = (
        df_action.groupby('cell_name')['Digital_Twin_Alert'].cumsum()
        / df_action['Cum_Points']
    ) * 100.0
    valid_crossings = df_action[
        (df_action['Cum_Alert_Pct'] > ALERT_FREQUENCY_THRESHOLD_PCT)
        & (df_action['Cum_Points'] >= 100)
    ]
    time_alert_day = valid_crossings.groupby('cell_name')['Exposure_Days'].min()
    summary['threshold_pct_day'] = np.where(
        summary['alert_freq_pct'] > ALERT_FREQUENCY_THRESHOLD_PCT,
        time_alert_day.reindex(summary.index).to_numpy(),
        np.nan,
    )

    cols_to_keep = [
        'extrinsic_failure', 'Diagnostic_Status', 'alert_freq_pct',
        'alert_pce_pct', 'alert_pff_pct',
        'combined_survival_days', 'threshold_pct_day',
        'survival_days_pce', 't80_failure_date_pce',
        'survival_days_pff', 't80_failure_date_pff',
        'combined_failure_date', 'PCE_initial', 'pFF_initial',
        'T80_threshold_PCE', 'T80_threshold_pFF',
    ]
    return summary[[c for c in cols_to_keep if c in summary.columns]].sort_values(
        by='alert_freq_pct', ascending=False
    )


# ==============================================================================
# PHASE 1: GRID SEARCH
# ==============================================================================
def _empty_grid_row(w: float, cohort_size: int) -> Dict[str, Any]:
    """
    Produce a fully-NaN grid row for a candidate window that could not be
    evaluated (either too few surviving cells or an unexpected error during
    the evaluation).

    Rationale: the plateau detection downstream uses the full window list.
    If the grid search silently drops a window from `df_results`, the
    printed table and the plateau calculation can disagree without warning.
    Emitting an explicit NaN row keeps the two artefacts consistent.
    """
    return {
        'Window_Days': w,
        'Healthy_Cells_N': cohort_size,
        'N_Folds_PCE': 0,
        'N_Folds_pFF': 0,
        'Threshold_PCE_Layered': np.nan,
        'Threshold_pFF_Layered': np.nan,
        'MAE_PCE': np.nan,
        'SE_PCE': np.nan,
        'MAE_pFF': np.nan,
        'SE_pFF': np.nan,
    }


def run_burn_in_grid_search(
    df: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    windows: tuple = BURN_IN_GRID_WINDOWS,
) -> pd.DataFrame:
    """
    Phase 1: identify the structural cohort plateau of the burn-in gate.

    The winner is NOT chosen by MAE. MAE is reported as validation only.
    The selection rule is:
        1. Compute the survivor cohort for every candidate W.
        2. Find the longest run of consecutive W with the same cohort.
        3. Select the geometric midpoint of that plateau.
    This avoids circularity (W chosen by a metric that depends on W) and
    the small-cohort artifact (tiny cohorts trivially achieve low MAE
    because they exclude the difficult cells).

    The threshold columns record the LAYERED threshold for each metric
    (i.e. max of alert level, dynamic floor, absolute floor), not the raw
    alert quantile. To inspect the three layers separately, use
    `_threshold_components` on the OOF residuals of the corresponding W.
    """
    print("\n" + "=" * 80)
    print(" PHASE 1: BURN-IN GRID SEARCH (Structural Cohort Plateau)")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    for w in windows:
        cohort_w = t80_metrics[t80_metrics['combined_survival_days'] > w].index.tolist()

        if len(cohort_w) < 2:
            logger.warning(
                f"Window {w}d leaves too few healthy cells (N={len(cohort_w)}). Skipping."
            )
            results.append(_empty_grid_row(w, len(cohort_w)))
            continue

        print(f"\n>>> Evaluating candidate window: {w} days (Cohort N={len(cohort_w)}): {cohort_w}")
        try:
            df_twin, thresholds, _, _ = train_and_evaluate_censored_twin(
                df, t80_metrics, healthy_cells=cohort_w, burn_in_days=w,
            )
            loocv_df = execute_loocv_validation(df_twin, cohort_w, burn_in_days=w)

            # MAE pooled from holdout errors is reported for VALIDATION only.
            # It is NOT the selection criterion.
            #
            # NO_DATA folds produce Fold_MAE_PCE = NaN, which dropna() removes
            # before the mean. This keeps indeterminate folds out of the
            # aggregate without contaminating the statistic.
            per_fold_pce = loocv_df['Fold_MAE_PCE'].dropna().to_numpy()
            per_fold_pff = loocv_df['Fold_MAE_PFF'].dropna().to_numpy()

            n_folds_pce, n_folds_pff = len(per_fold_pce), len(per_fold_pff)
            # MAE reported as the mean of per-fold MAEs (mean-of-means), so
            # every cell contributes equally regardless of how many action-
            # window points it has. This is the honest generalization metric.
            mae_pce = float(np.mean(per_fold_pce)) if n_folds_pce > 0 else np.nan
            mae_pff = float(np.mean(per_fold_pff)) if n_folds_pff > 0 else np.nan
            # SE across folds (cells), not across points. Points within a
            # cell are not independent, so the point-level SE would under-
            # estimate the true uncertainty by 1-2 orders of magnitude.
            se_pce = (
                float(np.std(per_fold_pce, ddof=1) / np.sqrt(n_folds_pce))
                if n_folds_pce > 1 else np.nan
            )
            se_pff = (
                float(np.std(per_fold_pff, ddof=1) / np.sqrt(n_folds_pff))
                if n_folds_pff > 1 else np.nan
            )

            results.append({
                'Window_Days': w,
                'Healthy_Cells_N': len(cohort_w),
                'N_Folds_PCE': n_folds_pce,
                'N_Folds_pFF': n_folds_pff,
                'Threshold_PCE_Layered': thresholds.get('pce', np.nan),
                'Threshold_pFF_Layered': thresholds.get('pff', np.nan),
                'MAE_PCE': mae_pce,
                'SE_PCE': se_pce,
                'MAE_pFF': mae_pff,
                'SE_pFF': se_pff,
            })
        except Exception as e:
            logger.error(f"Error evaluating window of {w} days: {e}")
            # Emit a NaN row so the grid and the plateau stay aligned.
            results.append(_empty_grid_row(w, len(cohort_w)))

    df_results = pd.DataFrame(results).round(4)

    if df_results.empty:
        logger.error("No candidate window produced a viable cohort. Aborting.")
        return df_results

    # --- Structural selection ---------------------------------------------
    plateau, plateau_cohort = _find_cohort_plateau(t80_metrics, windows)
    selected_w = _plateau_center(plateau)

    df_results['In_Plateau'] = df_results['Window_Days'].isin(plateau)
    df_results['Is_Selected'] = df_results['Window_Days'] == selected_w
    df_results = df_results.sort_values('Window_Days').reset_index(drop=True)

    print("\n[GRID SEARCH RESULTS]")
    print(df_results.to_string(index=False))

    print(
        f"\n[PLATEAU] W ∈ {plateau} "
        f"(structural selection, N={len(plateau_cohort)} cells)"
    )
    if df_results['Is_Selected'].any():
        sel = df_results[df_results['Is_Selected']].iloc[0]
        print(
            f"[SELECTED] W={selected_w:.0f}d — geometric midpoint of the "
            f"plateau. MAE_PCE={sel['MAE_PCE']:.4f} "
            f"(reported for validation, not used for selection)."
        )
    return df_results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    from src.config import (
        FILE_MERGED_FEATURES,
        FILE_T80_TRUTH,
        FILE_HEALTHY_COHORT,
        FILE_SCREENING_ARTIFACTS,
    )

    FILE_HEALTHY_COHORT.parent.mkdir(parents=True, exist_ok=True)
    FILE_SCREENING_ARTIFACTS.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_final = pd.read_parquet(FILE_MERGED_FEATURES)
        t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    except FileNotFoundError:
        logger.error("Missing input files. Ensure 06_jv_mppt_t80_tracker.py has been executed.")
        return

    # PHASE 1: GRID SEARCH
    grid_results = run_burn_in_grid_search(df_final, t80_metrics, windows=BURN_IN_GRID_WINDOWS)
    grid_parquet_path = FILE_SCREENING_ARTIFACTS.parent / "07_burn_in_grid_search.parquet"
    grid_results.to_parquet(grid_parquet_path, engine='pyarrow')
    logger.info(f"Grid search results exported to: {grid_parquet_path.name}")

    if not grid_results.empty and grid_results['Is_Selected'].any():
        selected_w = float(
            grid_results.loc[grid_results['Is_Selected'], 'Window_Days'].iloc[0]
        )
        if abs(selected_w - BURN_IN_DAYS) > 1e-6:
            logger.warning(
                f"Structural plateau selects W={selected_w:.0f}d, but "
                f"BURN_IN_DAYS={BURN_IN_DAYS:.0f}d. Update config.py to "
                f"align the pipeline with the empirical evidence."
            )

    # PHASE 2: PRODUCTION
    print(f"\nInitializing Phase 2 (Production) with optimal window: {BURN_IN_DAYS} days")

    # 1. Initial Screening Model
    df_twin_p1, _, screening_cohort, _ = train_and_evaluate_censored_twin(
        df_final, t80_metrics, burn_in_days=BURN_IN_DAYS,
    )

    initial_summary = generate_diagnostic_summary(
        df_twin_p1, t80_metrics, burn_in_days=BURN_IN_DAYS,
    )

    # 2. Extract cells that physically failed during burn-in.
    #
    # Note: the physical gate is already applied inside
    # train_and_evaluate_censored_twin when healthy_cells=None (auto-detection
    # keeps only cells with combined_survival_days > burn_in_days). So by
    # construction, no cell in `failed_physical` can appear in
    # `screening_cohort`, and `physical_survivors` is exactly `screening_cohort`.
    #
    # The explicit filter below is kept as a defensive check: if the upstream
    # contract ever changes (e.g. auto-detection switched off), the cohort is
    # still guaranteed to exclude early-death cells.
    failed_physical = initial_summary[
        initial_summary['combined_survival_days'] <= BURN_IN_DAYS
    ].index.tolist()
    physical_survivors = [c for c in screening_cohort if c not in failed_physical]
    print(f"\n[PHYSICAL GATE] Evicted due to early T80 death: {failed_physical}")

    # 3. Leave-One-Out Cross-Validation on physical survivors to catch ML anomalies
    loocv_screening = execute_loocv_validation(
        df_twin_p1, physical_survivors, burn_in_days=BURN_IN_DAYS,
    )

    # Explicit FAIL: the cell has data and exceeds the alert-frequency threshold.
    failed_by_loocv = loocv_screening.loc[
        loocv_screening['Validation_Status'] == 'FAIL',
        'Holdout_Cell',
    ].tolist()

    # NO_DATA: the cell has no points in the action window. Its status is
    # indeterminate: we cannot decide whether it is healthy or anomalous.
    # These cells are NOT evicted automatically, but must be reviewed.
    no_data_cells = loocv_screening.loc[
        loocv_screening['Validation_Status'] == 'NO_DATA',
        'Holdout_Cell',
    ].tolist()

    print(f"[LOOCV GATE] Evicted due to ML Anomalies: {failed_by_loocv}")
    if no_data_cells:
        logger.warning(
            f"[LOOCV GATE] Cells with NO_DATA in the action window: {no_data_cells}. "
            f"These cells have indeterminate validation status and are NOT evicted "
            f"by the gate. Review manually before including in production."
        )

    # 4. Define final production cohort
    failed = list(set(failed_physical + failed_by_loocv))
    production_cohort = [c for c in screening_cohort if c not in failed]
    print(f"\n[FINAL GATE] Production cohort secured: {production_cohort}\n")

    # 5. Final Production Training
    df_twin_final, final_thresholds, _, dt_models_final = train_and_evaluate_censored_twin(
        df_final, t80_metrics, healthy_cells=production_cohort, burn_in_days=BURN_IN_DAYS,
    )

    summary_table = generate_diagnostic_summary(
        df_twin_final, t80_metrics, burn_in_days=BURN_IN_DAYS,
    )
    print("\n--- Final Diagnostic Summary ---")
    print(summary_table.to_string())

    # Save Artifacts for Dashboard integration
    joblib.dump({
        "summary_table": summary_table,
        "screening_cohort": screening_cohort,
        # Legacy key kept for pre-v3 consumers. Semantically equals the union
        # of the two split lists below.
        "gated_out_cells": failed,
        # v3: explicit split of the gate failure modes.
        "gated_out_by_physical": failed_physical,
        "gated_out_by_loocv": failed_by_loocv,
        "healthy_cohort": production_cohort,
        "alert_thresholds": final_thresholds,
        "model_pce": dt_models_final['pce'],
        "model_pff": dt_models_final['pff'],
        # ---- Downstream contract (v3) --------------------------------
        # t80_target_metric declares WHICH survival column downstream
        # modules must use as ground truth for RUL forecasting. The gate
        # uses 'combined_survival_days' for conservatism, but the RUL
        # engine models physical PCE damage, so its target is PCE-based.
        # t80_target_days is the pre-computed dict {cell: days} so that
        # downstream modules do not have to re-derive it from t80_metrics.
        "schema_version": SCREENING_SCHEMA_VERSION,
        "t80_target_metric": "survival_days_pce",
        "t80_target_days": summary_table["survival_days_pce"].to_dict(),
        "gate_metric": "combined_survival_days",
    }, FILE_SCREENING_ARTIFACTS)

    df_twin_final.to_parquet(FILE_HEALTHY_COHORT, engine='pyarrow')
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()