"""
Module: src/anomaly_detection.py
Description: Short-term predictive engine and thermodynamic anomaly detection framework.
Trains an XGBoost Digital Twin using a Two-Pass Pipeline (Screening -> LOOCV Gate -> Production)
exclusively on the mature phase (>14d) of healthy perovskite cells.
Applies right-censoring at the T80 degradation threshold (3-day streak) and extracts 
physical root-cause diagnostics utilizing Explainable AI (Surrogate Decision Trees).
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
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
logger = logging.getLogger("AnomalyDetection")


# =====================================================================
# 1. DIGITAL TWIN: TRAINING & INFERENCE PIPELINE
# =====================================================================
#
# BURN-IN CRITERION (14 DAYS):
#   - TRAINING: The model only learns from healthy cells with
#     Exposure_Days > burn_in_days ("mature" behavior).
#   - ACTION/ALERT: The model only generates alerts within the
#     Exposure_Days <= burn_in_days window (data the model NEVER
#     saw during training). This enables the detection of premature 
#     failures by comparing the actual initial performance of each cell 
#     against the baseline learned from the mature phase.
#
# DESIGN NOTE (Production): This function is agnostic to whether 
# 'healthy_cells' stems from the raw T80 criterion or a cohort already 
# refined by the LOOCV gate. Its internal logic remains strictly the same.
#
def train_and_evaluate_censored_twin(
    df: pd.DataFrame,
    healthy_cells: list | None = None,
    irradiance_threshold: float = 100.0,
    burn_in_days: float = 14.0
) -> tuple[pd.DataFrame, float, dict, dict, pd.DataFrame, list]:
    
    print("--- Initializing Digital Twin Training (T80 Right-Censored) ---")

    df_proc = df.copy()
    if df_proc.index.name == "Timestamp":
        df_proc = df_proc.reset_index()

    df_proc['Datetime'] = pd.to_datetime(df_proc['Timestamp'])
    df_proc['Date_Day'] = df_proc['Datetime'].dt.date
    df_proc['Day_Zero'] = df_proc.groupby('cell_name')['Datetime'].transform('min')
    df_proc['Exposure_Days'] = (df_proc['Datetime'] - df_proc['Day_Zero']).dt.total_seconds() / 86400.0

    df_daylight = df_proc[df_proc['POA_Irradiance_W_m2'] > irradiance_threshold].copy()

    # Daily series: peak performance and maximum exposure per cell/day
    df_daily = (
        df_daylight.groupby(['cell_name', 'Date_Day'])
        .agg(
            pseudo_FF_max=('pseudo_FF', 'max'), 
            Exposure_Days_max=('Exposure_Days', 'max'),
            Datetime_max=('Datetime', 'max') 
        )
        .reset_index()
        .sort_values(by=['cell_name', 'Date_Day'])
    )

    # Initial peak (first 3 days) -> reference point for the T80 threshold
    first_3_days = df_daily.groupby('cell_name').head(3)
    idx_max_initial = first_3_days.groupby('cell_name')['pseudo_FF_max'].idxmax()
    initial_peak = first_3_days.loc[idx_max_initial, ['cell_name', 'Exposure_Days_max', 'pseudo_FF_max']].rename(
        columns={'pseudo_FF_max': 'pFF_initial', 'Exposure_Days_max': 'Peak_Day'}
    )
    df_daily = df_daily.merge(initial_peak, on='cell_name', how='left')
    df_daily['T80_threshold'] = df_daily['pFF_initial'] * 0.80
    
    df_daily['Is_Below_T80'] = (
        (df_daily['Exposure_Days_max'] >= df_daily['Peak_Day']) &
        (df_daily['pseudo_FF_max'] < df_daily['T80_threshold'])
    ).astype(int)

    n_consecutive_days = 3
    survival_days = {}
    t80_dates = {}   

    for cell, group in df_daily.groupby('cell_name'):
        group = group.sort_values('Date_Day').reset_index(drop=True)
        group['Streak'] = group['Is_Below_T80'].rolling(window=n_consecutive_days, min_periods=n_consecutive_days).sum()

        if (group['Streak'] == n_consecutive_days).any():
            idx_confirmation = int((group['Streak'] == n_consecutive_days).idxmax())
            idx_streak_start = idx_confirmation - n_consecutive_days + 1
            survival_days[cell] = float(str(group.loc[idx_streak_start, 'Exposure_Days_max']))
            t80_dates[cell] = group.loc[idx_streak_start, 'Datetime_max']  
        else:
            survival_days[cell] = np.inf
            t80_dates[cell] = pd.NaT

    # Automatic healthy cohort designated BEFORE model interaction
    if healthy_cells is None:
        healthy_cells = [str(cell) for cell, days in survival_days.items() if days > burn_in_days]
        print(f"    Auto-detected healthy cohort (Survival_Days > {burn_in_days}): {healthy_cells}")

    # Right-censoring application
    df_daylight['Death_Day'] = df_daylight['cell_name'].map(survival_days)
    df_censored = (
        df_daylight[df_daylight['Exposure_Days'] <= df_daylight['Death_Day']]
        .drop(columns=['Death_Day'])
        .sort_values(by=['cell_name', 'Timestamp'])
        .reset_index(drop=True)
    )

    features = ['POA_Irradiance_W_m2', 'ModuleTemp_C', 'AbsoluteHumidity_g_m3']
    target = 'pseudo_FF'
    df_censored = df_censored.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target]).reset_index(drop=True)

    # Training logic: strictly limited to healthy cells AND mature phase data
    train_mask = (
        df_censored['cell_name'].isin(healthy_cells) &
        (df_censored['Exposure_Days'] > burn_in_days)
    )
    X_train, y_train = df_censored.loc[train_mask, features], df_censored.loc[train_mask, target]
    print(f"    Training rows (healthy, Exposure_Days > {burn_in_days}): {len(X_train)}")

    dt_model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42, n_jobs=-1
    )
    dt_model.fit(X_train, y_train)

    # Predictions mapped over the entire censored dataset
    df_censored['Twin_pFF_Pred'] = dt_model.predict(df_censored[features])
    df_censored['Underperformance'] = df_censored['Twin_pFF_Pred'] - df_censored[target]
    df_censored['Absolute_Residual'] = df_censored['Underperformance'].abs()

    base_mae = float(mean_absolute_error(y_train.to_numpy(), dt_model.predict(X_train)))
    alert_threshold = float(base_mae * 3.0)

    # Isolate alerts to the action window
    action_mask = df_censored['Exposure_Days'] <= burn_in_days
    df_censored['Digital_Twin_Alert'] = False
    df_censored.loc[action_mask, 'Digital_Twin_Alert'] = (
        df_censored.loc[action_mask, 'Underperformance'] > alert_threshold
    )
    df_censored['In_Action_Window'] = action_mask

    df_censored['T80_Threshold'] = df_censored['cell_name'].map(
        df_daily.drop_duplicates('cell_name').set_index('cell_name')['T80_threshold']
    )

    return df_censored, alert_threshold, survival_days, t80_dates, df_daily, healthy_cells


# =====================================================================
# 2. LOOCV (LEAVE-ONE-OUT CROSS-VALIDATION) FRAMEWORK
# =====================================================================
def execute_loocv_validation(
    df_censored: pd.DataFrame,
    healthy_cells: list,
    burn_in_days: float = 14.0,
    alert_freq_limit: float = 15.0
) -> pd.DataFrame:
    features = ['POA_Irradiance_W_m2', 'ModuleTemp_C', 'AbsoluteHumidity_g_m3']
    target = 'pseudo_FF'
    loocv_results = []

    for holdout_cell in healthy_cells:
        train_cells = [c for c in healthy_cells if c != holdout_cell]

        train_mask = (
            df_censored['cell_name'].isin(train_cells) &
            (df_censored['Exposure_Days'] > burn_in_days)
        )
        # Validation isolated to the holdout cell's action window
        test_mask = (
            (df_censored['cell_name'] == holdout_cell) &
            (df_censored['Exposure_Days'] <= burn_in_days)
        )

        X_train, y_train = df_censored.loc[train_mask, features], df_censored.loc[train_mask, target]
        X_test, y_test = df_censored.loc[test_mask, features], df_censored.loc[test_mask, target]

        if len(X_test) == 0:
            continue

        loocv_model = xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42, n_jobs=-1
        )
        loocv_model.fit(X_train, y_train)

        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)
        base_mae = float(mean_absolute_error(y_train_np, loocv_model.predict(X_train)))
        alert_threshold = float(base_mae * 3.0)

        test_preds = loocv_model.predict(X_test)
        holdout_mae = float(mean_absolute_error(y_test_np, test_preds))
        
        underperformance = test_preds - y_test_np
        alert_pct = (underperformance > alert_threshold).sum() / len(y_test_np) * 100.0

        loocv_results.append({
            'Holdout_Cell': holdout_cell,
            'Train_MAE': base_mae,
            'Holdout_MAE': holdout_mae,
            'Action_Window_Points': len(y_test_np),
            'Alert_Freq_Pct': float(alert_pct),
            'Validation_Status': 'PASS' if alert_pct <= alert_freq_limit else 'FAIL'
        })

    return pd.DataFrame(loocv_results).round(4)


# =====================================================================
# 3. DIAGNOSTIC RISK ASSESSMENT GENERATOR (STREAMLINED)
# =====================================================================
def generate_diagnostic_summary(
    df_twin: pd.DataFrame,
    survival_days: dict,
    t80_dates: dict, # NUEVO: Recibir t80_dates
    burn_in_days: float = 14.0,
    min_points_required: int = 100,
    early_transient_days: float = 3.0,
    alert_freq_limit: float = 15.0
) -> pd.DataFrame:
    
    df_action = df_twin[df_twin['Exposure_Days'] <= burn_in_days].copy()
    df_action = df_action.sort_values(by=['cell_name', 'Datetime'])

    df_action['Alert_Int'] = df_action['Digital_Twin_Alert'].astype(int)
    df_action['Cum_Alert_Count'] = df_action.groupby('cell_name')['Alert_Int'].cumsum()
    df_action['Cum_Points'] = df_action.groupby('cell_name').cumcount() + 1
    df_action['Cum_Alert_Pct'] = (df_action['Cum_Alert_Count'] / df_action['Cum_Points']) * 100.0

    valid_crossing = (
        (df_action['Cum_Alert_Pct'] > alert_freq_limit) &
        (df_action['Cum_Points'] >= min_points_required) &
        (df_action['Exposure_Days'] > early_transient_days)
    )
    
    time_15pct_day = df_action[valid_crossing].groupby('cell_name')['Exposure_Days'].min()
    time_15pct_date = df_action[valid_crossing].groupby('cell_name')['Datetime'].first() # NUEVO

    summary = df_action.groupby('cell_name').agg(
        Alert_Count=('Digital_Twin_Alert', 'sum'),
        Data_Points=('pseudo_FF', 'count')
    )
    
    summary['alert_freq_pct'] = (summary['Alert_Count'] / summary['Data_Points']) * 100.0
    summary['survival_days'] = [survival_days.get(str(cell), np.inf) for cell in summary.index]
    summary['t80_failure_date'] = [t80_dates.get(str(cell), pd.NaT) for cell in summary.index] # NUEVO

    early_collapse = summary['survival_days'] <= burn_in_days
    summary['extrinsic_failure'] = early_collapse | (summary['alert_freq_pct'] > alert_freq_limit)

    summary['threshold_15pct_day'] = np.where(summary['extrinsic_failure'], time_15pct_day.reindex(summary.index).to_numpy(), np.nan)
    no_alert_dates = pd.Series(pd.NaT, index=summary.index, dtype='datetime64[ns]').to_numpy()
    summary['ml_alert_date'] = np.where(
        summary['extrinsic_failure'],
        time_15pct_date.reindex(summary.index).to_numpy(),
        no_alert_dates
    )

    summary = summary.drop(columns=['Alert_Count', 'Data_Points']).sort_values(by='alert_freq_pct', ascending=False)
    
    numeric_cols = summary.select_dtypes(include=['float64', 'float32']).columns
    summary[numeric_cols] = summary[numeric_cols].round(3)
    
    return summary


# =====================================================================
# 4. LOOCV GATE — Filters the training cohort BEFORE production
# =====================================================================
def apply_loocv_gate(
    loocv_table: pd.DataFrame,
    healthy_cells: list,
    alert_freq_limit: float = 15.0
) -> tuple[list, list]:
    """
    Evicts any cell from the training cohort whose Alert_Freq_Pct 
    in the LOOCV evaluation exceeds the alert_freq_limit.
    """
    if loocv_table.empty:
        return list(healthy_cells), []

    failed = loocv_table.loc[
        loocv_table['Alert_Freq_Pct'] > alert_freq_limit, 'Holdout_Cell'
    ].tolist()
    gated_cohort = [c for c in healthy_cells if c not in failed]
    return gated_cohort, failed


# =====================================================================
# 5. PRODUCTION ORCHESTRATOR: SCREENING -> LOOCV GATE -> RETRAINING
# =====================================================================
def run_production_pipeline(
    df: pd.DataFrame,
    burn_in_days: float = 14.0,
    irradiance_threshold: float = 100.0,
    alert_freq_limit: float = 15.0,
    verbose: bool = True
) -> dict:
    
    def log(msg=""):
        if verbose: print(msg)

    # ---------------- PASS 1: SCREENING ----------------
    log("=" * 90)
    log("PASS 1 — SCREENING: T80 cohort + preliminary digital twin")
    log("=" * 90)

    df_twin_p1, _alert_thr_p1, survival_days, t80_dates_p1, df_daily, screening_cohort = train_and_evaluate_censored_twin(
        df=df, healthy_cells=None,
        irradiance_threshold=irradiance_threshold, burn_in_days=burn_in_days
    )
    log(f"Screening cohort (Survival_Days > {burn_in_days}): {screening_cohort}\n")

    log("-" * 90)
    log("LOOCV on the screening cohort (this acts as the entry GATE to production)")
    log("-" * 90)
    loocv_screening = execute_loocv_validation(
        df_twin_p1, screening_cohort, burn_in_days=burn_in_days, alert_freq_limit=alert_freq_limit
    )
    log(loocv_screening.to_string(index=False))

    # ---------------- GATE ----------------
    production_cohort, gated_out_cells = apply_loocv_gate(
        loocv_screening, screening_cohort, alert_freq_limit=alert_freq_limit
    )

    log("")
    log("-" * 90)
    if gated_out_cells:
        log(f"Cells EVICTED from training by LOOCV (FAIL, Alert_Freq_Pct > {alert_freq_limit}%): {gated_out_cells}")
    else:
        log("No cells were evicted: the entire screening cohort passed LOOCV.")
    log(f"PRODUCTION cohort (the final robust set used for retraining): {production_cohort}")
    log("-" * 90 + "\n")

    # ---------------- PASS 2: PRODUCTION ----------------
    log("=" * 90)
    log("PASS 2 — PRODUCTION: retraining with the refined cohort")
    log("=" * 90)
    df_twin_final, alert_threshold, survival_days_final, t80_dates_final, df_daily_final, _ = train_and_evaluate_censored_twin(
        df=df, healthy_cells=production_cohort,
        irradiance_threshold=irradiance_threshold, burn_in_days=burn_in_days
    )

    log("")
    log("-" * 90)
    log("Confirmation LOOCV on the production cohort (audit only, not a second gate)")
    log("-" * 90)
    loocv_production = execute_loocv_validation(
        df_twin_final, production_cohort, burn_in_days=burn_in_days, alert_freq_limit=alert_freq_limit
    )
    log(loocv_production.to_string(index=False))

    if not loocv_production.empty and (loocv_production['Validation_Status'] == 'FAIL').any():
        still_failing = loocv_production.loc[
            loocv_production['Validation_Status'] == 'FAIL', 'Holdout_Cell'
        ].tolist()
        log(f"\n⚠ Warning: after the gate, the following still fail LOOCV: {still_failing}.")
        log("  They are not automatically evicted in a second cycle (to avoid threshold instability);")
        log("  manual review is advised to determine if explicit exclusion is needed.")
    log("=" * 90 + "\n")

    # ---------------- FINAL DIAGNOSTICS ----------------
    summary_table = generate_diagnostic_summary(
        df_twin_final, survival_days_final, t80_dates_final, 
        burn_in_days=burn_in_days, alert_freq_limit=alert_freq_limit
    )

    validated_cells = summary_table[~summary_table['extrinsic_failure']].index.tolist()
    defective_cells = summary_table[summary_table['extrinsic_failure']].index.tolist()

    return {
        'df_twin': df_twin_final,
        'alert_threshold': alert_threshold,
        'survival_days': survival_days_final,
        'df_daily': df_daily_final,
        'screening_cohort': screening_cohort,
        'gated_out_cells': gated_out_cells,
        'production_cohort': production_cohort,
        'loocv_screening': loocv_screening,
        'loocv_production': loocv_production,
        'summary_table': summary_table,
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
    features = ['ModuleTemp_C', 'POA_Irradiance_W_m2', 'AbsoluteHumidity_g_m3']
    target = 'Digital_Twin_Alert'

    if window_data[target].nunique() < 2:
        logger.warning(f"[{cell_name}] Insufficient variance in anomaly classifications to train surrogate tree.")
        return None, None, None

    X, y = window_data[features], window_data[target].astype(int)
    surrogate_tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    surrogate_tree.fit(X, y)

    tree_rules = export_text(surrogate_tree, feature_names=features, decimals=1)
    
    print("\n" + "="*85)
    print(f" SURROGATE RULE EXTRACTION: [{cell_name}] (Critical Window up to Day {threshold_day + window_days:.1f})")
    print("-" * 85)
    print(tree_rules)
    print("="*85 + "\n")

    return surrogate_tree, features, tree_rules


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

        # Execute the 2-pass orchestrator
        results = run_production_pipeline(
            df=df_final,
            burn_in_days=BURN_IN_DAYS,
            irradiance_threshold=100.0,
            alert_freq_limit=ALERT_FREQ_LIMIT,
            verbose=True
        )

        summary_table = results['summary_table']
        df_twin_results = results['df_twin']
        alert_threshold = results['alert_threshold']

        print("\n" + "=" * 90)
        print("                                PREMATURE FAILURE RISK ASSESSMENT                                ")
        print("=" * 90)
        cols_to_show = ['extrinsic_failure', 'alert_freq_pct', 'threshold_15pct_day', 'survival_days']
        print(summary_table[cols_to_show].to_string())
        print("=" * 90)
        print(f"""
[DIAGNOSTIC METRICS LEGEND]
• Extrinsic_Failure   : True if cell suffers early T80 collapse (<={BURN_IN_DAYS:.0f} days burn-in) OR exhibits a
                        persistent thermodynamic anomaly (Alert_Freq_Pct > {ALERT_FREQ_LIMIT:.1f}%) DURING THE ACTION
                        WINDOW (Exposure_Days <= {BURN_IN_DAYS:.0f}), i.e. the data never seen during training.
• Alert_Freq_Pct      : Percentage of daylight observations, restricted to Exposure_Days <= {BURN_IN_DAYS:.0f},
                        where performance underperformed (>3x Base MAE) relative to the PRODUCTION model,
                        trained only on the LOOCV-gated healthy cohort's mature phase
                        (Exposure_Days > {BURN_IN_DAYS:.0f}).
• Threshold_15pct_Day : Exposure day (within the action window) when the cumulative alert rate crossed the
                        >{ALERT_FREQ_LIMIT:.0f}% threshold. Requires >=100 action-window data points to ensure stability.
• Survival_Days       : Exposure days to reach T80 structural collapse (inf = device remained healthy throughout).

[PRODUCTION COHORT TRACE]
• Screening cohort (T80 only):        {results['screening_cohort']}
• Expelled by LOOCV gate:             {results['gated_out_cells'] if results['gated_out_cells'] else 'none'}
• Production cohort (trained model):  {results['production_cohort']}
""" + "=" * 90 + "\n")

        # ---------------------------------------------------------------------
        # DATA EXPORT 1: Enriched Parquet Dataset
        # ---------------------------------------------------------------------
        logger.info("Enriching the original dataset architecture with dynamic anomaly metrics...")
        
        df_enriched_full = df_final.merge(
            df_twin_results[['cell_name', 'Timestamp', 'Twin_pFF_Pred', 'Underperformance', 'Digital_Twin_Alert', 'In_Action_Window']],
            on=['cell_name', 'Timestamp'], how='left'
        )

        df_enriched_full = df_enriched_full.merge(
            summary_table[['extrinsic_failure', 'threshold_15pct_day', 'survival_days']],
            left_on='cell_name', right_index=True, how='left'
        )
        df_enriched_full['Digital_Twin_Alert'] = df_enriched_full['Digital_Twin_Alert'].fillna(False)

        enriched_parquet_path = ANOMALY_DIR / "anomaly_scored_dataset.parquet"
        df_enriched_full.to_parquet(enriched_parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"Exported anomaly-scored telemetry matrix ({len(df_enriched_full)} rows) to: {enriched_parquet_path.name}")

        # ---------------------------------------------------------------------
        # XAI ROOT CAUSE DIAGNOSTICS (Rule Extraction Only)
        # ---------------------------------------------------------------------
        defective_cells = results['defective_cells']
        print(f"--> Extracting surrogate rules for identified defective devices: {defective_cells}\n")
        
        xai_diagnostics_dict = {}

        for cell_id in defective_cells:
            raw_t_day = summary_table.loc[cell_id, 'threshold_15pct_day']
            t_day = float(str(raw_t_day)) if pd.notna(raw_t_day) else np.nan
            
            # XAI Surrogate Rules extraction
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
        # DATA EXPORT 3: Joblib Artifacts for Streamlit Dashboard
        # ---------------------------------------------------------------------
        ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")
        ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            "summary_table": summary_table,
            "healthy_cohort": results['production_cohort'],
            "alert_threshold": alert_threshold
        }
        
        joblib.dump(artifacts, ARTIFACTS_PATH)
        logger.info(f"Exported Digital Twin brain artifacts to: {ARTIFACTS_PATH}")