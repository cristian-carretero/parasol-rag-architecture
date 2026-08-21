"""
Module: src/anomaly_detection.py
Description: Short-term predictive engine and anomaly detection framework.
Trains an XGBoost Digital Twin on automatically selected healthy perovskite cells.
Applies right-censoring at T80 and extracts physical root-cause diagnostics 
using Explainable AI (Surrogate Decision Trees).
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from viz_anomaly import plot_censored_digital_twin, plot_surrogate_diagnostics

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AnomalyDetection")

# =====================================================================
# 1. DIGITAL TWIN: TRAINING & INFERENCE PIPELINE
# =====================================================================

def train_and_evaluate_censored_twin(
    df: pd.DataFrame,
    irradiance_threshold: float = 100.0,
    burn_in_days: int = 14
) -> tuple[pd.DataFrame, float, dict, pd.DataFrame, list]:
    
    logger.info("Initializing Digital Twin Training (T80 Right-Censored)...")

    # 1.1 Temporal Feature Engineering
    df_proc = df.copy()
    df_proc['Datetime'] = pd.to_datetime(df_proc['Timestamp'])
    df_proc['Date_Day'] = df_proc['Datetime'].dt.date
    df_proc['Day_Zero'] = df_proc.groupby('cell_name')['Datetime'].transform('min')
    df_proc['Exposure_Days'] = (df_proc['Datetime'] - df_proc['Day_Zero']).dt.total_seconds() / 86400.0

    # 1.2 Daytime Pre-filtering
    df_daylight = df_proc[df_proc['POA_Irradiance_W_m2'] > irradiance_threshold].copy()

    # 1.3 Daily Max Aggregation & T80 Threshold Determination
    df_daily = (
        df_daylight.groupby(['cell_name', 'Date_Day'])
        .agg(pseudo_FF_max=('pseudo_FF', 'max'), Exposure_Days_max=('Exposure_Days', 'max'))
        .reset_index()
        .sort_values(by=['cell_name', 'Date_Day'])
    )

    first_3_days = df_daily.groupby('cell_name').head(3)
    idx_max_initial = first_3_days.groupby('cell_name')['pseudo_FF_max'].idxmax()

    initial_peak_info = first_3_days.loc[idx_max_initial, ['cell_name', 'Exposure_Days_max', 'pseudo_FF_max']].rename(
        columns={'pseudo_FF_max': 'pFF_initial', 'Exposure_Days_max': 'Peak_Day'}
    )

    df_daily = df_daily.merge(initial_peak_info, on='cell_name', how='left')
    df_daily['T80_threshold'] = df_daily['pFF_initial'] * 0.80

    df_daily['Event'] = (
        (df_daily['Exposure_Days_max'] >= df_daily['Peak_Day']) &
        (df_daily['pseudo_FF_max'] < df_daily['T80_threshold'])
    ).astype(int)

    death_days = {}
    for cell, group in df_daily.groupby('cell_name'):
        if group['Event'].sum() > 0:
            first_failure_idx = group['Event'].idxmax()
            death_days[cell] = group.loc[first_failure_idx, 'Exposure_Days_max']
        else:
            death_days[cell] = np.inf

    # Automated selection of the healthy baseline cohort
    healthy_cells = [cell for cell, t_death in death_days.items() if t_death > burn_in_days]
    logger.info(f"Auto-detected Healthy Cohort (Survival > {burn_in_days} days): {healthy_cells}")

    # 1.4 Vectorized Right-Censoring
    df_daylight['Death_Day'] = df_daylight['cell_name'].map(death_days)
    df_censored = (
        df_daylight[df_daylight['Exposure_Days'] <= df_daylight['Death_Day']]
        .drop(columns=['Death_Day'])
        .sort_values(by=['cell_name', 'Timestamp'])
        .reset_index(drop=True)
    )

    features = ['POA_Irradiance_W_m2', 'ModuleTemp_C', 'AbsoluteHumidity_g_m3']
    target = 'pseudo_FF'

    # Sanitize data to prevent XGBoost errors with NaNs or Infinities
    df_censored = df_censored.replace([np.inf, -np.inf], np.nan)
    df_censored = df_censored.dropna(subset=features + [target]).reset_index(drop=True)

    # 1.5 Model Training on the Selected Baseline Cohort
    train_mask = df_censored['cell_name'].isin(healthy_cells)
    X_train, y_train = df_censored.loc[train_mask, features], df_censored.loc[train_mask, target]

    dt_model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42, n_jobs=-1
    )
    dt_model.fit(X_train, y_train)

    # 1.6 Inference & Directional Anomaly Quantification
    df_censored['Twin_pFF_Pred'] = dt_model.predict(df_censored[features])
    df_censored['Underperformance'] = df_censored['Twin_pFF_Pred'] - df_censored[target]
    df_censored['Absolute_Residual'] = df_censored['Underperformance'].abs()

    base_mae = mean_absolute_error(y_train, dt_model.predict(X_train))
    alert_threshold = float(base_mae * 3)
    df_censored['Digital_Twin_Alert'] = df_censored['Underperformance'] > alert_threshold

    t80_lookup = df_daily.drop_duplicates('cell_name').set_index('cell_name')['T80_threshold'].to_dict()
    df_censored['T80_Threshold'] = df_censored['cell_name'].map(t80_lookup)

    return df_censored, alert_threshold, death_days, df_daily, healthy_cells


# =====================================================================
# 2. LOOCV (LEAVE-ONE-OUT CROSS-VALIDATION) FRAMEWORK
# =====================================================================

def execute_loocv_validation(df_censored: pd.DataFrame, healthy_cells: list) -> pd.DataFrame:
    features = ['POA_Irradiance_W_m2', 'ModuleTemp_C', 'AbsoluteHumidity_g_m3']
    target = 'pseudo_FF'
    loocv_results = []

    for holdout_cell in healthy_cells:
        train_cells = [c for c in healthy_cells if c != holdout_cell]

        train_mask = df_censored['cell_name'].isin(train_cells)
        test_mask = df_censored['cell_name'] == holdout_cell

        X_train, y_train = df_censored.loc[train_mask, features], df_censored.loc[train_mask, target]
        X_test, y_test = df_censored.loc[test_mask, features], df_censored.loc[test_mask, target]

        loocv_model = xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42, n_jobs=-1
        )
        loocv_model.fit(X_train, y_train)

        train_preds = loocv_model.predict(X_train)
        base_mae = mean_absolute_error(y_train, train_preds)
        alert_threshold = float(base_mae * 3)

        test_preds = loocv_model.predict(X_test)
        holdout_mae = mean_absolute_error(y_test, test_preds)

        underperformance = test_preds - y_test

        alert_count = (underperformance > alert_threshold).sum()
        alert_pct = (alert_count / len(y_test)) * 100.0 if len(y_test) > 0 else 0.0

        loocv_results.append({
            'Holdout_Cell': holdout_cell,
            'Train_MAE': base_mae,
            'Holdout_MAE': holdout_mae,
            'Alert_Freq_Pct': alert_pct,
            'Validation_Status': 'PASS' if alert_pct <= 15.0 else 'FAIL'
        })

    return pd.DataFrame(loocv_results).round(4)


# =====================================================================
# 3. DIAGNOSTIC RISK ASSESSMENT GENERATOR
# =====================================================================

def generate_diagnostic_summary(df_twin: pd.DataFrame, death_days: dict, burn_in_days: int = 14, min_points_required: int = 100) -> pd.DataFrame:
    df_sorted = df_twin.sort_values(by=['cell_name', 'Datetime']).copy()

    # 1. Cumulative alert rate calculation
    df_sorted['Alert_Int'] = df_sorted['Digital_Twin_Alert'].astype(int)
    df_sorted['Cum_Alert_Count'] = df_sorted.groupby('cell_name')['Alert_Int'].cumsum()
    df_sorted['Cum_Points'] = df_sorted.groupby('cell_name').cumcount() + 1
    df_sorted['Cum_Alert_Pct'] = (df_sorted['Cum_Alert_Count'] / df_sorted['Cum_Points']) * 100.0

    # 2. Minimum sample size & Burn-in filter (Ignores transient data during the initial 3 days)
    valid_crossing = (
        (df_sorted['Cum_Alert_Pct'] > 15.0) &
        (df_sorted['Cum_Points'] >= min_points_required) &
        (df_sorted['Exposure_Days'] > 3.0)
    )
    crossed_15 = df_sorted[valid_crossing]
    time_15pct = crossed_15.groupby('cell_name')['Exposure_Days'].min()

    # 3. Aggregated diagnostic summary
    summary = df_twin.groupby('cell_name').agg(
        Alert_Count=('Digital_Twin_Alert', 'sum'),
        Data_Points=('pseudo_FF', 'count')
    )

    summary['Alert_Freq_Pct'] = (summary['Alert_Count'] / summary['Data_Points']) * 100.0
    summary['Survival_Days'] = [death_days.get(c, np.inf) for c in summary.index]
    early_collapse = summary['Survival_Days'] <= burn_in_days

    # Extrinsic failure condition (early T80 collapse OR severe anomaly rate >15%)
    summary['Extrinsic_Failure'] = early_collapse | (summary['Alert_Freq_Pct'] > 15.0)

    # Conditional mapping: Timestamp of crossing only for confirmed extrinsic failures
    raw_time_15pct = summary.index.map(time_15pct)
    summary['Threshold_15pct_Day'] = np.where(summary['Extrinsic_Failure'], raw_time_15pct, np.nan)

    # Clean up and sort by alert severity
    summary = summary.drop(columns=['Alert_Count', 'Data_Points']).sort_values(by='Alert_Freq_Pct', ascending=False)
    return summary.round(3)


# =====================================================================
# 5. EXPLAINABILITY & ROOT CAUSE ANALYSIS (SURROGATE TREES)
# =====================================================================

def extract_surrogate_rules(df_twin: pd.DataFrame, cell_name: str, threshold_day: float, window_days: float = 2.0):
    if pd.isna(threshold_day):
        return None, None, None

    # Isolate the critical temporal window
    window_data = df_twin[(df_twin['cell_name'] == cell_name) & 
                          (df_twin['Exposure_Days'] <= threshold_day + window_days)].copy()
    
    features = ['ModuleTemp_C', 'POA_Irradiance_W_m2', 'AbsoluteHumidity_g_m3']
    target = 'Digital_Twin_Alert'

    # Verify adequate variance to fit the tree
    if window_data[target].nunique() < 2:
        logger.warning(f"[{cell_name}] Insufficient variance to train surrogate tree.")
        return None, None, None

    X = window_data[features]
    y = window_data[target].astype(int)

    # Train a shallow decision tree to ensure human interpretability
    surrogate_tree = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    surrogate_tree.fit(X, y)

    # Extract rules as text for console output and JSON logging
    tree_rules = export_text(surrogate_tree, feature_names=features, decimals=1)
    print("\n" + "="*85)
    print(f" SURROGATE RULE EXTRACTION: [{cell_name}] (Critical Window up to Day {threshold_day + window_days:.1f})")
    print("-" * 85)
    print(tree_rules)
    print("="*85 + "\n")

    return surrogate_tree, features, tree_rules


# =====================================================================
# 6. EXECUTION PIPELINE
# =====================================================================

if __name__ == "__main__":
    # Define absolute paths explicitly to avoid nesting inside /data
    PROJECT_ROOT = Path(r"C:\Users\crica\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\parasol-rag-architecture-main")
    DATA_DIR = PROJECT_ROOT / "data" / "processed" / "outdoor"
    survival_file = DATA_DIR / "survival_dataset.parquet"
    
    # Configure figure export directory strictly in /outputs/figures/anomaly_detection
    FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "anomaly_detection"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    if not survival_file.exists():
        logger.error(f"Survival dataset not found at: {survival_file}")
    else:
        logger.info(f"Loading survival dataset from: {survival_file}")
        df_final = pd.read_parquet(survival_file)

        # 1. Automated baseline training and healthy cohort selection
        df_twin_results, alert_threshold, death_days, df_daily_master, healthy_cohort = train_and_evaluate_censored_twin(
            df=df_final,
            burn_in_days=14
        )

        # 2. Cross-validation across the nominal baseline cohort
        print("\n" + "=" * 90)
        print("                    LOOCV STABILITY VALIDATION (HEALTHY COHORT)                    ")
        print("=" * 90)
        loocv_table = execute_loocv_validation(df_twin_results, healthy_cohort)
        print(loocv_table.to_string(index=False))
        print("=" * 90 + "\n")

        # 3. Diagnostic risk assessment generation
        summary_table = generate_diagnostic_summary(df_twin_results, death_days, burn_in_days=14)

        print("\n" + "=" * 90)
        print("                     PREMATURE FAILURE RISK ASSESSMENT                             ")
        print("=" * 90)
        cols_to_show = ['Extrinsic_Failure', 'Alert_Freq_Pct', 'Threshold_15pct_Day', 'Survival_Days']
        print(summary_table[cols_to_show].to_string())
        print("=" * 90)

        print("""
[DIAGNOSTIC METRICS LEGEND]
• Extrinsic_Failure   : True if the device exhibits premature T80 collapse (<=14 days burn-in) 
                        OR a persistent thermodynamic anomaly (Alert_Freq_Pct > 15.0%).
• Alert_Freq_Pct      : Percentage of daylight observations exhibiting underperformance (>3x Base MAE), 
                        evaluated strictly up to the T80 failure point (right-censored) or across the full dataset if surviving.
• Threshold_15pct_Day : Exposure day when the cumulative anomaly density permanently exceeded 15% 
                        (enforcing a >=100 daylight sample threshold to eliminate small-sample warm-up bias).
• Survival_Days       : Exposure duration required to reach T80 structural collapse (inf indicates nominal survival).
""")

        # ---------------------------------------------------------------------
        # DATA EXPORT 1: Enriched Parquet Dataset (KEEPING ALL ORIGINAL ROWS)
        # ---------------------------------------------------------------------
        logger.info("Enriching the original dataset with anomaly metrics without dropping rows...")
        
        # Merge 1: Inject dynamic twin predictions mapped by cell and timestamp
        df_enriched_full = df_final.merge(
            df_twin_results[['cell_name', 'Timestamp', 'Twin_pFF_Pred', 'Underperformance', 'Digital_Twin_Alert']],
            on=['cell_name', 'Timestamp'],
            how='left'
        )

        # Merge 2: Inject static cell-level diagnostics
        df_enriched_full = df_enriched_full.merge(
            summary_table[['Extrinsic_Failure', 'Threshold_15pct_Day', 'Survival_Days']],
            left_on='cell_name',
            right_index=True,
            how='left'
        )
        
        # Ensure boolean defaults for non-daylight rows that Twin didn't evaluate
        df_enriched_full['Digital_Twin_Alert'] = df_enriched_full['Digital_Twin_Alert'].fillna(False)

        enriched_parquet_path = DATA_DIR / "anomaly_scored_dataset.parquet"
        df_enriched_full.to_parquet(enriched_parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"Exported full anomaly-scored dataset ({len(df_enriched_full)} rows) to: {enriched_parquet_path.name}")

        # 4. Anomaly visualization and XAI diagnostic profiling via viz_anomaly.py
        defective_cells = summary_table[summary_table['Extrinsic_Failure']].index.tolist()
        logger.info(f"Generating anomaly profiles and root-cause diagnostics for defective devices: {defective_cells}")
        logger.info(f"Saving figures to: {FIGURES_DIR}")

        xai_diagnostics_dict = {}

        for cell_id in defective_cells:
            t_day = summary_table.loc[cell_id, 'Threshold_15pct_Day']
            
            # Step 1: Render and export Digital Twin tracking plot
            plot_censored_digital_twin(df_twin_results, cell_id, alert_threshold, t_day, FIGURES_DIR)
            
            # Step 2: Fit surrogate decision tree and extract explicit rule hierarchy
            tree_model, feat_names, tree_rules = extract_surrogate_rules(df_twin_results, cell_id, t_day)
            
            # Step 3: Render and export surrogate tree diagnostics and feature attributions
            if tree_model is not None:
                plot_surrogate_diagnostics(tree_model, feat_names, cell_id, t_day, FIGURES_DIR)
                
                # Store XAI information for JSON export
                xai_diagnostics_dict[cell_id] = {
                    "Threshold_15pct_Day": float(t_day) if pd.notna(t_day) else None,
                    "Features_Used": feat_names,
                    "Extracted_Rules": tree_rules.strip().split('\n')
                }

        # ---------------------------------------------------------------------
        # DATA EXPORT 2: XAI Surrogate Rules JSON
        # ---------------------------------------------------------------------
        if xai_diagnostics_dict:
            xai_json_path = DATA_DIR / "xai_surrogate_rules.json"
            with open(xai_json_path, 'w', encoding='utf-8') as f:
                json.dump(xai_diagnostics_dict, f, indent=4)
            logger.info(f"Exported XAI surrogate rules to: {xai_json_path.name}")

        validated_cells = summary_table[~summary_table['Extrinsic_Failure']].index.tolist()
        logger.info(f"Final validated cohort secured for long-term prognostic modeling: {validated_cells}")