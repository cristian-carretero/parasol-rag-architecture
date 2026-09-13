"""
Module: src/config.py
Description: Central configuration for the ParaSol pipeline, ordered to
follow the execution flow: telemetry alignment -> filtering/QC ->
T80 survival labeling -> early screening (digital twin) -> XAI surrogates
-> clustering -> RUL forecasting -> multivariate trajectory forecasting -> dashboard.
Every tunable threshold, tolerance, or hyperparameter lives here so changing
one value updates every module that depends on it.
"""

import pandas as pd
from pathlib import Path

# ==========================================
# FILE SYSTEM REGISTRY (PIPELINE ARTIFACTS)
# ==========================================
BASE_DATA_DIR = Path("data")
RAW_DIR = BASE_DATA_DIR / "raw/outdoor"
# Raw input: per-device CSVs (<device_id>/{mpp,jv,meteo}.csv). Immutable source of truth.

# ---------------------------------------------------------------------------
# Outputs root (declared early so downstream file paths can reference it).
# These artifacts are regenerable and gitignored, distinct from `data/`
# pipeline artifacts. Kept under a single root to make cleanup trivial.
# ---------------------------------------------------------------------------
BASE_OUTPUTS_DIR = Path("outputs")
FIGURES_DIR = BASE_OUTPUTS_DIR / "figures"
DIAGNOSTICS_DIR = BASE_OUTPUTS_DIR / "diagnostics"

# Per-module figure directories, named after the pipeline stage they illustrate.
FIGURES_FILTERING_DIR = FIGURES_DIR / "filtering"
FIGURES_CLUSTERING_DIR = FIGURES_DIR / "clustering"
FIGURES_SCREENING_DIR = FIGURES_DIR / "screening"
FIGURES_XAI_DIR = FIGURES_DIR / "xai"
FIGURES_RUL_DIR = FIGURES_DIR / "rul"

# ==========================================
# PIPELINE ARTIFACT PATHS (per module)
# ==========================================

# [01_ingest_raw.py] ── Ingestion & Parquet serialization
DIR_PROCESSED = BASE_DATA_DIR / "processed/outdoor"
# Per-device Parquet artifacts: <DIR_PROCESSED>/<device_id>/<device_id>_{mpp,jv,meteo}.parquet
# Columns: Timestamp (UTC), Voltage_V, Current_A, Power_W (mpp) | ScanDirection (jv) |
#          POA_Irradiance_W_m2, ModuleTemp_C, AbsoluteHumidity_g_m3 (meteo).

# [02_jv_filtering.py] ── Physical QC, artifact removal, & physics extraction
DIR_FILTERED = BASE_DATA_DIR / "filtered/outdoor"
FILE_JV_FILTERED = DIR_FILTERED / "02_jv_filtered.parquet"
# Point-level J-V sweeps with QC boolean masks and extracted physical scalars
# (Voc, Jsc, FF, P_mpp).
FILE_FILTERING_META = BASE_DATA_DIR / "filtered/metadata/outdoor/02_valid_curves_summary.json"
# Per-cell valid curve counts: {"<cell_name>": <n_valid>, ...}.

# [03_jv_clustering.py] ── Morphological state extraction (PCA + K-Medoids)
DIR_CLUSTERED = BASE_DATA_DIR / "clustered/outdoor"
FILE_JV_LABELED = DIR_CLUSTERED / "03_jv_clustered.parquet"
# FILE_JV_FILTERED + {label_curve (int, -1 = pruned), pseudo_FF (float)}.
FILE_CLUSTERING_ARTIFACTS = BASE_DATA_DIR / "clustered/artifacts/03_clustering_artifacts.joblib"
# Serialized dict: pca_model, kmedoids_model, labels, cluster_metrics, ...

# [04_mppt_aggregation.py] ── 10-min fleet telemetry + PCE
DIR_AGGREGATED = BASE_DATA_DIR / "aggregated/outdoor"
FILE_TELEMETRY_10MIN = DIR_AGGREGATED / "04_meteo_mppt_10min.parquet"
# Fleet-wide 10-min resampled telemetry. Columns: POA_Irradiance_W_m2,
# AbsoluteHumidity_g_m3, ModuleTemp_{Mean,Median,Min,Max}_C, and per-device
# {PCE, Power_W, ModuleTemp}_<device_id>.

# [05_merge_jv_mppt.py] ── Feature matrix (X): J-V curves + aligned telemetry
DIR_DATASET = BASE_DATA_DIR / "dataset/outdoor"
FILE_MERGED_FEATURES = DIR_DATASET / "05_jv_mppt_merged.parquet"
# One row per J-V curve with aligned telemetry, physical parameters
# (Voc, Jsc, FF), and cumulative stressors.

# [06_jv_mppt_t80_tracker.py] ── Ground truth (y): T80 collapse day per cell
FILE_T80_TRUTH = DIR_DATASET / "06_t80_ground_truth.parquet"
# Per-cell survival metrics. Columns: {PCE,pFF}_initial, T80_threshold_*,
# survival_days_{pce,pff}, t80_failure_date_{pce,pff},
# combined_survival_days, combined_failure_date.

# [07_jv_mppt_early_screening.py] ── Digital Twin + LOOCV gate
# Layout: data/screening/{outdoor/, artifacts/}
#   - outdoor/    → action-window cohort parquet (data per cell)
#   - artifacts/  → serialized models, thresholds, and grid-search results
DIR_SCREENING = BASE_DATA_DIR / "screening/outdoor"
DIR_SCREENING_ARTIFACTS = BASE_DATA_DIR / "screening/artifacts"

FILE_HEALTHY_COHORT = DIR_SCREENING / "07_early_screened_cohort.parquet"
# Action-window telemetry with Digital Twin predictions and ML alert flags.
# Columns: cell_name, Timestamp, Exposure_Days, PCE, pFF, environmental
# features, Twin_{PCE,pFF}_Pred, Underperformance_*, Alert_*, Digital_Twin_Alert.

FILE_SCREENING_ARTIFACTS = DIR_SCREENING_ARTIFACTS / "07_early_screening_artifacts.joblib"
# Serialized dict: summary_table, {screening,gated_out,healthy}_cohort,
# alert_thresholds, model_pce, model_pff.

FILE_BURN_IN_GRID = DIR_SCREENING_ARTIFACTS / "07_burn_in_grid_search.parquet"
# Phase-1 sensitivity search over candidate burn-in windows (BURN_IN_GRID_WINDOWS).
# Columns: Window_Days, Healthy_Cells_N, Q98_PCE_Raw, Q98_pFF_Raw,
#          LOOCV_MAE_PCE, LOOCV_MAE_pFF.

# [08_mppt_rul_forecasting.py] ── RUL prognosis & API calibration
DIR_RUL = BASE_DATA_DIR / "rul"
FILE_RUL_TARGETS = DIR_RUL / "08_mppt_rul_targets.parquet"
# Consolidated RUL target per cell: {cell_name, last_anchor_day,
# rul_pred_days, true_survival_days}.

FILE_RUL_COEFFS_CALIBRATED = DIAGNOSTICS_DIR / "rul_coeffs_calibrated.json"
# Optimized kinematic coefficients (written by rul_calibration_optimizer.py).
# When present, the 08 module loads these instead of the hardcoded baselines.

# [09_jv_mppt_trajectory_forecasting.py] ── Multivariate kinematic trajectory engines
FILE_TRAJECTORY_MODELS = DIR_RUL / "09_trajectory_models.joblib"
# Serialized dict {param -> fitted model}. Model family per parameter is
# selected via MODEL_TYPE_PER_PARAM in the 09 module (XGBoost for PCE,
# RandomForest for FF/Jsc/Voc), per the trajectory audit.

FILE_TRAJECTORY_LOOCV = DIR_RUL / "09_trajectory_loocv.parquet"
# Blind-model trajectories (LOOCV): model trained without the cell, sensor weather.
# Columns: cell_name, Date_Day, Exposure_Days, Actual_*, Pred_*.

FILE_TRAJECTORY_PRODUCTION = DIR_RUL / "09_trajectory_production.parquet"
# Production trajectories: model trained on 100% of cohort, API-calibrated weather.
# Columns: cell_name, Date_Day, Exposure_Days, Actual_*, Pred_*.

FILE_TRAJECTORY_COEFFS_CALIBRATED = DIAGNOSTICS_DIR / "trajectory_coeffs_calibrated.json"
# Per-parameter k_blend calibrated by trajectory_calibration_optimizer.py.
# When present, the 09 module loads these instead of the hardcoded defaults.

# ==============================================================================
# 0. PHYSICAL / OPTICAL CONSTANTS (used across every stage)
# ==============================================================================
# Active photovoltaic cell area (0.64 cm^2), converted to m^2 for use in
# PCE = Power_W / (Irradiance_W_m2 * CELL_AREA_M2) * 100.
CELL_AREA_M2 = 0.64 / 10000.0

# Irradiance (W/m^2) below which readings are treated as night-time / low-light
# and excluded from PCE calculation and daylight filtering.
DAYLIGHT_IRRADIANCE_MIN_W_M2 = 100.0

# Floor applied to PCE_initial before using it as a denominator (e.g.
# PCE_Relative = PCE / PCE_initial), to avoid blow-ups on a noisy first
# measurement close to zero.
PCE_INITIAL_REF_FLOOR = 1e-3

RANDOM_STATE = 42

# ==============================================================================
# 1. TELEMETRY ALIGNMENT
# ==============================================================================
# Tolerance for the causal backward merge_asof between J-V curves and
# aggregated telemetry, to prevent stale sensor data from crossing gaps.
TELEMETRY_MERGE_TOLERANCE = pd.Timedelta("20min")

# ==============================================================================
# 2. FILTERING / QUALITY CONTROL
# ==============================================================================
# Inclusive daylight operational window used to exclude night-time noise from
# both the QC pipeline and its diagnostic plots.
OPERATIONAL_HOUR_START = 6
OPERATIONAL_HOUR_END = 22

# ==============================================================================
# 3. T80 PHYSICAL SURVIVAL TRACKING
# ==============================================================================
# Fraction of the initial (Day 0-3) peak PCE that defines the T80 death
# threshold (80% of peak). pFF is also tracked but is a morphological
# descriptor, not a physical parameter.
T80_FRACTION = 0.80

# Number of initial daily observations used to establish the peak baseline.
T80_INITIAL_PEAK_DAYS = 3

# Minimum irradiance required to confirm a PCE T80 drop is not low-light noise.
T80_PCE_CONFIRM_IRRADIANCE_MIN_W_M2 = 400.0

# Consecutive days below the T80 threshold required to confirm structural
# collapse (vs. a transient dip).
T80_CONFIRM_CONSECUTIVE_DAYS = 3

# ==============================================================================
# 4. SHARED MODEL FEATURES
# ==============================================================================
FEATURES = [
    'POA_Irradiance_W_m2',
    'ModuleTemp_C',
    'AbsoluteHumidity_g_m3',
    'Delta_Temp_C_per_h',
    'Delta_Hum_g_m3_per_h',
    'Hour_Sin',
    'Hour_Cos',
    'Day_Sin',
    'Day_Cos'
]

# Subset of purely physical features used by the XAI Explainer Trees to prevent
# them from lazily splitting on temporal confounding variables (sin/cos).
XAI_PHYSICAL_FEATURES = [
    'POA_Irradiance_W_m2',
    'ModuleTemp_C',
    'AbsoluteHumidity_g_m3',
    'Delta_Temp_C_per_h',
    'Delta_Hum_g_m3_per_h'
]

# ==============================================================================
# 5. EARLY SCREENING / DIGITAL TWIN
# ==============================================================================
# Length of the "action window": how many days of early exposure are
# evaluated for ML anomaly alerts. Also used by the dashboard as the
# burn-in / audit cutoff, and by rul_forecasting.py to gate backtesting.
BURN_IN_DAYS = 14.0

# Candidate burn-in windows evaluated during the Phase 1 sensitivity search.
BURN_IN_GRID_WINDOWS = (5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 35.0)

# Minimum physically-meaningful residual (MAE) floor for each metric, so the
# Digital Twin never derives an alert threshold tighter than the model's
# own irreducible noise.
MIN_PHYSICAL_MAE_PCE = 0.085
MIN_PHYSICAL_MAE_PFF = 0.035

# Percentage of in-action-window points flagged as anomalous above which a
# cell is considered to be failing.
ALERT_FREQUENCY_THRESHOLD_PCT = 50.0

# Quantile of in-sample residuals used to set the per-metric alert threshold
# (e.g. 0.98 -> "98th percentile"). Used both for training and for the empirical
# audit chart.
RESIDUAL_ALERT_QUANTILE = 0.98

# XGBoost hyperparameters for the Dual Digital Twin (shared by the production
# fit and every LOOCV fold).
XGB_PCE_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1,
)
XGB_PFF_PARAMS = dict(
    n_estimators=100, learning_rate=0.05, max_depth=3,
    subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1,
)

# ==============================================================================
# 6. XAI SURROGATE MODELS
# ==============================================================================
# Shared DecisionTreeClassifier hyperparameters for both the physical-forensic
# and ML-alert surrogate explainers.
SURROGATE_TREE_PARAMS = dict(
    max_depth=3, min_samples_leaf=5, class_weight="balanced",
    random_state=RANDOM_STATE,
)

# Minimum feature importance to report in a surrogate's explanation.
FEATURE_IMPORTANCE_MIN = 0.05

# Width (days) of the "prodromal" window immediately before physical T80
# collapse, contrasted against the cell's own healthy history.
PRODROMAL_WINDOW_DAYS = 3.0

# Cells whose T80 collapse happens at or before this many days are treated as
# early/infant-mortality failures worth forensic analysis.
EARLY_FAILURE_WINDOW_DAYS = BURN_IN_DAYS

# Sample size used to keep the global SHAP beeswarm plot tractable.
SHAP_SAMPLE_SIZE = 5000

# ==============================================================================
# 7. CLUSTERING
# ==============================================================================
# Target cumulative explained variance for automatic PCA component selection
# and its corresponding diagnostic reference line.
PCA_TARGET_EXPLAINED_VARIANCE = 0.90

# ==============================================================================
# 8. RUL FORECASTING & MULTIVARIATE TRAJECTORIES
# ==============================================================================
DEFAULT_LAT = 41.6833
DEFAULT_LON = -0.8833

# Hyperparameters for the RUL damage-increment engine (PCE; also reused by the
# trajectory audit for its XGBoost comparison baseline). The trajectory
# forecasting engine itself uses RandomForest instead, per the audit findings.
XGB_PARAMS_RUL_PCE = dict(
    n_estimators=200, learning_rate=0.03, max_depth=3,
    subsample=0.75, colsample_bytree=0.8, reg_lambda=10.0,
    reg_alpha=0.5, random_state=RANDOM_STATE, n_jobs=-1
)

# --- MULTIVARIATE TRAJECTORY FORECASTING ---
ROLLING_WINDOW = 7         # Days of thermal/radiative inertia memory
ANCHOR_DAY = 14.0          # Calibration window (burn-in) before forecasting starts

# Evaluation horizon: how many days AFTER the anchor are compared against
# ground truth to compute MAE. Kept at 14 so all cells share a comparable
# window (this is the number used in the technical report).
EVALUATION_HORIZON = 14

# Simulation horizon: how many days AFTER the anchor are simulated for the
# dashboard. Days beyond EVALUATION_HORIZON are pure extrapolation: they are
# plotted but not scored (no ground truth needed).
SIMULATION_HORIZON = 30

# Backward-compatible alias. Prefer EVALUATION_HORIZON in new code.
FORECAST_HORIZON = EVALUATION_HORIZON

TARGET_PARAMS = ["PCE", "FF", "Jsc", "Voc"]

# ==============================================================================
# 9. DEPLOYMENT CONTEXT
# ==============================================================================
# Deployment-local timezone. Raw timestamps are stored in UTC for DST-safe
# merging, but every operational-hour window (daylight filtering, chronometric
# diagnostics) is defined in local time and must be converted before use.
DEPLOYMENT_TIMEZONE = "Europe/Madrid"