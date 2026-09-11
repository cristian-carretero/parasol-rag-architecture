"""
Module: src/config.py
Description: Central configuration for the ParaSol pipeline, ordered to
follow the execution flow: telemetry alignment -> filtering/QC ->
T80 survival labeling -> early screening (digital twin) -> XAI surrogates
-> clustering -> RUL forecasting -> dashboard. Every tunable threshold,
tolerance, or hyperparameter lives here so changing one value updates
every module that depends on it.
"""

import pandas as pd

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
# 1. TELEMETRY ALIGNMENT (survival_dataset.py)
# ==============================================================================
# Tolerance for the causal backward merge_asof between J-V curves and
# aggregated telemetry, to prevent stale sensor data from crossing gaps.
TELEMETRY_MERGE_TOLERANCE = pd.Timedelta("20min")

# ==============================================================================
# 2. FILTERING / QUALITY CONTROL (filtering.py, viz_filtering.py)
# ==============================================================================
# Inclusive daylight operational window used to exclude night-time noise from
# both the QC pipeline and its diagnostic plots.
OPERATIONAL_HOUR_START = 6
OPERATIONAL_HOUR_END = 22

# ==============================================================================
# 3. T80 PHYSICAL SURVIVAL TRACKING (t80_survival_tracker.py)
# ==============================================================================
# Fraction of the initial (Day 0-3) peak PCE/pFF that defines the T80 death
# threshold (80% of peak).
T80_FRACTION = 0.80

# Number of initial daily observations used to establish the peak baseline.
T80_INITIAL_PEAK_DAYS = 3

# Minimum irradiance required to confirm a PCE T80 drop is not low-light noise.
T80_PCE_CONFIRM_IRRADIANCE_MIN_W_M2 = 400.0

# Consecutive days below the T80 threshold required to confirm structural
# collapse (vs. a transient dip).
T80_CONFIRM_CONSECUTIVE_DAYS = 3

# ==============================================================================
# 4. SHARED MODEL FEATURES (early_screening.py, xai_*.py)
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
# 5. EARLY SCREENING / DIGITAL TWIN (early_screening.py)
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
# (e.g. 0.98 -> "98th percentile"). Used both for training (early_screening.py)
# and for the empirical audit chart (streamlit_app.py).
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
# 6. XAI SURROGATE MODELS (xai_physical_forensic.py, xai_digital_twin.py)
# ==============================================================================
# Shared DecisionTreeClassifier hyperparameters for both the physical-forensic
# and ML-alert surrogate explainers.
SURROGATE_TREE_PARAMS = dict(
    max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE,
)

# Minimum feature importance to report in a surrogate's explanation.
FEATURE_IMPORTANCE_MIN = 0.05

# Width (days) of the "prodromal" window immediately before physical T80
# collapse, contrasted against the cell's own healthy history.
PRODROMAL_WINDOW_DAYS = 3.0

# Cells whose T80 collapse happens at or before this many days are treated as
# early/infant-mortality failures worth forensic analysis. Reuses BURN_IN_DAYS
# since both represent "the early-life action window".
EARLY_FAILURE_WINDOW_DAYS = BURN_IN_DAYS

# Sample size used to keep the global SHAP beeswarm plot tractable.
SHAP_SAMPLE_SIZE = 5000

# ==============================================================================
# 7. CLUSTERING (clustering.py, viz_clustering.py)
# ==============================================================================
# Target cumulative explained variance for automatic PCA component selection
# and its corresponding diagnostic reference line.
PCA_TARGET_EXPLAINED_VARIANCE = 0.90

# ==============================================================================
# 8. RUL FORECASTING & API CALIBRATION (rul_forecasting.py)
# ==============================================================================
DEFAULT_LAT = 41.6833
DEFAULT_LON = -0.8833

# Hiperparámetros separados por métrica: reg_lambda=10.0 compartido colapsaba
# el modelo de pFF a una constante (feature_importances_ = [0,0,0,0,0,0]),
# porque su target tiene ~52% de ceros y escala pequeña (0.003-0.07) frente
# al de PCE. Confirmado por diagnóstico antes de separar.
XGB_PARAMS_RUL_PCE = dict(
    n_estimators=200, learning_rate=0.03, max_depth=3,
    subsample=0.75, colsample_bytree=0.8, reg_lambda=10.0,
    reg_alpha=0.5, random_state=RANDOM_STATE, n_jobs=-1
)
XGB_PARAMS_RUL_PFF = dict(
    n_estimators=200, learning_rate=0.03, max_depth=3,
    subsample=0.75, colsample_bytree=0.8, reg_lambda=1.0,
    reg_alpha=0.0, random_state=RANDOM_STATE, n_jobs=-1
)