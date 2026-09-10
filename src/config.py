"""
Module: src/config.py
Description: Central configuration for the ParaSol pipeline (aggregation,
survival tracking, early screening, XAI, dashboard). Every tunable
threshold, tolerance, or model hyperparameter lives here so that changing
one value updates every module that depends on it — no more chasing the
same literal across five files (see the 15%% vs 30%% alert-threshold drift
this replaces).
"""

import pandas as pd

# ==============================================================================
# PHYSICAL / OPTICAL CONSTANTS
# ==============================================================================
# Active photovoltaic cell area (0.64 cm^2), converted to m^2 for use in
# PCE = Power_W / (Irradiance_W_m2 * CELL_AREA_M2) * 100.
CELL_AREA_M2 = 0.64 / 10000.0

# Irradiance (W/m^2) below which readings are treated as night-time / low-light
# and excluded. Used both to keep instantaneous PCE numerically stable and to
# filter daylight rows before daily aggregation (T80 tracking, early screening).
DAYLIGHT_IRRADIANCE_MIN_W_M2 = 100.0

# Floor applied to PCE_initial before using it as a denominator (e.g.
# PCE_Relative = PCE / PCE_initial), to avoid blow-ups on near-zero
# reference values from a noisy first measurement.
PCE_INITIAL_REF_FLOOR = 1e-3

# ==============================================================================
# SHARED MODEL FEATURES
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
# T80 PHYSICAL SURVIVAL TRACKING (t80_survival_tracker.py)
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
# EARLY SCREENING / DIGITAL TWIN (early_screening.py)
# ==============================================================================
# Length of the "action window": how many days of early exposure are
# evaluated for ML anomaly alerts. Also used by the dashboard as the
# burn-in / audit cutoff.
BURN_IN_DAYS = 21.0

# Candidate burn-in windows evaluated during the Phase 1 sensitivity search.
BURN_IN_GRID_WINDOWS = (5.0, 7.0, 10.0, 14.0, 21.0, 28.0, 35.0)

# Minimum physically-meaningful residual (MAE) floor for each metric, so the
# Digital Twin never derives an alert threshold tighter than the model's
# own irreducible noise.
MIN_PHYSICAL_MAE_PCE = 0.085
MIN_PHYSICAL_MAE_PFF = 0.035

# Percentage of in-action-window points flagged as anomalous above which a
# cell is considered to be failing. This is the value that used to be baked
# into the column name "threshold_pct_day" — change it here only.
ALERT_FREQUENCY_THRESHOLD_PCT = 50.0

# Quantile of in-sample residuals used to set the per-metric alert threshold
# (e.g. 0.98 -> "98th percentile"). Used both for training (early_screening.py)
# and for the empirical audit chart (streamlit_app.py).
RESIDUAL_ALERT_QUANTILE = 0.98

# XGBoost hyperparameters for the Dual Digital Twin (shared by the production
# fit and every LOOCV fold).
RANDOM_STATE = 42
XGB_PCE_PARAMS = dict(
    n_estimators=150, learning_rate=0.05, max_depth=5,
    subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1,
)
XGB_PFF_PARAMS = dict(
    n_estimators=100, learning_rate=0.05, max_depth=3,
    subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1,
)

# ==============================================================================
# XAI SURROGATE MODELS (xai_physical_forensic.py, xai_digital_twin.py)
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
# TELEMETRY ALIGNMENT (survival_dataset.py)
# ==============================================================================
# Tolerance for the causal backward merge_asof between J-V curves and
# aggregated telemetry, to prevent stale sensor data from crossing gaps.
TELEMETRY_MERGE_TOLERANCE = pd.Timedelta("20min")

# ==============================================================================
# FILTERING / QUALITY CONTROL (filtering.py, viz_filtering.py)
# ==============================================================================
# Inclusive daylight operational window used to exclude night-time noise from
# both the QC pipeline and its diagnostic plots.
OPERATIONAL_HOUR_START = 6
OPERATIONAL_HOUR_END = 22

# ==============================================================================
# CLUSTERING (clustering.py, viz_clustering.py)
# ==============================================================================
# Target cumulative explained variance for automatic PCA component selection
# and its corresponding diagnostic reference line.
PCA_TARGET_EXPLAINED_VARIANCE = 0.90