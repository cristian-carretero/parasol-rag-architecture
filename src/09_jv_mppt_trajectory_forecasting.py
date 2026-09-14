"""
Module: src/09_jv_mppt_trajectory_forecasting.py
Description: Multivariate autoregressive trajectory forecasting for physical
parameters (PCE, FF, Jsc, Voc). Trains one independent regressor per parameter
(XGBoost or RandomForest, selected per parameter from the trajectory audit),
using relative normalization and LOOCV to ensure out-of-sample physical
generalization. Produces both the production models and the LOOCV trajectory
artifacts consumed by the dashboard.

             The engine has two stages:

               1. Kinematic motor (per parameter): a regressor that predicts
                  the normalized daily increment, integrated forward over the
                  forecast horizon. The model family is chosen per parameter:

                    - PCE : XGBoost (better on absolute 14-day MAE).
                    - FF  : RandomForest (per audit and empirical MAE).
                    - Jsc : RandomForest (only family with GroupKFold signal).
                    - Voc : RandomForest.

               2. Persistence blend: the raw motor forecast is mixed with a
                  "persistence" baseline (the parameter's value at the anchor
                  day) using a per-parameter coefficient k_blend:

                     Pred_final = (1 - k_blend) * Pred_motor + k_blend * Persistence

                  When the motor carries no signal for a parameter (e.g. FF),
                  the blend with a high k_blend collapses the forecast to the
                  trivial "value stays constant" baseline, which is optimal.

                  Per the trajectory audit (src/trajectory_xgb_audit.py):
                    - Jsc: real, transferable signal -> low k_blend.
                    - PCE, Voc: signal in-cell but not across cells -> moderate k.
                    - FF: no signal -> high k.

             The k_blend coefficients are calibrated empirically by
             src/trajectory_calibration_optimizer.py, which writes a JSON file
             at FILE_TRAJECTORY_COEFFS_CALIBRATED. At import time this module
             loads that file if present, overriding the hardcoded baselines.

             Forecast horizon policy (two horizons, not one):
               - EVALUATION_HORIZON (14 d): window used to compute the MAE.
                 Kept fixed so all cells share a comparable metric.
               - SIMULATION_HORIZON (30 d): total days simulated for the
                 dashboard. Days beyond EVALUATION_HORIZON are pure
                 extrapolation: they are plotted but not scored.

               Per-cell behavior:
                 * Sensor weather (LOOCV): simulation is bounded by the cell's
                   own sensor data, so SIMULATION_HORIZON is a ceiling that may
                   never be reached.
                 * API weather (production): simulation runs the full
                   SIMULATION_HORIZON because Open-Meteo is unbounded.
                 * Cells with fewer than MIN_FORECAST_HORIZON future days are
                   discarded entirely. Cells with fewer than EVALUATION_HORIZON
                   are flagged Forecast_Truncated=True and excluded from the
                   aggregate MAE.
"""

import json
import logging
from typing import Any, Dict, List, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.config import (
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    FILE_TRAJECTORY_MODELS,
    FILE_TRAJECTORY_LOOCV,
    FILE_TRAJECTORY_PRODUCTION,
    FILE_TRAJECTORY_COEFFS_CALIBRATED,
    DEFAULT_LAT,
    DEFAULT_LON,
    ROLLING_WINDOW,
    ANCHOR_DAY,
    EVALUATION_HORIZON,
    SIMULATION_HORIZON,
    FORECAST_HORIZON,     # alias of EVALUATION_HORIZON, kept for compatibility
    RANDOM_STATE,
    XGB_PARAMS_RUL_PCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrajectoryForecasting")


# ==============================================================================
# MODULE-LEVEL CONFIGURATION
# ==============================================================================
# Physical parameters to forecast. FF is the physical Fill Factor (from the
# filtering stage, step 02), NOT the morphological pFF descriptor produced by
# the clustering stage (step 03). Using pFF here would forecast a structural
# descriptor rather than a physical quantity.
TARGET_PARAMS = ["PCE", "FF", "Jsc", "Voc"]

BASE_FEATURES = [
    "Daily_Irradiance_Dose",
    "Daily_Max_Temp_C",
    "Daily_Median_Humidity",
    "Rolling_Irradiance",
    "Rolling_Thermal_load",
]

# ------------------------------------------------------------------------------
# Per-parameter model family selection.
# Rationale (empirical, on the 14-day absolute MAE):
#   - PCE : XGBoost beats RF.
#   - FF  : RF beats XGBoost.
#   - Jsc : RF beats XGBoost, and is the only family with cross-cell signal.
#   - Voc : RF beats XGBoost.
# ------------------------------------------------------------------------------
MODEL_TYPE_PER_PARAM: Dict[str, str] = {
    "PCE": "xgboost",
    "FF":  "random_forest",
    "Jsc": "random_forest",
    "Voc": "random_forest",
}

# ------------------------------------------------------------------------------
# Model hyperparameters.
# ------------------------------------------------------------------------------
RF_PARAMS: Dict[str, Any] = dict(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ------------------------------------------------------------------------------
# Forecast horizon policy.
# ------------------------------------------------------------------------------
# Hard floor: below this many future days, the MAE is dominated by noise and
# the cell is discarded entirely. At or above this floor, the cell is
# simulated over its available window and flagged as truncated.
MIN_FORECAST_HORIZON = 3

# ------------------------------------------------------------------------------
# Persistence-blend coefficients (per parameter).
# ------------------------------------------------------------------------------
# k_blend = 0   -> pure motor
# k_blend = 1   -> pure persistence (forecast stays at the anchor value)
#
# The values below are the immutable baseline. If the calibration optimizer
# (src/trajectory_calibration_optimizer.py) has been run, its JSON payload
# overrides them at import time.
K_BLEND_HARDCODED: Dict[str, float] = {
    "PCE": 0.0,
    "FF": 0.0,
    "Jsc": 0.0,
    "Voc": 0.0,
}


def _load_k_blend_defaults() -> Dict[str, float]:
    """
    Load calibrated k_blend from JSON if present; else fall back to the
    hardcoded baselines. The JSON is written by
    src/trajectory_calibration_optimizer.py.
    """
    values = dict(K_BLEND_HARDCODED)
    if FILE_TRAJECTORY_COEFFS_CALIBRATED.exists():
        try:
            with FILE_TRAJECTORY_COEFFS_CALIBRATED.open(encoding="utf-8") as f:
                payload = json.load(f)
            calibrated = payload.get("k_blend_per_param", {})
            for p in values:
                if p in calibrated:
                    values[p] = float(calibrated[p])
            logger.info(f"Calibrated k_blend loaded from JSON -> {values}")
        except Exception as e:
            logger.warning(
                f"Failed to read calibrated k_blend ({e}); using hardcoded values."
            )
    else:
        logger.info(f"No calibrated k_blend found; using hardcoded -> {values}")
    return values


K_BLEND_DEFAULTS: Dict[str, float] = _load_k_blend_defaults()


# ==============================================================================
# 1. DATA PREPARATION & RELATIVE NORMALIZATION
# ==============================================================================
def build_trajectory_matrix(df_healthy: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Group telemetry by day, normalize each physical parameter against its own
    initial baseline, and engineer lag/delta states for autoregressive modeling.
    """
    df = df_healthy.copy()
    if "Datetime" not in df.columns:
        df["Datetime"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["Date_Day"] = df["Datetime"].dt.date

    available_targets = [p for p in TARGET_PARAMS if p in df.columns]

    agg_funcs = {
        "POA_Irradiance_W_m2": "sum",
        "ModuleTemp_C": "max",
        "AbsoluteHumidity_g_m3": "median",
        "Exposure_Days": "max",
    }
    for param in available_targets:
        agg_funcs[param] = "mean"

    df_daily = df.groupby(["cell_name", "Date_Day"]).agg(agg_funcs).reset_index()
    df_daily.rename(
        columns={
            "POA_Irradiance_W_m2": "Daily_Irradiance_Dose",
            "ModuleTemp_C": "Daily_Max_Temp_C",
            "AbsoluteHumidity_g_m3": "Daily_Median_Humidity",
        },
        inplace=True,
    )

    df_daily = df_daily.sort_values(by=["cell_name", "Date_Day"])

    df_daily["Rolling_Irradiance"] = (
        df_daily.groupby("cell_name")["Daily_Irradiance_Dose"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    df_daily["Rolling_Thermal_load"] = (
        df_daily.groupby("cell_name")["Daily_Max_Temp_C"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )

    for param in available_targets:
        init_vals = df_daily.groupby("cell_name").head(3).groupby("cell_name")[param].mean()
        df_daily[f"{param}_Initial"] = df_daily["cell_name"].map(init_vals)

        df_daily[f"{param}_Norm"] = df_daily[param] / (df_daily[f"{param}_Initial"] + 1e-9)

        df_daily[f"{param}_Smooth"] = (
            df_daily.groupby("cell_name")[f"{param}_Norm"]
            .rolling(ROLLING_WINDOW, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df_daily[f"{param}_Lag1"] = df_daily.groupby("cell_name")[f"{param}_Smooth"].shift(1)
        df_daily[f"{param}_Delta"] = df_daily[f"{param}_Smooth"] - df_daily[f"{param}_Lag1"]

    cleaned_df = df_daily.dropna().copy()
    return cleaned_df, available_targets


# ==============================================================================
# 2. KINEMATIC ENGINE: TRAINING & SIMULATION
# ==============================================================================
def train_multivariate_engine(
    df_train: pd.DataFrame,
    targets: List[str],
) -> Dict[str, Any]:
    """
    Train one independent regressor per physical parameter.

    The model family is selected per parameter via MODEL_TYPE_PER_PARAM:
      - "xgboost"       -> xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
      - "random_forest" -> RandomForestRegressor(**RF_PARAMS)

    Returns a dict mapping parameter -> fitted model.
    """
    models: Dict[str, Any] = {}
    for param in targets:
        features = BASE_FEATURES + [f"{param}_Lag1"]
        model_type = MODEL_TYPE_PER_PARAM.get(param, "random_forest")

        if model_type == "xgboost":
            model = XGBRegressor(**XGB_PARAMS_RUL_PCE)
        else:
            model = RandomForestRegressor(**RF_PARAMS)

        model.fit(df_train[features], df_train[f"{param}_Delta"])
        models[param] = model
    return models


def simulate_trajectory(
    initial_norm_states: Dict[str, float],
    initial_raw_values: Dict[str, float],
    future_weather: pd.DataFrame,
    models: Dict[str, Any],
    historical_weather: pd.DataFrame,
    targets: List[str],
    k_blend_per_param: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Iterate stepwise into the future, predicting normalized increments and
    updating each parameter's state autoregressively.

    Args:
        initial_norm_states: State at the anchor day, per parameter, in
            normalized units (value / initial value).
        initial_raw_values: Initial baseline values in physical units.
        future_weather: Daily weather over the forecast horizon. Its length
            defines how many days the simulation runs (may be shorter than
            SIMULATION_HORIZON if the underlying data source is bounded).
        models: One trained regressor per parameter.
        historical_weather: History up to the anchor (for rolling features).
        targets: Parameters to simulate.
        k_blend_per_param: Optional dict mapping parameter -> persistence blend
            weight. If None or missing a key, the value in K_BLEND_DEFAULTS
            is used.
    """
    k_blend = dict(K_BLEND_DEFAULTS)
    if k_blend_per_param:
        k_blend.update(k_blend_per_param)

    # Persistence baseline: the parameter's value at the anchor day, in
    # physical units. Constant across the forecast horizon. Represents the
    # "value stays as observed" naive forecast.
    persistence_value: Dict[str, float] = {
        p: initial_raw_values[p] * initial_norm_states[p] for p in targets
    }

    simulated_days = []
    current_states = initial_norm_states.copy()

    rolling_irr = list(historical_weather["Daily_Irradiance_Dose"].tail(ROLLING_WINDOW))
    rolling_temp = list(historical_weather["Daily_Max_Temp_C"].tail(ROLLING_WINDOW))

    for _, row in future_weather.iterrows():
        day_irr = row["Daily_Irradiance_Dose"]
        day_temp = row["Daily_Max_Temp_C"]

        rolling_irr.append(day_irr)
        rolling_temp.append(day_temp)
        if len(rolling_irr) > ROLLING_WINDOW:
            rolling_irr.pop(0)
            rolling_temp.pop(0)

        daily_result = {
            "Date_Day": row["Date_Day"],
            "Exposure_Days": row["Exposure_Days"],
        }

        for param in targets:
            features_sim = pd.DataFrame([{
                "Daily_Irradiance_Dose": day_irr,
                "Daily_Max_Temp_C": day_temp,
                "Daily_Median_Humidity": row["Daily_Median_Humidity"],
                "Rolling_Irradiance": np.median(rolling_irr),
                "Rolling_Thermal_load": np.median(rolling_temp),
                f"{param}_Lag1": current_states[param],
            }])

            pred_delta = float(models[param].predict(features_sim)[0])

            new_norm_state = max(0.0, current_states[param] + pred_delta)
            current_states[param] = new_norm_state

            pred_motor = new_norm_state * initial_raw_values[param]

            # Persistence blend (per parameter).
            k = k_blend.get(param, 0.0)
            if k > 0.0:
                pred_final = (1.0 - k) * pred_motor + k * persistence_value[param]
            else:
                pred_final = pred_motor

            daily_result[f"Pred_{param}"] = pred_final

        simulated_days.append(daily_result)

    return pd.DataFrame(simulated_days)


def fetch_and_calibrate_api_weather(
    df_sensor_daily: pd.DataFrame,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Fetch Open-Meteo historical archive and calibrate it against local sensor
    readings (linear regression per variable). Returns a per-day DataFrame with
    Daily_Irradiance_Dose, Daily_Max_Temp_C, Daily_Median_Humidity already in
    sensor-equivalent units.

    Strategy:
      1. Try the on-disk cache written by module 08 (FILE_API_HISTORY_CACHE).
         Reuse it only if its stored (min_date, max_date) covers the range we
         need AND it is at most 24 h old.
      2. Otherwise, hit the live archive endpoint. The end date is clipped to
         today (the archive rejects future dates). On success, write the
         cache so future runs don't need the network.
    """
    import json
    import time
    from pathlib import Path

    import requests
    from sklearn.linear_model import LinearRegression

    from src.config import FILE_API_HISTORY_CACHE

    logger.info("Fetching and calibrating Open-Meteo API weather...")

    min_date_needed = pd.Timestamp(df_sensor_daily["Date_Day"].min()).date()
    today = pd.Timestamp.now("UTC").date()
    requested_max = (
        pd.Timestamp(df_sensor_daily["Date_Day"].max())
        + pd.Timedelta(days=SIMULATION_HORIZON)
    ).date()
    max_date_needed = min(requested_max, today)

    df_api_daily: pd.DataFrame | None = None

    # ---- 1. Cache lookup -----------------------------------------------------
    cache_meta_path = Path(FILE_API_HISTORY_CACHE).with_suffix(".meta.json")
    if FILE_API_HISTORY_CACHE.exists() and cache_meta_path.exists():
        try:
            meta = json.loads(cache_meta_path.read_text())
            cache_min = pd.Timestamp(meta["min_date"]).date()
            cache_max = pd.Timestamp(meta["max_date"]).date()
            cache_age_h = (
                pd.Timestamp.now("UTC") - pd.Timestamp(meta["fetched_at"])
            ).total_seconds() / 3600.0
            covers_range = cache_min <= min_date_needed and cache_max >= max_date_needed
            if covers_range and cache_age_h < 24.0:
                df_api_daily = pd.read_parquet(FILE_API_HISTORY_CACHE)
                logger.info(
                    f"Reusing cached API history from {cache_meta_path.name} "
                    f"(age {cache_age_h:.1f} h, covers {cache_min}..{cache_max})"
                )
            else:
                logger.info(
                    f"API cache stale or insufficient: age={cache_age_h:.1f} h, "
                    f"cache range {cache_min}..{cache_max}, "
                    f"needed {min_date_needed}..{max_date_needed}"
                )
        except Exception as exc:
            logger.warning(f"Failed to read API cache ({exc}); will fetch live.")

    # ---- 2. Live fetch if no usable cache ------------------------------------
    if df_api_daily is None:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={min_date_needed}&end_date={max_date_needed}"
            f"&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
            f"&timezone=Europe%2FMadrid"
        )

        last_exc: Exception | None = None
        payload = None
        for attempt in (1, 2, 3):
            try:
                resp = requests.get(url, timeout=90)
                # Some Open-Meteo 5xx responses are served with a 200 status and
                # an empty body. Reject those explicitly instead of letting
                # resp.json() raise an opaque "Expecting value" error.
                if not resp.content or not resp.content.strip():
                    raise RuntimeError(
                        f"Open-Meteo returned an empty body "
                        f"(status={resp.status_code}, len=0)"
                    )
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:
                last_exc = exc
                wait = 5 * attempt
                logger.warning(
                    f"Open-Meteo request failed (attempt {attempt}/3): {exc}. "
                    f"Retrying in {wait} s..."
                )
                if attempt < 3:
                    time.sleep(wait)

        if payload is None:
            raise RuntimeError(
                f"Open-Meteo request failed after 3 attempts: {last_exc}"
            )

        if "hourly" not in payload:
            raise RuntimeError(
                f"Open-Meteo archive did not return hourly data. "
                f"Keys: {list(payload.keys())}, "
                f"range: {min_date_needed} to {max_date_needed}"
            )
        data = payload["hourly"]

        df_api = pd.DataFrame({
            "Timestamp": pd.to_datetime(data["time"]),
            "API_Temp_C": data["temperature_2m"],
            "RH_pct": data["relative_humidity_2m"],
            "API_GHI_W_m2": data["shortwave_radiation"],
        })
        df_api["Date_Day"] = df_api["Timestamp"].dt.date

        df_api_daily = df_api.groupby("Date_Day").agg(
            API_Daily_Max_Temp=("API_Temp_C", "max"),
            API_Daily_Irr_Dose=("API_GHI_W_m2", "sum"),
            Daily_Mean_RH=("RH_pct", "mean"),
        ).reset_index()

        # Write the cache so subsequent runs (or other modules) don't need the network.
        try:
            from src.config import FILE_API_HISTORY_CACHE as _CACHE
            _CACHE.parent.mkdir(parents=True, exist_ok=True)
            df_api_daily.to_parquet(_CACHE, index=False)
            cache_meta_path.write_text(json.dumps({
                "min_date": str(min_date_needed),
                "max_date": str(max_date_needed),
                "fetched_at": pd.Timestamp.now("UTC").isoformat(),
            }, indent=2))
            logger.info(f"API history cached -> {_CACHE}")
        except Exception as exc:
            logger.warning(f"Failed to cache API history: {exc}")

    # ---- 3. Calibration against sensor overlap -------------------------------
    sensor_daily = df_sensor_daily[[
        "Date_Day", "Daily_Max_Temp_C", "Daily_Irradiance_Dose"
    ]].drop_duplicates()

    merged = pd.merge(sensor_daily, df_api_daily, on="Date_Day", how="inner").dropna()
    if merged.empty:
        raise RuntimeError("No overlapping days between sensors and API for calibration.")

    reg_temp = LinearRegression().fit(
        merged[["API_Daily_Max_Temp"]], merged["Daily_Max_Temp_C"]
    )
    reg_irr = LinearRegression().fit(
        merged[["API_Daily_Irr_Dose"]], merged["Daily_Irradiance_Dose"]
    )

    df_api_daily = df_api_daily.copy()
    df_api_daily["Daily_Max_Temp_C"] = reg_temp.predict(df_api_daily[["API_Daily_Max_Temp"]])
    df_api_daily["Daily_Irradiance_Dose"] = reg_irr.predict(df_api_daily[["API_Daily_Irr_Dose"]])

    p_sat = 6.112 * np.exp(
        (17.67 * df_api_daily["Daily_Max_Temp_C"]) /
        (df_api_daily["Daily_Max_Temp_C"] + 243.5)
    )
    df_api_daily["Daily_Median_Humidity"] = (
        (216.68 * (p_sat * (df_api_daily["Daily_Mean_RH"] / 100.0))) /
        (df_api_daily["Daily_Max_Temp_C"] + 273.15)
    )

    return df_api_daily


# ==============================================================================
# 3. ROBUST EVALUATION (LEAVE-ONE-CELL-OUT)
# ==============================================================================
def run_loocv_evaluation(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    targets: List[str],
    k_blend_per_param: Dict[str, float] | None = None,
) -> List[dict]:
    """
    Leave-One-Cell-Out evaluation: for each cell, train on the rest of the
    cohort and forecast its trajectory using sensor weather (ground truth).

    Horizon policy:
      * Simulation runs for min(SIMULATION_HORIZON, sensor_days_available).
        Since sensor weather dies with the cell, the effective horizon is
        bounded by the physical lifespan.
      * MAE is computed on the first EVALUATION_HORIZON days only.
      * Cells whose evaluation window is shorter than EVALUATION_HORIZON are
        flagged Forecast_Truncated=True and excluded from the aggregate MAE.

    Returns the per-day simulated vs. actual records for dashboard plotting.
    """
    print("\n" + "=" * 85)
    print(f" MULTIVARIATE EVALUATION: LEAVE-ONE-CELL-OUT "
          f"(eval={EVALUATION_HORIZON}d, sim={SIMULATION_HORIZON}d)")
    print("=" * 85)

    all_metrics: List[dict] = []
    trajectory_records: List[dict] = []

    for test_cell in healthy_cohort:
        df_train = df_daily[df_daily["cell_name"] != test_cell]
        df_test = df_daily[df_daily["cell_name"] == test_cell]

        if df_test.empty:
            continue

        blind_models = train_multivariate_engine(df_train, targets)

        hist_cutoff = df_test[df_test["Exposure_Days"] <= ANCHOR_DAY]
        future_all = df_test[df_test["Exposure_Days"] > ANCHOR_DAY]

        if hist_cutoff.empty:
            logger.warning(f"[{test_cell}] No data before anchor day {ANCHOR_DAY}. Skipping.")
            continue

        n_available = len(future_all)
        if n_available < MIN_FORECAST_HORIZON:
            logger.warning(
                f"[{test_cell}] Only {n_available} future days available "
                f"(< {MIN_FORECAST_HORIZON}). Skipping."
            )
            continue

        # Simulate as many days as we can: min(SIMULATION_HORIZON, sensor data).
        n_simulate = min(SIMULATION_HORIZON, n_available)
        future_ground_truth = future_all.head(n_simulate).copy()

        # Evaluation window: always the first EVALUATION_HORIZON days.
        n_evaluate = min(EVALUATION_HORIZON, n_simulate)
        truncated = n_evaluate < EVALUATION_HORIZON

        if truncated:
            logger.info(
                f"[{test_cell}] Evaluation window truncated to {n_evaluate} days "
                f"(of {EVALUATION_HORIZON}). Excluded from aggregate MAE."
            )
        elif n_simulate > EVALUATION_HORIZON:
            logger.info(
                f"[{test_cell}] Extended simulation to {n_simulate} days "
                f"(MAE evaluated on first {EVALUATION_HORIZON})."
            )

        initial_norm_states = {
            p: float(hist_cutoff.iloc[-1][f"{p}_Smooth"]) for p in targets
        }
        initial_raw_values = {
            p: float(hist_cutoff.iloc[-1][f"{p}_Initial"]) for p in targets
        }

        future_weather = future_ground_truth[[
            "Date_Day", "Exposure_Days",
            "Daily_Irradiance_Dose", "Daily_Max_Temp_C", "Daily_Median_Humidity",
        ]]

        df_sim = simulate_trajectory(
            initial_norm_states, initial_raw_values,
            future_weather, blind_models, hist_cutoff, targets,
            k_blend_per_param=k_blend_per_param,
        )

        # --- Calibration history (Actual only; Pred = NaN) ---
        hist_actual = {
            p: (hist_cutoff[f"{p}_Smooth"] * hist_cutoff[f"{p}_Initial"]).to_numpy(dtype=float)
            for p in targets
        }
        hist_exposure = hist_cutoff["Exposure_Days"].to_numpy(dtype=float)
        hist_dates = hist_cutoff["Date_Day"].to_numpy()

        for i in range(len(hist_cutoff)):
            record = {
                "cell_name": test_cell,
                "Date_Day": hist_dates[i],
                "Exposure_Days": float(hist_exposure[i]),
                "Phase": "Calibration",
                "Forecast_Horizon_Used": n_simulate,
                "Evaluation_Horizon_Used": n_evaluate,
                "Forecast_Truncated": truncated,
            }
            for param in targets:
                record[f"Actual_{param}"] = float(hist_actual[param][i])
                record[f"Pred_{param}"] = float("nan")
            trajectory_records.append(record)

        # --- Forecast window (Actual + Pred) ---
        truth_actual = {
            p: (future_ground_truth[f"{p}_Smooth"] * future_ground_truth[f"{p}_Initial"]).to_numpy(dtype=float)
            for p in targets
        }
        truth_exposure = future_ground_truth["Exposure_Days"].to_numpy(dtype=float)
        truth_dates = future_ground_truth["Date_Day"].to_numpy()
        pred_arrays = {
            p: df_sim[f"Pred_{p}"].to_numpy(dtype=float) for p in targets
        }

        for i in range(len(df_sim)):
            record = {
                "cell_name": test_cell,
                "Date_Day": truth_dates[i],
                "Exposure_Days": float(truth_exposure[i]),
                "Phase": "Forecast",
                "Forecast_Horizon_Used": n_simulate,
                "Evaluation_Horizon_Used": n_evaluate,
                "Forecast_Truncated": truncated,
            }
            for param in targets:
                record[f"Actual_{param}"] = float(truth_actual[param][i])
                record[f"Pred_{param}"] = float(pred_arrays[param][i])
            trajectory_records.append(record)

        # --- MAE metrics per parameter (evaluated on first n_evaluate days) ---
        for param in targets:
            mae = float(mean_absolute_error(
                truth_actual[param][:n_evaluate],
                pred_arrays[param][:n_evaluate],
            ))
            all_metrics.append({
                "Test_Cell": test_cell,
                "Parameter": param,
                "MAE": mae,
                "Horizon_Used": n_evaluate,
                "Truncated": truncated,
            })

    # --- Aggregate report (complete cells vs truncated cells) ---
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_complete = df_metrics[~df_metrics["Truncated"]]
        df_trunc = df_metrics[df_metrics["Truncated"]]

        if not df_complete.empty:
            res_full = df_complete.groupby("Parameter")["MAE"].mean().reset_index()
            n_full = df_complete["Test_Cell"].nunique()
            print(f"\nMEAN ABSOLUTE ERROR (MAE) @ {EVALUATION_HORIZON}-DAY HORIZON "
                  f"— complete cells only (N={n_full}):")
            print(res_full.to_string(index=False))

        if not df_trunc.empty:
            n_trunc = df_trunc["Test_Cell"].nunique()
            print(f"\nTRUNCATED CELLS — excluded from the aggregate above (N={n_trunc}):")
            print(df_trunc[["Test_Cell", "Parameter", "MAE", "Horizon_Used"]].to_string(index=False))

        print("=" * 85 + "\n")
    else:
        print("\n[WARNING] No cell had enough data to run the LOOCV.")
        print("=" * 85 + "\n")

    return trajectory_records


def run_production_backtesting(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    targets: List[str],
    production_models: Dict[str, Any],
    api_weather: pd.DataFrame,
    k_blend_per_param: Dict[str, float] | None = None,
) -> List[dict]:
    """
    Production-model backtest: uses the 100%-trained models with API-calibrated
    weather (instead of sensor weather) for the future window.

    Horizon policy:
      * Simulation runs for SIMULATION_HORIZON days (unbounded, because the
        API can supply weather indefinitely).
      * Sensor ground truth only exists for the days the cell lived. Beyond
        that, `Actual_*` is left as NaN so the dashboard can distinguish
        prediction from measured reality.
      * MAE is computed on the first min(EVALUATION_HORIZON, sensor_days) days.
    """
    records: List[dict] = []

    for cell in healthy_cohort:
        cell_data = df_daily[df_daily["cell_name"] == cell].sort_values("Exposure_Days")
        if cell_data.empty:
            continue

        hist = cell_data[cell_data["Exposure_Days"] <= ANCHOR_DAY]
        if hist.empty:
            logger.warning(f"[{cell}] No history before anchor day. Skipping.")
            continue

        # Sensor ground truth: bounded by the cell's lifespan.
        sensor_future = cell_data[cell_data["Exposure_Days"] > ANCHOR_DAY]
        n_sensor_available = len(sensor_future)

        # Evaluation window: first EVALUATION_HORIZON days of sensor truth.
        n_evaluate = min(EVALUATION_HORIZON, n_sensor_available)
        truncated = n_evaluate < EVALUATION_HORIZON

        # Full sensor ground truth for the parquet's Actual_* column: extends
        # up to SIMULATION_HORIZON or wherever the sensor stops, whichever
        # comes first. This matches the LOOCV parquet's Actual coverage so
        # both charts show the same physical reality.
        n_actual_show = min(SIMULATION_HORIZON, n_sensor_available)
        future_truth_full = sensor_future.head(n_actual_show).copy()

        if n_evaluate == 0:
            logger.warning(f"[{cell}] No sensor ground truth after anchor. MAE will be NaN.")
        elif truncated:
            logger.info(
                f"[{cell}] Evaluation window truncated to {n_evaluate} days "
                f"(of {EVALUATION_HORIZON})."
            )

        # API weather: unbounded, so we simulate SIMULATION_HORIZON days.
        anchor_date = hist["Date_Day"].max()
        anchor_day = float(hist["Exposure_Days"].max())

        future_dates_api = pd.date_range(
            anchor_date + pd.Timedelta(days=1),
            periods=SIMULATION_HORIZON,
            freq="D",
        ).date.tolist()

        api_slice = api_weather[api_weather["Date_Day"].isin(future_dates_api)].copy()
        api_slice = api_slice.sort_values("Date_Day").head(SIMULATION_HORIZON)

        if len(api_slice) < MIN_FORECAST_HORIZON:
            logger.warning(f"[{cell}] API weather missing days. Skipping production backtest.")
            continue

        # Attach Exposure_Days: real value where sensor exists, else anchor + offset.
        exposure_map = dict(zip(sensor_future["Date_Day"], sensor_future["Exposure_Days"]))
        api_slice = api_slice.copy()
        api_slice["Exposure_Days"] = [
            exposure_map.get(d, anchor_day + (d - anchor_date).days)
            for d in api_slice["Date_Day"]
        ]

        future_weather = api_slice
        n_simulate = len(future_weather)
        if n_simulate > EVALUATION_HORIZON:
            logger.info(
                f"[{cell}] Extended production simulation to {n_simulate} days "
                f"(MAE evaluated on first {n_evaluate})."
            )

        initial_norm_states = {
            p: float(hist.iloc[-1][f"{p}_Smooth"]) for p in targets
        }
        initial_raw_values = {
            p: float(hist.iloc[-1][f"{p}_Initial"]) for p in targets
        }

        df_sim = simulate_trajectory(
            initial_norm_states, initial_raw_values,
            future_weather, production_models, hist, targets,
            k_blend_per_param=k_blend_per_param,
        )

        # --- Calibration history (Actual only; Pred = NaN) ---
        hist_actual = {
            p: (hist[f"{p}_Smooth"] * hist[f"{p}_Initial"]).to_numpy(dtype=float)
            for p in targets
        }
        hist_exposure = hist["Exposure_Days"].to_numpy(dtype=float)
        hist_dates = hist["Date_Day"].to_numpy()

        for i in range(len(hist)):
            record = {
                "cell_name": cell,
                "Date_Day": hist_dates[i],
                "Exposure_Days": float(hist_exposure[i]),
                "Phase": "Calibration",
                "Forecast_Horizon_Used": n_simulate,
                "Evaluation_Horizon_Used": n_evaluate,
                "Forecast_Truncated": truncated,
            }
            for param in targets:
                record[f"Actual_{param}"] = float(hist_actual[param][i])
                record[f"Pred_{param}"] = float("nan")
            records.append(record)

        # --- Forecast window: Pred for all n_simulate days, Actual for every
        #     day the sensor actually covers (up to SIMULATION_HORIZON). The
        #     MAE is still evaluated on the first n_evaluate days only. ---
        truth_actual_full = {
            p: (future_truth_full[f"{p}_Smooth"] * future_truth_full[f"{p}_Initial"]).to_numpy(dtype=float)
            for p in targets
        }
        pred_arrays = {p: df_sim[f"Pred_{p}"].to_numpy(dtype=float) for p in targets}
        sim_dates = df_sim["Date_Day"].to_numpy()
        sim_exposure = df_sim["Exposure_Days"].to_numpy(dtype=float)

        for i in range(n_simulate):
            record = {
                "cell_name": cell,
                "Date_Day": sim_dates[i],
                "Exposure_Days": float(sim_exposure[i]),
                "Phase": "Forecast",
                "Forecast_Horizon_Used": n_simulate,
                "Evaluation_Horizon_Used": n_evaluate,
                "Forecast_Truncated": truncated,
            }
            for param in targets:
                # Actual_* is filled for every day the sensor covers, even
                # beyond the evaluation window. This keeps Chart 2's Actual
                # aligned with Chart 1's, and lets the dashboard shade only
                # the truly unknown tail (past the last sensor observation).
                actual = (
                    float(truth_actual_full[param][i])
                    if i < n_actual_show
                    else float("nan")
                )
                record[f"Actual_{param}"] = actual
                record[f"Pred_{param}"] = float(pred_arrays[param][i])
            records.append(record)

    return records


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    try:
        logger.info("Loading Digital Twin healthy cohort...")
        df_healthy = pd.read_parquet(FILE_HEALTHY_COHORT)
        artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)
        healthy_cohort = cast(List[str], artifacts["healthy_cohort"])

        df_daily, available_targets = build_trajectory_matrix(df_healthy)
        logger.info(f"Target parameters detected for forecasting: {available_targets}")
        logger.info(f"Model family per parameter: {MODEL_TYPE_PER_PARAM}")
        logger.info(f"Persistence blend coefficients in use: {K_BLEND_DEFAULTS}")
        logger.info(
            f"Horizon policy: evaluation = {EVALUATION_HORIZON} d, "
            f"simulation = {SIMULATION_HORIZON} d, "
            f"hard floor = {MIN_FORECAST_HORIZON} d"
        )

        # --- Phase 1: blind-model LOOCV trajectories ---
        loocv_records = run_loocv_evaluation(df_daily, healthy_cohort, available_targets)

        # --- Phase 2: production models trained on 100% of the cohort ---
        logger.info("Training final production models with 100% of the healthy cohort...")
        production_models = train_multivariate_engine(df_daily, available_targets)

        # --- Fetch API weather (calibrated to sensors) for production backtest ---
        try:
            api_weather = fetch_and_calibrate_api_weather(df_daily)
            production_records = run_production_backtesting(
                df_daily, healthy_cohort, available_targets, production_models, api_weather,
            )
            if not production_records:
                logger.warning(
                    "Production backtest produced 0 records. "
                    "This usually means Open-Meteo did not cover the anchor+horizon window "
                    "(e.g. the anchor is too close to today)."
                )
        except Exception as e:
            logger.warning(f"Production backtest skipped: {e}")
            logger.warning(
                "  LOOCV artifacts are still valid and will be serialized. "
                "Re-run when Open-Meteo is reachable."
            )
            production_records = []

        # --- Serialization ---
        FILE_TRAJECTORY_MODELS.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(production_models, FILE_TRAJECTORY_MODELS)
        logger.info(f"Trajectory models serialized -> {FILE_TRAJECTORY_MODELS}")

        if loocv_records:
            FILE_TRAJECTORY_LOOCV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(loocv_records).to_parquet(FILE_TRAJECTORY_LOOCV, index=False)
            logger.info(f"LOOCV trajectories serialized -> {FILE_TRAJECTORY_LOOCV}")
        else:
            logger.warning("No LOOCV trajectories to serialize.")

        if production_records:
            FILE_TRAJECTORY_PRODUCTION.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(production_records).to_parquet(FILE_TRAJECTORY_PRODUCTION, index=False)
            logger.info(f"Production trajectories serialized -> {FILE_TRAJECTORY_PRODUCTION}")
        else:
            logger.warning("No production trajectories to serialize.")

    except Exception as e:
        logger.error(f"Execution aborted: {e}")


if __name__ == "__main__":
    main()