"""
Module: src/08_mppt_rul_forecasting.py
Description: PCE Remaining Useful Life (RUL) forecasting and API calibration.

             Implements a hybrid prognostic engine that combines a data-driven
             XGBoost velocity with an explicit physical kinematics layer. The
             kinematics layer provides three stabilising mechanisms:
                 - A deep rolling-median anti-ratchet filter on the damage signal.
                 - An asymptotic structural floor that accelerates the forecast
                   as the damage approaches the T80 threshold.
                 - A soft-countdown reconciliation that stabilises consecutive
                   weekly predictions.

             API-sensor calibration is performed in walk-forward mode, with
             strict temporal and cell-level anti-leakage filtering.

             The six kinematic coefficients are exposed as module-level
             constants so that external optimisers (e.g. LOOCV-based grid
             search) can override them without duplicating the simulation
             logic. See `src/rul_calibration_optimizer.py`.
"""

import logging
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from src.config import (
    BURN_IN_DAYS, T80_FRACTION, DEFAULT_LAT, DEFAULT_LON,
    XGB_PARAMS_RUL_PCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PCE_Forecasting")

# ==============================================================================
# MODULE-LEVEL CONFIGURATION
# ==============================================================================
SMOOTHING_WINDOW = 2   # ventana de la mediana móvil
ANCHOR_SPACING   = 7   # días entre anclas de backtesting
SIMULATION_WINDOW = 14
T80_DAMAGE_LIMIT = 1.0 - T80_FRACTION  # Universal 0.20
MIN_VELOCITY = 1e-4
RUL_MAX_DAYS = 3650

FEATURES_RUL_PCE = [
    "Daily_Irradiance_Dose",
    "Daily_Max_Temp_C",
    "Daily_Median_Humidity",
    "Rolling_Irradiance",
    "Rolling_Thermal_load",
    "Cumulative_Damage_Lag1",
]

# ------------------------------------------------------------------------------
# Hybrid kinematics coefficients
# ------------------------------------------------------------------------------
# Two-tier system:
#   - _HARDCODED_* values are the immutable baseline (used by the optimizer
#     as reference and by the 08 when no calibration is available).
#   - *_DEFAULT values are the effective defaults; they start as the hardcoded
#     baseline but are overridden if a calibrated JSON exists.

# Structural floor:  v_floor(rho) = PHI_0 + PHI_1 * exp(PHI_2 * rho)
PHI_0_HARDCODED = 0.0015
PHI_1_HARDCODED = 0.0025
PHI_2_HARDCODED = 1.5

# Structural weight: w(rho) = min(1, max(0, rho)) ** LAMBDA_W
LAMBDA_W_HARDCODED = 1.0

# Soft-countdown:    RUL_final = MU * RUL_raw + (1 - MU) * RUL_expected
MU_HARDCODED = 0.5

# Monotonicity lax:  RUL_final <= RUL_prev + EPS
EPS_HARDCODED = 1.0

# Temporal baseline blend (Strategy B): mixes engine RUL with clock RUL
K_BLEND_HARDCODED = 0.0   # 0 = pure engine, 1 = pure clock
T_REF_HARDCODED = 54.0    # median lifetime of the cohort (days)


# ------------------------------------------------------------------------------
# Calibrated coefficient loader
# ------------------------------------------------------------------------------
def _load_calibrated_coefficients() -> dict:
    """
    Load optimized kinematic coefficients from the JSON written by
    `rul_calibration_optimizer.py`. Returns an empty dict if the file does
    not exist or is malformed.
    """
    import json
    from pathlib import Path
    from src.config import FILE_RUL_COEFFS_CALIBRATED

    path = Path(FILE_RUL_COEFFS_CALIBRATED)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        required = {"phi_0", "phi_1", "phi_2", "lambda_w", "mu", "eps"}
        optional = {"k_blend", "t_ref"}
        missing = required - set(data.keys())
        if missing:
            logger.warning(
                f"Calibrated coefficients JSON is missing keys {missing}. "
                f"Falling back to hardcoded defaults."
            )
            return {}
        # Optional keys (Strategy B): fall back silently if absent
        result = {k: float(data[k]) for k in required}
        for k in optional:
            if k in data:
                result[k] = float(data[k])
        # Smoothing_window for consistency checks
        if "smoothing_window" in data:
            result["smoothing_window"] = int(data["smoothing_window"])
        return result
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(f"Failed to load calibrated coefficients JSON: {exc}")
        return {}


# Effective defaults: start as hardcoded baseline, override if calibrated.
PHI_0_DEFAULT = PHI_0_HARDCODED
PHI_1_DEFAULT = PHI_1_HARDCODED
PHI_2_DEFAULT = PHI_2_HARDCODED
LAMBDA_W_DEFAULT = LAMBDA_W_HARDCODED
MU_DEFAULT = MU_HARDCODED
EPS_DEFAULT = EPS_HARDCODED
K_BLEND_DEFAULT = K_BLEND_HARDCODED
T_REF_DEFAULT = T_REF_HARDCODED

_CALIBRATED = _load_calibrated_coefficients()
if _CALIBRATED:
    PHI_0_DEFAULT = _CALIBRATED["phi_0"]
    PHI_1_DEFAULT = _CALIBRATED["phi_1"]
    PHI_2_DEFAULT = _CALIBRATED["phi_2"]
    LAMBDA_W_DEFAULT = _CALIBRATED["lambda_w"]
    MU_DEFAULT = _CALIBRATED["mu"]
    EPS_DEFAULT = _CALIBRATED["eps"]
    K_BLEND_DEFAULT = _CALIBRATED.get("k_blend", K_BLEND_HARDCODED)
    T_REF_DEFAULT = _CALIBRATED.get("t_ref", T_REF_HARDCODED)
    _cal_w = _CALIBRATED.get("smoothing_window")
    if _cal_w is not None and _cal_w != SMOOTHING_WINDOW:
        logger.warning(
            f"Calibrated coefficients JSON was produced for "
            f"SMOOTHING_WINDOW={_cal_w}, but the module is running with "
            f"SMOOTHING_WINDOW={SMOOTHING_WINDOW}. The coefficients are NOT "
            f"optimal for this W. Re-run the optimizer or restore "
            f"SMOOTHING_WINDOW={_cal_w}."
        )
    logger.info(
        f"Using CALIBRATED kinematic coefficients: "
        f"phi_0={PHI_0_DEFAULT}, phi_1={PHI_1_DEFAULT}, phi_2={PHI_2_DEFAULT}, "
        f"lambda_w={LAMBDA_W_DEFAULT}, mu={MU_DEFAULT}, eps={EPS_DEFAULT}, "
        f"k_blend={K_BLEND_DEFAULT}, t_ref={T_REF_DEFAULT}"
    )
else:
    logger.info("Using HARDCODED kinematic coefficients (no calibrated JSON found).")


# ==============================================================================
# 1. DATA PIPELINE (NATURAL FLUCTUATION, NO CUMMAX)
# ==============================================================================
def build_rul_matrix(
    df_twin: pd.DataFrame,
    healthy_cohort: List[str],
    smoothing_window: int = SMOOTHING_WINDOW,
) -> pd.DataFrame:
    """
    Build the daily RUL feature matrix from the raw digital twin.

    Aggregates sub-daily measurements into daily statistics and applies a
    rolling median filter (anti-ratchet) instead of a destructive cumulative
    maximum.
    """
    df = df_twin[df_twin["cell_name"].isin(healthy_cohort)].copy()
    if "Datetime" not in df.columns:
        df["Datetime"] = pd.to_datetime(df["Timestamp"], utc=True)
    df["Date_Day"] = df["Datetime"].dt.date

    if "PCE_initial" not in df.columns:
        raise KeyError(
            "Missing 'PCE_initial' in df_twin. It must be merged from t80_metrics."
        )

    df_daily = (
        df.groupby(["cell_name", "Date_Day"])
        .agg(
            Daily_Irradiance_Dose=("POA_Irradiance_W_m2", "sum"),
            Daily_Max_Temp_C=("ModuleTemp_C", "max"),
            Daily_Median_Humidity=("AbsoluteHumidity_g_m3", "median"),
            Daily_PCE=("PCE", "max"),
            Exposure_Days=("Exposure_Days", "max"),
            PCE_Initial=("PCE_initial", "first"),
        )
        .reset_index()
        .sort_values(by=["cell_name", "Date_Day"])
    )

    # 1. Instantaneous real loss relative to the initial PCE
    df_daily["Instant_Loss"] = 1.0 - df_daily["Daily_PCE"] / df_daily["PCE_Initial"]

    # 2. Rolling median (W=1 = identity, no smoothing) to stabilise noise.
    #    Anti-ratchet: does not accumulate noise as permanent damage.
    df_daily["Cumulative_Damage"] = (
        df_daily.groupby("cell_name")["Instant_Loss"]
        .rolling(smoothing_window, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )

    df_daily["Cumulative_Damage_Lag1"] = (
        df_daily.groupby("cell_name")["Cumulative_Damage"].shift(1).fillna(0.0)
    )

    # 3. Increment may be negative (photo-annealing recoveries or optimal weather)
    df_daily["Daily_Damage_Increment"] = (
        df_daily["Cumulative_Damage"] - df_daily["Cumulative_Damage_Lag1"]
    )

    df_daily["Rolling_Irradiance"] = (
        df_daily.groupby("cell_name")["Daily_Irradiance_Dose"]
        .rolling(smoothing_window, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )
    df_daily["Rolling_Thermal_load"] = (
        df_daily.groupby("cell_name")["Daily_Max_Temp_C"]
        .rolling(smoothing_window, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )

    required_cols = list(set(FEATURES_RUL_PCE + ["Daily_Damage_Increment"]))
    return df_daily.dropna(subset=required_cols).copy()


def fetch_api_history(
    df_sensor_daily: pd.DataFrame,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """
    Download raw historical data from Open-Meteo for the deployment coordinates.

    Returns daily aggregated API variables. No calibration is applied here.

    The requested range is padded (400 days back, SIMULATION_WINDOW+30 days
    forward, clipped to today) so the resulting cache file can serve module 09
    without a second live fetch. On success, the response is also persisted to
    FILE_API_HISTORY_CACHE for downstream reuse.
    """
    logger.info("Fetching raw historical data from Open-Meteo API...")

    raw_min = pd.Timestamp(df_sensor_daily["Date_Day"].min())
    raw_max = pd.Timestamp(df_sensor_daily["Date_Day"].max())
    today = pd.Timestamp.now("UTC")
    min_date = (raw_min - pd.Timedelta(days=400)).date()
    max_date = min(
        (raw_max + pd.Timedelta(days=SIMULATION_WINDOW + 30)).date(),
        today.date(),
    )

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={min_date}&end_date={max_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
        f"&timezone=Europe%2FMadrid"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if "hourly" not in payload:
            raise RuntimeError(
                f"Open-Meteo archive did not return hourly data. "
                f"Response keys: {list(payload.keys())}. "
                f"Requested range: {min_date} to {max_date}"
            )
        data = payload["hourly"]
        df_api = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(data["time"]),
                "API_Temp_C": data["temperature_2m"],
                "RH_pct": data["relative_humidity_2m"],
                "API_GHI_W_m2": data["shortwave_radiation"],
            }
        )
        df_api["Date_Day"] = df_api["Timestamp"].dt.date
        df_api_daily = (
            df_api.groupby("Date_Day")
            .agg(
                API_Daily_Max_Temp=("API_Temp_C", "max"),
                API_Daily_Irr_Dose=("API_GHI_W_m2", "sum"),
                Daily_Mean_RH=("RH_pct", "mean"),
            )
            .reset_index()
        )
        logger.info(f"API history fetched: {len(df_api_daily)} daily records.")

        # Persist raw API history so downstream modules (e.g. 09) can reuse it
        # without a second live fetch. Keyed by (min_date, max_date) so a
        # later run with a wider range invalidates the cache automatically.
        try:
            import json
            from src.config import FILE_API_HISTORY_CACHE

            FILE_API_HISTORY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df_api_daily.to_parquet(FILE_API_HISTORY_CACHE, index=False)
            cache_meta = {
                "min_date": str(min_date),
                "max_date": str(max_date),
                "fetched_at": pd.Timestamp.now("UTC").isoformat(),
            }
            (FILE_API_HISTORY_CACHE.with_suffix(".meta.json")).write_text(
                json.dumps(cache_meta, indent=2)
            )
            logger.info(f"API history cached -> {FILE_API_HISTORY_CACHE}")
        except Exception as exc:
            logger.warning(f"Failed to cache API history: {exc}")

        return df_api_daily
    except Exception as exc:
        logger.warning(f"API fetch failed: {exc}")
        return pd.DataFrame()


def fit_calibration(
    df_sensor_daily: pd.DataFrame,
    df_api_raw: pd.DataFrame,
    max_date,
    train_cells: Optional[List[str]] = None,
    window_days: Optional[int] = None,
) -> Tuple[LinearRegression, LinearRegression]:
    """
    Fit API->sensor linear regressions using ONLY data with Date_Day <= max_date
    and ONLY the cells provided in `train_cells`. This prevents temporal and
    cell-level information leakage during LOOCV backtesting.

    Args:
        df_sensor_daily: Daily sensor dataframe (must contain 'cell_name',
                         'Date_Day', 'Daily_Max_Temp_C', 'Daily_Irradiance_Dose').
        df_api_raw:      Raw (uncalibrated) daily API dataframe.
        max_date:        Upper date bound for calibration (inclusive).
        train_cells:     Optional list of cells to use. None = all cells.
        window_days:     Optional rolling window length. None = expanding window.
    """
    sensor = df_sensor_daily.copy()
    if train_cells is not None:
        sensor = sensor[sensor["cell_name"].isin(train_cells)]
    sensor = sensor[sensor["Date_Day"] <= max_date]
    if window_days is not None:
        min_allowed = (pd.Timestamp(max_date) - pd.Timedelta(days=window_days)).date()
        sensor = sensor[sensor["Date_Day"] >= min_allowed]

    sensor_daily = sensor[
        ["Date_Day", "Daily_Max_Temp_C", "Daily_Irradiance_Dose"]
    ].drop_duplicates()
    api = df_api_raw[df_api_raw["Date_Day"] <= max_date]
    merged = pd.merge(sensor_daily, api, on="Date_Day", how="inner").dropna()
    if len(merged) < 10:
        raise ValueError(f"Not enough data to fit calibration ({len(merged)} rows).")

    reg_temp = LinearRegression().fit(
        merged[["API_Daily_Max_Temp"]], merged["Daily_Max_Temp_C"]
    )
    reg_irr = LinearRegression().fit(
        merged[["API_Daily_Irr_Dose"]], merged["Daily_Irradiance_Dose"]
    )
    return reg_temp, reg_irr


def apply_calibration(
    reg_temp: LinearRegression,
    reg_irr: LinearRegression,
    df_api_subset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply calibration regressions to a subset of raw API data and derive
    absolute humidity via the Clausius-Clapeyron approximation.

    Returns a copy; input is not modified.
    """
    out = df_api_subset.copy()
    out["Daily_Max_Temp_C"] = reg_temp.predict(out[["API_Daily_Max_Temp"]])
    out["Daily_Irradiance_Dose"] = reg_irr.predict(out[["API_Daily_Irr_Dose"]])
    p_sat = 6.112 * np.exp(
        (17.67 * out["Daily_Max_Temp_C"]) / (out["Daily_Max_Temp_C"] + 243.5)
    )
    out["Daily_Median_Humidity"] = (
        216.68 * (p_sat * (out["Daily_Mean_RH"] / 100.0))
    ) / (out["Daily_Max_Temp_C"] + 273.15)
    return out


# ==============================================================================
# 2. PROGNOSTIC ENGINE (HYBRID VELOCITY KINEMATICS)
# ==============================================================================
def train_rul_engine(df_daily: pd.DataFrame) -> xgb.XGBRegressor:
    """Fit an XGBoost regressor to predict the daily damage increment."""
    model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
    model.fit(df_daily[FEATURES_RUL_PCE], df_daily["Daily_Damage_Increment"])
    return model


def simulate_rul_kinematics(
    current_damage: float,
    rolling_irr: list,
    rolling_temp: list,
    future_weather: pd.DataFrame,
    model: xgb.XGBRegressor,
    phi_0: float = PHI_0_DEFAULT,
    phi_1: float = PHI_1_DEFAULT,
    phi_2: float = PHI_2_DEFAULT,
    lambda_w: float = LAMBDA_W_DEFAULT,
) -> float:
    """
    Forward-simulate damage over the future weather window and convert the
    integrated velocity into an RUL estimate (days until T80 threshold).

    Includes the asymptotic structural floor that accelerates near end-of-life.
    All six coefficients are overridable to support LOOCV-based calibration.
    """
    if current_damage >= T80_DAMAGE_LIMIT:
        return 0.0

    loop_damage = current_damage
    for _, row in future_weather.iterrows():
        day_irr = row["Daily_Irradiance_Dose"]
        day_temp = row["Daily_Max_Temp_C"]
        rolling_irr.append(day_irr)
        rolling_temp.append(day_temp)
        if len(rolling_irr) > SMOOTHING_WINDOW:
            rolling_irr.pop(0)
            rolling_temp.pop(0)

        x_sim = pd.DataFrame(
            [{
                "Daily_Irradiance_Dose": day_irr,
                "Daily_Max_Temp_C": day_temp,
                "Daily_Median_Humidity": row["Daily_Median_Humidity"],
                "Rolling_Irradiance": np.median(rolling_irr),
                "Rolling_Thermal_load": np.median(rolling_temp),
                "Cumulative_Damage_Lag1": loop_damage,
            }]
        )

        loop_damage += float(model.predict(x_sim)[0])

    velocity = max(
        MIN_VELOCITY, (loop_damage - current_damage) / len(future_weather)
    )

    # Structural floor: gentle in healthy phases, dominant near the death threshold
    damage_ratio = max(0.0, current_damage) / T80_DAMAGE_LIMIT
    structural_floor = phi_0 + (phi_1 * np.exp(damage_ratio * phi_2))
    weight_structural = min(1.0, max(0.0, damage_ratio)) ** lambda_w

    final_velocity = (
        (1.0 - weight_structural) * velocity
        + weight_structural * max(velocity, structural_floor)
    )
    final_velocity = np.maximum(final_velocity, MIN_VELOCITY)

    remaining_damage = max(0.0, T80_DAMAGE_LIMIT - current_damage)
    return min(remaining_damage / final_velocity, RUL_MAX_DAYS)


# ==============================================================================
# 3. BACKTESTING & EVALUATION
# ==============================================================================
def run_dynamic_backtesting(
    df_daily: pd.DataFrame,
    df_api_raw: pd.DataFrame,
    cell: str,
    model_pce: xgb.XGBRegressor,
    t80_metrics: pd.DataFrame,
    train_cells: Optional[List[str]] = None,
    mu: float = MU_DEFAULT,
    eps: float = EPS_DEFAULT,
    k_blend: float = K_BLEND_DEFAULT,
    t_ref: float = T_REF_DEFAULT,
    kinematic_coeffs: Optional[dict] = None,
    target_days: Optional[dict] = None,     
) -> List[dict]:
    """
    Run dynamic backtesting for a single cell across weekly anchors.

    Args:
        ...
        target_days: optional dict {cell_name: survival_days} loaded from
                     the screening artifact (schema v2). When provided, it
                     overrides the local re-derivation of the ground truth
                     from t80_metrics, making the 07->08 contract explicit.
    """
    coeffs = kinematic_coeffs or {}
    phi_0 = coeffs.get("phi_0", PHI_0_DEFAULT)
    phi_1 = coeffs.get("phi_1", PHI_1_DEFAULT)
    phi_2 = coeffs.get("phi_2", PHI_2_DEFAULT)
    lambda_w = coeffs.get("lambda_w", LAMBDA_W_DEFAULT)

    cell_data = df_daily[df_daily["cell_name"] == cell].sort_values("Exposure_Days")
    if cell_data.empty:
        return []

    true_survival_days: Optional[float] = None
    if target_days is not None and cell in target_days:
        # v2 contract: ground truth comes from the screening artifact.
        value = target_days[cell]
        if value is not None and np.isfinite(value):
            true_survival_days = float(value)
    elif cell in t80_metrics.index:
        # Legacy fallback (pre-v2 artifact): re-derive from t80_metrics.
        value = pd.to_numeric(
            t80_metrics.loc[cell, "survival_days_pce"],
            errors="coerce",
        )
        if pd.notna(value):
            true_survival_days = float(value)

    max_days = cell_data["Exposure_Days"].max()
    anchors = list(range(int(BURN_IN_DAYS), int(max_days) + 1, ANCHOR_SPACING))
    if int(max_days) not in anchors:
        anchors.append(int(max_days))

    records: List[dict] = []

    # --------------------------------------------------------------------------
    # 3.1 SENSOR BACKTESTING (no calibration involved)
    # --------------------------------------------------------------------------
    print(f"\n[{cell}] HISTORICAL PCE RUL TRACKING (Limit: {T80_DAMAGE_LIMIT*100:.1f}%)")
    print("-" * 85)

    prev_anchor_sensor: Optional[float] = None
    prev_rul_sensor: Optional[float] = None
    for anchor in anchors:
        hist_cutoff = cell_data[cell_data["Exposure_Days"] <= anchor]
        if hist_cutoff.empty:
            continue
        actual_day = float(hist_cutoff.iloc[-1]["Exposure_Days"])
        cum_damage = float(hist_cutoff.iloc[-1]["Cumulative_Damage"])

        if cum_damage >= T80_DAMAGE_LIMIT:
            print(
                f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% | "
                f"[DEGRADED CELL] Simulation aborted."
            )
            break

        future = cell_data[cell_data["Exposure_Days"] > anchor].head(SIMULATION_WINDOW)
        if future.empty:
            rul_val = float("nan")
            rul_str = "  N/A"
            real_rul_str = "  N/A"
        else:
            rul_val = simulate_rul_kinematics(
                cum_damage,
                list(hist_cutoff["Daily_Irradiance_Dose"].tail(SMOOTHING_WINDOW)),
                list(hist_cutoff["Daily_Max_Temp_C"].tail(SMOOTHING_WINDOW)),
                future,
                model_pce,
                phi_0=phi_0, phi_1=phi_1, phi_2=phi_2, lambda_w=lambda_w,
            )

            if prev_rul_sensor is not None and prev_anchor_sensor is not None:
                elapsed = actual_day - prev_anchor_sensor
                expected_rul = max(0.0, prev_rul_sensor - elapsed)
                # Soft-countdown: blend raw kinematics with the strict passage of time
                rul_val = mu * rul_val + (1.0 - mu) * expected_rul
                # Allow a slight rebound under optimal weather but keep a descending ramp
                rul_val = min(rul_val, prev_rul_sensor + eps)

            # Strategy B: blend with temporal baseline
            if k_blend > 0.0:
                rul_temporal = max(0.0, t_ref - actual_day)
                rul_val = (1.0 - k_blend) * rul_val + k_blend * rul_temporal

            prev_anchor_sensor, prev_rul_sensor = actual_day, rul_val
            rul_str = f"{rul_val:5.1f} Days"

            if true_survival_days is not None:
                real_rul = true_survival_days - actual_day
                real_rul_str = f"{real_rul:5.1f} Days"
            else:
                real_rul_str = "  N/A"

        records.append(
            {
                "cell_name": cell,
                "Anchor_Day": actual_day,
                "True_Survival_Days": true_survival_days,
                "RUL_Pred": rul_val,
                "Type": "Sensor",
            }
        )
        print(
            f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% -> "
            f"Pred RUL: {rul_str} | True RUL: {real_rul_str}"
        )

    # --------------------------------------------------------------------------
    # 3.2 API BACKTESTING (walk-forward calibration, leakage-free)
    # --------------------------------------------------------------------------
    if not df_api_raw.empty:
        print(f"[{cell}] API CALIBRATED BACKTESTING (walk-forward)")
        prev_anchor_api: Optional[float] = None
        prev_rul_api: Optional[float] = None
        for anchor in anchors:
            hist_cutoff = cell_data[cell_data["Exposure_Days"] <= anchor]
            if hist_cutoff.empty:
                continue
            actual_day = float(hist_cutoff.iloc[-1]["Exposure_Days"])
            cum_damage = float(hist_cutoff.iloc[-1]["Cumulative_Damage"])
            anchor_date = hist_cutoff.iloc[-1]["Date_Day"]

            if cum_damage >= T80_DAMAGE_LIMIT:
                break

            # Walk-forward fit: only past data and only training cells are visible.
            try:
                reg_temp_wf, reg_irr_wf = fit_calibration(
                    df_daily,
                    df_api_raw,
                    max_date=anchor_date,
                    train_cells=train_cells,
                    window_days=90,  # rolling window; set None for expanding window
                )
            except ValueError:
                continue

            future_raw = df_api_raw[df_api_raw["Date_Day"] > anchor_date].head(
                SIMULATION_WINDOW
            )
            if future_raw.empty:
                rul_val = float("nan")
                rul_str = "  N/A"
                real_rul_str = "  N/A"
            else:
                future = apply_calibration(reg_temp_wf, reg_irr_wf, future_raw)
                rul_val = simulate_rul_kinematics(
                    cum_damage,
                    list(hist_cutoff["Daily_Irradiance_Dose"].tail(SMOOTHING_WINDOW)),
                    list(hist_cutoff["Daily_Max_Temp_C"].tail(SMOOTHING_WINDOW)),
                    future,
                    model_pce,
                    phi_0=phi_0, phi_1=phi_1, phi_2=phi_2, lambda_w=lambda_w,
                )

                if prev_rul_api is not None and prev_anchor_api is not None:
                    elapsed = actual_day - prev_anchor_api
                    expected_rul = max(0.0, prev_rul_api - elapsed)
                    rul_val = mu * rul_val + (1.0 - mu) * expected_rul
                    rul_val = min(rul_val, prev_rul_api + eps)

                # Strategy B: blend with temporal baseline
                if k_blend > 0.0:
                    rul_temporal = max(0.0, t_ref - actual_day)
                    rul_val = (1.0 - k_blend) * rul_val + k_blend * rul_temporal

                prev_anchor_api, prev_rul_api = actual_day, rul_val
                rul_str = f"{rul_val:5.1f} Days"

                if true_survival_days is not None:
                    real_rul = true_survival_days - actual_day
                    real_rul_str = f"{real_rul:5.1f} Days"
                else:
                    real_rul_str = "  N/A"

            records.append(
                {
                    "cell_name": cell,
                    "Anchor_Day": actual_day,
                    "True_Survival_Days": true_survival_days,
                    "RUL_Pred": rul_val,
                    "Type": "API",
                }
            )
            print(
                f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% -> "
                f"Pred RUL: {rul_str} | True RUL: {real_rul_str}"
            )

    return records


def evaluate_model_performance(all_records: List[dict]) -> None:
    """Aggregate MAE across LOOCV folds for sensor and API modes."""
    if not all_records:
        return
    df_eval = pd.DataFrame(all_records)
    df_eval["RUL_Real"] = df_eval["True_Survival_Days"] - df_eval["Anchor_Day"]
    valid = df_eval[df_eval["RUL_Real"] > 0].copy()
    if valid.empty:
        return

    print("\n" + "=" * 75)
    print(" MODEL PERFORMANCE EVALUATION")
    print("=" * 75)
    for m_type in ["Sensor", "API"]:
        sub = valid[valid["Type"] == m_type]
        if not sub.empty:
            mae = np.mean(np.abs(sub["RUL_Pred"] - sub["RUL_Real"]))
            print(f"[{m_type:6s}] MAE: {mae:5.2f} days (N={len(sub)})")
    print("=" * 75 + "\n")


# ==============================================================================
# 4. LIVE PRODUCTION FORECAST
# ==============================================================================
def run_live_production_forecast(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    model_pce: xgb.XGBRegressor,
    reg_temp: LinearRegression,
    reg_irr: LinearRegression,
) -> None:
    """
    Execute a live 14-day production forecast using the Open-Meteo forecast API
    and the production calibration regressions fitted on the full history.
    """
    print("=" * 75)
    print(" LIVE PRODUCTION FORECAST (API 14-Day Outlook)")
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={DEFAULT_LAT}&longitude={DEFAULT_LON}"
        f"&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
        f"&timezone=Europe%2FMadrid&forecast_days={SIMULATION_WINDOW}"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if "hourly" not in payload:
            raise RuntimeError(
                f"Open-Meteo forecast did not return hourly data. "
                f"Response keys: {list(payload.keys())}"
            )
        data = payload["hourly"]
        df_api = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(data["time"]),
                "API_Temp_C": data["temperature_2m"],
                "RH_pct": data["relative_humidity_2m"],
                "API_GHI_W_m2": data["shortwave_radiation"],
            }
        )
        df_api["Date_Day"] = df_api["Timestamp"].dt.date
        df_forecast = (
            df_api.groupby("Date_Day")
            .agg(
                API_Daily_Max_Temp=("API_Temp_C", "max"),
                API_Daily_Irr_Dose=("API_GHI_W_m2", "sum"),
                Daily_Mean_RH=("RH_pct", "mean"),
            )
            .reset_index()
        )

        df_forecast["Daily_Max_Temp_C"] = reg_temp.predict(
            df_forecast[["API_Daily_Max_Temp"]]
        )
        df_forecast["Daily_Irradiance_Dose"] = reg_irr.predict(
            df_forecast[["API_Daily_Irr_Dose"]]
        )
        p_sat = 6.112 * np.exp(
            (17.67 * df_forecast["Daily_Max_Temp_C"])
            / (df_forecast["Daily_Max_Temp_C"] + 243.5)
        )
        df_forecast["Daily_Median_Humidity"] = (
            216.68 * (p_sat * (df_forecast["Daily_Mean_RH"] / 100.0))
        ) / (df_forecast["Daily_Max_Temp_C"] + 273.15)

        for cell in healthy_cohort:
            cell_data = (
                df_daily[df_daily["cell_name"] == cell]
                .sort_values("Exposure_Days")
            )
            if cell_data.empty:
                continue
            last = cell_data.iloc[-1]
            actual_day = float(last["Exposure_Days"])
            cum_damage = float(last["Cumulative_Damage"])

            if cum_damage >= T80_DAMAGE_LIMIT:
                rul_str = "[DEGRADED]"
            else:
                rul_val = simulate_rul_kinematics(
                    cum_damage,
                    list(cell_data["Daily_Irradiance_Dose"].tail(SMOOTHING_WINDOW)),
                    list(cell_data["Daily_Max_Temp_C"].tail(SMOOTHING_WINDOW)),
                    df_forecast,
                    model_pce,
                )
                # Strategy B: blend with temporal baseline
                if K_BLEND_DEFAULT > 0.0:
                    rul_temporal = max(0.0, T_REF_DEFAULT - actual_day)
                    rul_val = (1.0 - K_BLEND_DEFAULT) * rul_val + K_BLEND_DEFAULT * rul_temporal
                rul_str = f"{rul_val:5.1f} Days"

            print(
                f"[{cell}] Current: Day {actual_day:4.1f} | "
                f"Damage: {cum_damage*100:4.1f}% (Limit: {T80_DAMAGE_LIMIT*100:4.1f}%) "
                f"-> RUL: {rul_str}"
            )
    except Exception as exc:
        logger.error(f"Live forecast failed: {exc}")


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    from src.config import (
        FILE_HEALTHY_COHORT,
        FILE_T80_TRUTH,
        FILE_SCREENING_ARTIFACTS,
        FILE_RUL_TARGETS,
    )

    FILE_RUL_TARGETS.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
        t80_metrics = pd.read_parquet(FILE_T80_TRUTH)

        artifact = joblib.load(FILE_SCREENING_ARTIFACTS)
        schema_version = artifact.get("schema_version", 1)
        if schema_version < 2:
            logger.warning(
                "Screening artifact has no schema_version (legacy v1). "
                "Falling back to hardcoded 'survival_days_pce' derivation. "
                "Re-run module 07 to produce a v2 artifact."
            )

        healthy_cohort = artifact["healthy_cohort"]
        target_metric = artifact.get("t80_target_metric", "survival_days_pce")
        target_days = artifact.get("t80_target_days", {})
        logger.info(
            f"Screening artifact loaded (schema v{schema_version}, "
            f"target_metric='{target_metric}', n_targets={len(target_days)})"
        )

        # Inject PCE_initial into df_twin from t80_metrics before building the matrix
        if "PCE_initial" not in df_twin.columns:
            df_twin = df_twin.merge(
                t80_metrics[["PCE_initial"]],
                left_on="cell_name",
                right_index=True,
                how="left",
            )

        df_daily = build_rul_matrix(df_twin, healthy_cohort)
        df_daily.to_parquet(
            FILE_RUL_TARGETS.parent / "08_rul_features_matrix.parquet"
        )

        # Single raw download of the API (uncalibrated)
        df_api_raw = fetch_api_history(df_daily)

        # ----------------------------------------------------------------------
        # 1. HONEST EVALUATION: LOOCV with walk-forward calibration
        # ----------------------------------------------------------------------
        print("\n" + "=" * 85)
        print(" STARTING CROSS-VALIDATION: LEAVE-ONE-CELL-OUT (LOOCV)")
        print("=" * 85)

        all_records: List[dict] = []
        for cell in healthy_cohort:
            train_cells = [c for c in healthy_cohort if c != cell]
            df_train_loocv = df_daily[df_daily["cell_name"].isin(train_cells)]
            blind_model_pce = train_rul_engine(df_train_loocv)

            records = run_dynamic_backtesting(
                df_daily,
                df_api_raw,
                cell,
                blind_model_pce,
                t80_metrics,
                train_cells=train_cells,
                target_days=target_days,
            )
            all_records.extend(records)

        # This MAE represents the true generalization capability of the model
        evaluate_model_performance(all_records)

        # ----------------------------------------------------------------------
        # 2. PRODUCTION MODEL: train on 100% of the data
        # ----------------------------------------------------------------------
        print("\n" + "=" * 85)
        print(" TRAINING FINAL PRODUCTION MODEL (100% DATA)")
        print("=" * 85)
        final_prod_model_pce = train_rul_engine(df_daily)

        # Production calibration + live forecast require the API history.
        # If Open-Meteo was unreachable, skip both gracefully so the LOOCV
        # sensor artifacts and the consolidated RUL targets are still saved.
        if not df_api_raw.empty:
            # Final calibration: full history, all cells
            # (correct for production inference: there is no future to filter)
            reg_temp_prod, reg_irr_prod = fit_calibration(
                df_daily, df_api_raw, max_date=df_daily["Date_Day"].max()
            )

            run_live_production_forecast(
                df_daily,
                healthy_cohort,
                final_prod_model_pce,
                reg_temp_prod,
                reg_irr_prod,
            )
        else:
            logger.warning(
                "API history unavailable; skipping production calibration and "
                "live forecast. LOOCV sensor artifacts will still be saved."
            )

        # ----------------------------------------------------------------------
        # 3. Result persistence
        # ----------------------------------------------------------------------
        if all_records:
            df_results = pd.DataFrame(all_records)
            for m_type, fname in [
                ("Sensor", "08_rul_sim_sensor.parquet"),
                ("API", "08_rul_sim_api.parquet"),
            ]:
                sub = df_results[df_results["Type"] == m_type]
                if not sub.empty:
                    sub.to_parquet(FILE_RUL_TARGETS.parent / fname, index=False)

            # Consolidate official per-cell RUL target
            rul_targets = (
                df_results[df_results["Type"] == "Sensor"]
                .sort_values(["cell_name", "Anchor_Day"])
                .groupby("cell_name")
                .last()
                .reset_index()
                [["cell_name", "Anchor_Day", "RUL_Pred", "True_Survival_Days"]]
                .rename(
                    columns={
                        "Anchor_Day": "last_anchor_day",
                        "RUL_Pred": "rul_pred_days",
                        "True_Survival_Days": "true_survival_days",
                    }
                )
            )
            rul_targets.to_parquet(FILE_RUL_TARGETS, index=False)
            logger.info(f"RUL targets consolidated -> {FILE_RUL_TARGETS}")

    except Exception as exc:
        logger.error(f"Execution aborted: {exc}")


if __name__ == "__main__":
    main()