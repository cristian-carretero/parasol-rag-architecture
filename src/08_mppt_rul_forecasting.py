"""
Module: src/08_mppt_rul_forecasting.py
Description: PCE Remaining Useful Life (RUL) Forecasting & API Calibration.
             Implements bidirectional kinematics without cummax artifacts,
             walk-forward API-sensor calibration (leakage-free), and soft-countdown
             RUL reconciliation mechanics.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PCE_Forecasting")

ROLLING_WINDOW = 7
SIMULATION_WINDOW = 14
T80_DAMAGE_LIMIT = 1.0 - T80_FRACTION  # Universal 0.20
MIN_VELOCITY = 1e-4
RUL_MAX_DAYS = 3650

FEATURES_RUL_PCE = [
    'Daily_Irradiance_Dose', 'Daily_Max_Temp_C', 'Daily_Median_Humidity',
    'Rolling_Irradiance', 'Rolling_Thermal_load', 'Cumulative_Damage_Lag1'
]


# ==============================================================================
# 1. DATA PIPELINE (NATURAL FLUCTUATION, NO CUMMAX)
# ==============================================================================
def build_rul_matrix(df_twin: pd.DataFrame, healthy_cohort: List[str]) -> pd.DataFrame:
    """
    Build the daily RUL feature matrix from the raw digital twin.
    Aggregates sub-daily measurements into daily statistics and applies a rolling
    median filter (anti-ratchet) instead of a destructive cumulative maximum.
    """
    df = df_twin[df_twin['cell_name'].isin(healthy_cohort)].copy()
    if 'Datetime' not in df.columns:
        df['Datetime'] = pd.to_datetime(df['Timestamp'], utc=True)
    df['Date_Day'] = df['Datetime'].dt.date

    if 'PCE_initial' not in df.columns:
        raise KeyError("Missing 'PCE_initial' in df_twin. It must be merged from t80_metrics.")

    df_daily = df.groupby(['cell_name', 'Date_Day']).agg(
        Daily_Irradiance_Dose=('POA_Irradiance_W_m2', 'sum'),
        Daily_Max_Temp_C=('ModuleTemp_C', 'max'),
        Daily_Median_Humidity=('AbsoluteHumidity_g_m3', 'median'),
        Daily_PCE=('PCE', 'max'),
        Exposure_Days=('Exposure_Days', 'max'),
        PCE_Initial=('PCE_initial', 'first'),
    ).reset_index().sort_values(by=['cell_name', 'Date_Day'])

    # 1. Instantaneous real loss relative to the initial PCE
    df_daily['Instant_Loss'] = (1.0 - df_daily['Daily_PCE'] / df_daily['PCE_Initial'])

    # 2. Deep rolling median (7 days) to stabilize high-frequency noise (anti-ratchet)
    df_daily['Cumulative_Damage'] = (
        df_daily.groupby('cell_name')['Instant_Loss']
        .rolling(ROLLING_WINDOW, min_periods=1).median()
        .reset_index(level=0, drop=True)
    )

    df_daily['Cumulative_Damage_Lag1'] = df_daily.groupby('cell_name')['Cumulative_Damage'].shift(1).fillna(0.0)

    # 3. Increment may be negative (photo-annealing recoveries or optimal weather)
    df_daily['Daily_Damage_Increment'] = df_daily['Cumulative_Damage'] - df_daily['Cumulative_Damage_Lag1']

    df_daily['Rolling_Irradiance'] = df_daily.groupby('cell_name')['Daily_Irradiance_Dose'].rolling(ROLLING_WINDOW, min_periods=1).median().reset_index(level=0, drop=True)
    df_daily['Rolling_Thermal_load'] = df_daily.groupby('cell_name')['Daily_Max_Temp_C'].rolling(ROLLING_WINDOW, min_periods=1).median().reset_index(level=0, drop=True)

    required_cols = list(set(FEATURES_RUL_PCE + ['Daily_Damage_Increment']))
    return df_daily.dropna(subset=required_cols).copy()


def fetch_api_history(df_sensor_daily: pd.DataFrame, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> pd.DataFrame:
    """
    Download raw historical data from Open-Meteo for the deployment coordinates.
    Returns daily aggregated API variables. No calibration is applied here.
    """
    logger.info("Fetching raw historical data from Open-Meteo API...")
    min_date, max_date = df_sensor_daily['Date_Day'].min(), df_sensor_daily['Date_Day'].max()
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={min_date}&end_date={max_date}"
           f"&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation&timezone=Europe%2FMadrid")
    try:
        data = requests.get(url, timeout=15).json()["hourly"]
        df_api = pd.DataFrame({
            "Timestamp": pd.to_datetime(data["time"]),
            "API_Temp_C": data["temperature_2m"],
            "RH_pct": data["relative_humidity_2m"],
            "API_GHI_W_m2": data["shortwave_radiation"],
        })
        df_api["Date_Day"] = df_api["Timestamp"].dt.date
        df_api_daily = df_api.groupby("Date_Day").agg(
            API_Daily_Max_Temp=('API_Temp_C', 'max'),
            API_Daily_Irr_Dose=('API_GHI_W_m2', 'sum'),
            Daily_Mean_RH=('RH_pct', 'mean'),
        ).reset_index()
        logger.info(f"API history fetched: {len(df_api_daily)} daily records.")
        return df_api_daily
    except Exception as e:
        logger.warning(f"API fetch failed: {e}")
        return pd.DataFrame()


def fit_calibration(df_sensor_daily: pd.DataFrame,
                    df_api_raw: pd.DataFrame,
                    max_date,
                    train_cells: Optional[List[str]] = None,
                    window_days: Optional[int] = None
                    ) -> Tuple[LinearRegression, LinearRegression]:
    """
    Fit API->sensor linear regressions using ONLY data with Date_Day <= max_date
    and ONLY the cells provided in `train_cells`. This prevents temporal and
    cell-level information leakage during LOOCV backtesting.

    Args:
        df_sensor_daily: Daily sensor dataframe (must contain 'cell_name', 'Date_Day',
                         'Daily_Max_Temp_C', 'Daily_Irradiance_Dose').
        df_api_raw:      Raw (uncalibrated) daily API dataframe.
        max_date:        Upper date bound for calibration (inclusive).
        train_cells:     Optional list of cells to use. None = all cells.
        window_days:     Optional rolling window length. None = expanding window.
    """
    sensor = df_sensor_daily.copy()
    if train_cells is not None:
        sensor = sensor[sensor['cell_name'].isin(train_cells)]
    sensor = sensor[sensor['Date_Day'] <= max_date]
    if window_days is not None:
        min_allowed = (pd.Timestamp(max_date) - pd.Timedelta(days=window_days)).date()
        sensor = sensor[sensor['Date_Day'] >= min_allowed]

    sensor_daily = sensor[['Date_Day', 'Daily_Max_Temp_C', 'Daily_Irradiance_Dose']].drop_duplicates()
    api = df_api_raw[df_api_raw['Date_Day'] <= max_date]
    merged = pd.merge(sensor_daily, api, on='Date_Day', how='inner').dropna()
    if len(merged) < 10:
        raise ValueError(f"Not enough data to fit calibration ({len(merged)} rows).")

    reg_temp = LinearRegression().fit(merged[['API_Daily_Max_Temp']], merged['Daily_Max_Temp_C'])
    reg_irr = LinearRegression().fit(merged[['API_Daily_Irr_Dose']], merged['Daily_Irradiance_Dose'])
    return reg_temp, reg_irr


def apply_calibration(reg_temp: LinearRegression,
                      reg_irr: LinearRegression,
                      df_api_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Apply calibration regressions to a subset of raw API data and derive
    absolute humidity via the Clausius-Clapeyron approximation.
    Returns a copy; input is not modified.
    """
    out = df_api_subset.copy()
    out['Daily_Max_Temp_C'] = reg_temp.predict(out[['API_Daily_Max_Temp']])
    out['Daily_Irradiance_Dose'] = reg_irr.predict(out[['API_Daily_Irr_Dose']])
    p_sat = 6.112 * np.exp((17.67 * out["Daily_Max_Temp_C"]) / (out["Daily_Max_Temp_C"] + 243.5))
    out["Daily_Median_Humidity"] = (216.68 * (p_sat * (out["Daily_Mean_RH"] / 100.0))) / (out["Daily_Max_Temp_C"] + 273.15)
    return out


# ==============================================================================
# 2. PROGNOSTIC ENGINE (HYBRID VELOCITY KINEMATICS)
# ==============================================================================
def train_rul_engine(df_daily: pd.DataFrame) -> xgb.XGBRegressor:
    """Fit an XGBoost regressor to predict the daily damage increment."""
    model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
    model.fit(df_daily[FEATURES_RUL_PCE], df_daily['Daily_Damage_Increment'])
    return model


def simulate_rul_kinematics(
    current_damage: float,
    rolling_irr: list,
    rolling_temp: list,
    future_weather: pd.DataFrame,
    model: xgb.XGBRegressor,
) -> float:
    """
    Forward-simulate damage over the future weather window and convert the
    integrated velocity into an RUL estimate (days until T80 threshold).
    Includes the asymptotic structural floor that accelerates near end-of-life.
    """
    if current_damage >= T80_DAMAGE_LIMIT:
        return 0.0

    loop_damage = current_damage
    for _, row in future_weather.iterrows():
        day_irr, day_temp = row['Daily_Irradiance_Dose'], row['Daily_Max_Temp_C']
        rolling_irr.append(day_irr)
        rolling_temp.append(day_temp)
        if len(rolling_irr) > ROLLING_WINDOW:
            rolling_irr.pop(0)
            rolling_temp.pop(0)

        x_sim = pd.DataFrame([{
            'Daily_Irradiance_Dose': day_irr, 'Daily_Max_Temp_C': day_temp,
            'Daily_Median_Humidity': row['Daily_Median_Humidity'],
            'Rolling_Irradiance': np.median(rolling_irr),
            'Rolling_Thermal_load': np.median(rolling_temp),
            'Cumulative_Damage_Lag1': loop_damage,
        }])

        loop_damage += float(model.predict(x_sim)[0])

    velocity = max(MIN_VELOCITY, (loop_damage - current_damage) / len(future_weather))

    # Structural floor: gentle in healthy phases, dominant near the death threshold
    damage_ratio = max(0.0, current_damage) / T80_DAMAGE_LIMIT
    structural_floor = 0.0015 + (0.0025 * np.exp(damage_ratio * 1.5))
    weight_structural = min(1.0, max(0.0, damage_ratio))

    final_velocity = (1.0 - weight_structural) * velocity + weight_structural * max(velocity, structural_floor)
    final_velocity = np.maximum(final_velocity, MIN_VELOCITY)

    remaining_damage = max(0.0, T80_DAMAGE_LIMIT - current_damage)
    return min(remaining_damage / final_velocity, RUL_MAX_DAYS)


# ==============================================================================
# 3. BACKTESTING & EVALUATION
# ==============================================================================
def run_dynamic_backtesting(df_daily: pd.DataFrame,
                            df_api_raw: pd.DataFrame,
                            cell: str,
                            model_pce: xgb.XGBRegressor,
                            t80_metrics: pd.DataFrame,
                            train_cells: Optional[List[str]] = None) -> List[dict]:
    """
    Run dynamic backtesting for a single cell across weekly anchors.
    - Sensor mode: uses the cell's real future weather (oracle upper bound).
    - API mode:    calibrates walk-forward at each anchor using only past data
                   from `train_cells`, then applies it to future API weather.
    """
    cell_data = df_daily[df_daily['cell_name'] == cell].sort_values('Exposure_Days')
    if cell_data.empty:
        return []

    true_survival_days = t80_metrics.loc[cell, 'survival_days_pce'] if cell in t80_metrics.index else np.nan
    max_days = cell_data['Exposure_Days'].max()
    anchors = list(range(int(BURN_IN_DAYS), int(max_days) + 1, ROLLING_WINDOW))
    if int(max_days) not in anchors:
        anchors.append(int(max_days))

    records = []

    # --------------------------------------------------------------------------
    # 3.1 SENSOR BACKTESTING (no calibration involved)
    # --------------------------------------------------------------------------
    print(f"\n[{cell}] HISTORICAL PCE RUL TRACKING (Limit: {T80_DAMAGE_LIMIT*100:.1f}%)")
    print("-" * 85)

    prev_anchor_sensor, prev_rul_sensor = None, None
    for anchor in anchors:
        hist_cutoff = cell_data[cell_data['Exposure_Days'] <= anchor]
        if hist_cutoff.empty:
            continue
        actual_day = hist_cutoff.iloc[-1]['Exposure_Days']
        cum_damage = hist_cutoff.iloc[-1]['Cumulative_Damage']

        if cum_damage >= T80_DAMAGE_LIMIT:
            print(f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% | [DEGRADED CELL] Simulation aborted.")
            break

        future = cell_data[cell_data['Exposure_Days'] > anchor].head(SIMULATION_WINDOW)
        if future.empty:
            rul_val, rul_str = np.nan, "  N/A"
            real_rul_str = "  N/A"
        else:
            rul_val = simulate_rul_kinematics(
                cum_damage,
                list(hist_cutoff['Daily_Irradiance_Dose'].tail(ROLLING_WINDOW)),
                list(hist_cutoff['Daily_Max_Temp_C'].tail(ROLLING_WINDOW)),
                future, model_pce)

            if prev_rul_sensor is not None:
                elapsed = actual_day - prev_anchor_sensor
                expected_rul = max(0.0, prev_rul_sensor - elapsed)
                # Soft-countdown: blend raw kinematics with the strict passage of time
                rul_val = (0.5 * rul_val) + (0.5 * expected_rul)
                # Allow a slight rebound under optimal weather but keep a descending ramp
                rul_val = min(rul_val, prev_rul_sensor + 1.0)

            prev_anchor_sensor, prev_rul_sensor = actual_day, rul_val
            rul_str = f"{rul_val:5.1f} Days"
            real_rul = true_survival_days - actual_day if pd.notna(true_survival_days) else np.nan
            real_rul_str = f"{real_rul:5.1f} Days" if pd.notna(real_rul) else "  N/A"

        records.append({
            'cell_name': cell,
            'Anchor_Day': actual_day,
            'True_Survival_Days': true_survival_days,
            'RUL_Pred': rul_val,
            'Type': 'Sensor',
        })
        print(f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% -> Pred RUL: {rul_str} | True RUL: {real_rul_str}")

    # --------------------------------------------------------------------------
    # 3.2 API BACKTESTING (walk-forward calibration, leakage-free)
    # --------------------------------------------------------------------------
    if not df_api_raw.empty:
        print(f"[{cell}] API CALIBRATED BACKTESTING (walk-forward)")
        prev_anchor_api, prev_rul_api = None, None
        for anchor in anchors:
            hist_cutoff = cell_data[cell_data['Exposure_Days'] <= anchor]
            if hist_cutoff.empty:
                continue
            actual_day = hist_cutoff.iloc[-1]['Exposure_Days']
            cum_damage = hist_cutoff.iloc[-1]['Cumulative_Damage']
            anchor_date = hist_cutoff.iloc[-1]['Date_Day']

            if cum_damage >= T80_DAMAGE_LIMIT:
                break

            # Walk-forward fit: only past data and only training cells are visible.
            try:
                reg_temp_wf, reg_irr_wf = fit_calibration(
                    df_daily, df_api_raw,
                    max_date=anchor_date,
                    train_cells=train_cells,
                    window_days=90,   # rolling window; set None for expanding window
                )
            except ValueError:
                continue

            future_raw = df_api_raw[df_api_raw['Date_Day'] > anchor_date].head(SIMULATION_WINDOW)
            if future_raw.empty:
                rul_val, rul_str = np.nan, "  N/A"
                real_rul_str = "  N/A"
            else:
                future = apply_calibration(reg_temp_wf, reg_irr_wf, future_raw)
                rul_val = simulate_rul_kinematics(
                    cum_damage,
                    list(hist_cutoff['Daily_Irradiance_Dose'].tail(ROLLING_WINDOW)),
                    list(hist_cutoff['Daily_Max_Temp_C'].tail(ROLLING_WINDOW)),
                    future, model_pce)

                if prev_rul_api is not None:
                    elapsed = actual_day - prev_anchor_api
                    expected_rul = max(0.0, prev_rul_api - elapsed)
                    rul_val = (0.5 * rul_val) + (0.5 * expected_rul)
                    rul_val = min(rul_val, prev_rul_api + 1.0)

                prev_anchor_api, prev_rul_api = actual_day, rul_val
                rul_str = f"{rul_val:5.1f} Days"
                real_rul = true_survival_days - actual_day if pd.notna(true_survival_days) else np.nan
                real_rul_str = f"{real_rul:5.1f} Days" if pd.notna(real_rul) else "  N/A"

            records.append({
                'cell_name': cell,
                'Anchor_Day': actual_day,
                'True_Survival_Days': true_survival_days,
                'RUL_Pred': rul_val,
                'Type': 'API',
            })
            print(f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% -> Pred RUL: {rul_str} | True RUL: {real_rul_str}")

    return records


def evaluate_model_performance(all_records: List[dict]):
    """Aggregate MAE across LOOCV folds for sensor and API modes."""
    if not all_records:
        return
    df_eval = pd.DataFrame(all_records)
    df_eval['RUL_Real'] = df_eval['True_Survival_Days'] - df_eval['Anchor_Day']
    valid = df_eval[df_eval['RUL_Real'] > 0].copy()
    if valid.empty:
        return

    print("\n===========================================================================")
    print(" MODEL PERFORMANCE EVALUATION")
    print("===========================================================================")
    for m_type in ['Sensor', 'API']:
        sub = valid[valid['Type'] == m_type]
        if not sub.empty:
            mae = np.mean(np.abs(sub['RUL_Pred'] - sub['RUL_Real']))
            print(f"[{m_type:6s}] MAE: {mae:5.2f} days (N={len(sub)})")
    print("===========================================================================\n")


# ==============================================================================
# 4. LIVE PRODUCTION FORECAST
# ==============================================================================
def run_live_production_forecast(df_daily: pd.DataFrame,
                                 healthy_cohort: List[str],
                                 model_pce: xgb.XGBRegressor,
                                 reg_temp: LinearRegression,
                                 reg_irr: LinearRegression):
    """
    Execute a live 14-day production forecast using the Open-Meteo forecast API
    and the production calibration regressions fitted on the full history.
    """
    print("===========================================================================")
    print(" LIVE PRODUCTION FORECAST (API 14-Day Outlook)")
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={DEFAULT_LAT}&longitude={DEFAULT_LON}"
           f"&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
           f"&timezone=Europe%2FMadrid&forecast_days={SIMULATION_WINDOW}")
    try:
        data = requests.get(url, timeout=10).json()["hourly"]
        df_api = pd.DataFrame({
            "Timestamp": pd.to_datetime(data["time"]),
            "API_Temp_C": data["temperature_2m"],
            "RH_pct": data["relative_humidity_2m"],
            "API_GHI_W_m2": data["shortwave_radiation"],
        })
        df_api["Date_Day"] = df_api["Timestamp"].dt.date
        df_forecast = df_api.groupby("Date_Day").agg(
            API_Daily_Max_Temp=('API_Temp_C', 'max'),
            API_Daily_Irr_Dose=('API_GHI_W_m2', 'sum'),
            Daily_Mean_RH=('RH_pct', 'mean'),
        ).reset_index()

        df_forecast['Daily_Max_Temp_C'] = reg_temp.predict(df_forecast[['API_Daily_Max_Temp']])
        df_forecast['Daily_Irradiance_Dose'] = reg_irr.predict(df_forecast[['API_Daily_Irr_Dose']])
        p_sat = 6.112 * np.exp((17.67 * df_forecast["Daily_Max_Temp_C"]) / (df_forecast["Daily_Max_Temp_C"] + 243.5))
        df_forecast["Daily_Median_Humidity"] = (216.68 * (p_sat * (df_forecast["Daily_Mean_RH"] / 100.0))) / (df_forecast["Daily_Max_Temp_C"] + 273.15)

        for cell in healthy_cohort:
            cell_data = df_daily[df_daily['cell_name'] == cell].sort_values('Exposure_Days')
            if cell_data.empty:
                continue
            last = cell_data.iloc[-1]
            actual_day, cum_damage = last['Exposure_Days'], last['Cumulative_Damage']

            rul_str = "[DEGRADED]" if cum_damage >= T80_DAMAGE_LIMIT else \
                f"{simulate_rul_kinematics(cum_damage, list(cell_data['Daily_Irradiance_Dose'].tail(ROLLING_WINDOW)), list(cell_data['Daily_Max_Temp_C'].tail(ROLLING_WINDOW)), df_forecast, model_pce):5.1f} Days"
            print(f"[{cell}] Current: Day {actual_day:4.1f} | Damage: {cum_damage*100:4.1f}% (Limit: {T80_DAMAGE_LIMIT*100:4.1f}%) -> RUL: {rul_str}")
    except Exception as e:
        logger.error(f"Live forecast failed: {e}")


def main():
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
        healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)['healthy_cohort']

        # CRITICAL: Inject PCE_initial into df_twin from t80_metrics before building the matrix
        if 'PCE_initial' not in df_twin.columns:
            df_twin = df_twin.merge(
                t80_metrics[['PCE_initial']],
                left_on='cell_name',
                right_index=True,
                how='left',
            )

        df_daily = build_rul_matrix(df_twin, healthy_cohort)
        df_daily.to_parquet(FILE_RUL_TARGETS.parent / "08_rul_features_matrix.parquet")

        # Single raw download of the API (uncalibrated)
        df_api_raw = fetch_api_history(df_daily)

        # ----------------------------------------------------------------------
        # 1. HONEST EVALUATION: LOOCV with walk-forward calibration
        # ----------------------------------------------------------------------
        print("\n" + "=" * 85)
        print(" INICIANDO VALIDACIÓN CRUZADA: LEAVE-ONE-CELL-OUT (LOOCV)")
        print("=" * 85)

        all_records = []
        for cell in healthy_cohort:
            train_cells = [c for c in healthy_cohort if c != cell]
            df_train_loocv = df_daily[df_daily['cell_name'].isin(train_cells)]
            blind_model_pce = train_rul_engine(df_train_loocv)

            records = run_dynamic_backtesting(
                df_daily, df_api_raw, cell, blind_model_pce, t80_metrics,
                train_cells=train_cells,
            )
            all_records.extend(records)

        # This MAE represents the true generalization capability of the model
        evaluate_model_performance(all_records)

        # ----------------------------------------------------------------------
        # 2. PRODUCTION MODEL: train on 100% of the data
        # ----------------------------------------------------------------------
        print("\n" + "=" * 85)
        print(" ENTRENANDO MODELO FINAL DE PRODUCCIÓN (100% DATOS)")
        print("=" * 85)
        final_prod_model_pce = train_rul_engine(df_daily)

        # Final calibration: full history, all cells (correct for production inference)
        reg_temp_prod, reg_irr_prod = fit_calibration(
            df_daily, df_api_raw, max_date=df_daily['Date_Day'].max()
        )

        run_live_production_forecast(
            df_daily, healthy_cohort, final_prod_model_pce,
            reg_temp_prod, reg_irr_prod,
        )

        # ----------------------------------------------------------------------
        # 3. Result persistence
        # ----------------------------------------------------------------------
        if all_records:
            df_results = pd.DataFrame(all_records)
            for m_type, fname in [('Sensor', "08_rul_sim_sensor.parquet"),
                                  ('API', "08_rul_sim_api.parquet")]:
                sub = df_results[df_results['Type'] == m_type]
                if not sub.empty:
                    sub.to_parquet(FILE_RUL_TARGETS.parent / fname, index=False)

            # Consolidate official per-cell RUL target
            rul_targets = (
                df_results[df_results['Type'] == 'Sensor']
                .sort_values(['cell_name', 'Anchor_Day'])
                .groupby('cell_name')
                .last()
                .reset_index()
                [['cell_name', 'Anchor_Day', 'RUL_Pred', 'True_Survival_Days']]
                .rename(columns={
                    'Anchor_Day': 'last_anchor_day',
                    'RUL_Pred': 'rul_pred_days',
                    'True_Survival_Days': 'true_survival_days',
                })
            )
            rul_targets.to_parquet(FILE_RUL_TARGETS, index=False)
            logger.info(f"RUL targets consolidated → {FILE_RUL_TARGETS}")

    except Exception as e:
        logger.error(f"Execution aborted: {e}")


if __name__ == "__main__":
    main()