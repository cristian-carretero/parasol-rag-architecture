"""
Module: src/rul_forecasting.py
Description: PCE Remaining Useful Life (RUL) Forecasting & API Calibration.
Implements bidirectional kinematics without cummax artifacts and soft-countdown RUL mechanics.
"""

import logging
from pathlib import Path
from typing import List, Tuple

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

    # 1. Pérdida instantánea real
    df_daily['Instant_Loss'] = (1.0 - df_daily['Daily_PCE'] / df_daily['PCE_Initial'])

    # 2. Mediana móvil a 7 DÍAS para estabilizar el ruido (sin cummax destructivo)
    df_daily['Cumulative_Damage'] = (
        df_daily.groupby('cell_name')['Instant_Loss']
        .rolling(ROLLING_WINDOW, min_periods=1).median()
        .reset_index(level=0, drop=True)
    )
    
    df_daily['Cumulative_Damage_Lag1'] = df_daily.groupby('cell_name')['Cumulative_Damage'].shift(1).fillna(0.0)
    
    # 3. El incremento puede ser negativo (recuperaciones por foto-annealing o clima óptimo)
    df_daily['Daily_Damage_Increment'] = df_daily['Cumulative_Damage'] - df_daily['Cumulative_Damage_Lag1']

    df_daily['Rolling_Irradiance'] = df_daily.groupby('cell_name')['Daily_Irradiance_Dose'].rolling(ROLLING_WINDOW, min_periods=1).median().reset_index(level=0, drop=True)
    df_daily['Rolling_Thermal_load'] = df_daily.groupby('cell_name')['Daily_Max_Temp_C'].rolling(ROLLING_WINDOW, min_periods=1).median().reset_index(level=0, drop=True)

    required_cols = list(set(FEATURES_RUL_PCE + ['Daily_Damage_Increment']))
    return df_daily.dropna(subset=required_cols).copy()


def calibrate_api_to_sensors(
    df_sensor_daily: pd.DataFrame, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON
) -> Tuple[LinearRegression, LinearRegression, pd.DataFrame]:
    logger.info("Calibrating Open-Meteo API parameters to local sensors...")
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
            "API_GHI_W_m2": data["shortwave_radiation"]
        })
        df_api["Date_Day"] = df_api["Timestamp"].dt.date
        df_api_daily = df_api.groupby("Date_Day").agg(
            API_Daily_Max_Temp=('API_Temp_C', 'max'),
            API_Daily_Irr_Dose=('API_GHI_W_m2', 'sum'),
            Daily_Mean_RH=('RH_pct', 'mean')
        ).reset_index()

        df_merged = pd.merge(df_sensor_daily[['Date_Day', 'Daily_Max_Temp_C', 'Daily_Irradiance_Dose']].drop_duplicates(), df_api_daily, on='Date_Day', how='inner').dropna()

        reg_temp = LinearRegression().fit(df_merged[['API_Daily_Max_Temp']], df_merged['Daily_Max_Temp_C'])
        reg_irr = LinearRegression().fit(df_merged[['API_Daily_Irr_Dose']], df_merged['Daily_Irradiance_Dose'])

        df_api_daily['Daily_Max_Temp_C'] = reg_temp.predict(df_api_daily[['API_Daily_Max_Temp']])
        df_api_daily['Daily_Irradiance_Dose'] = reg_irr.predict(df_api_daily[['API_Daily_Irr_Dose']])

        p_sat = 6.112 * np.exp((17.67 * df_api_daily["Daily_Max_Temp_C"]) / (df_api_daily["Daily_Max_Temp_C"] + 243.5))
        df_api_daily["Daily_Median_Humidity"] = (216.68 * (p_sat * (df_api_daily["Daily_Mean_RH"] / 100.0))) / (df_api_daily["Daily_Max_Temp_C"] + 273.15)
        return reg_temp, reg_irr, df_api_daily
    except Exception as e:
        logger.warning(f"API Calibration failed: {e}")
        return LinearRegression().fit([[20]], [25]), LinearRegression().fit([[100]], [105]), pd.DataFrame()


# ==============================================================================
# 2. PROGNOSTIC ENGINE (HYBRID VELOCITY KINEMATICS)
# ==============================================================================
def train_rul_engine(df_daily: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
    model.fit(df_daily[FEATURES_RUL_PCE], df_daily['Daily_Damage_Increment'])
    return model

def simulate_rul_kinematics(
    current_damage: float, rolling_irr: list, rolling_temp: list, future_weather: pd.DataFrame, model: xgb.XGBRegressor
) -> float:
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
            'Cumulative_Damage_Lag1': loop_damage
        }])

        loop_damage += float(model.predict(x_sim)[0])

    velocity = max(MIN_VELOCITY, (loop_damage - current_damage) / len(future_weather))
    
    # Suelo estructural calibrado: suave en fases sanas, dominante cerca de la muerte
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
def run_dynamic_backtesting(df_daily: pd.DataFrame, df_api_hist: pd.DataFrame, cell: str, model_pce, t80_metrics: pd.DataFrame) -> List[dict]:
    cell_data = df_daily[df_daily['cell_name'] == cell].sort_values('Exposure_Days')
    if cell_data.empty: return []

    true_survival_days = t80_metrics.loc[cell, 'survival_days_pce'] if cell in t80_metrics.index else np.nan
    max_days = cell_data['Exposure_Days'].max()
    anchors = list(range(int(BURN_IN_DAYS), int(max_days) + 1, ROLLING_WINDOW))
    if int(max_days) not in anchors: anchors.append(int(max_days))

    print(f"\n[{cell}] HISTORICAL PCE RUL TRACKING (Limit: {T80_DAMAGE_LIMIT*100:.1f}%)")
    print("-" * 85)

    records = []
    
    # 3.1 Sensor Backtesting
    prev_anchor_sensor = None
    prev_rul_sensor = None

    for anchor in anchors:
        hist_cutoff = cell_data[cell_data['Exposure_Days'] <= anchor]
        if hist_cutoff.empty: continue
        actual_day = hist_cutoff.iloc[-1]['Exposure_Days']
        cum_damage = hist_cutoff.iloc[-1]['Cumulative_Damage']

        if cum_damage >= T80_DAMAGE_LIMIT: 
            print(f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% | [DEGRADED CELL] Simulation aborted.")
            break

        future = cell_data[cell_data['Exposure_Days'] > anchor].head(SIMULATION_WINDOW)
        if future.empty:
            rul_val, rul_str = np.nan, "  N/A"
        else:
            rul_val = simulate_rul_kinematics(cum_damage, list(hist_cutoff['Daily_Irradiance_Dose'].tail(ROLLING_WINDOW)), list(hist_cutoff['Daily_Max_Temp_C'].tail(ROLLING_WINDOW)), future, model_pce)
            
            if prev_rul_sensor is not None:
                elapsed = actual_day - prev_anchor_sensor
                expected_rul = max(0.0, prev_rul_sensor - elapsed)
                # Soft Countdown: Promedia la cinemática pura con el paso estricto del tiempo
                rul_val = (0.5 * rul_val) + (0.5 * expected_rul)
                # Permite un leve rebote si el clima es óptimo, pero mantiene la rampa descendente general
                rul_val = min(rul_val, prev_rul_sensor + 1.0)
            
            prev_anchor_sensor = actual_day
            prev_rul_sensor = rul_val
            rul_str = f"{rul_val:5.1f} Days"
            
            real_rul = true_survival_days - actual_day if pd.notna(true_survival_days) else np.nan
            real_rul_str = f"{real_rul:5.1f} Days" if pd.notna(real_rul) else "  N/A"
            records.append({'cell_name': cell, 'Anchor_Day': actual_day, 'True_Survival_Days': true_survival_days, 'RUL_Pred': rul_val, 'Type': 'Sensor'})
            print(f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% -> Pred RUL: {rul_str} | True RUL: {real_rul_str}")

    # 3.2 API Backtesting
    if not df_api_hist.empty:
        print(f"[{cell}] API CALIBRATED BACKTESTING")
        prev_anchor_api = None
        prev_rul_api = None

        for anchor in anchors:
            hist_cutoff = cell_data[cell_data['Exposure_Days'] <= anchor]
            if hist_cutoff.empty: continue
            actual_day, cum_damage, anchor_date = hist_cutoff.iloc[-1]['Exposure_Days'], hist_cutoff.iloc[-1]['Cumulative_Damage'], hist_cutoff.iloc[-1]['Date_Day']

            if cum_damage >= T80_DAMAGE_LIMIT: break

            future = df_api_hist[df_api_hist['Date_Day'] > anchor_date].head(SIMULATION_WINDOW)
            if future.empty:
                rul_val, rul_str = np.nan, "  N/A"
            else:
                rul_val = simulate_rul_kinematics(cum_damage, list(hist_cutoff['Daily_Irradiance_Dose'].tail(ROLLING_WINDOW)), list(hist_cutoff['Daily_Max_Temp_C'].tail(ROLLING_WINDOW)), future, model_pce)
                
                if prev_rul_api is not None:
                    elapsed = actual_day - prev_anchor_api
                    expected_rul = max(0.0, prev_rul_api - elapsed)
                    rul_val = (0.5 * rul_val) + (0.5 * expected_rul)
                    rul_val = min(rul_val, prev_rul_api + 1.0)

                prev_anchor_api = actual_day
                prev_rul_api = rul_val
                rul_str = f"{rul_val:5.1f} Days"
                real_rul = true_survival_days - actual_day if pd.notna(true_survival_days) else np.nan
                real_rul_str = f"{real_rul:5.1f} Days" if pd.notna(real_rul) else "  N/A"
                records.append({'cell_name': cell, 'Anchor_Day': actual_day, 'True_Survival_Days': true_survival_days, 'RUL_Pred': rul_val, 'Type': 'API'})
                print(f"Day {actual_day:4.1f} | Damage: {cum_damage*100:5.1f}% -> Pred RUL: {rul_str} | True RUL: {real_rul_str}")

    return records


def evaluate_model_performance(all_records: List[dict]):
    if not all_records: return
    df_eval = pd.DataFrame(all_records)
    df_eval['RUL_Real'] = df_eval['True_Survival_Days'] - df_eval['Anchor_Day']
    valid = df_eval[df_eval['RUL_Real'] > 0].copy()
    if valid.empty: return

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
def run_live_production_forecast(df_daily: pd.DataFrame, healthy_cohort: List[str], model_pce, reg_temp, reg_irr):
    print("===========================================================================")
    print(" LIVE PRODUCTION FORECAST (API 14-Day Outlook)")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={DEFAULT_LAT}&longitude={DEFAULT_LON}&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation&timezone=Europe%2FMadrid&forecast_days={SIMULATION_WINDOW}"
    try:
        data = requests.get(url, timeout=10).json()["hourly"]
        df_api = pd.DataFrame({"Timestamp": pd.to_datetime(data["time"]), "API_Temp_C": data["temperature_2m"], "RH_pct": data["relative_humidity_2m"], "API_GHI_W_m2": data["shortwave_radiation"]})
        df_api["Date_Day"] = df_api["Timestamp"].dt.date
        df_forecast = df_api.groupby("Date_Day").agg(API_Daily_Max_Temp=('API_Temp_C', 'max'), API_Daily_Irr_Dose=('API_GHI_W_m2', 'sum'), Daily_Mean_RH=('RH_pct', 'mean')).reset_index()

        df_forecast['Daily_Max_Temp_C'] = reg_temp.predict(df_forecast[['API_Daily_Max_Temp']])
        df_forecast['Daily_Irradiance_Dose'] = reg_irr.predict(df_forecast[['API_Daily_Irr_Dose']])
        p_sat = 6.112 * np.exp((17.67 * df_forecast["Daily_Max_Temp_C"]) / (df_forecast["Daily_Max_Temp_C"] + 243.5))
        df_forecast["Daily_Median_Humidity"] = (216.68 * (p_sat * (df_forecast["Daily_Mean_RH"] / 100.0))) / (df_forecast["Daily_Max_Temp_C"] + 273.15)

        for cell in healthy_cohort:
            cell_data = df_daily[df_daily['cell_name'] == cell].sort_values('Exposure_Days')
            if cell_data.empty: continue
            last = cell_data.iloc[-1]
            actual_day, cum_damage = last['Exposure_Days'], last['Cumulative_Damage']
            
            rul_str = "[DEGRADED]" if cum_damage >= T80_DAMAGE_LIMIT else f"{simulate_rul_kinematics(cum_damage, list(cell_data['Daily_Irradiance_Dose'].tail(7)), list(cell_data['Daily_Max_Temp_C'].tail(7)), df_forecast, model_pce):5.1f} Days"
            print(f"[{cell}] Current: Day {actual_day:4.1f} | Damage: {cum_damage*100:4.1f}% (Limit: {T80_DAMAGE_LIMIT*100:4.1f}%) -> RUL: {rul_str}")
    except Exception as e:
        logger.error(f"Live forecast failed: {e}")


def main():
    ANOMALY_DIR = Path("data/anomaly/outdoor")
    SURVIVAL_DIR = Path("data/survival/outdoor")
    ARTIFACTS_DIR = Path("data/anomaly/artifacts")
    RUL_DIR = Path("data/rul")
    RUL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df_twin = pd.read_parquet(ANOMALY_DIR / "anomaly_scored_dataset.parquet")
        t80_metrics = pd.read_parquet(SURVIVAL_DIR / "t80_metrics_table.parquet")
        healthy_cohort = joblib.load(ARTIFACTS_DIR / "early_failure_artifacts.joblib")['healthy_cohort']

        df_daily = build_rul_matrix(df_twin, healthy_cohort)
        
        df_daily.to_parquet(RUL_DIR / "rul_features_matrix.parquet")
        reg_temp, reg_irr, df_api_hist = calibrate_api_to_sensors(df_daily)
        rul_model_pce = train_rul_engine(df_daily)

        all_records = []
        for cell in healthy_cohort:
            records = run_dynamic_backtesting(df_daily, df_api_hist, cell, rul_model_pce, t80_metrics)
            all_records.extend(records)

        evaluate_model_performance(all_records)
        run_live_production_forecast(df_daily, healthy_cohort, rul_model_pce, reg_temp, reg_irr)

        if all_records:
            df_results = pd.DataFrame(all_records)
            for m_type, fname in [('Sensor', "rul_sim_sensor.parquet"), ('API', "rul_sim_api.parquet")]:
                sub = df_results[df_results['Type'] == m_type]
                if not sub.empty: sub.to_parquet(RUL_DIR / fname, index=False)
    except Exception as e:
        logger.error(f"Execution aborted: {e}")

if __name__ == "__main__":
    main()