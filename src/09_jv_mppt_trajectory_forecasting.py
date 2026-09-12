"""
Module: src/09_jv_mppt_trajectory_forecasting.py
Description: Multivariate Autoregressive Trajectory Forecasting.
Trains independent XGBoost kinematics engines for physical parameters (PCE, pFF, Voc, Jsc).
Uses relative normalization and LOOCV to ensure robust out-of-sample physical generalization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.config import (
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    DIR_RUL,
    XGB_PARAMS_RUL_PCE,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrajectoryForecasting")

ROLLING_WINDOW = 7
TARGET_PARAMS = ['PCE', 'pFF', 'Jsc', 'Voc']

BASE_FEATURES = [
    'Daily_Irradiance_Dose', 'Daily_Max_Temp_C', 'Daily_Median_Humidity',
    'Rolling_Irradiance', 'Rolling_Thermal_load'
]


# ==============================================================================
# 1. DATA PREPARATION & RELATIVE NORMALIZATION
# ==============================================================================
def build_trajectory_matrix(df_healthy: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Groups telemetry by day, normalizes each physical parameter relative to its 
    initial baseline, and extracts lag and delta states for autoregressive modeling.
    """
    df = df_healthy.copy()
    if 'Datetime' not in df.columns:
        df['Datetime'] = pd.to_datetime(df['Timestamp'], utc=True)
    df['Date_Day'] = df['Datetime'].dt.date

    available_targets = [p for p in TARGET_PARAMS if p in df.columns]
    
    agg_funcs = {
        'POA_Irradiance_W_m2': 'sum',
        'ModuleTemp_C': 'max',
        'AbsoluteHumidity_g_m3': 'median',
        'Exposure_Days': 'max'
    }
    for param in available_targets:
        agg_funcs[param] = 'mean'

    df_daily = df.groupby(['cell_name', 'Date_Day']).agg(agg_funcs).reset_index()
    df_daily.rename(columns={
        'POA_Irradiance_W_m2': 'Daily_Irradiance_Dose', 
        'ModuleTemp_C': 'Daily_Max_Temp_C',
        'AbsoluteHumidity_g_m3': 'Daily_Median_Humidity'
    }, inplace=True)
    
    df_daily = df_daily.sort_values(by=['cell_name', 'Date_Day'])

    df_daily['Rolling_Irradiance'] = df_daily.groupby('cell_name')['Daily_Irradiance_Dose'].rolling(ROLLING_WINDOW, min_periods=1).median().reset_index(level=0, drop=True)
    df_daily['Rolling_Thermal_load'] = df_daily.groupby('cell_name')['Daily_Max_Temp_C'].rolling(ROLLING_WINDOW, min_periods=1).median().reset_index(level=0, drop=True)

    for param in available_targets:
        init_vals = df_daily.groupby('cell_name').head(3).groupby('cell_name')[param].mean()
        df_daily[f'{param}_Initial'] = df_daily['cell_name'].map(init_vals)
        
        df_daily[f'{param}_Norm'] = df_daily[param] / (df_daily[f'{param}_Initial'] + 1e-9)
        
        df_daily[f'{param}_Smooth'] = (
            df_daily.groupby('cell_name')[f'{param}_Norm']
            .rolling(ROLLING_WINDOW, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        
        df_daily[f'{param}_Lag1'] = df_daily.groupby('cell_name')[f'{param}_Smooth'].shift(1)
        df_daily[f'{param}_Delta'] = df_daily[f'{param}_Smooth'] - df_daily[f'{param}_Lag1']

    cleaned_df = df_daily.dropna().copy()
    return cleaned_df, available_targets


# ==============================================================================
# 2. KINEMATIC ENGINE: TRAINING & SIMULATION
# ==============================================================================
def train_multivariate_engine(df_train: pd.DataFrame, targets: List[str]) -> Dict[str, xgb.XGBRegressor]:
    """Trains an independent XGBoost estimator for each physical parameter."""
    models = {}
    for param in targets:
        features = BASE_FEATURES + [f'{param}_Lag1']
        
        model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
        model.fit(df_train[features], df_train[f'{param}_Delta'])
        models[param] = model
        
    return models


def simulate_trajectory(
    initial_norm_states: Dict[str, float], 
    initial_raw_values: Dict[str, float],
    future_weather: pd.DataFrame, 
    models: Dict[str, xgb.XGBRegressor],
    historical_weather: pd.DataFrame,
    targets: List[str]
) -> pd.DataFrame:
    """Iterates stepwise into the future, predicting normalized increments and updating states."""
    simulated_days = []
    current_states = initial_norm_states.copy()
    
    rolling_irr = list(historical_weather['Daily_Irradiance_Dose'].tail(ROLLING_WINDOW))
    rolling_temp = list(historical_weather['Daily_Max_Temp_C'].tail(ROLLING_WINDOW))

    for _, row in future_weather.iterrows():
        day_irr, day_temp = row['Daily_Irradiance_Dose'], row['Daily_Max_Temp_C']
        
        rolling_irr.append(day_irr)
        rolling_temp.append(day_temp)
        if len(rolling_irr) > ROLLING_WINDOW:
            rolling_irr.pop(0)
            rolling_temp.pop(0)

        daily_result = {'Date_Day': row['Date_Day'], 'Exposure_Days': row['Exposure_Days']}
        
        for param in targets:
            features_sim = pd.DataFrame([{
                'Daily_Irradiance_Dose': day_irr, 
                'Daily_Max_Temp_C': day_temp,
                'Daily_Median_Humidity': row['Daily_Median_Humidity'],
                'Rolling_Irradiance': np.median(rolling_irr),
                'Rolling_Thermal_load': np.median(rolling_temp), 
                f'{param}_Lag1': current_states[param]
            }])
            
            pred_delta = float(models[param].predict(features_sim)[0])
            
            new_norm_state = current_states[param] + pred_delta
            new_norm_state = max(0.0, new_norm_state)
            current_states[param] = new_norm_state
            
            daily_result[f'Pred_{param}'] = new_norm_state * initial_raw_values[param]
            
        simulated_days.append(daily_result)

    return pd.DataFrame(simulated_days)


# ==============================================================================
# 3. ROBUST EVALUATION (LEAVE-ONE-CELL-OUT)
# ==============================================================================
def run_loocv_evaluation(df_daily: pd.DataFrame, healthy_cohort: List[str], targets: List[str]) -> None:
    print("\n" + "="*85)
    print(" MULTIVARIATE EVALUATION: LEAVE-ONE-CELL-OUT (30-DAY FORECAST HORIZON)")
    print("="*85)
    
    all_metrics = []
    FORECAST_HORIZON = 30
    
    for test_cell in healthy_cohort:
        df_train = df_daily[df_daily['cell_name'] != test_cell]
        df_test = df_daily[df_daily['cell_name'] == test_cell]
        
        if df_test.empty:
            continue
            
        blind_models = train_multivariate_engine(df_train, targets)
        
        anchor_day = 30.0
        hist_cutoff = df_test[df_test['Exposure_Days'] <= anchor_day]
        future_ground_truth = df_test[df_test['Exposure_Days'] > anchor_day].head(FORECAST_HORIZON)
        
        if hist_cutoff.empty or len(future_ground_truth) < FORECAST_HORIZON:
            continue
            
        initial_norm_states = {p: float(hist_cutoff.iloc[-1][f'{p}_Smooth']) for p in targets}
        initial_raw_values = {p: float(hist_cutoff.iloc[-1][f'{p}_Initial']) for p in targets}
        
        future_weather = future_ground_truth[['Date_Day', 'Exposure_Days', 'Daily_Irradiance_Dose', 'Daily_Max_Temp_C', 'Daily_Median_Humidity']]
        
        df_sim = simulate_trajectory(initial_norm_states, initial_raw_values, future_weather, blind_models, hist_cutoff, targets)
        
        for param in targets:
            y_true_s = future_ground_truth[f'{param}_Smooth'] * future_ground_truth[f'{param}_Initial']
            y_true = np.asarray(y_true_s, dtype=float)
            y_pred = np.asarray(df_sim[f'Pred_{param}'], dtype=float)
            mae = float(mean_absolute_error(y_true, y_pred))
            all_metrics.append({'Test_Cell': test_cell, 'Parameter': param, 'MAE': mae})
            
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        res = df_metrics.groupby('Parameter')['MAE'].mean().reset_index()
        print("\nMEAN ABSOLUTE ERROR (MAE) AT 30-DAY HORIZON (PHYSICAL UNITS):")
        print(res.to_string(index=False))
        print("="*85 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    try:
        logger.info("Loading Digital Twin Healthy Cohort...")
        df_healthy = pd.read_parquet(FILE_HEALTHY_COHORT)
        artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)
        healthy_cohort = cast(List[str], artifacts['healthy_cohort'])

        df_daily, available_targets = build_trajectory_matrix(df_healthy)
        logger.info(f"Target parameters detected for forecasting: {available_targets}")
        
        run_loocv_evaluation(df_daily, healthy_cohort, available_targets)
        
        logger.info("Training final production models with 100% of the healthy cohort...")
        production_models = train_multivariate_engine(df_daily, available_targets)
        
        DIR_RUL.mkdir(parents=True, exist_ok=True)
        joblib.dump(production_models, DIR_RUL / "09_trajectory_models.joblib")
        logger.info("Trajectory forecasting pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Execution aborted: {e}")


if __name__ == "__main__":
    main()