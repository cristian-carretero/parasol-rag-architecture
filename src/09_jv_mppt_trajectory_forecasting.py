"""
Module: src/09_jv_mppt_trajectory_forecasting.py
Description: Multivariate autoregressive trajectory forecasting for physical
parameters (PCE, pFF, Voc, Jsc). Trains independent XGBoost kinematics engines
per parameter, using relative normalization and LOOCV to ensure out-of-sample
physical generalization. Produces both the production models and the LOOCV
trajectory artifacts consumed by the dashboard.
"""

import logging
from typing import Dict, List, Tuple, cast

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from src.config import (
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    FILE_TRAJECTORY_MODELS,
    FILE_TRAJECTORY_LOOCV,
    FILE_TRAJECTORY_PRODUCTION,   
    XGB_PARAMS_RUL_PCE,
    DEFAULT_LAT,                 
    DEFAULT_LON,     
    ROLLING_WINDOW,       
    ANCHOR_DAY,            
    FORECAST_HORIZON,    
    TARGET_PARAMS                
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("TrajectoryForecasting")


# ==============================================================================
# MODULE-LEVEL CONFIGURATION
# ==============================================================================
TARGET_PARAMS = ["PCE", "pFF", "Jsc", "Voc"]

BASE_FEATURES = [
    "Daily_Irradiance_Dose",
    "Daily_Max_Temp_C",
    "Daily_Median_Humidity",
    "Rolling_Irradiance",
    "Rolling_Thermal_load",
]


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
) -> Dict[str, xgb.XGBRegressor]:
    """Train one independent XGBoost estimator per physical parameter."""
    models = {}
    for param in targets:
        features = BASE_FEATURES + [f"{param}_Lag1"]
        model = xgb.XGBRegressor(**XGB_PARAMS_RUL_PCE)
        model.fit(df_train[features], df_train[f"{param}_Delta"])
        models[param] = model
    return models


def simulate_trajectory(
    initial_norm_states: Dict[str, float],
    initial_raw_values: Dict[str, float],
    future_weather: pd.DataFrame,
    models: Dict[str, xgb.XGBRegressor],
    historical_weather: pd.DataFrame,
    targets: List[str],
) -> pd.DataFrame:
    """
    Iterate stepwise into the future, predicting normalized increments and
    updating each parameter's state autoregressively.
    """
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

            daily_result[f"Pred_{param}"] = new_norm_state * initial_raw_values[param]

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
    """
    import requests
    from sklearn.linear_model import LinearRegression

    logger.info("Fetching and calibrating Open-Meteo API weather...")
    min_date = df_sensor_daily["Date_Day"].min()
    max_date = df_sensor_daily["Date_Day"].max()

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={min_date}&end_date={max_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation"
        f"&timezone=Europe%2FMadrid"
    )

    data = requests.get(url, timeout=20).json()["hourly"]
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

    # Calibrate against sensors using overlapping days
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

    df_api_daily["Daily_Max_Temp_C"] = reg_temp.predict(df_api_daily[["API_Daily_Max_Temp"]])
    df_api_daily["Daily_Irradiance_Dose"] = reg_irr.predict(df_api_daily[["API_Daily_Irr_Dose"]])

    # Absolute humidity via Magnus-Tetens
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
) -> List[dict]:
    """
    Leave-One-Cell-Out evaluation: for each cell, train on the rest of the
    cohort and forecast its trajectory using sensor weather (ground truth).
    Returns the per-day simulated vs. actual records for dashboard plotting.
    """
    print("\n" + "=" * 85)
    print(f" MULTIVARIATE EVALUATION: LEAVE-ONE-CELL-OUT ({FORECAST_HORIZON}-DAY FORECAST HORIZON)")
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
        future_ground_truth = (
            df_test[df_test["Exposure_Days"] > ANCHOR_DAY].head(FORECAST_HORIZON)
        )

        if hist_cutoff.empty:
            logger.warning(f"[{test_cell}] No data before anchor day {ANCHOR_DAY}. Skipping.")
            continue
        if len(future_ground_truth) < FORECAST_HORIZON:
            logger.warning(
                f"[{test_cell}] Only {len(future_ground_truth)} future days available "
                f"(need {FORECAST_HORIZON}). Skipping."
            )
            continue

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
            }
            for param in targets:
                record[f"Actual_{param}"] = float(truth_actual[param][i])
                record[f"Pred_{param}"] = float(pred_arrays[param][i])
            trajectory_records.append(record)

        # --- MAE metrics per parameter ---
        for param in targets:
            mae = float(mean_absolute_error(truth_actual[param], pred_arrays[param]))
            all_metrics.append({"Test_Cell": test_cell, "Parameter": param, "MAE": mae})

    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        res = df_metrics.groupby("Parameter")["MAE"].mean().reset_index()
        print(f"\nMEAN ABSOLUTE ERROR (MAE) AT {FORECAST_HORIZON}-DAY HORIZON (PHYSICAL UNITS):")
        print(res.to_string(index=False))
        print("=" * 85 + "\n")
    else:
        print("\n[WARNING] No cell had enough data to run the LOOCV.")
        print("=" * 85 + "\n")

    return trajectory_records

def run_production_backtesting(
    df_daily: pd.DataFrame,
    healthy_cohort: List[str],
    targets: List[str],
    production_models: Dict[str, xgb.XGBRegressor],
    api_weather: pd.DataFrame,
) -> List[dict]:
    """
    Production-model backtest: uses the 100%-trained models with API-calibrated
    weather (instead of sensor weather) for the future window.
    """
    records: List[dict] = []

    for cell in healthy_cohort:
        cell_data = df_daily[df_daily["cell_name"] == cell].sort_values("Exposure_Days")
        if cell_data.empty:
            continue

        hist = cell_data[cell_data["Exposure_Days"] <= ANCHOR_DAY]
        future_truth = cell_data[cell_data["Exposure_Days"] > ANCHOR_DAY].head(FORECAST_HORIZON)

        if hist.empty or len(future_truth) < FORECAST_HORIZON:
            logger.warning(f"[{cell}] Insufficient history/future for production backtest. Skipping.")
            continue

        # Take API weather for the same dates as the future window
        future_dates = future_truth["Date_Day"].tolist()
        api_slice = api_weather[api_weather["Date_Day"].isin(future_dates)].copy()
        api_slice = api_slice.sort_values("Date_Day").head(FORECAST_HORIZON)

        if len(api_slice) < FORECAST_HORIZON:
            logger.warning(f"[{cell}] API weather missing days. Skipping production backtest.")
            continue

        # Attach Exposure_Days (from sensors) to API weather
        future_weather = api_slice.merge(
            future_truth[["Date_Day", "Exposure_Days"]],
            on="Date_Day", how="left",
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
            }
            for param in targets:
                record[f"Actual_{param}"] = float(hist_actual[param][i])
                record[f"Pred_{param}"] = float("nan")
            records.append(record)

        # --- Forecast window (Actual + Pred) ---
        truth_actual = {
            p: (future_truth[f"{p}_Smooth"] * future_truth[f"{p}_Initial"]).to_numpy(dtype=float)
            for p in targets
        }
        truth_exposure = future_truth["Exposure_Days"].to_numpy(dtype=float)
        truth_dates = future_truth["Date_Day"].to_numpy()
        pred_arrays = {p: df_sim[f"Pred_{p}"].to_numpy(dtype=float) for p in targets}

        for i in range(len(df_sim)):
            record = {
                "cell_name": cell,
                "Date_Day": truth_dates[i],
                "Exposure_Days": float(truth_exposure[i]),
                "Phase": "Forecast",
            }
            for param in targets:
                record[f"Actual_{param}"] = float(truth_actual[param][i])
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
        except Exception as e:
            logger.warning(f"Production backtest skipped: {e}")
            production_records = []

        # --- Serialization ---
        FILE_TRAJECTORY_MODELS.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(production_models, FILE_TRAJECTORY_MODELS)
        logger.info(f"Trajectory models serialized → {FILE_TRAJECTORY_MODELS}")

        if loocv_records:
            FILE_TRAJECTORY_LOOCV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(loocv_records).to_parquet(FILE_TRAJECTORY_LOOCV, index=False)
            logger.info(f"LOOCV trajectories serialized → {FILE_TRAJECTORY_LOOCV}")
        else:
            logger.warning("No LOOCV trajectories to serialize.")

        if production_records:
            FILE_TRAJECTORY_PRODUCTION.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(production_records).to_parquet(FILE_TRAJECTORY_PRODUCTION, index=False)
            logger.info(f"Production trajectories serialized → {FILE_TRAJECTORY_PRODUCTION}")
        else:
            logger.warning("No production trajectories to serialize.")

    except Exception as e:
        logger.error(f"Execution aborted: {e}")


if __name__ == "__main__":
    main()