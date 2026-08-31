"""
Module: src/early_failure_detection.py
Description: Genera y exporta el cerebro del Gemelo Digital en formato .joblib.
Extrae las fechas exactas de colapso termodinámico (T80) y alertas de Machine Learning
necesarias para el dashboard interactivo de Streamlit.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EarlyFailureDT")

# =====================================================================
# RUTAS ABSOLUTAS (Alineadas con app.py)
# =====================================================================
SURVIVAL_DATA_PATH = Path("data/survival/outdoor/survival_dataset.parquet")

# Carpeta de salida para el artefacto
ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")
ARTIFACTS_DIR = ARTIFACTS_PATH.parent

def train_and_export_digital_twin():
    logger.info("Iniciando pipeline del Gemelo Digital...")

    # 1. Carga de datos
    if not SURVIVAL_DATA_PATH.exists():
        logger.error(f"Dataset no encontrado en: {SURVIVAL_DATA_PATH}")
        return

    df = pd.read_parquet(SURVIVAL_DATA_PATH)
    if df.index.name == "Timestamp":
        df = df.reset_index()
    
    df['Datetime'] = pd.to_datetime(df['Timestamp'])
    df['Date_Day'] = df['Datetime'].dt.date
    df['Day_Zero'] = df.groupby('cell_name')['Datetime'].transform('min')
    df['Exposure_Days'] = (df['Datetime'] - df['Day_Zero']).dt.total_seconds() / 86400.0

    # 2. Detección T80 (Cálculo de Fechas Exactas)
    df_daylight = df[df['POA_Irradiance_W_m2'] > 100.0].copy()

    df_daily = (
        df_daylight.groupby(['cell_name', 'Date_Day'])
        .agg(
            pseudo_FF_max=('pseudo_FF', 'max'), 
            Exposure_Days_max=('Exposure_Days', 'max'),
            Datetime_max=('Datetime', 'max')  # Capturamos la hora exacta
        )
        .reset_index()
        .sort_values(by=['cell_name', 'Date_Day'])
    )

    first_3_days = df_daily.groupby('cell_name').head(3)
    idx_max_initial = first_3_days.groupby('cell_name')['pseudo_FF_max'].idxmax()
    initial_peak = first_3_days.loc[idx_max_initial, ['cell_name', 'Exposure_Days_max', 'pseudo_FF_max']].rename(
        columns={'pseudo_FF_max': 'pFF_initial', 'Exposure_Days_max': 'Peak_Day'}
    )

    df_daily = df_daily.merge(initial_peak, on='cell_name', how='left')
    df_daily['T80_threshold'] = df_daily['pFF_initial'] * 0.80

    df_daily['Event'] = (
        (df_daily['Exposure_Days_max'] >= df_daily['Peak_Day']) &
        (df_daily['pseudo_FF_max'] < df_daily['T80_threshold'])
    ).astype(int)

    # Variables de salida para T80
    t80_dates = {}
    survival_days = {}

    for cell, group in df_daily.groupby('cell_name'):
        if group['Event'].sum() > 0:
            first_fail_idx = group['Event'].idxmax()
            t80_dates[cell] = group.loc[first_fail_idx, 'Datetime_max']
            survival_days[cell] = group.loc[first_fail_idx, 'Exposure_Days_max']
        else:
            t80_dates[cell] = pd.NaT
            survival_days[cell] = np.nan

    # Cohorte sana (sobrevive > 14 días)
    healthy_cells = [cell for cell, days in survival_days.items() if pd.isna(days) or days > 14.0]
    logger.info(f"Cohorte sana detectada para entrenamiento ML: {healthy_cells}")

    # 3. Preparación de datos censurados para XGBoost
    df_daylight['Death_Day'] = df_daylight['cell_name'].map(survival_days).fillna(np.inf)
    df_censored = df_daylight[df_daylight['Exposure_Days'] <= df_daylight['Death_Day']].copy()
    
    features = ['POA_Irradiance_W_m2', 'ModuleTemp_C', 'AbsoluteHumidity_g_m3']
    target = 'pseudo_FF'

    df_censored = df_censored.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [target])

    # 4. Entrenamiento XGBoost
    train_mask = df_censored['cell_name'].isin(healthy_cells)
    X_train = df_censored.loc[train_mask, features]
    y_train = df_censored.loc[train_mask, target]

    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Inferencia y Umbrales
    df_censored['Twin_pFF_Pred'] = model.predict(df_censored[features])
    df_censored['Underperformance'] = df_censored['Twin_pFF_Pred'] - df_censored[target]
    
    base_mae = mean_absolute_error(y_train, model.predict(X_train))
    alert_threshold = base_mae * 3.0
    
    df_censored['Digital_Twin_Alert'] = df_censored['Underperformance'] > alert_threshold

    # 6. Cálculo Dinámico de Alertas (>15%)
    df_sorted = df_censored.sort_values(by=['cell_name', 'Datetime']).copy()
    df_sorted['Alert_Int'] = df_sorted['Digital_Twin_Alert'].astype(int)
    df_sorted['Cum_Alert_Count'] = df_sorted.groupby('cell_name')['Alert_Int'].cumsum()
    df_sorted['Cum_Points'] = df_sorted.groupby('cell_name').cumcount() + 1
    df_sorted['Cum_Alert_Pct'] = (df_sorted['Cum_Alert_Count'] / df_sorted['Cum_Points']) * 100.0

    valid_crossing = (
        (df_sorted['Cum_Alert_Pct'] > 15.0) &
        (df_sorted['Cum_Points'] >= 100) &
        (df_sorted['Exposure_Days'] > 3.0)
    )
    
    crossed_15 = df_sorted[valid_crossing].groupby('cell_name').first()
    
    ml_alert_dates = crossed_15['Datetime'].to_dict()
    ml_alert_days = crossed_15['Exposure_Days'].to_dict()

    # 7. Construcción de la Tabla de Resumen para Streamlit
    summary = df_censored.groupby('cell_name').agg(
        Alert_Count=('Digital_Twin_Alert', 'sum'),
        Data_Points=('pseudo_FF', 'count')
    )
    
    summary['alert_freq_pct'] = (summary['Alert_Count'] / summary['Data_Points']) * 100.0
    summary['survival_days'] = summary.index.map(survival_days)
    summary['t80_failure_date'] = summary.index.map(t80_dates)
    
    summary['ml_alert_date'] = summary.index.map(lambda x: ml_alert_dates.get(x, pd.NaT))
    summary['threshold_15pct_day'] = summary.index.map(lambda x: ml_alert_days.get(x, np.nan))

    summary['extrinsic_failure'] = (summary['survival_days'] <= 14.0) | (summary['alert_freq_pct'] > 15.0)

    # Formateo final esperado por app.py
    cols_export = ['alert_freq_pct', 'threshold_15pct_day', 'survival_days', 't80_failure_date', 'ml_alert_date', 'extrinsic_failure']
    df_summary_final = summary[cols_export].copy()

    # 8. Exportación del Artefacto
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "summary_table": df_summary_final,
        "model": model,
        "healthy_cohort": healthy_cells,
        "alert_threshold": alert_threshold
    }
    
    joblib.dump(artifacts, ARTIFACTS_PATH)
    logger.info(f"Artefactos exportados exitosamente en: {ARTIFACTS_PATH}")
    logger.info(f"Dispositivos procesados:\n{df_summary_final[['alert_freq_pct', 't80_failure_date', 'ml_alert_date']]}")

if __name__ == "__main__":
    train_and_export_digital_twin()