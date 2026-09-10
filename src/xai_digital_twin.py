"""
Module: src/xai_digital_twin.py
Description: Explainability (XAI) strictly for the Digital Twin ML models.
1. Global SHAP: Validates the thermodynamic logic learned by XGBoost on healthy cells.
2. Local Surrogate: Explains the environmental triggers behind early ML anomalies.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import (
    EARLY_FAILURE_WINDOW_DAYS,
    FEATURE_IMPORTANCE_MIN,
    FEATURES,
    RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    SURROGATE_TREE_PARAMS,
    XAI_PHYSICAL_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("XAI_DigitalTwin")

plt.rcParams.update({'font.size': 12, 'figure.facecolor': 'white'})


def generate_global_shap(model_pce, model_pff, df_data: pd.DataFrame, healthy_cells: list, output_dir: Path):
    logger.info("Generating Global SHAP footprint for the Digital Twin...")
    
    healthy_data = df_data[
        (df_data['cell_name'].isin(healthy_cells)) & 
        (df_data['Exposure_Days'] > EARLY_FAILURE_WINDOW_DAYS)
    ].sample(n=min(SHAP_SAMPLE_SIZE, len(df_data)), random_state=RANDOM_STATE)

    # SHAP audita al Gemelo Digital: necesita usar TODAS las FEATURES (Seno/Coseno incluidos)
    X_eval = healthy_data[FEATURES]
    clean_features = [f.replace('_', ' ').replace('C', '(°C)').replace('W m2', '(W/m²)').replace('g m3', '(g/m³)') for f in FEATURES]

    for model, name in [(model_pce, "PCE"), (model_pff, "pFF")]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_eval)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_eval, feature_names=clean_features, show=False, plot_type="dot")
        plt.title(f"Global SHAP Impact on Healthy Baseline ({name})", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_global_beeswarm_{name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()


def extract_twin_surrogate(cell_data: pd.DataFrame, alert_day: float) -> Dict[str, Any]:
    action_window = cell_data[cell_data['In_Action_Window'] == True].copy()
    if action_window.empty:
        return {}

    # Caso extremo: Varianza cero (ej. celda M0)
    if action_window['Digital_Twin_Alert'].nunique() < 2:
        if action_window['Digital_Twin_Alert'].iloc[0]:
            return {
                "ML_Alert_Day": float(alert_day),
                "Feature_Importances": {"Catastrophic continuous failure (all features)": 1.0},
                "Rules": ["Catastrophic Early Failure: The cell triggered ML anomalies continuously across all environmental conditions."]
            }
        return {}

    # ÁRBOL LOCAL: Explica la anomalía usando SOLO física (prohibido el tiempo)
    X = action_window[XAI_PHYSICAL_FEATURES]
    y = action_window['Digital_Twin_Alert'].astype(int)

    # Desempaquetado explícito para satisfacer al Type Checker
    tree = DecisionTreeClassifier(
        max_depth=int(SURROGATE_TREE_PARAMS.get("max_depth", 3)),
        min_samples_leaf=int(SURROGATE_TREE_PARAMS.get("min_samples_leaf", 5)),
        class_weight=str(SURROGATE_TREE_PARAMS.get("class_weight", "balanced")),
        random_state=int(SURROGATE_TREE_PARAMS.get("random_state", 42))
    )
    tree.fit(X, y)
    
    if tree.tree_.node_count <= 1:
        return {}

    rules = export_text(tree, feature_names=XAI_PHYSICAL_FEATURES, decimals=1)
    importances = {feat: float(imp) for feat, imp in zip(XAI_PHYSICAL_FEATURES, tree.feature_importances_) if imp > FEATURE_IMPORTANCE_MIN}
    sorted_imps = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    return {
        "ML_Alert_Day": float(alert_day),
        "Feature_Importances": sorted_imps,
        "Rules": rules.strip().split('\n')
    }


if __name__ == "__main__":
    ANOMALY_DIR = Path("data/anomaly/outdoor")
    ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")
    DIAGNOSTICS_DIR = Path("data/anomaly/diagnostics")
    SHAP_OUT_DIR = Path("outputs/figures/xai")
    
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_scored = pd.read_parquet(ANOMALY_DIR / "anomaly_scored_dataset.parquet")
    artifacts = joblib.load(ARTIFACTS_PATH)
    
    summary_table = artifacts.get("summary_table")
    model_pce = artifacts.get("model_pce")
    model_pff = artifacts.get("model_pff")
    healthy_cohort = artifacts.get("healthy_cohort", [])

    # 1. SHAP Global
    if healthy_cohort:
        generate_global_shap(model_pce, model_pff, df_scored, healthy_cohort, SHAP_OUT_DIR)

    # 2. Surrogate Local (Solo para celdas con alerta ML)
    ml_failed_cells = summary_table[summary_table['threshold_pct_day'].notna()].copy()
    logger.info(f"Extracting ML Surrogate Rules for {len(ml_failed_cells)} anomaly devices...")

    report_dict = {}
    for cell, row in ml_failed_cells.iterrows():
        cell_data = df_scored[df_scored['cell_name'] == str(cell)]
        alert_day = float(row['threshold_pct_day'])
        
        report = extract_twin_surrogate(cell_data, alert_day)
        if report:
            report_dict[str(cell)] = report

    out_file = DIAGNOSTICS_DIR / "twin_surrogate_rules.json"
    with open(out_file, 'w') as f:
        json.dump(report_dict, f, indent=4)
        
    logger.info(f"Digital Twin XAI complete. Saved to {out_file.name}")