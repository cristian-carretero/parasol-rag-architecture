"""
Module: src/shap_early_screening.py
Description: Global and local explainability using SHAP values for infant mortality.
Deconstructs the XGBoost Dual Digital Twin predictions to quantify 
the exact thermodynamic drivers of early degradation (Thermal vs. Moisture)
during the 14-day burn-in window.
"""

import logging
from pathlib import Path

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SHAP_EarlyScreening")

FEATURES = [
    'POA_Irradiance_W_m2', 
    'ModuleTemp_C', 
    'AbsoluteHumidity_g_m3', 
    'Cum_Light_Dose_Wh_m2', 
    'Cum_Humidity_Exposure'
]

def generate_shap_diagnostics(
    model_pce, model_pff, df_data: pd.DataFrame, cell_id: str, output_dir: Path
):
    """
    Computes SHAP values for a specific cell to identify the root cause 
    of its divergence from the healthy baseline during the burn-in phase.
    """
    logger.info(f"Generating SHAP thermodynamic footprint for defective device: [{cell_id}]")
    
    # Isolate the cell's data (focusing on the action window where the early failure occurred)
    cell_data = df_data[(df_data['cell_name'] == cell_id) & (df_data['In_Action_Window'] == True)].copy()
    
    if cell_data.empty:
        logger.warning(f"No action-window data available for {cell_id}.")
        return

    X_eval = cell_data[FEATURES]
    
    # 1. Initialize TreeExplainers
    explainer_pce = shap.TreeExplainer(model_pce)
    explainer_pff = shap.TreeExplainer(model_pff)
    
    shap_values_pce = explainer_pce.shap_values(X_eval)
    shap_values_pff = explainer_pff.shap_values(X_eval)

    # 2. Format Feature Names for Plotting
    clean_features = [
        f.replace('_', ' ').replace('C', '(°C)').replace('W m2', '(W/m²)').replace('g m3', '(g/m³)')
        for f in FEATURES
    ]

    # 3. Generate SHAP Summary Plot (Beeswarm) for pFF (Structural Health)
    # We focus the visual plot on pFF because structural mutation precedes power loss
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_pff, 
        X_eval, 
        feature_names=clean_features,
        show=False,
        plot_type="dot"
    )
    plt.title(f"SHAP Impact on Early Structural Fatigue (pFF) - [{cell_id}]", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    pff_plot_path = output_dir / f"shap_beeswarm_pff_{cell_id}.png"
    plt.savefig(pff_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Generate SHAP Bar Plot for PCE (Power Output)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_pce, 
        X_eval, 
        feature_names=clean_features,
        show=False,
        plot_type="bar",
        color="#3B82F6"
    )
    plt.title(f"Mean Absolute SHAP Impact on Early Power Loss (PCE) - [{cell_id}]", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    pce_plot_path = output_dir / f"shap_bar_pce_{cell_id}.png"
    plt.savefig(pce_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"SHAP diagnostics saved to {output_dir.name}/")

if __name__ == "__main__":
    ANOMALY_DIR = Path("data/anomaly/outdoor")
    ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")
    SHAP_OUT_DIR = Path("outputs/figures/xai")
    SHAP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = ANOMALY_DIR / "anomaly_scored_dataset.parquet"

    if not parquet_path.exists() or not ARTIFACTS_PATH.exists():
        logger.error("Missing required datasets or ML artifacts. Ensure src/early_screening.py has run.")
        exit(1)

    # Load Data and Models
    df_scored = pd.read_parquet(parquet_path)
    artifacts = joblib.load(ARTIFACTS_PATH)
    
    summary_table = artifacts.get("summary_table")
    model_pce = artifacts.get("model_pce")
    model_pff = artifacts.get("model_pff")

    if model_pce is None or model_pff is None:
        logger.error("XGBoost models not found in artifacts. Update early_screening.py to export them.")
        exit(1)

    # Filter strictly for cells that failed the Early Screening LOOCV Gate
    defective_cells = summary_table[summary_table['extrinsic_failure']].index.tolist()
    
    logger.info(f"Initializing SHAP Explainer for early-failure cohort: {defective_cells}")

    for cell in defective_cells:
        generate_shap_diagnostics(
            model_pce=model_pce,
            model_pff=model_pff,
            df_data=df_scored,
            cell_id=cell,
            output_dir=SHAP_OUT_DIR
        )