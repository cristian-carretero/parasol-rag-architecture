import logging
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np

# MLOps logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VizAnomaly")

def plot_censored_digital_twin(
    df_twin: pd.DataFrame, 
    cell_name: str, 
    alert_thresh_pce: float, 
    alert_thresh_pff: float, 
    threshold_day: float, 
    output_dir: Path
) -> None:
    """
    Generates the Dual Digital Twin telemetry tracking plot (PCE + pFF).
    """
    cell_data = df_twin[df_twin['cell_name'] == cell_name].copy()
    if cell_data.empty: 
        return

    t80_thresh = cell_data['T80_Threshold'].iloc[0]
    daily_peaks = cell_data.loc[cell_data.groupby('Date_Day')['PCE'].idxmax()]

    # Ampliamos a 3 paneles: 1. PCE, 2. pFF, 3. Underperformance Normalizado
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1.5]})

    # --- TOP PLOT: PCE (Operational Twin) ---
    ax1.plot(cell_data['Exposure_Days'], cell_data['Twin_PCE_Pred'], color='forestgreen', alpha=0.75, linewidth=1.5, label='Twin Baseline (PCE)')
    ax1.scatter(cell_data['Exposure_Days'], cell_data['PCE'], color='royalblue', alpha=0.3, s=12, rasterized=True, label='Actual PCE')
    ax1.scatter(daily_peaks['Exposure_Days'], daily_peaks['PCE'], color='gold', edgecolors='black', s=65, zorder=5, label='Daily Peak')
    
    # Alertas específicas de PCE (Ignorando los NaNs nocturnos)
    alerts_pce = cell_data[cell_data['Alert_PCE'] == True]
    ax1.scatter(alerts_pce['Exposure_Days'], alerts_pce['PCE'], color='crimson', marker='x', s=45, rasterized=True, label='PCE Anomaly')
    ax1.axhline(y=t80_thresh, color='black', linestyle='--', linewidth=2, label=f'T80 Collapse ({t80_thresh:.2f})')

    ax1.set_title(f"Dual Digital Twin Tracking: [{cell_name}]", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Efficiency (PCE %)", fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # --- MIDDLE PLOT: pFF (Structural Twin) ---
    ax2.plot(cell_data['Exposure_Days'], cell_data['Twin_pFF_Pred'], color='teal', alpha=0.75, linewidth=1.5, label='Twin Baseline (pFF)')
    ax2.scatter(cell_data['Exposure_Days'], cell_data['pFF'], color='mediumpurple', alpha=0.3, s=12, rasterized=True, label='Actual pFF')
    
    # Alertas específicas de pFF (Ignorando los NaNs nocturnos)
    alerts_pff = cell_data[cell_data['Alert_pFF'] == True]
    ax2.scatter(alerts_pff['Exposure_Days'], alerts_pff['pFF'], color='darkorange', marker='X', s=45, rasterized=True, label='pFF Anomaly (Structural)')
    
    ax2.set_ylabel("Morphology (pFF)", fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # --- BOTTOM PLOT: Normalized Residuals ---
    # Normalizamos por su respectivo umbral. Si ratio > 1.0, es una alerta.
    cell_data['Norm_Resid_PCE'] = cell_data['Underperformance_PCE'] / alert_thresh_pce
    cell_data['Norm_Resid_pFF'] = cell_data['Underperformance_pFF'] / alert_thresh_pff

    ax3.plot(cell_data['Exposure_Days'], cell_data['Norm_Resid_PCE'], color='crimson', alpha=0.7, linewidth=1, label='PCE Criticality Ratio')
    ax3.plot(cell_data['Exposure_Days'], cell_data['Norm_Resid_pFF'], color='darkorange', alpha=0.7, linewidth=1, label='pFF Criticality Ratio')
    ax3.axhline(y=1.0, color='red', linestyle=':', linewidth=2, label='Alert Trigger Threshold (Ratio = 1)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    if pd.notna(threshold_day):
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=threshold_day, color='purple', linestyle='-.', linewidth=2)
        ax3.text(threshold_day + 0.1, 1.1, 'Cumulative 15% Alert Limit', color='purple', fontweight='bold')

    ax3.set_xlabel("Exposure Time (Days)", fontsize=11)
    ax3.set_ylabel("Criticality Ratio", fontsize=11)
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = output_dir / f"01_{cell_name}_dual_twin_anomaly.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved anomaly tracking plot: {save_path.name}")
    plt.close()


def plot_surrogate_diagnostics(
    surrogate_tree, 
    features: list, 
    cell_name: str, 
    threshold_day: float, 
    output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [2.5, 1]})
    fig.suptitle(f"Root Cause Analysis & Environmental Stressors: [{cell_name}]", fontsize=16, fontweight='bold')

    plot_tree(
        surrogate_tree, 
        feature_names=features, 
        class_names=['Nominal (0)', 'Anomaly (1)'],
        filled=True, rounded=True, ax=axes[0], proportion=False, fontsize=10, impurity=False, precision=2
    )
    axes[0].set_title(f"Decision Path to Thermodynamic Failure (Action Window: {threshold_day:.2f} Days)", fontsize=13)

    importances = surrogate_tree.feature_importances_
    indices = np.argsort(importances)
    
    # Mapa de colores actualizado con los gradientes térmicos y de humedad
    color_map = {
        'ModuleTemp_C': '#d62728',          # Red (Instant Heat)
        'POA_Irradiance_W_m2': '#ff7f0e',   # Orange (Instant Light)
        'AbsoluteHumidity_g_m3': '#1f77b4', # Blue (Instant Moisture)
        'Delta_Temp_C_per_h': '#8c564b',    # Brown (Thermal Shock / Gradient)
        'Delta_Hum_g_m3_per_h': '#9467bd'   # Purple (Moisture Ingress Rate)
    }
    colors = [color_map.get(features[i], '#7f7f7f') for i in indices]

    axes[1].barh(range(len(indices)), importances[indices], color=colors, align='center', edgecolor='black')
    axes[1].set_yticks(range(len(indices)))
    
    # Limpieza de strings dinámica actualizada
    clean_labels = [
        features[i].replace('_', ' ')
                   .replace('ModuleTemp C', 'Temp Módulo (°C)')
                   .replace('POA Irradiance W m2', 'Irradiancia (W/m²)')
                   .replace('AbsoluteHumidity g m3', 'Humedad (g/m³)')
                   .replace('Delta Temp C per h', 'Choque Térmico (ΔT/h)')
                   .replace('Delta Hum g m3 per h', 'Ingreso Humedad (ΔH/h)')
        for i in indices
    ]
    
    axes[1].set_yticklabels(clean_labels, fontsize=11)
    axes[1].set_title("Primary Degradation Drivers", fontsize=13)
    axes[1].set_xlabel("Relative Feature Importance", fontsize=11)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = output_dir / f"02_{cell_name}_surrogate_tree.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved surrogate diagnostic plot: {save_path.name}")
    plt.close()


if __name__ == "__main__":
    from src.early_screening import extract_surrogate_rules
    import joblib
    
    ANOMALY_DIR = Path("data/anomaly/outdoor")
    ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")
    FIGURES_DIR = Path("outputs/figures/early_screening")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = ANOMALY_DIR / "anomaly_scored_dataset.parquet"

    if not parquet_path.exists() or not ARTIFACTS_PATH.exists():
        logger.error("Missing artifacts. Please execute src/early_screening.py first.")
        exit(1)

    logger.info("Loading anomaly dataset and ML artifacts...")
    df_twin_results = pd.read_parquet(parquet_path)
    artifacts = joblib.load(ARTIFACTS_PATH)
    
    summary_table = artifacts["summary_table"]
    
    # AHORA EXTRAEMOS AMBOS UMBRALES
    alert_threshold_pce = artifacts["alert_thresholds"]["pce"]
    alert_threshold_pff = artifacts["alert_thresholds"]["pff"]

    defective_cells = summary_table[summary_table['extrinsic_failure']].index.tolist()
    logger.info(f"Generating visual diagnostic profiles for rejected devices: {defective_cells}")

    for cell_id in defective_cells:
        raw_t_day = summary_table.loc[cell_id, 'threshold_15pct_day']
        t_day = float(raw_t_day) if pd.notna(raw_t_day) else np.nan
        
        # 1. Digital Twin Telemetry Plot (Pasamos ambos umbrales)
        plot_censored_digital_twin(df_twin_results, cell_id, alert_threshold_pce, alert_threshold_pff, t_day, FIGURES_DIR)
        
        # 2. Extract logic rules and Plot Surrogate Tree (XAI)
        tree_model, feat_names, _ = extract_surrogate_rules(df_twin_results, cell_id, t_day)
        if tree_model is not None and feat_names is not None:
            plot_surrogate_diagnostics(tree_model, feat_names, cell_id, t_day, FIGURES_DIR)
            
    logger.info("All anomaly tracking visualizations generated successfully.")