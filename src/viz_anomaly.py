"""
Module: src/viz_anomaly.py
Description: Visualization functions for the Digital Twin anomaly detection
and Surrogate Tree explainability. Saves outputs to outputs/figures/.
"""

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("VizAnomaly")

def plot_censored_digital_twin(df_twin: pd.DataFrame, cell_name: str, alert_threshold: float, threshold_day: float, output_dir: Path):
    """Generates and saves the Digital Twin telemetry plot."""
    cell_data = df_twin[df_twin['cell_name'] == cell_name].copy()
    if cell_data.empty: return

    t80_thresh = cell_data['T80_Threshold'].iloc[0]
    daily_peaks = cell_data.loc[cell_data.groupby('Date_Day')['pseudo_FF'].idxmax()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top Plot: Telemetry vs Twin
    ax1.plot(cell_data['Exposure_Days'], cell_data['Twin_pFF_Pred'], color='forestgreen', alpha=0.75, linewidth=1.5, label='Digital Twin Baseline')
    ax1.scatter(cell_data['Exposure_Days'], cell_data['pseudo_FF'], color='royalblue', alpha=0.3, s=12, label='Actual Telemetry')
    ax1.scatter(daily_peaks['Exposure_Days'], daily_peaks['pseudo_FF'], color='gold', edgecolors='black', s=65, zorder=5, label='Daily Peak')
    
    alerts = cell_data[cell_data['Digital_Twin_Alert']]
    ax1.scatter(alerts['Exposure_Days'], alerts['pseudo_FF'], color='crimson', marker='x', s=45, label='Directional Anomaly Alert')
    ax1.axhline(y=t80_thresh, color='black', linestyle='--', linewidth=2, label=f'T80 Collapse ({t80_thresh:.3f})')

    if pd.notna(threshold_day):
        ax1.axvline(x=threshold_day, color='purple', linestyle='-.', linewidth=2, label=f'15% Alert Crossed (Day {threshold_day:.3f})')

    ax1.set_title(f"Thermodynamic Anomaly Tracking: [{cell_name}]", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Performance (pseudo_FF)", fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Bottom Plot: Residuals
    ax2.plot(cell_data['Exposure_Days'], cell_data['Underperformance'], color='darkorange', linewidth=1.5, label='Directional Underperformance')
    ax2.axhline(y=alert_threshold, color='red', linestyle=':', linewidth=2, label=f'Alert Threshold ({alert_threshold:.3f})')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    if pd.notna(threshold_day):
        ax2.axvline(x=threshold_day, color='purple', linestyle='-.', linewidth=2)

    ax2.set_xlabel("Exposure Time (Days)", fontsize=11)
    ax2.set_ylabel("Underperformance Magnitude", fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    
    # Save Figure
    save_path = output_dir / f"06_{cell_name}_digital_twin_anomaly.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved anomaly tracking plot: {save_path.name}")
    plt.close()


def plot_surrogate_diagnostics(surrogate_tree, features, cell_name, threshold_day, output_dir: Path):
    """Generates and saves the visual Surrogate Tree and Feature Importance plots."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [2.5, 1]})
    fig.suptitle(f"Root Cause Analysis & Environmental Stressors: [{cell_name}]", fontsize=16, fontweight='bold')

    # Tree Plot
    plot_tree(surrogate_tree, 
              feature_names=features, 
              class_names=['Nominal (0)', 'Anomaly (1)'],
              filled=True, rounded=True, ax=axes[0], 
              proportion=False, fontsize=10, impurity=False, precision=1)
    axes[0].set_title(f"Decision Path to Thermodynamic Failure (Threshold Day: {threshold_day:.2f})", fontsize=13)

    # Feature Importance Plot
    importances = surrogate_tree.feature_importances_
    indices = np.argsort(importances)
    colors = ['#2ca02c' if 'Temp' in features[i] else '#ff7f0e' if 'Irradiance' in features[i] else '#1f77b4' for i in indices]

    axes[1].barh(range(len(indices)), importances[indices], color=colors, align='center', edgecolor='black')
    axes[1].set_yticks(range(len(indices)))
    
    clean_labels = [features[i].replace('_', ' ').replace('C', '(°C)').replace('W m2', '(W/m²)').replace('g m3', '(g/m³)') for i in indices]
    axes[1].set_yticklabels(clean_labels, fontsize=11)
    axes[1].set_title("Primary Degradation Drivers", fontsize=13)
    axes[1].set_xlabel("Relative Feature Importance", fontsize=11)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save Figure
    save_path = output_dir / f"07_{cell_name}_surrogate_tree.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved surrogate diagnostic plot: {save_path.name}")
    plt.close()