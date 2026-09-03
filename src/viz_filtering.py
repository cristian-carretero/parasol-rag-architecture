"""
Module: src/viz_filtering.py
Description: Automated generation of analytical plots for pre and post-filtering data audits.
"""

import math
from pathlib import Path
import logging
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Viz-Filtering")
sns.set_theme(style="whitegrid")

OUTPUT_DIR = Path("outputs/figures/filtering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_histogram(data: pd.Series, name: str, n_bins: int | str = 'auto') -> pd.DataFrame:
    clean_data = data.dropna()
    total = len(clean_data)
    if total == 0:
        return pd.DataFrame()

    counts, bin_edges = np.histogram(clean_data, bins=n_bins)
    df_hist = pd.DataFrame({
        'Start (mV)': bin_edges[:-1],
        'End (mV)': bin_edges[1:],
        'Count': counts,
        'Percent (%)': (counts / total) * 100
    })
    return df_hist

def plot_voltage_span_distribution(jv_df: pd.DataFrame, stage_name: str = "Pre-Filter", filename: str = "voltage_span_distribution.png"):
    logger.info(f"Computing voltage span distribution for stage: {stage_name}...")
    volt_diff_series = (jv_df.groupby(['cell_name', 'id_curve'])['Voltage_V'].agg(np.ptp) * 1000)
    visual_bins = 16
    
    plt.figure(figsize=(12, 7))
    device_names = jv_df['cell_name'].unique()
    plotted_count = 0

    for name in device_names:
        cell_data = volt_diff_series.get(name, pd.Series(dtype=float))
        if not cell_data.empty:
            sns.histplot(cell_data, bins=visual_bins, kde=True, alpha=0.3, label=name)
            plotted_count += 1

    if plotted_count == 0:
        plt.close()
        return

    plt.xlabel(r'$\Delta V = V_{max} - V_{min}$ (mV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Voltage Spans per Cell ({stage_name})', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot successfully saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    FILTERED_DIR = Path("data/filtered/outdoor")
    
    if not FILTERED_DIR.exists():
        logger.error(f"Directory {FILTERED_DIR} not found.")
        exit(1)

    # ==========================================
    # 1. PRE-FILTER VISUALIZATION
    # ==========================================
    device_dirs = [d.name for d in FILTERED_DIR.iterdir() if d.is_dir()]
    processed_dfs = []
    logger.info(f"Scanning raw devices for Pre-Filter visualization: {device_dirs}")

    for name in device_dirs:
        parquet_path = FILTERED_DIR / name / f"{name}_jv.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path, columns=['ScanDirection', 'Voltage_V', 'Timestamp'])
            if not df.empty:
                df['cell_name'] = name
                is_reverse_start = (df['ScanDirection'] == 'Reverse') & (df['ScanDirection'].shift(1) != 'Reverse')
                is_reverse_start.iloc[0] = True
                df['curve'] = is_reverse_start.cumsum()
                processed_dfs.append(df)

    if processed_dfs:
        jv_raw = pd.concat(processed_dfs, axis=0, ignore_index=True)
        jv_raw['id_curve'] = jv_raw.groupby(['cell_name', 'curve']).ngroup()
        jv_raw = jv_raw[jv_raw['Timestamp'].dt.time.between(pd.to_datetime('06:00').time(), pd.to_datetime('22:00').time())]
        plot_voltage_span_distribution(jv_df=jv_raw, stage_name="Pre-Filter (Raw)", filename="voltage_span_pre_filter.png")
        
        del jv_raw
        del processed_dfs
        gc.collect()

    # ==========================================
    # 2. POST-FILTER VISUALIZATION
    # ==========================================
    filtered_parquet_path = FILTERED_DIR / "jv_dataset_filtered.parquet"
    if filtered_parquet_path.exists():
        logger.info("Loading post-filter dataset...")
        
        jv_clean = pd.read_parquet(
            filtered_parquet_path,
            columns=['cell_name', 'id_curve', 'Voltage_V'],
            filters=[('is_curve_valid', '==', 1)]
        )
        
        plot_voltage_span_distribution(jv_df=jv_clean, stage_name="Post-Filter (Clean)", filename="voltage_span_post_filter.png")
    
    logger.info("Filtering visualizations complete.")