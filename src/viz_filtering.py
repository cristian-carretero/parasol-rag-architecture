"""
Module: src/viz_filtering.py
Description: Automated generation of analytical diagnostic plots for pre- and post-filtering 
data audits. Implements memory-efficient early aggregation to prevent Out-Of-Memory (OOM) 
failures when processing high-density raw telemetry.
"""

import math
from pathlib import Path
import logging
import gc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional MLOps logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Viz-Filtering")
sns.set_theme(style="whitegrid")

OUTPUT_DIR = Path("outputs/figures/filtering")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_voltage_span_distribution(spans_df: pd.DataFrame, stage_name: str, filename: str) -> None:
    """
    Renders the distribution of J-V sweep voltage spans (ΔV) to audit 
    hardware freezes and incomplete scan cycles.
    """
    logger.info(f"Rendering voltage span distribution for stage: {stage_name}...")
    visual_bins = 16
    
    plt.figure(figsize=(12, 7))
    device_names = spans_df['cell_name'].unique()
    plotted_count = 0

    for name in device_names:
        cell_data = spans_df.loc[spans_df['cell_name'] == name, 'v_span_mV']
        if isinstance(cell_data, pd.Series) and not cell_data.empty:
            sns.histplot(
                data=np.asarray(cell_data, dtype=float),
                bins=visual_bins,
                kde=True,
                alpha=0.3,
                label=name,
            )
            plotted_count += 1

    if plotted_count == 0:
        logger.warning(f"No valid data to plot for {stage_name}.")
        plt.close()
        return

    plt.xlabel(r'$\Delta V = V_{\mathrm{max}} - V_{\mathrm{min}}$ (mV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Voltage Spans per Cell ({stage_name})', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot successfully saved to: {output_path.name}")
    plt.close()

if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed/outdoor")
    FILTERED_DIR = Path("data/filtered/outdoor")
    
    if not PROCESSED_DIR.exists():
        logger.error(f"Directory {PROCESSED_DIR} not found. Ensure upstream processing is complete.")
        exit(1)

    # =========================================================================
    # 1. PRE-FILTER VISUALIZATION (Memory Optimized via Early Aggregation)
    # =========================================================================
    device_dirs = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    span_dfs = []
    logger.info(f"Scanning processed devices for Pre-Filter visualization: {device_dirs}")

    for name in device_dirs:
        parquet_path = PROCESSED_DIR / name / f"{name}_jv.parquet"
        if parquet_path.exists():
            # Load minimal columns to mitigate RAM footprint
            df = pd.read_parquet(parquet_path, columns=['ScanDirection', 'Voltage_V', 'Timestamp'])
            
            if not df.empty:
                # Enforce strict chronological sorting before evaluating directional shifts
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
                df = df.sort_values('Timestamp')
                
                is_reverse_start = (df['ScanDirection'] == 'Reverse') & (df['ScanDirection'].shift(1) != 'Reverse')
                is_reverse_start.iloc[0] = True
                df['curve'] = is_reverse_start.cumsum()
                
                # Filter to daylight operational hours
                df = df[df['Timestamp'].dt.hour.between(6, 22)]
                
                # EARLY AGGREGATION: Reduces millions of rows to ~1000 per device, preventing OOM
                spans = df.groupby('curve')['Voltage_V'].agg(lambda x: x.max() - x.min()) * 1000
                
                cell_df = spans.reset_index(name='v_span_mV')
                cell_df['cell_name'] = name
                span_dfs.append(cell_df)
                
                # Explicit garbage collection flush
                del df
                gc.collect()

    if span_dfs:
        pre_filter_spans = pd.concat(span_dfs, ignore_index=True)
        plot_voltage_span_distribution(
            spans_df=pre_filter_spans, 
            stage_name="Pre-Filter (Raw)", 
            filename="voltage_span_pre_filter.png"
        )
        
        del pre_filter_spans, span_dfs
        gc.collect()

    # =========================================================================
    # 2. POST-FILTER VISUALIZATION
    # =========================================================================
    filtered_parquet_path = FILTERED_DIR / "jv_dataset_filtered.parquet"
    if filtered_parquet_path.exists():
        logger.info("Loading post-filter dataset...")
        
        jv_clean = pd.read_parquet(
            filtered_parquet_path,
            columns=['cell_name', 'id_curve', 'Voltage_V'],
            filters=[('is_curve_valid', '==', 1)]
        )
        
        # Aggregate spans for the valid, clean dataset
        clean_spans = jv_clean.groupby(['cell_name', 'id_curve'])['Voltage_V'].agg(lambda x: x.max() - x.min()) * 1000
        clean_spans_df = clean_spans.reset_index(name='v_span_mV')
        
        plot_voltage_span_distribution(
            spans_df=clean_spans_df, 
            stage_name="Post-Filter (Clean)", 
            filename="voltage_span_post_filter.png"
        )
    
    logger.info("Filtering diagnostic visualizations complete.")