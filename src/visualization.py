"""
Module: src/visualization.py
Description: Automated generation and saving of analytical plots (pre and post filtering)
with professional MLOps logging standards.
"""

from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional logging configuration matching data_processing.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Visualization")

# Professional aesthetic configuration
sns.set_theme(style="whitegrid")

def analyze_histogram(data: pd.Series, name: str, n_bins: int | str = 'auto') -> pd.DataFrame:
    """Calculates numerical counts and percentage distributions for histogram bins."""
    clean_data = data.dropna()
    total = len(clean_data)

    if total == 0:
        logger.warning(f"No valid data available to analyze histogram for {name}.")
        return pd.DataFrame()

    counts, bin_edges = np.histogram(clean_data, bins=n_bins)

    df_hist = pd.DataFrame({
        'Start (mV)': bin_edges[:-1],
        'End (mV)': bin_edges[1:],
        'Count': counts,
        'Percent (%)': (counts / total) * 100
    })
    return df_hist

def plot_voltage_span_distribution(
    jv_df: pd.DataFrame, 
    stage_name: str = "Pre-Filter", 
    filename: str = "voltage_span_distribution.png"
) -> dict[str, pd.DataFrame]:
    """
    General function to plot and numerically audit the voltage span distribution (Delta V)
    grouped by cell and curve for both pre-filtered and post-filtered data.
    """
    logger.info(f"Computing voltage span distribution for stage: {stage_name}...")

    # Calculate voltage span (Delta V = Vmax - Vmin in mV)
    volt_diff_series = (
        jv_df.groupby(['cell_name', 'id_curve'])['Voltage_V']
          .agg(np.ptp) * 1000
    )

    visual_bins = 16
    plt.figure(figsize=(12, 7))

    device_names = jv_df['cell_name'].unique()
    plotted_count = 0

    for name in device_names:
        cell_data = volt_diff_series.get(name, pd.Series(dtype=float))
        if not cell_data.empty:
            sns.histplot(
                cell_data,
                bins=visual_bins,
                kde=True,
                alpha=0.3,
                label=name
            )
            plotted_count += 1

    if plotted_count == 0:
        logger.warning(f"No cell data found to plot for stage: {stage_name}")
        plt.close()
        return {}

    plt.xlabel(r'$\Delta V = V_{max} - V_{min}$ (mV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Voltage Spans per Cell ({stage_name})', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Automatically save figure
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot successfully saved to: {output_path}")
    plt.close()

    # Generate analytical tables per cell
    analysis_tables = {}
    for name in device_names:
        cell_data = volt_diff_series.get(name, pd.Series(dtype=float))
        if not cell_data.empty:
            analysis_tables[name] = analyze_histogram(cell_data, f"Cell {name}", n_bins=visual_bins)

    return analysis_tables


if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed/outdoor")
    
    if not PROCESSED_DIR.exists():
        logger.error(f"Directory {PROCESSED_DIR} not found. Run data_processing.py first.")
        raise FileNotFoundError(f"Directory {PROCESSED_DIR} not found.")

    device_dirs = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    processed_dfs = []

    logger.info(f"Scanning processed outdoor devices for Pre-Filter visualization: {device_dirs}")

    for name in device_dirs:
        parquet_path = PROCESSED_DIR / name / f"{name}_jv.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if df.empty:
                continue

            df['cell_name'] = name
            
            # Physical scan sequence curve identification (Reverse -> Forward cycle)
            is_reverse_start = (df['ScanDirection'] == 'Reverse') & (df['ScanDirection'].shift(1) != 'Reverse')
            is_reverse_start.iloc[0] = True
            df['curve'] = is_reverse_start.cumsum()

            processed_dfs.append(df)
        else:
            logger.warning(f"Parquet file not found for device: {name}")

    if processed_dfs:
        jv_raw = pd.concat(processed_dfs, axis=0, ignore_index=True)
        jv_raw['id_curve'] = jv_raw.groupby(['cell_name', 'curve']).ngroup()

        # Daylight window filtering (06:00 - 22:00)
        initial_records = len(jv_raw)
        jv_raw = jv_raw[
            jv_raw['Timestamp'].dt.time.between(
                pd.to_datetime('06:00').time(),
                pd.to_datetime('22:00').time()
            )
        ]
        logger.info(f"Applied daytime window filter: {len(jv_raw):,} / {initial_records:,} records retained.")

        # Execute pre-filter pipeline plotting and analysis
        plot_voltage_span_distribution(
            jv_df=jv_raw,
            stage_name="Pre-Filter (Raw Data)",
            filename="voltage_span_pre_filter.png"
        )
    else:
        logger.warning("No processed dataframes available for visualization.")