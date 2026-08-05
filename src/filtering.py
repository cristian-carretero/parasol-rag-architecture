"""
Module: src/filtering.py
Description: Physical filtering, artifact removal, and quality control pipeline for J-V curves.
"""

from pathlib import Path
import logging
import warnings
import gc
import pandas as pd
import numpy as np

# Professional logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Filtering")


def _process_single_cell(
    df: pd.DataFrame, 
    name: str, 
    cell_id: int, 
    v_span_thresh: int, 
    freeze_thresh: int, 
    i_span_thresh: float,
    corruption_tol: float = 0.90,
    spike_tol: int = 2
) -> pd.DataFrame:
    """
    Core engine to apply physical, range, and structural filters to a single cell's dataset.
    """
    df_c = df.copy()
    
    # --- 0. PREPARATION ---
    df_c['Voltage_V'] = df_c['Voltage_V'].astype('float32')
    if 'Current_A' in df_c.columns: 
        df_c['Current_A'] = df_c['Current_A'].astype('float32')
    if 'Power_mW' in df_c.columns: 
        df_c['Power_mW'] = df_c['Power_mW'].astype('float32')
    df_c['Timestamp'] = pd.to_datetime(df_c['Timestamp'])

    # --- 1. CURVE IDENTIFICATION ---
    # A new curve strictly begins when the scan sequence starts in 'Reverse'
    is_reverse_start = (df_c['ScanDirection'] == 'Reverse') & (df_c['ScanDirection'].shift(1) != 'Reverse')
    is_reverse_start.iloc[0] = True
    
    df_c['curve'] = is_reverse_start.cumsum()
    df_c['cell_name'] = name
    df_c['cell_id'] = cell_id
    df_c['is_reverse'] = (df_c['ScanDirection'] == 'Reverse').astype('int8') if 'ScanDirection' in df_c.columns else 0

    # --- 2. RANGE CALCULATION ---
    stats = df_c.groupby('curve')['Voltage_V'].agg(['max', 'min']).rename(columns={'max':'v_max', 'min':'v_min'})
    i_stats = df_c.groupby('curve')['Current_A'].agg(['max', 'min']).rename(columns={'max':'i_max', 'min':'i_min'})

    df_c = df_c.merge(stats, on='curve', how='left').merge(i_stats, on='curve', how='left')
    df_c['v_span_mV'] = (df_c['v_max'] - df_c['v_min']) * 1000
    df_c['i_span_A'] = df_c['i_max'] - df_c['i_min']

    # --- 3. NEGATIVE CURRENT FILTER ---
    df_c['is_negative_point'] = (df_c['Current_A'] < 0)
    df_c['is_corrupted'] = df_c.groupby('curve')['is_negative_point'].transform('mean') > corruption_tol
    df_c = df_c[~(df_c['is_negative_point'] & (~df_c['is_corrupted']))].copy()

    # --- 4. HARDWARE FREEZE FILTER ---
    v_span_window = df_c['Voltage_V'].rolling(freeze_thresh, min_periods=1).max() - df_c['Voltage_V'].rolling(freeze_thresh, min_periods=1).min()
    i_span_window = df_c['Current_A'].rolling(freeze_thresh, min_periods=1).max() - df_c['Current_A'].rolling(freeze_thresh, min_periods=1).min()

    df_c['is_curve_frozen'] = ((v_span_window <= 1e-4).groupby(df_c['curve']).transform('mean') > 0.9) | \
                              ((i_span_window <= 1e-6).groupby(df_c['curve']).transform('mean') > 0.9)

    # --- 5. TIME & LENGTH FILTERS ---
    df_c['is_in_time'] = (df_c['Timestamp'].dt.hour >= 6) & (df_c['Timestamp'].dt.hour <= 22)
    is_curve_in_time_window = df_c.groupby('curve')['is_in_time'].transform('all')

    df_c['is_low_voltage'] = df_c['v_span_mV'] <= v_span_thresh
    df_c['is_low_current'] = df_c['i_span_A'] <= i_span_thresh
    df_c['is_night_time'] = ~is_curve_in_time_window

    # --- 6. ARTIFACT & INTEGRITY FILTERS ---
    # A) Outliers Filter
    df_c['d_current_abs'] = df_c.groupby('curve')['Current_A'].diff().abs().bfill()
    df_c['is_outlier_point'] = (df_c['d_current_abs'] / (df_c['i_span_A'] + 1e-9)) > 0.15
    outlier_count = df_c.groupby('curve')['is_outlier_point'].transform('sum')
    
    df_c['is_spike_error'] = (outlier_count > spike_tol)
    df_c = df_c[~(df_c['is_outlier_point'] & (~df_c['is_spike_error']))].copy()

    # B) Structural Symmetry Filter
    fwd_pts = df_c[df_c['is_reverse'] == 0].groupby('curve')['Voltage_V'].count()
    rev_pts = df_c[df_c['is_reverse'] == 1].groupby('curve')['Voltage_V'].count()
    df_c['fwd_len'] = df_c['curve'].map(fwd_pts).fillna(0)
    df_c['rev_len'] = df_c['curve'].map(rev_pts).fillna(0)
    
    ratio = df_c['fwd_len'] / (df_c['rev_len'] + 1e-9)
    df_c['is_asymmetric'] = (df_c['fwd_len'] == 0) | (df_c['rev_len'] == 0) | (ratio < 0.85) | (ratio > 1.15)

    # C) Hysteresis Integrity Filter (Mean Collapse)
    fwd_mean = df_c[df_c['is_reverse'] == 0].groupby('curve')['Current_A'].mean()
    rev_mean = df_c[df_c['is_reverse'] == 1].groupby('curve')['Current_A'].mean()
    df_c['fwd_mean'] = df_c['curve'].map(fwd_mean).fillna(0)
    df_c['rev_mean'] = df_c['curve'].map(rev_mean).fillna(0)
    
    df_c['is_mean_mismatch'] = (df_c['fwd_mean'] - df_c['rev_mean']).abs() / (df_c[['fwd_mean', 'rev_mean']].max(axis=1) + 1e-9) > 0.45

    # --- 7. FINAL VALIDITY CRITERION ---
    invalid_mask = (
        df_c['is_low_voltage'] | df_c['is_low_current'] | df_c['is_night_time'] | 
        df_c['is_curve_frozen'] | df_c['is_corrupted'] | 
        df_c['is_spike_error'] | df_c['is_asymmetric'] | 
        df_c['is_mean_mismatch']
    )
    df_c['is_curve_valid'] = (~invalid_mask).astype('int8')

    return df_c


def process_all_jv_curves(
    raw_dfs_dict: dict[str, pd.DataFrame], 
    v_span_threshold_mv: int = 50, 
    freeze_run_threshold: int = 15, 
    i_span_threshold_A: float = 0.0001,
    spike_tol: int = 2
) -> pd.DataFrame:
    """
    Orchestrates the filtering pipeline across all provided cell DataFrames.
    """
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    logger.info(f"Initiating physical filtering pipeline for {len(raw_dfs_dict)} cells...")

    processed_dfs = []
    
    for cell_id, (name, df) in enumerate(raw_dfs_dict.items()):
        logger.info(f"Applying filter topology to cell: {name}")
        
        cleaned_df = _process_single_cell(
            df=df, 
            name=name, 
            cell_id=cell_id, 
            v_span_thresh=v_span_threshold_mv, 
            freeze_thresh=freeze_run_threshold, 
            i_span_thresh=i_span_threshold_A,
            spike_tol=spike_tol
        )
        processed_dfs.append(cleaned_df)
        gc.collect()

    # Consolidate dataset
    logger.info("Concatenating processed chunks into global DataFrame...")
    jv_all = pd.concat(processed_dfs, axis=0, ignore_index=True)
    jv_all['id_curve'] = jv_all.groupby(['cell_name', 'curve']).ngroup()

    # --- DIAGNOSTICS ---
    logger.info("Generating global diagnostics report...")
    summary = jv_all.groupby(['cell_name', 'curve']).first()

    try:
        val_counts = summary.groupby('cell_name')['is_curve_valid'].agg(['count', 'sum'])
        val_counts.columns = ['Total Cycles Detected', 'Valid Cycles']
        
        discard_cols = [
            'is_low_voltage', 'is_low_current', 'is_night_time', 
            'is_curve_frozen', 'is_corrupted', 'is_spike_error', 
            'is_asymmetric', 'is_mean_mismatch'
        ]
        
        report = (
            f"\n=== FINAL DIAGNOSTICS ===\n"
            f"Total cycles detected across all cells: {len(summary)}\n\n"
            f"Cycle Yield by Cell:\n{val_counts}\n\n"
            f"Global Rejection Triggers (Applied per cycle):\n{summary[discard_cols].sum().to_string()}"
        )
        logger.info(report)
        
    except Exception as e:
        logger.warning(f"Detailed diagnostics output failed: {e}")
        
    warnings.filterwarnings('default', category=RuntimeWarning)
    return jv_all


def analyze_curve_timings(jv_df: pd.DataFrame) -> None:
    """
    Extracts chronological statistics: average measurement durations and idle times.
    """
    logger.info("Analyzing chronometry (measurement durations & idle intervals)...")
    
    # Calculate boundary timestamps per curve
    all_curves_stats = jv_df.groupby(['cell_name', 'id_curve'])['Timestamp'].agg(['min', 'max']).sort_values('min')
    all_curves_stats['is_valid'] = jv_df.groupby(['cell_name', 'id_curve'])['is_curve_valid'].first()

    global_durations, global_intervals = [], []
    report_lines = ["\n=== TIME ANALYSIS PER CELL ==="]

    for cell in jv_df['cell_name'].unique():
        cell_stats = all_curves_stats.loc[cell]
        
        # Idle intervals (excluding massive gaps > 1 hour)
        intervals = cell_stats['min'].diff().dt.total_seconds()
        intervals_clean = intervals[intervals < 3600]
        avg_interval = intervals_clean.mean()
        global_intervals.extend(intervals_clean.dropna().tolist())

        # Sweep duration (valid curves only)
        valid_stats = cell_stats[cell_stats['is_valid'] == 1]
        durations = (valid_stats['max'] - valid_stats['min']).dt.total_seconds()
        avg_duration = durations.mean()
        global_durations.extend(durations.dropna().tolist())

        # Capacity metrics
        total_valid = len(valid_stats)
        valid_df = jv_df[(jv_df['cell_name'] == cell) & (jv_df['is_curve_valid'] == 1)]
        max_daily = valid_df.set_index('Timestamp').resample('D')['id_curve'].nunique().max() if not valid_df.empty else 0

        # Build report
        report_lines.extend([
            f"[{cell}]",
            f"  -> Valid curves detected: {total_valid}",
            f"  -> Avg duration (Rev+Fwd): {avg_duration:.2f} s" if total_valid > 0 else "  -> Avg duration: N/A",
            f"  -> Avg time between sweeps: {avg_interval:.2f} s" if not np.isnan(avg_interval) else "  -> Avg time between sweeps: N/A",
            f"  -> Max daily valid curves: {max_daily}\n"
        ])

    report_lines.extend([
        "=" * 35,
        "=== AGGREGATED GLOBAL AVERAGES ===",
        f"Global Sweep Duration: {np.mean(global_durations):.2f} s" if global_durations else "Global Sweep Duration: N/A",
        f"Global Hardware Idle:  {np.mean(global_intervals):.2f} s" if global_intervals else "Global Hardware Idle: N/A",
        "=" * 35
    ])

    logger.info("\n".join(report_lines))


if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed/outdoor")
    
    if not PROCESSED_DIR.exists():
        logger.error(f"Directory missing: {PROCESSED_DIR}. Ensure data_processing.py ran successfully.")
        raise FileNotFoundError(f"Directory {PROCESSED_DIR} not found.")

    device_dirs = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    raw_data_dict = {}

    logger.info(f"Discovered processed datasets for devices: {device_dirs}")
    
    for name in device_dirs:
        parquet_path = PROCESSED_DIR / name / f"{name}_jv.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if not df.empty:
                raw_data_dict[name] = df
        else:
            logger.warning(f"Expected Parquet file missing for device: {name}")

    if raw_data_dict:
        # 1. Execute the core filtering engine
        jv_dataset_filtered = process_all_jv_curves(
            raw_dfs_dict=raw_data_dict,
            v_span_threshold_mv=50,
            i_span_threshold_A=0.0001,
            spike_tol=2
        )

        # 2. Output final audit metrics
        logger.info("\n--- PIPELINE YIELD AUDIT ---")
        logger.info(f"Raw Datapoints (Rows):\n{jv_dataset_filtered.groupby('is_reverse')['id_curve'].count().to_string()}")
        logger.info(f"Gross Sweeps Detected:\n{jv_dataset_filtered.groupby('is_reverse')['id_curve'].nunique().to_string()}")
        logger.info(f"Net Valid Sweeps (Post-Filter):\n{jv_dataset_filtered[jv_dataset_filtered['is_curve_valid'] == 1].groupby('is_reverse')['id_curve'].nunique().to_string()}")

        # 3. Analyze time dynamics
        analyze_curve_timings(jv_dataset_filtered)
        
        # 4. Save the dataset to disk
        output_path = PROCESSED_DIR / "jv_dataset_filtered.parquet"
        logger.info(f"Saving filtered dataset to {output_path}...")
        
        # Save using pyarrow engine with snappy compression for max performance
        jv_dataset_filtered.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        logger.info("Filtered dataset successfully saved! Pipeline complete.")
        
    else:
        logger.error("No valid Parquet files were loaded into memory.")