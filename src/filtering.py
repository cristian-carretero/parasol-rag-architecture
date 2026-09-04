"""
Module: src/filtering.py
Description: Physical filtering, artifact removal, and quality control pipeline for J-V curves.
Implements vectorized thermodynamic boundary checks and robust chronometric sorting.
"""

from pathlib import Path
import logging
import warnings
import gc
import json
import pandas as pd
import numpy as np

# Professional MLOps logging configuration
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
    spike_tol: int = 2,
    frozen_ratio_tol: float = 0.90,
    mean_mismatch_tol: float = 0.60
) -> pd.DataFrame:
    """
    Applies a highly vectorized physical, temporal, and structural filtering topology 
    to a single device's raw J-V telemetry.
    
    Args:
        df: Raw DataFrame containing J-V telemetry.
        name: Device identifier.
        cell_id: Numeric mapping for the device.
        v_span_thresh: Minimum required voltage span (mV).
        freeze_thresh: Rolling window size for hardware freeze detection.
        i_span_thresh: Minimum required current span (A).
        corruption_tol: Max allowable percentage of anomalous negative current readings.
        spike_tol: Maximum allowable outlier spikes per curve.
        frozen_ratio_tol: Maximum allowable frozen-point ratio per curve.
        mean_mismatch_tol: Maximum allowable relative forward/reverse mean mismatch.
        
    Returns:
        pd.DataFrame: Processed dataset with the 'is_curve_valid' boolean mask applied.
    """
    df_c = df.copy()
    
    # --- 0. PREPARATION & STRICT CHRONOLOGICAL SORTING ---
    # Enforcing UTC prevents future daylight saving time (DST) merge collisions
    df_c['Timestamp'] = pd.to_datetime(df_c['Timestamp'], utc=True)
    df_c = df_c.sort_values('Timestamp').reset_index(drop=True)
    
    # Downcasting floats drastically reduces RAM footprint (~50% memory savings)
    df_c['Voltage_V'] = df_c['Voltage_V'].astype('float32')
    if 'Current_A' in df_c.columns: 
        df_c['Current_A'] = df_c['Current_A'].astype('float32')
    if 'Power_mW' in df_c.columns: 
        df_c['Power_mW'] = df_c['Power_mW'].astype('float32')

    # --- 1. CYCLE IDENTIFICATION ---
    # A new cycle strictly initiates upon detecting a 'Reverse' scan direction
    is_reverse_start = (df_c['ScanDirection'] == 'Reverse') & (df_c['ScanDirection'].shift(1) != 'Reverse')
    is_reverse_start.iloc[0] = True
    
    df_c['curve'] = is_reverse_start.cumsum()
    df_c['cell_name'] = name
    df_c['cell_id'] = cell_id
    df_c['is_reverse'] = (df_c['ScanDirection'] == 'Reverse').astype('int8') if 'ScanDirection' in df_c.columns else 0

    # --- 2. VECTORIZED POINT-LEVEL METRICS ---
    df_c['is_in_time'] = (df_c['Timestamp'].dt.hour >= 6) & (df_c['Timestamp'].dt.hour <= 22)
    
    # [PHYSICS FIX]: Negative Current Isolation
    # Current is naturally negative when exceeding Voc (diode injection regime). 
    # We strictly flag negative currents as anomalous ONLY if they occur at low voltages (< 0.5V).
    df_c['is_anomalous_negative'] = (df_c['Current_A'] < 0) & (df_c['Voltage_V'] < 0.5)
    
    # Hardware freeze detection (Rolling span)
    v_rolling = df_c.groupby('curve', sort=False)['Voltage_V'].rolling(freeze_thresh, min_periods=1)
    i_rolling = df_c.groupby('curve', sort=False)['Current_A'].rolling(freeze_thresh, min_periods=1)
    v_span_window = v_rolling.max().droplevel(0) - v_rolling.min().droplevel(0)
    i_span_window = i_rolling.max().droplevel(0) - i_rolling.min().droplevel(0)
    df_c['is_frozen_point'] = (v_span_window <= 1e-4) | (i_span_window <= 1e-6)

    # Spike detection preparation
    df_c['d_current_abs'] = df_c.groupby('curve')['Current_A'].diff().abs().bfill()

    # --- 3. CONSOLIDATED CURVE-LEVEL AGGREGATION ---
    # Performing a single massive aggregation eliminates extreme Pandas overhead
    curve_stats = df_c.groupby('curve').agg(
        v_max=('Voltage_V', 'max'),
        v_min=('Voltage_V', 'min'),
        i_max=('Current_A', 'max'),
        i_min=('Current_A', 'min'),
        neg_ratio=('is_anomalous_negative', 'mean'),
        in_time_window=('is_in_time', 'all'),
        frozen_ratio=('is_frozen_point', 'mean')
    )
    
    curve_stats['v_span_mV'] = (curve_stats['v_max'] - curve_stats['v_min']) * 1000
    curve_stats['i_span_A'] = curve_stats['i_max'] - curve_stats['i_min']
    
    # Evaluate global boundary conditions
    curve_stats['is_corrupted'] = curve_stats['neg_ratio'] > corruption_tol
    curve_stats['is_curve_frozen'] = curve_stats['frozen_ratio'] > frozen_ratio_tol
    curve_stats['is_low_voltage'] = curve_stats['v_span_mV'] <= v_span_thresh
    curve_stats['is_low_current'] = curve_stats['i_span_A'] <= i_span_thresh
    curve_stats['is_night_time'] = ~curve_stats['in_time_window']
    
    # Map current span back to evaluate relative spikes
    df_c['i_span_A'] = df_c['curve'].map(curve_stats['i_span_A'])
    df_c['is_outlier_point'] = (df_c['d_current_abs'] / df_c['i_span_A'].add(1e-9)) > 0.15
    curve_stats['is_spike_error'] = df_c.groupby('curve')['is_outlier_point'].sum() > spike_tol

    # --- 4. STRUCTURAL INTEGRITY (FORWARD VS REVERSE) ---
    # Fast unstacking handles symmetry and mismatch calculations securely
    dir_stats = df_c.groupby(['curve', 'is_reverse'])['Current_A'].agg(['count', 'mean']).unstack('is_reverse')
    dir_stats = pd.DataFrame(dir_stats)
    dir_stats.columns = [f"{stat}_{int(rev)}" for stat, rev in dir_stats.columns]
    
    # Failsafe dictionary extraction to prevent MultiIndex KeyError on unidirectional artifacts
    fwd_len = dir_stats.get('count_0', pd.Series(0, index=dir_stats.index))
    rev_len = dir_stats.get('count_1', pd.Series(0, index=dir_stats.index))
    fwd_mean = dir_stats.get('mean_0', pd.Series(0, index=dir_stats.index))
    rev_mean = dir_stats.get('mean_1', pd.Series(0, index=dir_stats.index))

    ratio = fwd_len / (rev_len + 1e-9)
    curve_stats['is_asymmetric'] = (fwd_len == 0) | (rev_len == 0) | (ratio < 0.85) | (ratio > 1.15)
    
    # [PHYSICS FIX]: Relaxed hysteresis discrepancy to 0.60 to preserve severely degraded cells
    max_mean = np.maximum(fwd_mean, rev_mean)
    curve_stats['is_mean_mismatch'] = (fwd_mean - rev_mean).abs() / (max_mean + 1e-9) > mean_mismatch_tol

    # --- 5. FINALIZE MASK & MEMORY CLEANUP ---
    invalid_mask = (
        curve_stats['is_low_voltage'] | curve_stats['is_low_current'] | curve_stats['is_night_time'] | 
        curve_stats['is_curve_frozen'] | curve_stats['is_corrupted'] | 
        curve_stats['is_spike_error'] | curve_stats['is_asymmetric'] | 
        curve_stats['is_mean_mismatch']
    )
    curve_stats['is_curve_valid'] = (~invalid_mask).astype('int8')
    
    # Map final validities back to point-level DataFrame
    validity_cols = ['is_low_voltage', 'is_low_current', 'is_night_time', 'is_curve_frozen', 
                     'is_corrupted', 'is_spike_error', 'is_asymmetric', 'is_mean_mismatch', 'is_curve_valid']
    
    df_c = df_c.merge(curve_stats[validity_cols], left_on='curve', right_index=True, how='left')
    
    # Point-level artifact dropping (Removes noise without dropping the entire curve)
    df_c = df_c[~(df_c['is_anomalous_negative'] & (~df_c['is_corrupted']))].copy()
    df_c = df_c[~(df_c['is_outlier_point'] & (~df_c['is_spike_error']))].copy()
    
    # Flush temporary heavy arrays from RAM
    df_c = df_c.drop(columns=['is_in_time', 'is_anomalous_negative', 'is_frozen_point', 'd_current_abs', 'is_outlier_point', 'i_span_A'])

    return df_c


def analyze_curve_timings(jv_df: pd.DataFrame) -> None:
    """
    Extracts and logs chronological metadata, including sweep durations and hardware idle intervals.
    """
    logger.info("Computing chronometric metadata (sweep durations and sensor idle intervals)...")
    
    all_curves_stats = jv_df.groupby(['cell_name', 'id_curve'])['Timestamp'].agg(['min', 'max']).sort_values('min')
    all_curves_stats['is_valid'] = jv_df.groupby(['cell_name', 'id_curve'])['is_curve_valid'].first()

    global_durations, global_intervals = [], []
    report_lines = ["\n=== TEMPORAL ANALYSIS PER DEVICE ==="]

    for cell in jv_df['cell_name'].unique():
        cell_stats = all_curves_stats.loc[cell]
        
        intervals = cell_stats['min'].diff().dt.total_seconds()
        intervals_clean = intervals[intervals < 3600]
        avg_interval = intervals_clean.mean()
        global_intervals.extend(intervals_clean.dropna().tolist())

        valid_stats = cell_stats[cell_stats['is_valid'] == 1]
        durations = (valid_stats['max'] - valid_stats['min']).dt.total_seconds()
        avg_duration = durations.mean()
        global_durations.extend(durations.dropna().tolist())

        total_valid = len(valid_stats)
        valid_df = jv_df[(jv_df['cell_name'] == cell) & (jv_df['is_curve_valid'] == 1)]
        max_daily = valid_df.set_index('Timestamp').resample('D')['id_curve'].nunique().max() if not valid_df.empty else 0

        report_lines.extend([
            f"[{cell}]",
            f"  -> Valid cycles retained: {total_valid}",
            f"  -> Mean duration (Rev+Fwd): {avg_duration:.2f} s" if total_valid > 0 else "  -> Mean duration: N/A",
            f"  -> Mean idle interval: {avg_interval:.2f} s" if not np.isnan(avg_interval) else "  -> Mean idle interval: N/A",
            f"  -> Peak daily cycle volume: {max_daily}\n"
        ])

    report_lines.extend([
        "=" * 40,
        "=== AGGREGATED GLOBAL TEMPORAL METRICS ===",
        f"Mean Global Sweep Duration: {np.mean(global_durations):.2f} s" if global_durations else "Mean Global Sweep Duration: N/A",
        f"Mean Global Hardware Idle:  {np.mean(global_intervals):.2f} s" if global_intervals else "Mean Global Hardware Idle: N/A",
        "=" * 40
    ])

    logger.info("\n".join(report_lines))


if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed/outdoor/")
    FILTERED_DIR = Path("data/filtered/outdoor/")
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR = Path("data/filtered/metadata/outdoor/")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DIR.exists():
        logger.error(f"Target directory missing: {PROCESSED_DIR}. Verify upstream execution.")
        raise FileNotFoundError(f"Directory {PROCESSED_DIR} not found.")

    device_dirs = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    logger.info(f"Discovered processed datasets for localized devices: {device_dirs}")
    
    all_cleaned_dfs = []
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Sequential loading and filtering per device to prevent OOM
    for cell_id, name in enumerate(device_dirs):
        parquet_path = PROCESSED_DIR / name / f"{name}_jv.parquet"
        if parquet_path.exists():
            logger.info(f"Loading raw telemetry for device: {name}")
            df = pd.read_parquet(parquet_path)
            
            if not df.empty:
                logger.info(f"Executing filter topology on device: {name}")
                cleaned_df = _process_single_cell(
                    df=df,
                    name=name,
                    cell_id=cell_id,
                    v_span_thresh=50,
                    freeze_thresh=15,
                    i_span_thresh=0.0001,
                    spike_tol=2
                )
                all_cleaned_dfs.append(cleaned_df)
                
                # Explicit Garbage Collection
                del df
                gc.collect()
            else:
                logger.warning(f"Parquet file for device {name} is empty.")
        else:
            logger.warning(f"Expected Parquet artifact missing for device: {name}")

    if all_cleaned_dfs:
        logger.info("Consolidating processed device chunks...")
        jv_dataset_filtered = pd.concat(all_cleaned_dfs, axis=0, ignore_index=True)
        
        del all_cleaned_dfs
        gc.collect()

        jv_dataset_filtered['id_curve'] = jv_dataset_filtered.groupby(['cell_name', 'curve']).ngroup()

        # --- DIAGNOSTICS REPORTING ---
        logger.info("Aggregating global diagnostics report...")
        summary = jv_dataset_filtered.groupby(['cell_name', 'curve']).first()

        try:
            val_counts = summary.groupby('cell_name')['is_curve_valid'].agg(['count', 'sum'])
            val_counts.columns = ['Total Cycles Detected', 'Valid Cycles Retained']
            
            discard_cols = [
                'is_low_voltage', 'is_low_current', 'is_night_time', 
                'is_curve_frozen', 'is_corrupted', 'is_spike_error', 
                'is_asymmetric', 'is_mean_mismatch'  
            ]
            
            report = (
                f"\n=== QUALITY CONTROL METRICS ===\n"
                f"Aggregate cycles evaluated globally: {len(summary)}\n\n"
                f"Cycle Yield Distribution by Device:\n{val_counts}\n\n"
                f"Global Rejection Frequencies (Applied per cycle):\n{summary[discard_cols].sum().to_string()}"
            )
            logger.info(report)
            
        except Exception as e:
            logger.warning(f"Diagnostics aggregation failed: {e}")

        # Output final audit metrics
        logger.info("\n--- PIPELINE RETENTION AUDIT ---")
        logger.info(f"Raw Vector Dimensions (Rows):\n{jv_dataset_filtered.groupby('is_reverse')['id_curve'].count().to_string()}")
        logger.info(f"Gross Cycle Count:\n{jv_dataset_filtered.groupby('is_reverse')['id_curve'].nunique().to_string()}")
        logger.info(f"Net Valid Cycle Count (Post-Filter):\n{jv_dataset_filtered[jv_dataset_filtered['is_curve_valid'] == 1].groupby('is_reverse')['id_curve'].nunique().to_string()}")

        analyze_curve_timings(jv_dataset_filtered)
        
        # --- QUALITY ASSURANCE & DATASET METADATA EXPORT ---
        logger.info("\n--- QUALITY ASSURANCE & DATASET METADATA EXPORT ---")
        valid_only_df = jv_dataset_filtered[jv_dataset_filtered['is_curve_valid'] == 1]
        valid_counts_dict = valid_only_df.groupby('cell_name')['id_curve'].nunique().to_dict()
        
        metadata_path = METADATA_DIR / "valid_curves_summary.json"
        with open(metadata_path, 'w') as f:
            json.dump(valid_counts_dict, f, indent=4)
        logger.info(f"Dataset structural metadata successfully serialized to: {metadata_path}")
        
        # Serialize dataset to disk
        output_path = FILTERED_DIR / "jv_dataset_filtered.parquet"
        logger.info(f"Serializing filtered dataset artifact to {output_path}...")
        
        jv_dataset_filtered.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        logger.info("Data engineering pipeline successfully terminated. Artifact serialized.")
        
    else:
        logger.error("Data ingestion failure: No valid Parquet artifacts were loaded into memory.")

    warnings.filterwarnings('default', category=RuntimeWarning)