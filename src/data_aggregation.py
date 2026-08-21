"""
Module: src/data_aggregation.py
Description: Downsamples and aggregates high-resolution Parquet data (MPP, Meteo) per device 
into 10-minute intervals, and builds a unified global fleet-wide meteorological dataset 
handling asynchronous active periods across all outdoor cells.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Logging configuration for professional traceability (MLOps standard)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataAggregation")

# Project base directory definitions
SOURCE_DIR = Path("data/processed/outdoor")
AGGREGATED_DIR = Path("data/aggregated/outdoor")

def aggregate_device_data(device_id: str) -> None:
    """
    Processes and aggregates MPP and Meteorological Parquet files for a specific device
    into lightweight 10-minute resolution files with robust error handling.
    """
    device_source_dir = SOURCE_DIR / device_id
    device_agg_dir = AGGREGATED_DIR / device_id
    device_agg_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Aggregating device data for: {device_id} ===")

    # 1. Aggregate MPPT Data (10-minute average with numeric_only=True & try/except)
    mpp_file = device_source_dir / f"{device_id}_mpp.parquet"
    output_path_mpp = device_agg_dir / f"{device_id}_mpp_10min.parquet"
    
    if mpp_file.exists():
        try:
            df_mpp = pd.read_parquet(mpp_file)
            if 'Timestamp' in df_mpp.columns:
                df_mpp['Timestamp'] = pd.to_datetime(df_mpp['Timestamp'], errors='coerce')
                df_mpp.set_index('Timestamp', inplace=True)
            
            df_mpp_10m = df_mpp.resample('10min').mean(numeric_only=True).dropna(how='all')
            
            output_path_mpp.parent.mkdir(parents=True, exist_ok=True)
            df_mpp_10m.to_parquet(output_path_mpp, engine='pyarrow', compression='snappy')
            logger.info(f" -> MPPT aggregated successfully: {output_path_mpp.name} ({len(df_mpp_10m):,} records)")
        except Exception as e:
            logger.error(f"Error processing MPP file for {device_id}: {e}")
    else:
        logger.warning(f"MPP file not found at: {mpp_file}")

    # 2. Aggregate Meteorological Data per device (10-minute average)
    meteo_file = device_source_dir / f"{device_id}_meteo.parquet"
    output_path_meteo = device_agg_dir / f"{device_id}_meteo_10min.parquet"
    
    if meteo_file.exists():
        try:
            df_meteo = pd.read_parquet(meteo_file)
            if 'Timestamp' in df_meteo.columns:
                df_meteo['Timestamp'] = pd.to_datetime(df_meteo['Timestamp'], errors='coerce')
                df_meteo.set_index('Timestamp', inplace=True)
            
            df_meteo_10m = df_meteo.resample('10min').mean(numeric_only=True).dropna(how='all')
            
            output_path_meteo.parent.mkdir(parents=True, exist_ok=True)
            df_meteo_10m.to_parquet(output_path_meteo, engine='pyarrow', compression='snappy')
            logger.info(f" -> Device Meteo aggregated successfully: {output_path_meteo.name} ({len(df_meteo_10m):,} records)")
        except Exception as e:
            logger.error(f"Error processing meteorological file for {device_id}: {e}")
    else:
        logger.warning(f"Meteorological file not found at: {meteo_file}")

    # 3. J-V Curve Data (Omitido temporalmente por corrección física)
    jv_file = device_source_dir / f"{device_id}_jv.parquet"
    if jv_file.exists():
        logger.info(f" -> J-V file found for {device_id} at {jv_file} (feature extraction omitted pending rigorous physical modeling).")
    else:
        logger.warning(f"J-V file not found at: {jv_file}")


def aggregate_fleet_meteo(device_ids: list) -> None:
    """
    Merges asynchronous temporal ranges, resamples to 10 minutes,
    unifies environmental parameters, and computes fleet-wide module 
    thermal statistics (Mean, Median, Min, Max).
    """
    logger.info("=== Aggregating unified fleet meteorological dataset ===")
    
    if not device_ids:
        logger.warning("No device list provided for fleet meteo aggregation.")
        return

    all_dfs = []

    for device_id in device_ids:
        meteo_file = SOURCE_DIR / device_id / f"{device_id}_meteo.parquet"
        
        if meteo_file.exists():
            try:
                df = pd.read_parquet(meteo_file)
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                    df.set_index('Timestamp', inplace=True)
                
                if 'ModuleTemp_C' in df.columns:
                    df = df.rename(columns={'ModuleTemp_C': f'ModuleTemp_{device_id}'})
                
                all_dfs.append(df)
                logger.info(f" -> Loaded meteo for fleet sync: {device_id}")
            except Exception as e:
                logger.error(f"Error loading meteorological file for {device_id}: {e}")
        else:
            logger.warning(f"Meteo file not found for {device_id} at {meteo_file}")

    if not all_dfs:
        logger.error("No valid meteorological data found across any device.")
        return

    master_df = pd.concat(all_dfs, axis=0).sort_index()

    module_temp_cols = [col for col in master_df.columns if col.startswith('ModuleTemp_')]
    env_cols = [col for col in master_df.columns if col not in module_temp_cols]

    # Resample
    resampled_env = master_df[env_cols].resample('10min').mean(numeric_only=True)
    resampled_modules = master_df[module_temp_cols].resample('10min').mean(numeric_only=True)

    # Cálculo de estadísticas (Mean, Median, Min, Max)
    fleet_module_mean = resampled_modules.mean(axis=1, skipna=True)
    fleet_module_median = resampled_modules.median(axis=1, skipna=True)
    fleet_module_min = resampled_modules.min(axis=1, skipna=True)
    fleet_module_max = resampled_modules.max(axis=1, skipna=True)

    # Construcción del DataFrame final
    fleet_meteo = resampled_env.copy()
    fleet_meteo['ModuleTemp_Mean_C'] = fleet_module_mean
    fleet_meteo['ModuleTemp_Median_C'] = fleet_module_median
    fleet_meteo['ModuleTemp_Min_C'] = fleet_module_min
    fleet_meteo['ModuleTemp_Max_C'] = fleet_module_max

    if 'POA_Irradiance_W_m2' in fleet_meteo.columns:
        fleet_meteo = fleet_meteo.dropna(subset=['POA_Irradiance_W_m2'], how='all')

    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGGREGATED_DIR / "meteo_10min.parquet"
    
    fleet_meteo.reset_index().to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"✅ Unified fleet dataset saved to: {output_path}")


def process_all_data() -> None:
    """
    Automatically scans SOURCE_DIR once, passes the discovered device list to 
    individual aggregations, and computes the unified fleet meteorological dataset.
    """
    if not SOURCE_DIR.exists():
        logger.error(f"Source directory '{SOURCE_DIR}' does not exist. Check the structure.")
        return

    device_dirs = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]
    
    if not device_dirs:
        logger.warning(f"No device directories found in {SOURCE_DIR}")
        return

    logger.info(f"Devices detected for aggregation: {device_dirs}")

    # 1. Process individual device files (MPP, device-level Meteo, and safe J-V handling)
    for device_id in device_dirs:
        aggregate_device_data(device_id)

    # 2. Process global fleet-wide meteorological dataset using the pre-scanned list
    aggregate_fleet_meteo(device_dirs)

    logger.info("Complete data aggregation pipeline successfully completed for all outdoor devices!")


if __name__ == "__main__":
    process_all_data()