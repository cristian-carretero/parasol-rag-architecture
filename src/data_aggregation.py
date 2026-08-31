"""
Module: src/data_aggregation.py
Description: Downsamples and aggregates high-resolution Parquet data (MPP, Meteo) per device 
into synchronized 10-minute intervals. Employs causal resampling (right-closed) and backward 
asof-merging to rigorously prevent data leakage. Computes PCE (Power Conversion Efficiency) 
and structurally compiles a unified global fleet-wide dataset.
"""

from pathlib import Path
import pandas as pd
import logging

# Professional MLOps logging configuration for pipeline traceability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataAggregation")

# Project base directory definitions
PROCESSED_DIR = Path("data/processed/outdoor")
AGGREGATED_DIR = Path("data/aggregated/outdoor")

def aggregate_device_data(device_id: str) -> None:
    """
    Processes and merges meteorological and MPPT data for a single physical device,
    resampling to a common 10-minute frequency and calculating continuous PCE.
    """
    device_PROCESSED_DIR = PROCESSED_DIR / device_id
    device_agg_dir = AGGREGATED_DIR / device_id
    device_agg_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Aggregating device telemetry for: {device_id} ===")

    df_meteo_10m = pd.DataFrame()
    df_mpp_10m = pd.DataFrame()

    # 1. Process Meteorological Data
    meteo_file = device_PROCESSED_DIR / f"{device_id}_meteo.parquet"
    if meteo_file.exists():
        try:
            df_meteo = pd.read_parquet(meteo_file)
            if 'Timestamp' in df_meteo.columns:
                df_meteo['Timestamp'] = pd.to_datetime(df_meteo['Timestamp'], errors='coerce')
                df_meteo.set_index('Timestamp', inplace=True)
            
            # Right-closed resampling ensures trailing causality (no future data leakage)
            df_meteo_10m = df_meteo.resample('10min', closed='right', label='right').mean(numeric_only=True).dropna(how='all')
        except Exception as e:
            logger.error(f"Error processing meteorological matrix for {device_id}: {e}")
    else:
        logger.warning(f"Meteorological artifact not found at: {meteo_file}")

    # 2. Process MPPT Data
    mpp_file = device_PROCESSED_DIR / f"{device_id}_mpp.parquet"
    if mpp_file.exists():
        try:
            df_mpp = pd.read_parquet(mpp_file)
            if 'Timestamp' in df_mpp.columns:
                df_mpp['Timestamp'] = pd.to_datetime(df_mpp['Timestamp'], errors='coerce')
                df_mpp.set_index('Timestamp', inplace=True)
            
            df_mpp_10m = df_mpp.resample('10min', closed='right', label='right').mean(numeric_only=True).dropna(how='all')
        except Exception as e:
            logger.error(f"Error processing MPPT matrix for {device_id}: {e}")
    else:
        logger.warning(f"MPPT artifact not found at: {mpp_file}")

    # 3. Merge streams avoiding Data Leakage and compute dynamic PCE
    if not df_meteo_10m.empty and not df_mpp_10m.empty:
        # Backward merge explicitly forces the MPPT point to align only with past/current weather
        df_merged = pd.merge_asof(
            df_meteo_10m.sort_index(),
            df_mpp_10m.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # Dynamic search heuristic to locate variable naming conventions for MPP Power
        power_col = next((col for col in df_merged.columns if 'power' in col.lower() or 'p_mpp' in col.lower()), None)
        
        if power_col and 'POA_Irradiance_W_m2' in df_merged.columns:
            mask_day = df_merged['POA_Irradiance_W_m2'] > 10 
            # Note: Absolute PCE mathematically requires active area normalization (Area in m²)
            df_merged.loc[mask_day, 'PCE'] = df_merged.loc[mask_day, power_col] / df_merged.loc[mask_day, 'POA_Irradiance_W_m2']
            df_merged['PCE'] = df_merged['PCE'].fillna(0) 
            logger.info(f" -> PCE vector successfully computed utilizing column: '{power_col}'")

        logger.info(f" -> Telemetry streams (Meteo + MPPT) successfully merged for {device_id}")
    elif not df_meteo_10m.empty:
        df_merged = df_meteo_10m
    elif not df_mpp_10m.empty:
        df_merged = df_mpp_10m
    else:
        return

    output_path = device_agg_dir / f"{device_id}_meteo_mppt_10min.parquet"
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f" -> Serialized localized device matrix: {output_path.name} ({len(df_merged):,} topological records)")

def aggregate_fleet_data(device_ids: list) -> None:
    """
    Constructs a unified, horizontally scaled DataFrame encompassing all active devices,
    computing systemic fleet-wide module temperatures and decoupled PCE columns.
    """
    logger.info("\n=== Compiling unified global fleet dataset (Meteo + MPPT) ===")
    
    if not device_ids:
        return

    all_meteo_dfs = []

    # 1. Gather global Meteorological Data
    for device_id in device_ids:
        meteo_file = PROCESSED_DIR / device_id / f"{device_id}_meteo.parquet"
        if meteo_file.exists():
            try:
                df = pd.read_parquet(meteo_file)
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                    df.set_index('Timestamp', inplace=True)
                
                # Suffixing avoids spatial collision across multiple devices in the fleet matrix
                if 'ModuleTemp_C' in df.columns:
                    df = df.rename(columns={'ModuleTemp_C': f'ModuleTemp_{device_id}'})
                all_meteo_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading meteorological payload for {device_id}: {e}")

    if not all_meteo_dfs:
        logger.error("Critical failure: No valid meteorological artifacts located to build the fleet baseline.")
        return

    # Base Global Meteorological Topology
    master_df = pd.concat(all_meteo_dfs, axis=0).sort_index()
    module_temp_cols = [col for col in master_df.columns if col.startswith('ModuleTemp_')]
    env_cols = [col for col in master_df.columns if col not in module_temp_cols]

    resampled_env = master_df[env_cols].resample('10min', closed='right', label='right').mean(numeric_only=True)
    resampled_modules = master_df[module_temp_cols].resample('10min', closed='right', label='right').mean(numeric_only=True)

    fleet_dataset = resampled_env.copy()
    
    # Calculate systemic fleet thermal variances
    fleet_dataset['ModuleTemp_Mean_C'] = resampled_modules.mean(axis=1, skipna=True)
    fleet_dataset['ModuleTemp_Median_C'] = resampled_modules.median(axis=1, skipna=True)
    fleet_dataset['ModuleTemp_Min_C'] = resampled_modules.min(axis=1, skipna=True)
    fleet_dataset['ModuleTemp_Max_C'] = resampled_modules.max(axis=1, skipna=True)

    # Prune entirely synthetic rows generated during temporal resampling limits
    if 'POA_Irradiance_W_m2' in fleet_dataset.columns:
        fleet_dataset = fleet_dataset.dropna(subset=['POA_Irradiance_W_m2'], how='all')

    # 2. Iteratively append isolated MPP Data per device & Compute distributed PCE
    for device_id in device_ids:
        mpp_file = PROCESSED_DIR / device_id / f"{device_id}_mpp.parquet"
        if mpp_file.exists():
            try:
                df_mpp = pd.read_parquet(mpp_file)
                if 'Timestamp' in df_mpp.columns:
                    df_mpp['Timestamp'] = pd.to_datetime(df_mpp['Timestamp'], errors='coerce')
                    df_mpp.set_index('Timestamp', inplace=True)
                
                df_mpp_10m = df_mpp.resample('10min', closed='right', label='right').mean(numeric_only=True).dropna(how='all')
                df_mpp_10m = df_mpp_10m.add_suffix(f'_{device_id}')

                fleet_dataset = pd.merge_asof(
                    fleet_dataset.sort_index(),
                    df_mpp_10m.sort_index(),
                    left_index=True,
                    right_index=True,
                    direction='backward'
                )
                
                # Localized dynamic heuristic string matching to map power variables specific to the appended device
                power_col = next((col for col in fleet_dataset.columns if ('power' in col.lower() or 'p_mpp' in col.lower()) and device_id.lower() in col.lower()), None)
                
                if power_col and 'POA_Irradiance_W_m2' in fleet_dataset.columns:
                    mask_day = fleet_dataset['POA_Irradiance_W_m2'] > 10
                    pce_col = f'PCE_{device_id}'
                    
                    fleet_dataset.loc[mask_day, pce_col] = fleet_dataset.loc[mask_day, power_col] / fleet_dataset.loc[mask_day, 'POA_Irradiance_W_m2']
                    fleet_dataset[pce_col] = fleet_dataset[pce_col].fillna(0)
                    logger.info(f" -> Component PCE_{device_id} mapped and quantified utilizing '{power_col}'.")

            except Exception as e:
                logger.error(f"Error executing horizontal merge for fleet device {device_id}: {e}")

    # 3. Serialize Unified Fleet Architecture
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGGREGATED_DIR / "meteo_mppt_10min.parquet"
    
    fleet_dataset.reset_index().to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"\n✅ Fleet consolidation successful. Master architecture serialized to: {output_path}")

def process_all_data() -> None:
    """
    Main orchestrator logic mapping isolated environments into continuous aggregated schemas.
    """
    if not PROCESSED_DIR.exists():
        logger.error(f"Target telemetry directory '{PROCESSED_DIR}' does not exist.")
        return

    device_dirs = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    if not device_dirs:
        return

    for device_id in device_dirs:
        aggregate_device_data(device_id)

    aggregate_fleet_data(device_dirs)
    logger.info("Data aggregation pipeline sequence successfully terminated.")

if __name__ == "__main__":
    process_all_data()