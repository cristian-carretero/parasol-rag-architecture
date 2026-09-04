"""
Module: src/data_aggregation.py
Description: Downsamples and aggregates high-resolution Parquet telemetry (MPP, Meteo) 
into synchronized 10-minute intervals per device. Employs causal resampling (right-closed) 
and strict outer joins on the shared temporal index. Computes active 
Power Conversion Efficiency (PCE) and compiles a unified fleet-wide dataset.
"""

from pathlib import Path
import pandas as pd
import logging

# Professional MLOps logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataAggregation")

PROCESSED_DIR = Path("data/processed/outdoor")
AGGREGATED_DIR = Path("data/aggregated/outdoor")

CELL_AREA_M2 = 0.64 / 10000.0

# Minimum irradiance threshold to prevent numerical instability in PCE calculation
PCE_IRRADIANCE_MIN_W_M2 = 100.0

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
                # Force UTC standard so both streams share the same temporal index
                df_meteo['Timestamp'] = pd.to_datetime(df_meteo['Timestamp'], errors='coerce', utc=True)
                df_meteo = df_meteo.sort_values('Timestamp').set_index('Timestamp')
            
            # Right-closed resampling ensures strict trailing causality 
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
                df_mpp['Timestamp'] = pd.to_datetime(df_mpp['Timestamp'], errors='coerce', utc=True)
                df_mpp = df_mpp.sort_values('Timestamp').set_index('Timestamp')
            
            df_mpp_10m = df_mpp.resample('10min', closed='right', label='right').mean(numeric_only=True).dropna(how='all')
        except Exception as e:
            logger.error(f"Error processing MPPT matrix for {device_id}: {e}")
    else:
        logger.warning(f"MPPT artifact not found at: {mpp_file}")

    # 3. Merge streams avoiding Data Leakage and compute dynamic PCE
    if not df_meteo_10m.empty and not df_mpp_10m.empty:
        df_merged = df_meteo_10m.join(df_mpp_10m, how='outer')
        
        power_col = next((col for col in df_merged.columns if 'power' in col.lower() or 'p_mpp' in col.lower()), None)

        if power_col is not None and 'POA_Irradiance_W_m2' in df_merged.columns:
            mask_day_gap = df_merged['POA_Irradiance_W_m2'] > PCE_IRRADIANCE_MIN_W_M2
            n_gaps = df_merged.loc[mask_day_gap, power_col].isna().sum()
            if n_gaps > 0:
                logger.warning(f"[{device_id}] {n_gaps} daylight rows lack MPPT values after the temporal join — possible sensor gap.")

            mask_day = df_merged['POA_Irradiance_W_m2'] > PCE_IRRADIANCE_MIN_W_M2
            
            # Absolute PCE mathematically requires active area normalization
            df_merged.loc[mask_day, 'PCE'] = (df_merged.loc[mask_day, power_col]) / (df_merged.loc[mask_day, 'POA_Irradiance_W_m2'] * CELL_AREA_M2) * 100.0

            # Nighttime rows are zeroed out to prevent NaN proliferation downstream
            df_merged.loc[~mask_day, 'PCE'] = df_merged.loc[~mask_day, 'PCE'].fillna(0)

            # Cap PCE to physical bounds to discard low-irradiance division noise
            df_merged['PCE'] = df_merged['PCE'].clip(lower=0, upper=100)
            logger.info(f" -> PCE vector computed utilizing column: '{power_col}'")

        logger.info(f" -> Telemetry streams (Meteo + MPPT) merged for {device_id}")
    elif not df_meteo_10m.empty:
        df_merged = df_meteo_10m
    elif not df_mpp_10m.empty:
        df_merged = df_mpp_10m
    else:
        return

    output_path = device_agg_dir / f"{device_id}_meteo_mppt_10min.parquet"
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f" -> Serialized localized device matrix: {output_path.name} ({len(df_merged):,} records)")

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
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
                    df = df.sort_values('Timestamp').set_index('Timestamp')
                
                # Suffixing avoids spatial collision across multiple devices
                if 'ModuleTemp_C' in df.columns:
                    df = df.rename(columns={'ModuleTemp_C': f'ModuleTemp_{device_id}'})
                all_meteo_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading meteorological payload for {device_id}: {e}")

    if not all_meteo_dfs:
        logger.error("Critical failure: No valid meteorological artifacts located.")
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

    if 'POA_Irradiance_W_m2' in fleet_dataset.columns:
        fleet_dataset = fleet_dataset.dropna(subset=['POA_Irradiance_W_m2'], how='all')

    # 2. Iteratively append isolated MPP Data per device & Compute distributed PCE
    for device_id in device_ids:
        mpp_file = PROCESSED_DIR / device_id / f"{device_id}_mpp.parquet"
        if mpp_file.exists():
            try:
                df_mpp = pd.read_parquet(mpp_file)
                if 'Timestamp' in df_mpp.columns:
                    df_mpp['Timestamp'] = pd.to_datetime(df_mpp['Timestamp'], errors='coerce', utc=True)
                    df_mpp = df_mpp.sort_values('Timestamp').set_index('Timestamp')
                
                df_mpp_10m = df_mpp.resample('10min', closed='right', label='right').mean(numeric_only=True).dropna(how='all')
                df_mpp_10m = df_mpp_10m.add_suffix(f'_{device_id}')

                fleet_dataset = fleet_dataset.join(df_mpp_10m, how='outer')
                
                power_col = next((col for col in fleet_dataset.columns if ('power' in col.lower() or 'p_mpp' in col.lower()) and device_id.lower() in col.lower()), None)

                if power_col is not None:
                    n_gaps = fleet_dataset[power_col].isna().sum()
                    if n_gaps > 0:
                        logger.warning(f"[{device_id}] {n_gaps} rows lack MPPT values after the temporal join.")

                if power_col and 'POA_Irradiance_W_m2' in fleet_dataset.columns:
                    mask_day = fleet_dataset['POA_Irradiance_W_m2'] > PCE_IRRADIANCE_MIN_W_M2
                    pce_col = f'PCE_{device_id}'
                    
                    fleet_dataset.loc[mask_day, pce_col] = fleet_dataset.loc[mask_day, power_col] / (fleet_dataset.loc[mask_day, 'POA_Irradiance_W_m2'] * CELL_AREA_M2) * 100.0
                    fleet_dataset.loc[~mask_day, pce_col] = fleet_dataset.loc[~mask_day, pce_col].fillna(0)
                    fleet_dataset[pce_col] = fleet_dataset[pce_col].clip(lower=0, upper=100)
                    logger.info(f" -> Component PCE_{device_id} mapped and quantified utilizing '{power_col}'.")

            except Exception as e:
                logger.error(f"Error executing horizontal merge for fleet device {device_id}: {e}")

    # 3. Serialize Unified Fleet Architecture
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGGREGATED_DIR / "meteo_mppt_10min.parquet"
    
    fleet_dataset.reset_index().to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"\n✅ Fleet consolidation successful. Master architecture serialized to: {output_path}")

def process_all_data() -> None:
    """Orchestrator logic mapping isolated environments into continuous aggregated schemas."""
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