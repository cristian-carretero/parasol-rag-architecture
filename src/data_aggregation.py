"""
Module: src/data_aggregation.py
Description: Downsamples and aggregates high-resolution Parquet data (MPP, Meteo) per device 
into 10-minute intervals. Uses causal resampling (right-closed) and backward asof-merging 
to rigorously prevent data leakage. Computes PCE (Power Conversion Efficiency) and builds 
a unified global fleet-wide dataset.
"""

from pathlib import Path
import pandas as pd
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
    device_source_dir = SOURCE_DIR / device_id
    device_agg_dir = AGGREGATED_DIR / device_id
    device_agg_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Aggregating device data for: {device_id} ===")

    df_meteo_10m = pd.DataFrame()
    df_mpp_10m = pd.DataFrame()

    # 1. Process Meteorological Data
    meteo_file = device_source_dir / f"{device_id}_meteo.parquet"
    if meteo_file.exists():
        try:
            df_meteo = pd.read_parquet(meteo_file)
            if 'Timestamp' in df_meteo.columns:
                df_meteo['Timestamp'] = pd.to_datetime(df_meteo['Timestamp'], errors='coerce')
                df_meteo.set_index('Timestamp', inplace=True)
            
            df_meteo_10m = df_meteo.resample('10min', closed='right', label='right').mean(numeric_only=True).dropna(how='all')
        except Exception as e:
            logger.error(f"Error processing meteo file for {device_id}: {e}")
    else:
        logger.warning(f"Meteorological file not found at: {meteo_file}")

    # 2. Process MPPT Data
    mpp_file = device_source_dir / f"{device_id}_mpp.parquet"
    if mpp_file.exists():
        try:
            df_mpp = pd.read_parquet(mpp_file)
            if 'Timestamp' in df_mpp.columns:
                df_mpp['Timestamp'] = pd.to_datetime(df_mpp['Timestamp'], errors='coerce')
                df_mpp.set_index('Timestamp', inplace=True)
            
            df_mpp_10m = df_mpp.resample('10min', closed='right', label='right').mean(numeric_only=True).dropna(how='all')
        except Exception as e:
            logger.error(f"Error processing MPP file for {device_id}: {e}")
    else:
        logger.warning(f"MPP file not found at: {mpp_file}")

    # 3. Merge avoiding Data Leakage and compute PCE
    if not df_meteo_10m.empty and not df_mpp_10m.empty:
        df_merged = pd.merge_asof(
            df_meteo_10m.sort_index(),
            df_mpp_10m.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # Búsqueda dinámica mejorada (detectará 'Power_W', 'power', 'p_mpp', etc.)
        power_col = next((col for col in df_merged.columns if 'power' in col.lower() or 'p_mpp' in col.lower()), None)
        
        if power_col and 'POA_Irradiance_W_m2' in df_merged.columns:
            mask_day = df_merged['POA_Irradiance_W_m2'] > 10 
            df_merged.loc[mask_day, 'PCE'] = df_merged.loc[mask_day, power_col] / df_merged.loc[mask_day, 'POA_Irradiance_W_m2']
            df_merged['PCE'] = df_merged['PCE'].fillna(0) 
            logger.info(f" -> PCE calculated using '{power_col}'")

        logger.info(f" -> Merged Meteo + MPP successfully for {device_id}")
    elif not df_meteo_10m.empty:
        df_merged = df_meteo_10m
    elif not df_mpp_10m.empty:
        df_merged = df_mpp_10m
    else:
        return

    output_path = device_agg_dir / f"{device_id}_merged_10min.parquet"
    df_merged.to_parquet(output_path, engine='pyarrow', compression='snappy')
    logger.info(f"✅ Saved single device file: {output_path.name} ({len(df_merged):,} records)")

def aggregate_fleet_data(device_ids: list) -> None:
    logger.info("=== Aggregating unified fleet dataset (Meteo + MPP) ===")
    
    if not device_ids:
        return

    all_meteo_dfs = []

    # 1. Gather Meteorological Data
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
                all_meteo_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading meteo file for {device_id}: {e}")

    if not all_meteo_dfs:
        logger.error("No valid meteorological data found to build fleet base.")
        return

    # Base Meteorológica Global
    master_df = pd.concat(all_meteo_dfs, axis=0).sort_index()
    module_temp_cols = [col for col in master_df.columns if col.startswith('ModuleTemp_')]
    env_cols = [col for col in master_df.columns if col not in module_temp_cols]

    resampled_env = master_df[env_cols].resample('10min', closed='right', label='right').mean(numeric_only=True)
    resampled_modules = master_df[module_temp_cols].resample('10min', closed='right', label='right').mean(numeric_only=True)

    fleet_dataset = resampled_env.copy()
    fleet_dataset['ModuleTemp_Mean_C'] = resampled_modules.mean(axis=1, skipna=True)
    fleet_dataset['ModuleTemp_Median_C'] = resampled_modules.median(axis=1, skipna=True)
    fleet_dataset['ModuleTemp_Min_C'] = resampled_modules.min(axis=1, skipna=True)
    fleet_dataset['ModuleTemp_Max_C'] = resampled_modules.max(axis=1, skipna=True)

    if 'POA_Irradiance_W_m2' in fleet_dataset.columns:
        fleet_dataset = fleet_dataset.dropna(subset=['POA_Irradiance_W_m2'], how='all')

    # 2. Add MPP Data per device & Compute PCE
    for device_id in device_ids:
        mpp_file = SOURCE_DIR / device_id / f"{device_id}_mpp.parquet"
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
                
                # BÚSQUEDA CORREGIDA: Busca columnas de potencia asociadas a este device_id
                power_col = next((col for col in fleet_dataset.columns if ('power' in col.lower() or 'p_mpp' in col.lower()) and device_id.lower() in col.lower()), None)
                
                if power_col and 'POA_Irradiance_W_m2' in fleet_dataset.columns:
                    mask_day = fleet_dataset['POA_Irradiance_W_m2'] > 10
                    pce_col = f'PCE_{device_id}'
                    # Nota física: Dependiendo de tu área, podrías tener que dividir por (Área en m2) aquí
                    fleet_dataset.loc[mask_day, pce_col] = fleet_dataset.loc[mask_day, power_col] / fleet_dataset.loc[mask_day, 'POA_Irradiance_W_m2']
                    fleet_dataset[pce_col] = fleet_dataset[pce_col].fillna(0)
                    logger.info(f" -> PCE_{device_id} successfully computed from '{power_col}'.")

            except Exception as e:
                logger.error(f"Error merging MPP data for fleet {device_id}: {e}")

    # 3. Save Unified Fleet Dataset
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AGGREGATED_DIR / "fleet_merged_10min.parquet"
    
    fleet_dataset.reset_index().to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    logger.info(f"✅ Unified fleet dataset saved to: {output_path}")

def process_all_data() -> None:
    if not SOURCE_DIR.exists():
        logger.error(f"Source directory '{SOURCE_DIR}' does not exist.")
        return

    device_dirs = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]
    if not device_dirs:
        return

    for device_id in device_dirs:
        aggregate_device_data(device_id)

    aggregate_fleet_data(device_dirs)
    logger.info("Pipeline successfully completed!")

if __name__ == "__main__":
    process_all_data()