"""
Module: src/data_processing.py
Description: Ingestion, cleaning, and mass conversion of raw data (CSV) 
into optimized columnar Parquet format for all outdoor cells.
"""

from pathlib import Path
import pandas as pd
import logging

# Logging configuration for professional traceability (MLOps standard)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataProcessing")

# Project base directory definitions adjusted for outdoor platform
RAW_DIR = Path("data/raw/outdoor")
PROCESSED_DIR = Path("data/processed/outdoor")

def process_device_data(device_id: str) -> None:
    """
    Processes MPP, J-V, and Meteorological files comprehensively 
    for a specific outdoor device/cell.
    """
    device_raw_dir = RAW_DIR / device_id
    device_processed_dir = PROCESSED_DIR / device_id
    device_processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Processing device: {device_id} ===")

    # 1. Process MPP Data (At 1 Hz - Chunked reading due to high volume)
    mpp_file = device_raw_dir / f"{device_id}_mpp.csv"
    output_path_mpp = device_processed_dir / f"{device_id}_mpp.parquet"
    
    if mpp_file.exists():
        chunk_size = 1_000_000
        chunks = []
        for chunk in pd.read_csv(mpp_file, chunksize=chunk_size):
            chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce')
            chunk = chunk.dropna(subset=['Timestamp', 'Voltage_V', 'Current_A', 'Power_W'])
            chunks.append(chunk)
            
        df_mpp = pd.concat(chunks, ignore_index=True)
        output_path_mpp.parent.mkdir(parents=True, exist_ok=True)
        df_mpp.to_parquet(output_path_mpp, engine='pyarrow', compression='snappy', index=False)
        logger.info(f" -> MPP file successfully saved to: {output_path_mpp} ({len(df_mpp):,} records)")
    else:
        logger.warning(f"MPP file not found at: {mpp_file}")

    # 2. Process J-V Curve Data (Captured every 10 minutes)
    jv_file = device_raw_dir / f"{device_id}_jv.csv"
    output_path_jv = device_processed_dir / f"{device_id}_jv.parquet"
    
    if jv_file.exists():
        df_jv = pd.read_csv(jv_file)
        df_jv['Timestamp'] = pd.to_datetime(df_jv['Timestamp'], errors='coerce')
        df_jv = df_jv.dropna(subset=['Timestamp', 'Voltage_V', 'Current_A', 'ScanDirection'])
        
        output_path_jv.parent.mkdir(parents=True, exist_ok=True)
        df_jv.to_parquet(output_path_jv, engine='pyarrow', compression='snappy', index=False)
        logger.info(f" -> J-V file successfully saved to: {output_path_jv} ({len(df_jv):,} records)")
    else:
        logger.warning(f"J-V file not found at: {jv_file}")

    # 3. Process Meteorological Data
    meteo_file = device_raw_dir / f"{device_id}_meteo.csv"
    output_path_meteo = device_processed_dir / f"{device_id}_meteo.parquet"
    
    if meteo_file.exists():
        df_meteo = pd.read_csv(meteo_file)
        df_meteo['Timestamp'] = pd.to_datetime(df_meteo['Timestamp'], errors='coerce')
        df_meteo = df_meteo.dropna(subset=['Timestamp', 'POA_Irradiance_W_m2', 'ModuleTemp_C'])
        
        output_path_meteo.parent.mkdir(parents=True, exist_ok=True)
        df_meteo.to_parquet(output_path_meteo, engine='pyarrow', compression='snappy', index=False)
        logger.info(f" -> Meteorological file successfully saved to: {output_path_meteo} ({len(df_meteo):,} records)")
    else:
        logger.warning(f"Meteorological file not found at: {meteo_file}")

def process_all_devices() -> None:
    """
    Automatically scans data/raw/outdoor/ to identify all available cell directories
    and execute the preprocessing pipeline.
    """
    if not RAW_DIR.exists():
        logger.error(f"Base directory '{RAW_DIR}' does not exist. Check the structure.")
        return

    device_dirs = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    
    if not device_dirs:
        logger.warning(f"No device directories found in {RAW_DIR}")
        return

    logger.info(f"Devices detected in disk: {device_dirs}")

    for device_id in device_dirs:
        process_device_data(device_id)

    logger.info("Parquet conversion pipeline successfully completed for all outdoor devices!")

if __name__ == "__main__":
    process_all_devices()