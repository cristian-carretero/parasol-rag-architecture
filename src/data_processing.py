"""
Module: src/data_processing.py
Description: Ingestion, cleaning, and mass conversion of raw telemetry (CSV) 
into optimized columnar Parquet format for outdoor cells.
Implements true disk-to-disk streaming to prevent Out-Of-Memory (OOM) failures,
enforces strict global UTC chronological ordering, and applies Magnus-Tetens 
approximations for environmental thermodynamics.
"""

from pathlib import Path
import logging
import tempfile
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Professional MLOps logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataProcessing")

# Project directory definitions
RAW_DIR = Path("data/raw/outdoor")
PROCESSED_DIR = Path("data/processed/outdoor")


def _write_globally_sorted_chunks(chunks, output_path: Path) -> int:
    """Write transformed chunks as one Parquet file globally sorted by Timestamp."""
    total_records = 0
    parquet_writer = None

    with tempfile.TemporaryDirectory(prefix="parasol-sort-") as temp_dir:
        sorted_chunk_paths = []
        for chunk_number, chunk in enumerate(chunks):
            if chunk.empty:
                continue
            chunk_path = Path(temp_dir) / f"chunk-{chunk_number:06d}.parquet"
            chunk.sort_values("Timestamp", kind="stable").to_parquet(
                chunk_path, index=False
            )
            sorted_chunk_paths.append(chunk_path)

        iterators = [pq.ParquetFile(path).iter_batches(batch_size=65_536)
                     for path in sorted_chunk_paths]
        batches = []
        for source, iterator in enumerate(iterators):
            try:
                batch = next(iterator).to_pandas()
            except StopIteration:
                batches.append(None)
                continue
            batches.append(batch)

        while any(batch is not None for batch in batches):
            active_batches = [batch for batch in batches if batch is not None]
            watermark = min(
                batch["Timestamp"].iloc[-1].value for batch in active_batches
            )

            merged = pd.concat(
                [batch.assign(_merge_source=source)
                 for source, batch in enumerate(batches) if batch is not None],
                ignore_index=True,
            ).sort_values("Timestamp", kind="stable")
            ready = merged[merged["Timestamp"].array.asi8 <= watermark]

            for start in range(0, len(ready), 65_536):
                output_chunk = ready.iloc[start:start + 65_536].drop(
                    columns="_merge_source"
                )
                table = pa.Table.from_pandas(output_chunk, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(
                        output_path, table.schema, compression="snappy"
                    )
                parquet_writer.write_table(table)
                total_records += len(output_chunk)

            for source, batch in enumerate(batches):
                if batch is None:
                    continue
                consumed = np.searchsorted(
                    batch["Timestamp"].array.asi8,
                    watermark,
                    side="right",
                )
                remaining = batch.iloc[consumed:]
                if len(remaining):
                    batches[source] = remaining
                    continue
                try:
                    batches[source] = next(iterators[source]).to_pandas()
                except StopIteration:
                    batches[source] = None

        if parquet_writer is not None:
            parquet_writer.close()
    return total_records
def process_device_data(device_id: str) -> None:
    """
    Executes the disk-to-disk preprocessing pipeline for MPP, J-V, and Meteorological
    files, applying memory downcasting and strict temporal normalization.
    """
    device_raw_dir = RAW_DIR / device_id
    device_processed_dir = PROCESSED_DIR / device_id
    device_processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Processing device: {device_id} ===")

    # ---------------------------------------------------------
    # 1. Process MPP Data (Disk-to-disk streaming via PyArrow)
    # ---------------------------------------------------------
    mpp_file = device_raw_dir / f"{device_id}_mpp.csv"
    output_path_mpp = device_processed_dir / f"{device_id}_mpp.parquet"
    
    if mpp_file.exists():
        chunk_size = 1_000_000
        def transformed_mpp_chunks():
            for chunk in pd.read_csv(mpp_file, chunksize=chunk_size):
                chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce', utc=True)
                chunk = chunk.dropna(subset=['Timestamp', 'Voltage_V', 'Current_A', 'Power_W'])

                float_cols = ['Voltage_V', 'Current_A', 'Power_W']
                chunk[float_cols] = chunk[float_cols].astype('float32')
                yield chunk

        total_records = _write_globally_sorted_chunks(transformed_mpp_chunks(), output_path_mpp)
        if total_records:
            logger.info(f" -> MPP telemetry successfully streamed to: {output_path_mpp.name} ({total_records:,} records)")
    else:
        logger.warning(f"MPP file not found at: {mpp_file}")

    # ---------------------------------------------------------
    # 2. Process J-V Curve Data (Disk-to-disk streaming)
    # ---------------------------------------------------------
    jv_file = device_raw_dir / f"{device_id}_jv.csv"
    output_path_jv = device_processed_dir / f"{device_id}_jv.parquet"
    
    if jv_file.exists():
        chunk_size = 1_000_000
        def transformed_jv_chunks():
            for chunk in pd.read_csv(jv_file, chunksize=chunk_size):
                chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce', utc=True)
                chunk = chunk.dropna(subset=['Timestamp', 'Voltage_V', 'Current_A', 'ScanDirection'])

                float_cols = ['Voltage_V', 'Current_A']
                chunk[float_cols] = chunk[float_cols].astype('float32')
                yield chunk

        total_records = _write_globally_sorted_chunks(transformed_jv_chunks(), output_path_jv)
        if total_records:
            logger.info(f" -> J-V structural sweeps successfully streamed to: {output_path_jv.name} ({total_records:,} records)")
    else:
        logger.warning(f"J-V file not found at: {jv_file}")

    # ---------------------------------------------------------
    # 3. Process Meteorological Data & Thermodynamic Variables
    # ---------------------------------------------------------
    meteo_file = device_raw_dir / f"{device_id}_meteo.csv"
    output_path_meteo = device_processed_dir / f"{device_id}_meteo.parquet"
    
    if meteo_file.exists():
        chunk_size = 500_000
        def transformed_meteo_chunks():
            for chunk in pd.read_csv(meteo_file, chunksize=chunk_size):
                chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce', utc=True)
                chunk = chunk.dropna(subset=['Timestamp', 'POA_Irradiance_W_m2', 'ModuleTemp_C'])

                if 'AmbientTemp_C' in chunk.columns and 'RelativeHumidity_pct' in chunk.columns:
                    T_amb = chunk['AmbientTemp_C'].to_numpy(dtype='float32')
                    rh_pct = chunk['RelativeHumidity_pct'].to_numpy(dtype='float32')
                    p_sat = 6.112 * np.exp((17.67 * T_amb) / (T_amb + 243.5))
                    p_a = p_sat * (rh_pct / 100.0)
                    chunk['AbsoluteHumidity_g_m3'] = (216.68 * p_a) / (T_amb + 273.15)
                    chunk['AbsoluteHumidity_g_m3'] = chunk['AbsoluteHumidity_g_m3'].astype('float32')

                float_cols = [c for c in chunk.columns if c != 'Timestamp']
                chunk[float_cols] = chunk[float_cols].astype('float32')
                yield chunk

        total_records = _write_globally_sorted_chunks(transformed_meteo_chunks(), output_path_meteo)
        if total_records:
            logger.info(f" -> Meteorological data (+ Thermodynamics) successfully streamed to: {output_path_meteo.name} ({total_records:,} records)")
    else:
        logger.warning(f"Meteorological file not found at: {meteo_file}")

def process_all_devices() -> None:
    """
    Iterates through the raw data directory to identify and process all available devices.
    """
    if not RAW_DIR.exists():
        logger.error(f"Critical Error: Base directory '{RAW_DIR}' does not exist. Check data pipeline ingestion.")
        return

    device_dirs = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    
    if not device_dirs:
        logger.warning(f"No device directories found in {RAW_DIR}")
        return

    logger.info(f"Target devices detected: {device_dirs}")

    for device_id in device_dirs:
        process_device_data(device_id)

    logger.info("Data Engineering pipeline successfully terminated. All artifacts serialized to Parquet.")

if __name__ == "__main__":
    process_all_devices()