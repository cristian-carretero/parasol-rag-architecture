"""
Module: src/survival_dataset.py
Description: Feature engineering for Survival Analysis. 
Groups J-V scans at the curve level and aligns them with instantaneous 
environmental stress metrics, maintaining strict temporal causality.
Cumulative doses have been removed to avoid age-bias in downstream ML models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Logging configuration for professional traceability (MLOps standard)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SurvivalDataset")

def build_survival_features(jv_labeled_path: Path, meteo_base_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Builds the design matrix (features) for the XGBoost predictive engine,
    focusing on instantaneous weather variables for downstream rolling-window processing.
    """
    # 1. Load the Labeled Dataset
    logger.info(f"Loading labeled J-V dataset from: {jv_labeled_path}")
    df_labeled = pd.read_parquet(jv_labeled_path)

    df_labeled = df_labeled[df_labeled["label_curve"] != -1].copy()
    logger.info(f"Filtered out invalid sweeps. Remaining records: {len(df_labeled)}")
    
    # 2. Group by curve (1 row = 1 J-V cycle)
    logger.info("Collapsing intra-day curves (dropping V, I, direction)...")
    # Group by identifiers and take the initial timestamp of the curve
    df_curves = df_labeled.groupby(['cell_name', 'cell_id', 'id_curve', 'label_curve']).agg(
        Timestamp=('Timestamp', 'min'),
        pseudo_FF=('pseudo_FF', 'first')  
    ).reset_index()
    
    # Safety chronological sort
    df_curves = df_curves.sort_values(by=['cell_name', 'Timestamp'])
    
    final_dfs = []
    unique_cells = df_curves['cell_name'].unique()
    
    # 3. Iterative alignment of meteorological variables per cell
    for cell in unique_cells:
        logger.info(f"Processing meteorological alignment for cell: {cell}")
        
        # Filter curves for the current cell
        cell_curves = df_curves[df_curves['cell_name'] == cell].copy()
        meteo_path = meteo_base_dir / cell / f"{cell}_meteo.parquet"
        
        if not meteo_path.exists():
            logger.warning(f"  -> Skipping: No telemetry found at {meteo_path}")
            continue
            
        df_meteo = pd.read_parquet(meteo_path)
        df_meteo = df_meteo.sort_values('Timestamp').reset_index(drop=True)
        
        # --- A. TIME METRICS (Kept for tracking, but not for ML accumulation) ---
        df_meteo['delta_t_h'] = df_meteo['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600.0
        df_meteo['exposure_time_h'] = df_meteo['delta_t_h'].cumsum()
        
        # Select instantaneous stress metrics and the join key (Absolute Humidity is already included)
        cols_meteo = [
            'Timestamp', 
            'exposure_time_h', 
            'POA_Irradiance_W_m2', 
            'ModuleTemp_C', 
            'AmbientTemp_C', 
            'AbsoluteHumidity_g_m3'
        ]
        df_meteo_stress = df_meteo[cols_meteo]
        
        # --- C. STRICT TEMPORAL CAUSAL JOIN ---
        # For each J-V curve, find the meteorological record that occurred EXACTLY 
        # at that moment or the immediately preceding one (direction='backward')
        cell_merged = pd.merge_asof(
            cell_curves,
            df_meteo_stress,
            on='Timestamp',
            direction='backward'
        )
        
        final_dfs.append(cell_merged)
        
    # 4. Consolidation and export
    if final_dfs:
        df_survival = pd.concat(final_dfs, ignore_index=True)
        
        # Drop records with no prior meteorological history
        df_survival = df_survival.dropna(subset=['POA_Irradiance_W_m2'])
        
        logger.info(f"Dataset successfully assembled: {len(df_survival)} J-V events linked.")
        
        # Export using pyarrow with snappy compression
        df_survival.to_parquet(output_path, engine='pyarrow', compression='snappy')
        logger.info(f"Survival matrix generated at: {output_path}")
        
        return df_survival
    else:
        logger.error("Critical failure: No dataframe generated (Missing meteo files).")
        return pd.DataFrame()


if __name__ == "__main__":
    # Path definitions adjusted for local structure
    PROCESSED_DIR = Path("data/processed/outdoor")
    LABELED_DIR = Path("data/clustered/outdoor")
    SURVIVAL_DIR = Path("data/survival/outdoor")
    SURVIVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    jv_labeled_file = LABELED_DIR / "jv_dataset_labeled.parquet"
    out_file = SURVIVAL_DIR / "survival_dataset.parquet"

    # Pipeline execution
    if jv_labeled_file.exists():
        df_final = build_survival_features(
            jv_labeled_path=jv_labeled_file, 
            meteo_base_dir=PROCESSED_DIR,   
            output_path=out_file
        )
        
        # Quick preview of the integrated data
        if not df_final.empty:
            print("\nPreview of the instantaneous integrated data:")
            print(df_final[['cell_name', 'Timestamp', 'POA_Irradiance_W_m2', 'ModuleTemp_C', 'AbsoluteHumidity_g_m3']])
    else:
        logger.error(f"Labeled dataset not found at: {jv_labeled_file}")