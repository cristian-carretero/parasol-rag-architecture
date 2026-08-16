"""
Module: src/survival_dataset.py
Description: Feature engineering for Survival Analysis. 
Groups J-V scans at the curve level, calculates accumulated environmental 
stress doses via numerical integration, and aligns both sources while 
maintaining strict temporal causality. Incorporates Magnus-Tetens approximation 
for Absolute Humidity.
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
    Builds the design matrix (features) for the XGBoost predictive engine.
    """
    # 1. Load the Labeled Dataset
    logger.info(f"Loading labeled J-V dataset from: {jv_labeled_path}")
    df_labeled = pd.read_parquet(jv_labeled_path)
    
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
    
    # 3. Iterative integration of meteorological variables per cell
    for cell in unique_cells:
        logger.info(f"Processing meteorological integration for cell: {cell}")
        
        # Filter curves for the current cell
        cell_curves = df_curves[df_curves['cell_name'] == cell].copy()
        meteo_path = meteo_base_dir / cell / f"{cell}_meteo.parquet"
        
        if not meteo_path.exists():
            logger.warning(f"  -> Skipping: No telemetry found at {meteo_path}")
            continue
            
        df_meteo = pd.read_parquet(meteo_path)
        df_meteo = df_meteo.sort_values('Timestamp').reset_index(drop=True)
        
        # --- A. ACCUMULATED DOSES CALCULATION (Integration) ---
        # Calculate delta t (in hours) relative to the previous row
        df_meteo['delta_t_h'] = df_meteo['Timestamp'].diff().dt.total_seconds().fillna(0) / 3600.0
        
        # Total exposure time since the first measurement
        df_meteo['exposure_time_h'] = df_meteo['delta_t_h'].cumsum()
        
        # Irradiance Dose (Wh/m2) = Sum(W/m2 * hours)
        df_meteo['irradiance_dose'] = (df_meteo['POA_Irradiance_W_m2'] * df_meteo['delta_t_h']).cumsum()
        
        # Thermal Dose = Sum(Temperature * hours)
        df_meteo['module_temp_dose'] = (df_meteo['ModuleTemp_C'] * df_meteo['delta_t_h']).cumsum()
        df_meteo['ambient_temp_dose'] = (df_meteo['AmbientTemp_C'] * df_meteo['delta_t_h']).cumsum()
        
        # --- ABSOLUTE HUMIDITY CALCULATION (Magnus-Tetens) ---
        T_amb = df_meteo['AmbientTemp_C']
        rh_pct = df_meteo['RelativeHumidity_pct']
        
        # 1. Saturation Vapor Pressure (p_sat) in hPa (mbar)
        p_sat = 6.112 * np.exp((17.67 * T_amb) / (T_amb + 243.5))
        
        # 2. Actual Vapor Pressure (p_a) in hPa
        p_a = p_sat * (rh_pct / 100.0)
        
        # 3. Absolute Humidity (AH) in g/m^3
        df_meteo['AbsoluteHumidity_g_m3'] = (216.68 * p_a) / (T_amb + 273.15)
        
        # Absolute Humidity Dose = Sum(g/m3 * hours)
        df_meteo['absolute_humidity_dose'] = (df_meteo['AbsoluteHumidity_g_m3'] * df_meteo['delta_t_h']).cumsum()
        
        # Select only the final stress metrics and the join key
        cols_meteo = [
            'Timestamp', 'exposure_time_h', 'irradiance_dose', 
            'module_temp_dose', 'ambient_temp_dose', 'absolute_humidity_dose'
        ]
        df_meteo_stress = df_meteo[cols_meteo]
        
        # --- B. STRICT TEMPORAL CAUSAL JOIN ---
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
        df_survival = df_survival.dropna(subset=['irradiance_dose'])
        
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
    BASE_DIR = Path(r"C:\Users\crica\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\parasol-rag-architecture-main\data\processed\outdoor")
    
    jv_labeled_file = BASE_DIR / "jv_dataset_labeled.parquet"
    out_file = BASE_DIR / "survival_dataset.parquet"
    
    # Pipeline execution
    if jv_labeled_file.exists():
        df_final = build_survival_features(
            jv_labeled_path=jv_labeled_file, 
            meteo_base_dir=BASE_DIR, 
            output_path=out_file
        )
        
        # Quick preview of the integrated data
        if not df_final.empty:
            print("\nPreview of the integrated data (Now with Absolute Humidity):")
            print(df_final.head())
    else:
        logger.error(f"Labeled dataset not found at: {jv_labeled_file}")