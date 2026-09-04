"""
Module: src/survival_dataset.py
Description: Feature engineering for predictive maintenance and survival analysis.
Groups J-V scans by curve, computes cumulative environmental stressors (doses), 
and aligns continuous telemetry using strict causal backward merging.
"""

import logging
from pathlib import Path

import pandas as pd

# Professional MLOps logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SurvivalDataset")

# Tolerance for backward merge to prevent stale telemetry propagation across sensor gaps
MERGE_ASOF_TOLERANCE = pd.Timedelta("20min")

def build_survival_features(jv_labeled_path: Path, aggregated_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Builds the design matrix (features) for the XGBoost predictive engine.
    Joins discretized J-V structural labels with continuous aggregated telemetry,
    incorporating cumulative physical degradation metrics.
    """
    # 1. Load the Labeled Dataset
    logger.info(f"Loading labeled J-V dataset from: {jv_labeled_path}")
    df_labeled = pd.read_parquet(jv_labeled_path)

    # Drop structurally invalid sweeps rejected by K-Medoids auto-pruning
    df_labeled = df_labeled[df_labeled["label_curve"] != -1].copy()
    logger.info(f"Filtered out invalid sweeps. Remaining records: {len(df_labeled)}")
    
    # 2. Collapse to curve-level data
    logger.info("Collapsing intra-day curves to extract structural features...")
    df_curves = df_labeled.groupby(['cell_name', 'cell_id', 'curve', 'label_curve']).agg(
        Timestamp=('Timestamp', 'min'),
        pFF=('pseudo_FF', 'first')
    ).reset_index()
    
    # Enforce UTC and chronological sort to prevent merge_asof timezone collapse
    df_curves['Timestamp'] = pd.to_datetime(df_curves['Timestamp'], utc=True)
    df_curves = df_curves.sort_values(by=['cell_name', 'Timestamp'])
    
    # 3. Load Aggregated Fleet/Device Telemetry
    logger.info(f"Loading aggregated telemetry from: {aggregated_path}")
    if not aggregated_path.exists():
        logger.error(f"Aggregated dataset not found at: {aggregated_path}")
        return pd.DataFrame()
        
    df_agg = pd.read_parquet(aggregated_path)
    if df_agg.index.name == "Timestamp":
        df_agg = df_agg.reset_index()
        
    # Enforce UTC and chronological sort
    df_agg['Timestamp'] = pd.to_datetime(df_agg['Timestamp'], utc=True)
    df_agg = df_agg.sort_values('Timestamp').reset_index(drop=True)

    # 4. Feature Engineering and alignment per cell via merge_asof (causal backward)
    logger.info("Computing cumulative physical stressors per cell (Light Dose, Humidity Exposure)...")
    final_dfs = []
    unique_cells = df_curves['cell_name'].unique()
    
    for cell in unique_cells:
        logger.info(f"Processing telemetry alignment for cell: {cell}")
        
        cell_curves = df_curves[df_curves['cell_name'] == cell].copy()
        
        # Map device-specific columns
        pce_col = f'PCE_{cell}'
        temp_col = f'ModuleTemp_{cell}'
        
        cols_to_keep = ['Timestamp']
        global_features = ['POA_Irradiance_W_m2', 'AbsoluteHumidity_g_m3']
                           
        for feat in global_features:
            if feat in df_agg.columns:
                cols_to_keep.append(feat)
            else:
                logger.warning(f"[{cell}] '{feat}' missing in aggregated dataset.")
        
        if temp_col in df_agg.columns:
            cols_to_keep.append(temp_col)
        elif 'ModuleTemp_Mean_C' in df_agg.columns:
            cols_to_keep.append('ModuleTemp_Mean_C')
            
        if pce_col in df_agg.columns:
            cols_to_keep.append(pce_col)
            
        df_subset = df_agg[cols_to_keep].dropna(subset=['Timestamp']).sort_values('Timestamp')

        # Start accumulating exposure at the cell's first measurement.
        birth_time = cell_curves['Timestamp'].min()
        df_subset = df_subset[df_subset['Timestamp'] >= birth_time].copy()

        if temp_col in df_subset.columns:
            df_subset = df_subset.rename(columns={temp_col: 'ModuleTemp_C'})
        elif 'ModuleTemp_Mean_C' in df_subset.columns:
            df_subset = df_subset.rename(columns={'ModuleTemp_Mean_C': 'ModuleTemp_C'})

        # Calculate physical memory from the birth of this cell only.
        dt_seconds = df_subset['Timestamp'].diff().dt.total_seconds()
        dt_hours = (
            dt_seconds.mask(dt_seconds > MERGE_ASOF_TOLERANCE.total_seconds(), 0)
            .fillna(600)
            / 3600.0
        )
        if 'ModuleTemp_C' in df_subset.columns:
            df_subset['Delta_Temp_C_per_h'] = df_subset['ModuleTemp_C'].diff() / dt_hours
            df_subset['Delta_Temp_C_per_h'] = df_subset['Delta_Temp_C_per_h'].fillna(0)
        if 'AbsoluteHumidity_g_m3' in df_subset.columns:
            df_subset['Delta_Hum_g_m3_per_h'] = (
                df_subset['AbsoluteHumidity_g_m3'].diff() / dt_hours
            )
            df_subset['Delta_Hum_g_m3_per_h'] = (
                df_subset['Delta_Hum_g_m3_per_h'].fillna(0)
            )
        if 'POA_Irradiance_W_m2' in df_subset.columns:
            df_subset['Cum_Light_Dose_Wh_m2'] = (
                df_subset['POA_Irradiance_W_m2'] * dt_hours
            ).cumsum()
        if 'AbsoluteHumidity_g_m3' in df_subset.columns:
            df_subset['Cum_Humidity_Exposure'] = (
                df_subset['AbsoluteHumidity_g_m3'] * dt_hours
            ).cumsum()
        
        # STRICT TEMPORAL CAUSAL JOIN: Match each J-V sweep with the most recent telemetry point
        cell_merged = pd.merge_asof(
            cell_curves,
            df_subset,
            on='Timestamp',
            direction='backward',
            tolerance=MERGE_ASOF_TOLERANCE
        )

        # Audit data gaps (missing matches within tolerance)
        if pce_col in cell_merged.columns:
            n_gaps = cell_merged[pce_col].isna().sum()
            if n_gaps > 0:
                logger.warning(f"[{cell}] {n_gaps} J-V sweeps lack telemetry within {MERGE_ASOF_TOLERANCE} tolerance (Sensor gap).")
        
        # Standardize features for XGBoost compatibility
        rename_dict = {}
        if pce_col in cell_merged.columns: 
            rename_dict[pce_col] = 'PCE'
        if temp_col in cell_merged.columns: 
            rename_dict[temp_col] = 'ModuleTemp_C'
        elif 'ModuleTemp_Mean_C' in cell_merged.columns: 
            rename_dict['ModuleTemp_Mean_C'] = 'ModuleTemp_C'
            
        cell_merged = cell_merged.rename(columns=rename_dict)
        final_dfs.append(cell_merged)
        
    # 6. Consolidation and export
    if final_dfs:
        df_survival = pd.concat(final_dfs, ignore_index=True)
        
        # Drop rows missing critical physics parameters
        df_survival = df_survival.dropna(subset=['POA_Irradiance_W_m2', 'PCE'])
        
        logger.info(f"Dataset successfully assembled: {len(df_survival)} J-V events linked with telemetry.")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_survival.to_parquet(output_path, engine='pyarrow', compression='snappy')
        logger.info(f"Survival design matrix generated at: {output_path}")
        
        return df_survival
        
    logger.error("Critical failure: No dataframe generated.")
    return pd.DataFrame()


if __name__ == "__main__":
    # Pipeline configuration paths
    LABELED_DIR = Path("data/clustered/outdoor")
    AGGREGATED_FILE = Path("data/aggregated/outdoor/meteo_mppt_10min.parquet")
    SURVIVAL_DIR = Path("data/survival/outdoor")
    
    jv_labeled_file = LABELED_DIR / "jv_dataset_labeled.parquet"
    out_file = SURVIVAL_DIR / "survival_dataset.parquet"

    # Pipeline execution
    if jv_labeled_file.exists() and AGGREGATED_FILE.exists():
        df_final = build_survival_features(
            jv_labeled_path=jv_labeled_file, 
            aggregated_path=AGGREGATED_FILE,   
            output_path=out_file
        )
        
        if not df_final.empty:
            print("\nPreview of the instantaneous and cumulative integrated data:")
            preview_cols = ['cell_name', 'Timestamp', 'POA_Irradiance_W_m2', 'PCE', 'Cum_Light_Dose_Wh_m2']
            print(df_final[preview_cols].head())
    else:
        logger.error(f"Missing required input datasets. Check paths:\n- {jv_labeled_file}\n- {AGGREGATED_FILE}")