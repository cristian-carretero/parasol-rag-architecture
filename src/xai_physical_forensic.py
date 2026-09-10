"""
Module: src/xai_physical_forensic.py
Description: Independent Physical Forensic Diagnostic.
Diagnoses the environmental triggers of physical T80 collapse by contrasting 
the 3-day prodromal (critical) window against the cell's own healthy history.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import (
    EARLY_FAILURE_WINDOW_DAYS,
    FEATURE_IMPORTANCE_MIN,
    PRODROMAL_WINDOW_DAYS,
    SURROGATE_TREE_PARAMS,
    XAI_PHYSICAL_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("XAI_Forensic")

def extract_physical_forensic_surrogate(cell_data: pd.DataFrame, failure_day: float) -> Dict[str, Any]:
    early_data = cell_data[cell_data['Exposure_Days'] <= failure_day].copy()
    if len(early_data) < 20:
        return {}

    critical_start = max(0.0, failure_day - PRODROMAL_WINDOW_DAYS) # Ventana prodrómica de 3 días
    early_data['Is_Critical'] = (early_data['Exposure_Days'] >= critical_start).astype(int)
    
    if early_data['Is_Critical'].nunique() < 2:
        return {}

    X = early_data[XAI_PHYSICAL_FEATURES]
    y = early_data['Is_Critical']

    # Desempaquetado explícito para satisfacer al Type Checker
    tree = DecisionTreeClassifier(
        max_depth=int(SURROGATE_TREE_PARAMS.get("max_depth", 3)),
        min_samples_leaf=int(SURROGATE_TREE_PARAMS.get("min_samples_leaf", 5)),
        class_weight=str(SURROGATE_TREE_PARAMS.get("class_weight", "balanced")),
        random_state=int(SURROGATE_TREE_PARAMS.get("random_state", 42))
    )
    tree.fit(X, y)
    
    if tree.tree_.node_count <= 1:
        return {
            "T80_Day": float(failure_day),
            "Feature_Importances": {},
            "Rules": ["Intrinsic Defect (No anomalous environmental triggers found separating the prodromal window from healthy history)"]
        }

    rules = export_text(tree, feature_names=XAI_PHYSICAL_FEATURES, decimals=1)
    
    importances = {feat: float(imp) for feat, imp in zip(XAI_PHYSICAL_FEATURES, tree.feature_importances_) if imp > FEATURE_IMPORTANCE_MIN}
    sorted_imps = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    return {
        "T80_Day": float(failure_day),
        "Feature_Importances": sorted_imps,
        "Rules": rules.strip().split('\n')
    }

if __name__ == "__main__":
    ANOMALY_DIR = Path("data/anomaly/outdoor")
    ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")
    DIAGNOSTICS_DIR = Path("data/anomaly/diagnostics")
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    df_scored = pd.read_parquet(ANOMALY_DIR / "anomaly_scored_dataset.parquet")
    artifacts = joblib.load(ARTIFACTS_PATH)
    summary_table = artifacts.get("summary_table")

    # Identificamos celdas que sufrieron muerte física prematura
    t80_failed_cells = summary_table[
        (summary_table['survival_days_pce'] <= EARLY_FAILURE_WINDOW_DAYS) | 
        (summary_table['survival_days_pff'] <= EARLY_FAILURE_WINDOW_DAYS)
    ].copy()
    
    logger.info(f"Extracting Physical Forensic Rules for {len(t80_failed_cells)} T80 collapsed devices...")

    report_dict = {}
    for cell, row in t80_failed_cells.iterrows():
        cell_data = df_scored[df_scored['cell_name'] == str(cell)]
        t80_day = min(row.get('survival_days_pce', 999), row.get('survival_days_pff', 999))
        
        report = extract_physical_forensic_surrogate(cell_data, t80_day)
        if report:
            report_dict[str(cell)] = report

    out_file = DIAGNOSTICS_DIR / "forensic_surrogate_rules.json"
    with open(out_file, 'w') as f:
        json.dump(report_dict, f, indent=4)
        
    logger.info(f"Physical Forensic pipeline complete. Saved to {out_file.name}")