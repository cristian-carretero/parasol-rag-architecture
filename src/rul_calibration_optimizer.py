"""
Module: src/rul_calibration_optimizer.py
Description: Empirical calibration of the six hardcoded coefficients of the
             hybrid kinematics engine in 08_mppt_rul_forecasting.py.

             Strategy:
               1. Pre-train the 4 blind XGBoost models (one per LOOCV fold).
               2. Pre-compute every simulation unit (cell x anchor x mode).
                  A simulation unit contains all the quantities that do NOT
                  depend on the kinematic coefficients, so a full LOOCV
                  re-evaluation reduces to re-running only the closed-form
                  simulation loop.
               3. Sweep coefficients via a 1D sensitivity analysis, then a
                  focused N-D grid search, using Sensor MAE as the objective.
               4. Persist the winning coefficients to a JSON file that the 08
                  module loads automatically on its next run.

             Outputs:
               - outputs/diagnostics/rul_coeffs_sensitivity.parquet
               - outputs/diagnostics/rul_coeffs_search.parquet
               - outputs/diagnostics/rul_coeffs_calibrated.json
               - outputs/diagnostics/rul_coeffs_optimization.txt (mirror of stdout)
"""

from __future__ import annotations

import importlib
import itertools
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    BURN_IN_DAYS,
    FILE_HEALTHY_COHORT,
    FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS,
    FILE_RUL_COEFFS_CALIBRATED,
    DIAGNOSTICS_DIR,
)

# ------------------------------------------------------------------------------
# Dynamic import of the RUL module (its filename starts with a digit).
# ------------------------------------------------------------------------------
rul = importlib.import_module("src.08_mppt_rul_forecasting")

build_rul_matrix = rul.build_rul_matrix
fetch_api_history = rul.fetch_api_history
fit_calibration = rul.fit_calibration
apply_calibration = rul.apply_calibration
train_rul_engine = rul.train_rul_engine
simulate_rul_kinematics = rul.simulate_rul_kinematics

# The optimizer always uses the immutable hardcoded baseline as reference,
# regardless of whether a previously calibrated JSON exists.
PHI_0_DEFAULT = rul.PHI_0_HARDCODED
PHI_1_DEFAULT = rul.PHI_1_HARDCODED
PHI_2_DEFAULT = rul.PHI_2_HARDCODED
LAMBDA_W_DEFAULT = rul.LAMBDA_W_HARDCODED
MU_DEFAULT = rul.MU_HARDCODED
EPS_DEFAULT = rul.EPS_HARDCODED

ROLLING_WINDOW = rul.ROLLING_WINDOW
SIMULATION_WINDOW = rul.SIMULATION_WINDOW
T80_DAMAGE_LIMIT = rul.T80_DAMAGE_LIMIT


# ==============================================================================
# Logging + report paths
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RUL-Calib-Optimizer")

REPORT_PATH = DIAGNOSTICS_DIR / "rul_coeffs_optimization.txt"
SENSITIVITY_PATH = DIAGNOSTICS_DIR / "rul_coeffs_sensitivity.parquet"
SEARCH_PATH = DIAGNOSTICS_DIR / "rul_coeffs_search.parquet"
CALIBRATED_COEFFS_PATH = FILE_RUL_COEFFS_CALIBRATED


class _Tee:
    """File-like object duplicating every write/flush to multiple streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


# ==============================================================================
# Baseline registry (used for reporting only)
# ==============================================================================
BASELINE_REGISTRY = {
    "phi_0": PHI_0_DEFAULT,
    "phi_1": PHI_1_DEFAULT,
    "phi_2": PHI_2_DEFAULT,
    "lambda_w": LAMBDA_W_DEFAULT,
    "mu": MU_DEFAULT,
    "eps": EPS_DEFAULT,
}


# ==============================================================================
# Pre-computed simulation units
# ==============================================================================
@dataclass
class SimulationUnit:
    cell: str
    mode: str                                  # 'Sensor' or 'API'
    anchor_day: float
    current_damage: float
    rolling_irr: list
    rolling_temp: list
    future_weather: pd.DataFrame
    model: xgb.XGBRegressor
    true_survival_days: Optional[float]


def _coerce_optional_float(value) -> Optional[float]:
    """Return a plain float or None; used to satisfy strict type checkers."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def precompute_units(
    df_daily: pd.DataFrame,
    df_api_raw: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    healthy_cohort: List[str],
    blind_models_by_fold: Dict[str, xgb.XGBRegressor],
) -> List[SimulationUnit]:
    """
    Pre-compute every (cell, anchor, mode) simulation unit. Everything that
    does NOT depend on the kinematic coefficients is computed once here.
    """
    units: List[SimulationUnit] = []

    for cell in healthy_cohort:
        cell_data = df_daily[df_daily["cell_name"] == cell].sort_values("Exposure_Days")
        if cell_data.empty:
            continue

        true_survival_days: Optional[float] = (
            _coerce_optional_float(t80_metrics.loc[cell, "survival_days_pce"])
            if cell in t80_metrics.index
            else None
        )

        max_days = cell_data["Exposure_Days"].max()
        anchors = list(range(int(BURN_IN_DAYS), int(max_days) + 1, ROLLING_WINDOW))
        if int(max_days) not in anchors:
            anchors.append(int(max_days))

        train_cells = [c for c in healthy_cohort if c != cell]
        model = blind_models_by_fold[cell]

        for anchor in anchors:
            hist_cutoff = cell_data[cell_data["Exposure_Days"] <= anchor]
            if hist_cutoff.empty:
                continue
            actual_day = float(hist_cutoff.iloc[-1]["Exposure_Days"])
            cum_damage = float(hist_cutoff.iloc[-1]["Cumulative_Damage"])

            if cum_damage >= T80_DAMAGE_LIMIT:
                break

            rolling_irr = list(hist_cutoff["Daily_Irradiance_Dose"].tail(ROLLING_WINDOW))
            rolling_temp = list(hist_cutoff["Daily_Max_Temp_C"].tail(ROLLING_WINDOW))

            # --- Sensor mode unit ---
            future_sensor = cell_data[cell_data["Exposure_Days"] > anchor].head(SIMULATION_WINDOW)
            if not future_sensor.empty:
                units.append(SimulationUnit(
                    cell=cell, mode="Sensor", anchor_day=actual_day,
                    current_damage=cum_damage,
                    rolling_irr=list(rolling_irr),
                    rolling_temp=list(rolling_temp),
                    future_weather=future_sensor,
                    model=model,
                    true_survival_days=true_survival_days,
                ))

            # --- API mode unit (walk-forward calibration does not depend on kinematic coeffs) ---
            if not df_api_raw.empty:
                anchor_date = hist_cutoff.iloc[-1]["Date_Day"]
                try:
                    reg_temp_wf, reg_irr_wf = fit_calibration(
                        df_daily, df_api_raw,
                        max_date=anchor_date,
                        train_cells=train_cells,
                        window_days=90,
                    )
                except ValueError:
                    continue

                future_raw = df_api_raw[df_api_raw["Date_Day"] > anchor_date].head(SIMULATION_WINDOW)
                if not future_raw.empty:
                    future_api = apply_calibration(reg_temp_wf, reg_irr_wf, future_raw)
                    units.append(SimulationUnit(
                        cell=cell, mode="API", anchor_day=actual_day,
                        current_damage=cum_damage,
                        rolling_irr=list(rolling_irr),
                        rolling_temp=list(rolling_temp),
                        future_weather=future_api,
                        model=model,
                        true_survival_days=true_survival_days,
                    ))

    return units


# ==============================================================================
# Fast LOOCV re-evaluation given pre-computed units
# ==============================================================================
def evaluate_coefficients(
    units: List[SimulationUnit],
    phi_0: float, phi_1: float, phi_2: float, lambda_w: float,
    mu: float, eps: float,
) -> Dict[str, float]:
    """
    Fast re-evaluation: applies the soft-countdown sequentially per
    (cell, mode) group and aggregates the LOOCV MAE for both modes.
    """
    grouped: Dict[tuple, List[SimulationUnit]] = {}
    for u in units:
        grouped.setdefault((u.cell, u.mode), []).append(u)

    records = []
    for (cell, mode), cell_units in grouped.items():
        cell_units.sort(key=lambda u: u.anchor_day)
        prev_anchor: Optional[float] = None
        prev_rul: Optional[float] = None

        for u in cell_units:
            rul_raw = simulate_rul_kinematics(
                u.current_damage,
                list(u.rolling_irr),
                list(u.rolling_temp),
                u.future_weather,
                u.model,
                phi_0=phi_0, phi_1=phi_1, phi_2=phi_2, lambda_w=lambda_w,
            )
            if prev_rul is not None and prev_anchor is not None:
                elapsed = u.anchor_day - prev_anchor
                expected = max(0.0, prev_rul - elapsed)
                rul_val = mu * rul_raw + (1.0 - mu) * expected
                rul_val = min(rul_val, prev_rul + eps)
            else:
                rul_val = rul_raw

            prev_anchor, prev_rul = u.anchor_day, rul_val
            records.append({
                "cell_name": cell,
                "Anchor_Day": u.anchor_day,
                "True_Survival_Days": u.true_survival_days,
                "RUL_Pred": rul_val,
                "Type": mode,
            })

    df = pd.DataFrame(records)
    df["RUL_Real"] = df["True_Survival_Days"] - df["Anchor_Day"]
    valid = df[df["RUL_Real"] > 0]

    def _mae(mode_label: str) -> float:
        sub = valid[valid["Type"] == mode_label]
        if sub.empty:
            return float("nan")
        return float(np.mean(np.abs(sub["RUL_Pred"] - sub["RUL_Real"])))

    return {
        "mae_sensor": _mae("Sensor"),
        "mae_api": _mae("API"),
        "n": int(len(valid)),
    }


# ==============================================================================
# Grids
# ==============================================================================
# Phase 1: 1D sensitivity analysis. Each coefficient is swept in isolation,
# with the others fixed at their production defaults.
GRID_1D = {
    "phi_0":    [0.0005, 0.0010, 0.0015, 0.0025, 0.0050, 0.0100],
    "phi_1":    [0.0010, 0.0025, 0.0050, 0.0100, 0.0200],
    "phi_2":    [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    "lambda_w": [0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
    "mu":       [0.2, 0.3, 0.5, 0.7, 0.9],
    "eps":      [0.0, 0.5, 1.0, 2.0, 5.0],
}

# Phase 2: focused N-D grid search on the most impactful coefficients.
# phi_1 and phi_2 are held at their defaults (see BASELINE_REGISTRY).
GRID_ND = {
    "phi_0":    [0.0075, 0.0100, 0.0125],
    "lambda_w": [0.5, 0.6, 0.7],
    "mu":       [0.55, 0.60, 0.65, 0.70],
    "eps":      [0.5, 1.0],
}


# ==============================================================================
# Phase 1 — 1D sensitivity analysis
# ==============================================================================
def run_phase_1_sensitivity(units: List[SimulationUnit]) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print(" PHASE 1: 1D SENSITIVITY ANALYSIS")
    print("=" * 90)
    print(
        f"  Baseline defaults: phi_0={PHI_0_DEFAULT}, phi_1={PHI_1_DEFAULT}, "
        f"phi_2={PHI_2_DEFAULT}, lambda_w={LAMBDA_W_DEFAULT}, "
        f"mu={MU_DEFAULT}, eps={EPS_DEFAULT}"
    )

    rows = []
    for coeff_name, values in GRID_1D.items():
        print(f"\n  --- Sweeping {coeff_name} ---")
        for v in values:
            kwargs = dict(
                phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
                phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
                mu=MU_DEFAULT, eps=EPS_DEFAULT,
            )
            kwargs[coeff_name] = v
            res = evaluate_coefficients(units, **kwargs)
            rows.append({"coefficient": coeff_name, "value": v, **res})
            print(
                f"    {coeff_name:<10s} = {v:<10.4f} | "
                f"MAE_sensor = {res['mae_sensor']:6.3f} | "
                f"MAE_api = {res['mae_api']:6.3f}"
            )

    return pd.DataFrame(rows)


# ==============================================================================
# Phase 2 — Focused N-D grid search
# ==============================================================================
def run_phase_2_grid(units: List[SimulationUnit]) -> pd.DataFrame:
    print("\n" + "=" * 90)
    print(" PHASE 2: FOCUSED N-D GRID SEARCH")
    print("=" * 90)

    keys = list(GRID_ND.keys())
    combos = list(itertools.product(*[GRID_ND[k] for k in keys]))
    print(f"  Evaluating {len(combos)} combinations on {len(units)} simulation units...")

    rows = []
    best_so_far = float("inf")
    for i, combo in enumerate(combos, 1):
        kwargs = dict(
            phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
            phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
            mu=MU_DEFAULT, eps=EPS_DEFAULT,
        )
        for k, v in zip(keys, combo):
            kwargs[k] = v
        res = evaluate_coefficients(units, **kwargs)
        rows.append({**dict(zip(keys, combo)), **res})
        if res["mae_sensor"] < best_so_far:
            best_so_far = res["mae_sensor"]
        if i % 10 == 0 or i == len(combos):
            print(
                f"  [{i:>4d}/{len(combos)}] best so far: "
                f"MAE_sensor = {best_so_far:.3f}"
            )

    df = pd.DataFrame(rows).sort_values("mae_sensor").reset_index(drop=True)
    print("\n  Top 10 combinations by Sensor MAE:")
    print(df.head(10).to_string(index=False))
    return df


# ==============================================================================
# Blind model pre-training
# ==============================================================================
def pretrain_blind_models(
    df_daily: pd.DataFrame, healthy_cohort: List[str]
) -> Dict[str, xgb.XGBRegressor]:
    print("\n" + "=" * 90)
    print(" PRE-TRAINING BLIND XGBOOST MODELS (ONE PER LOOCV FOLD)")
    print("=" * 90)
    models: Dict[str, xgb.XGBRegressor] = {}
    for cell in healthy_cohort:
        train_cells = [c for c in healthy_cohort if c != cell]
        df_train = df_daily[df_daily["cell_name"].isin(train_cells)]
        models[cell] = train_rul_engine(df_train)
        print(f"  [{cell}] trained on {len(df_train)} rows from {len(train_cells)} cells")
    return models


# ==============================================================================
# Final summary
# ==============================================================================
def print_final_summary(best: pd.Series, baseline: Dict[str, float]) -> None:
    print("\n" + "=" * 90)
    print(" BEST COMBINATION FOUND (by Sensor MAE)")
    print("=" * 90)

    print("  Swept coefficients:")
    for k in GRID_ND.keys():
        if k in best.index:
            base_val = BASELINE_REGISTRY[k]
            print(f"    {k:<10s} = {float(best[k]):<10.4f}   (baseline {base_val})")

    print("\n  Held coefficients (not swept in Phase 2):")
    for k in ["phi_1", "phi_2"]:
        if k not in GRID_ND:
            print(f"    {k:<10s} = {BASELINE_REGISTRY[k]:<10.4f}   (default)")

    print("\n  Metrics:")
    print(f"    MAE_sensor = {best['mae_sensor']:.3f}  (baseline {baseline['mae_sensor']:.3f})")
    print(f"    MAE_api    = {best['mae_api']:.3f}  (baseline {baseline['mae_api']:.3f})")

    improvement = baseline["mae_sensor"] - float(best["mae_sensor"])
    pct = 100.0 * improvement / baseline["mae_sensor"]
    print(f"\n  Improvement (Sensor) = {improvement:+.3f} days ({pct:+.1f}%)")
    print("=" * 90)


# ==============================================================================
# Persist calibrated coefficients
# ==============================================================================
def persist_calibrated_coefficients(
    best_row: pd.Series,
    baseline: Dict[str, float],
    grid_size: int,
) -> None:
    """
    Write the winning coefficients to a JSON file consumed by the 08 module.
    Includes a metadata block for traceability.
    """
    import json
    from datetime import datetime

    payload = {
        "phi_0":    float(best_row["phi_0"]) if "phi_0" in best_row.index else PHI_0_DEFAULT,
        "phi_1":    float(PHI_1_DEFAULT),
        "phi_2":    float(PHI_2_DEFAULT),
        "lambda_w": float(best_row["lambda_w"]) if "lambda_w" in best_row.index else LAMBDA_W_DEFAULT,
        "mu":       float(best_row["mu"]) if "mu" in best_row.index else MU_DEFAULT,
        "eps":      float(best_row["eps"]) if "eps" in best_row.index else EPS_DEFAULT,
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_by": "src/rul_calibration_optimizer.py",
            "grid_size": grid_size,
            "mae_sensor": float(best_row["mae_sensor"]),
            "mae_api": float(best_row["mae_api"]),
            "baseline_mae_sensor": float(baseline["mae_sensor"]),
            "baseline_mae_api": float(baseline["mae_api"]),
            "n_observations": int(best_row["n"]),
        },
    }

    CALIBRATED_COEFFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATED_COEFFS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  Calibrated coefficients persisted -> {CALIBRATED_COEFFS_PATH}")
    print("  The 08 module will automatically use these on its next run.")


# ==============================================================================
# Entry point
# ==============================================================================
def _run() -> None:
    # ---- Data ----
    logger.info("Loading inputs...")
    df_twin = pd.read_parquet(FILE_HEALTHY_COHORT)
    t80_metrics = pd.read_parquet(FILE_T80_TRUTH)
    healthy_cohort = joblib.load(FILE_SCREENING_ARTIFACTS)["healthy_cohort"]

    if "PCE_initial" not in df_twin.columns:
        df_twin = df_twin.merge(
            t80_metrics[["PCE_initial"]],
            left_on="cell_name", right_index=True, how="left",
        )

    df_daily = build_rul_matrix(df_twin, healthy_cohort)
    df_api_raw = fetch_api_history(df_daily)

    # ---- Pre-training ----
    blind_models = pretrain_blind_models(df_daily, healthy_cohort)

    # ---- Pre-compute simulation units ----
    print("\n" + "=" * 90)
    print(" PRE-COMPUTING SIMULATION UNITS")
    print("=" * 90)
    units = precompute_units(df_daily, df_api_raw, t80_metrics, healthy_cohort, blind_models)
    sensor_units = [u for u in units if u.mode == "Sensor"]
    api_units = [u for u in units if u.mode == "API"]
    print(f"  Sensor units: {len(sensor_units)}")
    print(f"  API units   : {len(api_units)}")
    print(f"  Total       : {len(units)}")

    # ---- Baseline ----
    baseline = evaluate_coefficients(
        units,
        phi_0=PHI_0_DEFAULT, phi_1=PHI_1_DEFAULT,
        phi_2=PHI_2_DEFAULT, lambda_w=LAMBDA_W_DEFAULT,
        mu=MU_DEFAULT, eps=EPS_DEFAULT,
    )
    print("\n" + "=" * 90)
    print(" BASELINE (current production defaults)")
    print("=" * 90)
    print(f"  MAE_sensor = {baseline['mae_sensor']:.3f} days")
    print(f"  MAE_api    = {baseline['mae_api']:.3f} days")
    print(f"  N          = {baseline['n']}")

    # ---- Phase 1 ----
    df_sens = run_phase_1_sensitivity(units)
    SENSITIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sens.to_parquet(SENSITIVITY_PATH, index=False)
    print(f"\n  Sensitivity table saved -> {SENSITIVITY_PATH}")

    # ---- Phase 2 ----
    df_grid = run_phase_2_grid(units)
    df_grid.to_parquet(SEARCH_PATH, index=False)
    print(f"\n  Grid search table saved -> {SEARCH_PATH}")

    # ---- Final summary ----
    print_final_summary(df_grid.iloc[0], baseline)

    # ---- Persist calibrated coefficients ----
    persist_calibrated_coefficients(
        best_row=df_grid.iloc[0],
        baseline=baseline,
        grid_size=len(df_grid),
    )


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            _run()
        finally:
            sys.stdout = original_stdout
    logger.info(f"Full optimization report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()