"""
Module: src/trajectory_xgb_audit.py
Description: Empirical audit of the multivariate trajectory forecasting engines
             (PCE, pFF, Jsc, Voc). Mirrors the diagnostic structure of
             rul_xgb_audit.py but operates on the normalized-increment targets
             used by 09_jv_mppt_trajectory_forecasting.py.

             The full audit report is automatically written to
             `outputs/diagnostics/trajectory_audit_summary.txt` while still
             being streamed to stdout.
"""

from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TextIO, cast

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import skew
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from src.config import (
    RANDOM_STATE,
    XGB_PARAMS_RUL_PCE,
    FILE_HEALTHY_COHORT,
    FILE_SCREENING_ARTIFACTS,
    DIAGNOSTICS_DIR,
)

# ------------------------------------------------------------------------------
# Module-level logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Trajectory-Audit")


# ------------------------------------------------------------------------------
# Dynamic import of the trajectory module (its filename starts with a digit).
# ------------------------------------------------------------------------------
_traj_mod = importlib.import_module("src.09_jv_mppt_trajectory_forecasting")
build_trajectory_matrix = _traj_mod.build_trajectory_matrix
BASE_FEATURES = _traj_mod.BASE_FEATURES


# ------------------------------------------------------------------------------
# Audit configuration
# ------------------------------------------------------------------------------
N_SPLITS = 5
N_PERMUTATION_REPEATS = 20
OVERFIT_GAP_RATIO = 0.30
VERDICT_THRESHOLDS = {"weak": 3.0, "real": 10.0}

AUDIT_REPORT_PATH: Path = DIAGNOSTICS_DIR / "trajectory_audit_summary.txt"

SCORING = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
    "MedAE": make_scorer(median_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score),
}
METRIC_SIGNS = {"MAE": -1, "RMSE": -1, "MedAE": -1, "R2": 1}

TARGET_PARAMS = ["PCE", "pFF", "Jsc", "Voc"]


# ------------------------------------------------------------------------------
# Stdout tee utility
# ------------------------------------------------------------------------------
class _Tee:
    """Duplicate every write/flush to multiple streams."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


# ------------------------------------------------------------------------------
# Target specification
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec:
    key: str
    target_column: str
    features: list[str]
    xgb_params: Mapping


def _build_specs() -> list[TargetSpec]:
    """One spec per physical parameter, each with its own lag feature."""
    specs = []
    for param in TARGET_PARAMS:
        specs.append(TargetSpec(
            key=param.lower(),
            target_column=f"{param}_Delta",
            features=BASE_FEATURES + [f"{param}_Lag1"],
            xgb_params=XGB_PARAMS_RUL_PCE,
        ))
    return specs


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------
def load_trajectory_matrix() -> pd.DataFrame:
    """Load the healthy cohort and build the normalized-increment matrix."""
    df_healthy = pd.read_parquet(FILE_HEALTHY_COHORT)
    artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)
    healthy_cohort = artifacts["healthy_cohort"]
    df_daily, _ = build_trajectory_matrix(df_healthy)
    df_daily = df_daily[df_daily["cell_name"].isin(healthy_cohort)].copy()
    return df_daily


# ------------------------------------------------------------------------------
# Descriptive statistics
# ------------------------------------------------------------------------------
def describe_target(y: pd.Series) -> pd.Series:
    n = len(y)
    return pd.Series({
        "n_samples": n,
        "mean": y.mean(),
        "median": y.median(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max(),
        "skew": skew(y),
        "pct_zero": 100 * (y == 0).sum() / n,
        "pct_negative": 100 * (y < 0).sum() / n,
    })


# ------------------------------------------------------------------------------
# Model zoo
# ------------------------------------------------------------------------------
def build_model_zoo(xgb_params: Mapping) -> dict[str, BaseEstimator]:
    xgb_alt_reg = dict(xgb_params)
    xgb_alt_reg["reg_lambda"] = 100.0
    xgb_alt_reg["reg_alpha"] = 1.0

    return {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "Dummy(median)": DummyRegressor(strategy="median"),
        "Ridge(alpha=0.1)": make_pipeline(StandardScaler(), Ridge(alpha=0.1, random_state=RANDOM_STATE)),
        "Ridge(alpha=1.0)": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "Ridge(alpha=10.0)": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        "Ridge(alpha=100.0)": make_pipeline(StandardScaler(), Ridge(alpha=100.0, random_state=RANDOM_STATE)),
        "RandomForest(200,md4)": RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost(production)": xgb.XGBRegressor(**xgb_params),
        "XGBoost(reg_lambda=100)": xgb.XGBRegressor(**xgb_alt_reg),
    }


def evaluate_cv(
    models: Mapping[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    cv_label: str,
) -> pd.DataFrame:
    rows = []
    for name, estimator in models.items():
        result = cross_validate(clone(estimator), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row: dict = {"cv": cv_label, "model": name}
        for metric, sign in METRIC_SIGNS.items():
            vals = result[f"test_{metric}"]
            row[f"{metric}_mean"] = sign * vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def print_cv_table(df_scores: pd.DataFrame, baseline_mae: float) -> None:
    for _, r in df_scores.iterrows():
        delta = baseline_mae - r["MAE_mean"]
        print(
            f"  [{r['cv']:>10s}] {r['model']:<28s} "
            f"MAE={r['MAE_mean']:.5f}±{r['MAE_std']:.5f}  "
            f"RMSE={r['RMSE_mean']:.5f}  MedAE={r['MedAE_mean']:.5f}  "
            f"R2={r['R2_mean']:+.4f}  ΔMAE_vs_baseline={delta:+.5f}"
        )


# ------------------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------------------
def compute_feature_importances(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    corr = X.corrwith(y)
    fitted = clone(model).fit(X, y)
    gain = pd.Series(fitted.feature_importances_, index=X.columns)

    perm = cast(
        Bunch,
        permutation_importance(
            fitted, X, y,
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        ),
    )
    perm_series = pd.Series(perm.importances_mean, index=X.columns)

    return pd.DataFrame({
        "pearson_corr": corr,
        "xgb_gain": gain,
        "permutation_importance": perm_series,
    }).sort_values("permutation_importance", ascending=False)


# ------------------------------------------------------------------------------
# Overfitting gap
# ------------------------------------------------------------------------------
def compute_overfit_gap(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_mae: float,
) -> tuple[float, float, str]:
    fitted = clone(model).fit(X, y)
    train_mae = mean_absolute_error(y, fitted.predict(X))
    gap = cv_mae - train_mae
    verdict = "possible overfit" if gap > OVERFIT_GAP_RATIO * cv_mae else "reasonable gap"
    return train_mae, gap, verdict


# ------------------------------------------------------------------------------
# Residual diagnostics
# ------------------------------------------------------------------------------
def compute_residual_diagnostics(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: pd.Series | None,
):
    oof_pred = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)
    residuals = y.to_numpy() - oof_pred
    resid_corr = pd.Series(
        {col: np.corrcoef(X[col], residuals)[0, 1] for col in X.columns}
    ).sort_values(key=np.abs, ascending=False)
    resid_by_group = None
    if groups is not None:
        resid_by_group = pd.Series(residuals, index=X.index).groupby(groups).mean().sort_values()
    return residuals, resid_corr, resid_by_group


# ------------------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------------------
def print_verdict(best_model: str, best_mae: float, baseline_mae: float) -> None:
    improvement_pct = 100 * (baseline_mae - best_mae) / baseline_mae
    if improvement_pct < VERDICT_THRESHOLDS["weak"]:
        verdict = "NO RELEVANT SIGNAL"
    elif improvement_pct < VERDICT_THRESHOLDS["real"]:
        verdict = "WEAK SIGNAL"
    else:
        verdict = "REAL PREDICTIVE SIGNAL"
    print(f"  Best model (KFold)  : {best_model} (MAE={best_mae:.5f})")
    print(f"  Improvement vs base : {improvement_pct:+.1f}%")
    print(f"  => {verdict}")


# ------------------------------------------------------------------------------
# Per-parameter diagnostic
# ------------------------------------------------------------------------------
def diagnose_target(df_daily: pd.DataFrame, spec: TargetSpec) -> None:
    print("=" * 90)
    print(f" TARGET: {spec.target_column} ({spec.key.upper()})")
    print("=" * 90)

    if spec.target_column not in df_daily.columns:
        print(f"  [SKIP] Column '{spec.target_column}' not present in the matrix.")
        print()
        return

    # Check that all required features exist
    missing = [f for f in spec.features if f not in df_daily.columns]
    if missing:
        print(f"  [SKIP] Missing features: {missing}")
        print()
        return

    X = df_daily[spec.features]
    y = df_daily[spec.target_column]
    groups = df_daily["cell_name"] if "cell_name" in df_daily.columns else None

    # 1. Descriptive stats
    print("\n--- 1. Target descriptive statistics ---")
    print(describe_target(y).to_string())

    # 2. Model comparison
    print("\n--- 2. Baseline vs models: KFold(shuffle) vs GroupKFold(cell_name) ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_model_zoo(dict(spec.xgb_params))

    scores_kfold = evaluate_cv(models, X, y, kf, "KFold")
    baseline_mae = scores_kfold.loc[scores_kfold["model"] == "Dummy(mean)", "MAE_mean"].iloc[0]
    print_cv_table(scores_kfold, baseline_mae)

    if groups is None:
        print("  (GroupKFold omitted: 'cell_name' column missing)")
    elif groups.nunique() < N_SPLITS:
        print(f"  (GroupKFold omitted: only {groups.nunique()} unique cells, "
              f"need at least {N_SPLITS} for {N_SPLITS}-fold)")
    else:
        gkf = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
        scores_groupkfold = evaluate_cv(models, X, y, list(gkf.split(X, y, groups)), "GroupKFold")
        print()
        print_cv_table(scores_groupkfold, baseline_mae)

    # 3. Feature importance
    print("\n--- 3. Feature importance (Pearson / XGBoost gain / permutation) ---")
    prod_model = xgb.XGBRegressor(**spec.xgb_params)
    print(compute_feature_importances(prod_model, X, y).to_string())

    # 4. Overfit gap
    print("\n--- 4. Overfitting gap (XGBoost production) ---")
    cv_mae_prod = scores_kfold.loc[scores_kfold["model"] == "XGBoost(production)", "MAE_mean"].iloc[0]
    train_mae, gap, overfit_verdict = compute_overfit_gap(prod_model, X, y, cv_mae_prod)
    print(f"  MAE full train     : {train_mae:.5f}")
    print(f"  MAE CV (KFold)     : {cv_mae_prod:.5f}")
    print(f"  Gap (CV - train)   : {gap:+.5f} ({overfit_verdict})")

    # 5. Residual diagnostics
    print("\n--- 5. Out-of-sample residuals (cross_val_predict) ---")
    residuals, resid_corr, resid_by_cell = compute_residual_diagnostics(prod_model, X, y, kf, groups)
    print(f"  Global bias (residual mean) : {residuals.mean():+.6f}")
    print(f"  Residual std                : {residuals.std():.6f}")
    print("\n  Residual-feature correlation:")
    print(resid_corr.to_string())
    if resid_by_cell is not None:
        print("\n  Mean residual bias per cell (out-of-fold):")
        print(resid_by_cell.to_string())

    # 6. Verdict
    print("\n--- 6. Automatic verdict ---")
    best_row = scores_kfold.iloc[0]
    print_verdict(best_row["model"], best_row["MAE_mean"], baseline_mae)
    print()


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
def _run_audit() -> None:
    logger.info("Loading trajectory feature matrix for multivariate audit...")
    df_daily = load_trajectory_matrix()
    logger.info(f"Rows: {len(df_daily)} | Cells: {df_daily['cell_name'].nunique()}")

    for spec in _build_specs():
        diagnose_target(df_daily, spec)


def main() -> None:
    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with open(AUDIT_REPORT_PATH, "w", encoding="utf-8") as report_file:
        sys.stdout = _Tee(original_stdout, report_file)
        try:
            _run_audit()
        finally:
            sys.stdout = original_stdout

    logger.info(f"Audit report saved to {AUDIT_REPORT_PATH}")


if __name__ == "__main__":
    main()