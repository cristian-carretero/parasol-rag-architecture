from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import joblib
import numpy as np
import pandas as pd
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
import xgboost as xgb

from src.config import RANDOM_STATE, XGB_PARAMS_RUL_PCE, XGB_PARAMS_RUL_PFF
from src.rul_forecasting import FEATURES_RUL_PCE, build_rul_matrix

# rul_forecasting.py solo expone FEATURES_RUL_PCE; el set de features para el
# target de pFF comparte el mismo vector de covariables ambientales, así que
# se define aquí en vez de importar un símbolo inexistente en el módulo.
FEATURES_RUL_PFF = list(FEATURES_RUL_PCE)

ANOMALY_DIR = Path("data/anomaly/outdoor")
ARTIFACTS_DIR = Path("data/anomaly/artifacts")
N_SPLITS = 5
N_PERMUTATION_REPEATS = 20
OVERFIT_GAP_RATIO = 0.30
VERDICT_THRESHOLDS = {"weak": 3.0, "real": 10.0}

SCORING = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(root_mean_squared_error, greater_is_better=False),
    "MedAE": make_scorer(median_absolute_error, greater_is_better=False),
    "R2": make_scorer(r2_score),
}


@dataclass(frozen=True)
class TargetSpec:
    key: str
    target_column: str
    features: list[str]
    xgb_params: Mapping[str, float | int]


TARGETS = [
    TargetSpec("pff", "Daily_Increment_pFF", list(FEATURES_RUL_PFF), XGB_PARAMS_RUL_PFF),
    TargetSpec("pce", "Daily_Damage_Increment", list(FEATURES_RUL_PCE), XGB_PARAMS_RUL_PCE),
]


def load_daily_matrix() -> pd.DataFrame:
    df_twin = pd.read_parquet(ANOMALY_DIR / "anomaly_scored_dataset.parquet")
    healthy_cohort = joblib.load(ARTIFACTS_DIR / "early_failure_artifacts.joblib")["healthy_cohort"]
    
    df_daily = build_rul_matrix(df_twin, healthy_cohort)
    
    if 'pFF' in df_twin.columns and 'Daily_Increment_pFF' not in df_daily.columns:
        df_pff = df_twin[df_twin['cell_name'].isin(healthy_cohort)].copy()
        if 'Datetime' not in df_pff.columns:
            df_pff['Datetime'] = pd.to_datetime(df_pff['Timestamp'], utc=True)
        df_pff['Date_Day'] = df_pff['Datetime'].dt.date
        
        df_pff_daily = df_pff.groupby(['cell_name', 'Date_Day']).agg(
            Daily_pFF=('pFF', 'mean')
        ).reset_index().sort_values(by=['cell_name', 'Date_Day'])
        
        # Al usar .groupby('cell_name')['Daily_pFF'].transform(...), pandas pasa un Series
        # (los valores de Daily_pFF del grupo), no un DataFrame. Por tanto, se opera directamente sobre el Series.
        df_pff_daily['pFF_Initial'] = (
            df_pff_daily.groupby('cell_name')['Daily_pFF']
            .transform(lambda s: s.head(7).median() if not s.empty else s.iloc[0])
        )
        
        df_pff_daily['Instant_Loss_pFF'] = 1.0 - (df_pff_daily['Daily_pFF'] / df_pff_daily['PFF_Initial'] if 'PFF_Initial' in df_pff_daily else df_pff_daily['Daily_pFF'] / df_pff_daily['pFF_Initial'])
        
        df_pff_daily['Smoothed_Loss_pFF'] = (
            df_pff_daily.groupby('cell_name')['Instant_Loss_pFF']
            .rolling(7, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        
        df_pff_daily['Cumulative_Damage_pFF'] = df_pff_daily['Smoothed_Loss_pFF'].clip(lower=0.0, upper=1.0)
        df_pff_daily['Lag_Damage_pFF'] = df_pff_daily.groupby('cell_name')['Cumulative_Damage_pFF'].shift(1).fillna(0.0)
        
        df_pff_daily['Daily_Increment_pFF'] = (
            df_pff_daily['Cumulative_Damage_pFF'] - df_pff_daily['Lag_Damage_pFF']
        ).clip(lower=0.0)
        
        df_daily = pd.merge(
            df_daily, 
            df_pff_daily[['cell_name', 'Date_Day', 'Daily_Increment_pFF']], 
            on=['cell_name', 'Date_Day'], 
            how='left'
        )
        
    return df_daily


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


def build_model_zoo(xgb_params: Mapping) -> dict[str, BaseEstimator]:
    xgb_high_l2 = dict(xgb_params)
    xgb_high_l2["reg_lambda"] = 10.0
    xgb_high_l2["reg_alpha"] = 0.5

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
        "XGBoost(producción)": xgb.XGBRegressor(**xgb_params),
        "XGBoost(reg_lambda=10.0)": xgb.XGBRegressor(**xgb_high_l2),
    }


def evaluate_cv(
    models: Mapping[str, BaseEstimator], X: pd.DataFrame, y: pd.Series,
    cv: BaseCrossValidator | list, cv_label: str,
) -> pd.DataFrame:
    rows = []
    for name, estimator in models.items():
        result = cross_validate(clone(estimator), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"cv": cv_label, "model": name}
        for metric, sign in zip(SCORING, (-1, -1, -1, 1)):
            vals = result[f"test_{metric}"]
            row[f"{metric}_mean"] = sign * vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_mean").reset_index(drop=True)


def print_cv_table(df_scores: pd.DataFrame, baseline_mae: float) -> None:
    for _, r in df_scores.iterrows():
        delta = baseline_mae - r["MAE_mean"]
        print(
            f"  [{r['cv']:>10s}] {r['model']:<26s} "
            f"MAE={r['MAE_mean']:.5f}±{r['MAE_std']:.5f}  "
            f"RMSE={r['RMSE_mean']:.5f}  MedAE={r['MedAE_mean']:.5f}  "
            f"R2={r['R2_mean']:+.4f}  ΔMAE_vs_baseline={delta:+.5f}"
        )


def compute_feature_importances(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    corr = X.corrwith(y)
    fitted = clone(model).fit(X, y)
    gain = pd.Series(fitted.feature_importances_, index=X.columns)
    # Con un único string en `scoring`, permutation_importance siempre devuelve
    # un Bunch (no dict[str, Bunch], que solo aplica si scoring es una lista/dict
    # de varios scorers); se fuerza el tipo porque el stub declara un Union.
    perm = cast(
        Bunch,
        permutation_importance(
            fitted, X, y, n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE, scoring="neg_mean_absolute_error", n_jobs=-1,
        ),
    )
    perm_series = pd.Series(perm.importances_mean, index=X.columns)
    df_importance = pd.DataFrame({
        "pearson_corr": corr, "xgb_gain": gain, "permutation_importance": perm_series,
    })
    return df_importance.sort_values("permutation_importance", ascending=False)


def compute_overfit_gap(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv_mae: float) -> tuple[float, float, str]:
    fitted = clone(model).fit(X, y)
    train_mae = mean_absolute_error(y, fitted.predict(X))
    gap = cv_mae - train_mae
    verdict = "posible sobreajuste" if gap > OVERFIT_GAP_RATIO * cv_mae else "gap razonable"
    return train_mae, gap, verdict


def compute_residual_diagnostics(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator, groups: pd.Series | None,
) -> tuple[np.ndarray, pd.Series, pd.Series | None]:
    oof_pred = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)
    residuals = y.to_numpy() - oof_pred
    resid_corr = pd.Series(
        {col: np.corrcoef(X[col], residuals)[0, 1] for col in X.columns}
    ).sort_values(key=np.abs, ascending=False)
    resid_by_group = None
    if groups is not None:
        resid_by_group = pd.Series(residuals, index=X.index).groupby(groups).mean().sort_values()
    return residuals, resid_corr, resid_by_group


def print_verdict(best_model: str, best_mae: float, baseline_mae: float) -> None:
    improvement_pct = 100 * (baseline_mae - best_mae) / baseline_mae
    if improvement_pct < VERDICT_THRESHOLDS["weak"]:
        verdict = "SIN SEÑAL RELEVANTE"
    elif improvement_pct < VERDICT_THRESHOLDS["real"]:
        verdict = "SEÑAL DÉBIL"
    else:
        verdict = "SEÑAL PREDICTIVA REAL"
    print(f"  Mejor modelo (KFold) : {best_model} (MAE={best_mae:.5f})")
    print(f"  Mejora vs baseline   : {improvement_pct:+.1f}%")
    print(f"  => {verdict}")


def diagnose_target(df_daily: pd.DataFrame, spec: TargetSpec) -> None:
    print("=" * 90)
    print(f" TARGET: {spec.target_column} ({spec.key.upper()})")
    print("=" * 90)

    X = df_daily[spec.features]
    y = df_daily[spec.target_column]
    groups = df_daily["cell_name"] if "cell_name" in df_daily.columns else None

    print("\n--- 1. Estadística descriptiva del target ---")
    print(describe_target(y).to_string())

    print("\n--- 2. Baseline vs modelos: KFold(shuffle) vs GroupKFold(cell_name) ---")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_model_zoo(dict(spec.xgb_params))

    scores_kfold = evaluate_cv(models, X, y, kf, "KFold")
    baseline_mae = scores_kfold.loc[scores_kfold["model"] == "Dummy(mean)", "MAE_mean"].iloc[0]
    print_cv_table(scores_kfold, baseline_mae)

    if groups is not None and groups.nunique() >= N_SPLITS:
        gkf = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
        scores_groupkfold = evaluate_cv(models, X, y, list(gkf.split(X, y, groups)), "GroupKFold")
        print()
        print_cv_table(scores_groupkfold, baseline_mae)
    else:
        print("  (GroupKFold omitido: sin 'cell_name' o celdas insuficientes)")

    print("\n--- 3. Importancia de variables (Pearson / ganancia XGBoost / permutación) ---")
    prod_model = xgb.XGBRegressor(**spec.xgb_params)
    print(compute_feature_importances(prod_model, X, y).to_string())

    print("\n--- 4. Overfitting gap (XGBoost producción) ---")
    cv_mae_prod = scores_kfold.loc[scores_kfold["model"] == "XGBoost(producción)", "MAE_mean"].iloc[0]
    train_mae, gap, overfit_verdict = compute_overfit_gap(prod_model, X, y, cv_mae_prod)
    print(f"  MAE train completo : {train_mae:.5f}")
    print(f"  MAE CV (KFold)     : {cv_mae_prod:.5f}")
    print(f"  Gap (CV - train)   : {gap:+.5f} ({overfit_verdict})")

    print("\n--- 5. Residuos fuera de muestra (cross_val_predict) ---")
    residuals, resid_corr, resid_by_cell = compute_residual_diagnostics(prod_model, X, y, kf, groups)
    print(f"  Sesgo global (media residuo) : {residuals.mean():+.6f}")
    print(f"  Std residuo                  : {residuals.std():.6f}")
    print("\n  Correlación residuo-feature:")
    print(resid_corr.to_string())
    if resid_by_cell is not None:
        print("\n  Sesgo medio del residuo por celda (out-of-fold):")
        print(resid_by_cell.to_string())

    print("\n--- 6. Veredicto automático ---")
    best_row = scores_kfold.iloc[0]
    print_verdict(best_row["model"], best_row["MAE_mean"], baseline_mae)
    print()


def main() -> None:
    df_daily = load_daily_matrix()
    print(f"Filas totales: {len(df_daily)} | Celdas: {df_daily['cell_name'].nunique()}\n")
    for spec in TARGETS:
        diagnose_target(df_daily, spec)


if __name__ == "__main__":
    main()