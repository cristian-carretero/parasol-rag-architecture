"""
Module: src/audit_threshold_phase_alignment.py
Description: Test whether the OOF residual distribution used to calibrate
alert thresholds (mature phase) matches the residual distribution actually
seen at runtime (early/action phase).

Populations, all computed with the SAME fold model (trained on other cells'
mature phase):
  A (calib)     : K-fold OOF |residuals| on training cells' mature phase
                  → this is what Q_alert is derived from.
  B (application): |residuals| on holdout cell's early phase (t <= W*)
                  → this is where alerts are actually emitted.
  C (control)   : |residuals| on holdout cell's mature phase (t > W*)
                  → isolates cell-identity generalization from phase.
"""
from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp

from src.config import (
    BURN_IN_DAYS, FEATURES, FILE_MERGED_FEATURES, FILE_T80_TRUTH,
    FILE_SCREENING_ARTIFACTS, XGB_PCE_PARAMS, XGB_PFF_PARAMS,
    FIGURES_SCREENING_DIR,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit.phase_alignment")

OUT_DIR = Path("outputs/diagnostics/screening")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = FIGURES_SCREENING_DIR / "phase_alignment"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# The screening module name starts with a digit; import via importlib.
_screening = importlib.import_module("src.07_jv_mppt_early_screening")
preprocess_telemetry_data = _screening.preprocess_telemetry_data
oof_abs_residuals = _screening.oof_abs_residuals


def _collect_populations(
    df: pd.DataFrame,
    t80_metrics: pd.DataFrame,
    cohort: List[str],
    burn_in_days: float = BURN_IN_DAYS,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Rebuild the LOOCV and collect A/B/C residual populations per metric."""
    df_dl = preprocess_telemetry_data(df)
    df_dl = df_dl.merge(
        t80_metrics[["PCE_initial", "combined_survival_days"]],
        left_on="cell_name", right_index=True, how="inner",
    )
    df_dl = df_dl[df_dl["Exposure_Days"] <= df_dl["combined_survival_days"]].copy()
    df_dl = df_dl.dropna(subset=FEATURES + ["PCE", "pFF", "PCE_initial"]).reset_index(drop=True)
    df_dl["PCE_Relative"] = df_dl["PCE"] / df_dl["PCE_initial"].clip(lower=1e-3)

    pops = {m: {"A": [], "B": [], "C": [], "per_cell": {}} for m in ("PCE", "pFF")}

    for holdout in cohort:
        train_cells = [c for c in cohort if c != holdout]
        train_mask = (
            df_dl["cell_name"].isin(train_cells)
            & (df_dl["Exposure_Days"] > burn_in_days)
        )
        early_mask = (
            (df_dl["cell_name"] == holdout)
            & (df_dl["Exposure_Days"] <= burn_in_days)
        )
        mature_mask = (
            (df_dl["cell_name"] == holdout)
            & (df_dl["Exposure_Days"] > burn_in_days)
        )

        Xtr = df_dl.loc[train_mask, FEATURES]
        ytr_pce = df_dl.loc[train_mask, "PCE_Relative"]
        ytr_pff = df_dl.loc[train_mask, "pFF"]

        m_pce = xgb.XGBRegressor(**XGB_PCE_PARAMS).fit(Xtr, ytr_pce)
        m_pff = xgb.XGBRegressor(**XGB_PFF_PARAMS).fit(Xtr, ytr_pff)

        # A: calibration (K-fold OOF on training cells' mature phase).
        A_pce = oof_abs_residuals(lambda: xgb.XGBRegressor(**XGB_PCE_PARAMS), Xtr, ytr_pce)
        A_pff = oof_abs_residuals(lambda: xgb.XGBRegressor(**XGB_PFF_PARAMS), Xtr, ytr_pff)

        # B: application (holdout's early phase).
        Xe = df_dl.loc[early_mask, FEATURES]
        ye_pce = df_dl.loc[early_mask, "PCE_Relative"].to_numpy()
        ye_pff = df_dl.loc[early_mask, "pFF"].to_numpy()
        B_pce = np.abs(ye_pce - m_pce.predict(Xe)) if len(Xe) else np.array([])
        B_pff = np.abs(ye_pff - m_pff.predict(Xe)) if len(Xe) else np.array([])

        # C: control (holdout's mature phase).
        Xm = df_dl.loc[mature_mask, FEATURES]
        ym_pce = df_dl.loc[mature_mask, "PCE_Relative"].to_numpy()
        ym_pff = df_dl.loc[mature_mask, "pFF"].to_numpy()
        C_pce = np.abs(ym_pce - m_pce.predict(Xm)) if len(Xm) else np.array([])
        C_pff = np.abs(ym_pff - m_pff.predict(Xm)) if len(Xm) else np.array([])

        for metric, A, B, C in (("PCE", A_pce, B_pce, C_pce), ("pFF", A_pff, B_pff, C_pff)):
            if A.size:
                pops[metric]["A"].append(A)
            if B.size:
                pops[metric]["B"].append(B)
            if C.size:
                pops[metric]["C"].append(C)

        pops["PCE"]["per_cell"][holdout] = {
            "n_A": int(A_pce.size), "n_B": int(B_pce.size), "n_C": int(C_pce.size),
            "p98_A": float(np.percentile(A_pce, 98)) if A_pce.size else None,
            "p98_B": float(np.percentile(B_pce, 98)) if B_pce.size else None,
            "p98_C": float(np.percentile(C_pce, 98)) if C_pce.size else None,
            "median_A": float(np.median(A_pce)) if A_pce.size else None,
            "median_B": float(np.median(B_pce)) if B_pce.size else None,
            "median_C": float(np.median(C_pce)) if C_pce.size else None,
        }

    return {
        m: {
            "A": np.concatenate(pops[m]["A"]) if pops[m]["A"] else np.array([]),
            "B": np.concatenate(pops[m]["B"]) if pops[m]["B"] else np.array([]),
            "C": np.concatenate(pops[m]["C"]) if pops[m]["C"] else np.array([]),
            "per_cell": pops[m]["per_cell"],
        }
        for m in ("PCE", "pFF")
    }


def _summarize(A: np.ndarray, B: np.ndarray, C: np.ndarray, label: str) -> dict:
    """Descriptive comparison of the three populations."""
    def q(x, p): return float(np.percentile(x, p)) if x.size else float("nan")
    ks_AB = ks_2samp(A, B) if A.size and B.size else None
    ks_AC = ks_2samp(A, C) if A.size and C.size else None
    ks_BC = ks_2samp(B, C) if B.size and C.size else None
    return {
        "label": label,
        "n": {"A": int(A.size), "B": int(B.size), "C": int(C.size)},
        "median": {"A": q(A, 50), "B": q(B, 50), "C": q(C, 50)},
        "p90":    {"A": q(A, 90), "B": q(B, 90), "C": q(C, 90)},
        "p98":    {"A": q(A, 98), "B": q(B, 98), "C": q(C, 98)},
        "p99":    {"A": q(A, 99), "B": q(B, 99), "C": q(C, 99)},
        # Effect sizes — ratios at the threshold-relevant quantile.
        "ratio_p98_B_over_A": q(B, 98) / q(A, 98) if A.size and B.size else float("nan"),
        "ratio_p98_B_over_C": q(B, 98) / q(C, 98) if B.size and C.size else float("nan"),
        "ratio_p98_C_over_A": q(C, 98) / q(A, 98) if A.size and C.size else float("nan"),
        "ks": {
            "A_vs_B": {"stat": float(ks_AB.statistic), "p": float(ks_AB.pvalue)} if ks_AB else None,
            "A_vs_C": {"stat": float(ks_AC.statistic), "p": float(ks_AC.pvalue)} if ks_AC else None,
            "B_vs_C": {"stat": float(ks_BC.statistic), "p": float(ks_BC.pvalue)} if ks_BC else None,
        },
    }


def _plot_ecdf(pops: Dict[str, np.ndarray], metric: str, out_path: Path) -> None:
    """ECDF of A, B, C on the same axes, log-x to expose tail behavior."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, color, lbl in (
        ("A", "#1f77b4", "A — calibración (madura, entrenamiento)"),
        ("B", "#d62728", "B — aplicación (temprana, holdout)"),
        ("C", "#2ca02c", "C — control (madura, holdout)"),
    ):
        x = pops[name]
        if x.size:
            xs = np.sort(x)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(xs, ys, label=lbl, color=color, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("|residuo| (escala log)")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Distribución de residuos — {metric}")
    ax.axhline(0.98, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(ax.get_xlim()[0], 0.985, " q=0.98", color="gray", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(FILE_MERGED_FEATURES)
    t80 = pd.read_parquet(FILE_T80_TRUTH)

    import joblib
    artifacts = joblib.load(FILE_SCREENING_ARTIFACTS)
    cohort = artifacts["healthy_cohort"]
    logger.info(f"Production cohort: {cohort}")

    pops = _collect_populations(df, t80, cohort)

    summary = {
        "cohort": cohort,
        "burn_in_days": BURN_IN_DAYS,
        "metrics": {
            "PCE": _summarize(pops["PCE"]["A"], pops["PCE"]["B"], pops["PCE"]["C"], "PCE"),
            "pFF": _summarize(pops["pFF"]["A"], pops["pFF"]["B"], pops["pFF"]["C"], "pFF"),
        },
        "per_cell_PCE": pops["PCE"]["per_cell"],
    }

    out_json = OUT_DIR / "phase_alignment_audit.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info(f"Summary written to {out_json}")

    for m in ("PCE", "pFF"):
        _plot_ecdf(pops[m], m, FIG_DIR / f"ecdf_phase_alignment_{m.lower()}.png")

    # Console verdict.
    print("\n" + "=" * 72)
    print(" VERDICT — Phase alignment of the alert threshold")
    print("=" * 72)
    for m in ("PCE", "pFF"):
        s = summary["metrics"][m]
        print(f"\n[{m}]")
        print(f"  n:  A={s['n']['A']:>6}  B={s['n']['B']:>6}  C={s['n']['C']:>6}")
        print(f"  p98: A={s['p98']['A']:.4f}  B={s['p98']['B']:.4f}  C={s['p98']['C']:.4f}")
        print(f"  median: A={s['median']['A']:.4f}  B={s['median']['B']:.4f}  C={s['median']['C']:.4f}")
        print(f"  ratio p98 (B/A) = {s['ratio_p98_B_over_A']:.3f}   "
              f"(B/C) = {s['ratio_p98_B_over_C']:.3f}   (C/A) = {s['ratio_p98_C_over_A']:.3f}")
        if s["ks"]["A_vs_B"]:
            print(f"  KS A~B: D={s['ks']['A_vs_B']['stat']:.3f}, "
                  f"p={s['ks']['A_vs_B']['p']:.2e}")


if __name__ == "__main__":
    main()