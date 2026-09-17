"""
Module: src/02_jv_filtering.py
Description: Physical filtering, artifact removal, and quality control pipeline
for J-V curves (memory-safe rewrite).

Three-layer architecture
------------------------
Layer 1 — Physical invariants : is_night, is_unphysical_point
Layer 2 — Per-curve scores    : v_span_ratio, snr_i, spike_count, hysteresis_index
Layer 3 — Decision            : is_curve_valid (single traceable boolean)

Memory strategy
---------------
Single load per device. No DataFrame.copy() in the hot path. Curve IDs assigned
once at load time. Per-curve validity mapped back via Series.map() instead of
merge(). Intermediate columns dropped as soon as they are consumed.

Revision history
----------------
v3 — Recalibrated against baseline filter's empirically-validated thresholds:
     - Spike detection reverted to absolute count (spike_count > 2), matching
       the original filter's spike_tol. v2's fractional threshold (2% of points)
       was 2.7x more permissive and admitted ~5,800 flat / dead curves.
       Audit of the delta confirmed only 2-20% were real cells.
     - JV_V_SPAN_RATIO_MIN relaxed from 0.25 to 0.10, matching the original
       filter's implicit threshold (50mV absolute / ~500mV median span).
       v2 rejected ~3,200 extra curves that the baseline accepted.

v2 — Corrected three calibration bugs found by audit:
     - spike_score reverted from MAD-of-gradient to dI/span.
       The MAD approach failed: J-V gradients vary systematically (flat near
       Isc, steep near Voc), so global MAD was tiny and the codo counted as
       a spike. Rejected 76% of curves in v1.
     - rej_unphysical now uses ratio > 0.90 instead of "any point".
       Rejected 9% of curves in v1 due to a single spurious negative reading.
     - JV_V_SPAN_RATIO_MIN relaxed from 0.5 to 0.25.

Guiding principle
-----------------
The baseline filter is rustic but has been empirically validated by two
independent audits (FP rate 0-1%, FN rate 0.11%). Any redesign must reproduce
the same output BEFORE introducing improvements. Changes that alter the
output require positive evidence they are better — not the reverse.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

from src.config import (
    OPERATIONAL_HOUR_END,
    OPERATIONAL_HOUR_START,
    CELL_AREA_M2,
    DIR_PROCESSED,
    FILE_JV_FILTERED,
    FILE_FILTERING_META,
    DEPLOYMENT_TIMEZONE,
    JV_SNR_MIN,
    JV_V_SPAN_RATIO_MIN,
    JV_SPIKE_RELATIVE_THRESHOLD,
    JV_SPIKE_MAX_POINTS,
    JV_UNPHYSICAL_RATIO_TOL,
    JV_FALLBACK_NOISE_FLOOR_I,
    JV_VOC_MIN_V,
    JV_JSC_MIN_MA_CM2,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Filtering")

CELL_AREA_CM2 = float(CELL_AREA_M2 * 10_000.0)

# ---------------------------------------------------------------------------
# Layer 2 / 3 thresholds — calibrated against validated baseline
# ---------------------------------------------------------------------------
# Signal-to-noise minimum: signal must exceed 3x the instrument noise floor.
JV_SNR_MIN = 3.0

# Voltage-span ratio minimum: curve sweep must be at least 10% of the
# population median for the same cell.
# Calibrated to match the original filter's v_span_thresh = 50 mV absolute
# (median span ~500 mV across devices → ratio ~0.10).
JV_V_SPAN_RATIO_MIN = 0.10

# Spike detection: point gradient as a fraction of the curve's current span.
# A point is a spike if |dI| exceeds this fraction of the total span.
JV_SPIKE_RELATIVE_THRESHOLD = 0.15

# Absolute count of spike points allowed per curve before rejection.
# Matches the original filter's spike_tol = 2. Empirically validated: the
# FP/FN audit showed this threshold catches broken traces without rejecting
# healthy perovskite curves with legitimate S-shape inflections.
JV_SPIKE_MAX_POINTS = 2

# Unphysical point ratio threshold. A curve is rejected only if more than
# 90% of its points violate I<0 & V<0.5. Isolated violations are removed
# at the point level (see cleanup block below).
JV_UNPHYSICAL_RATIO_TOL = 0.90

# Fallback noise floor when no night curves are available for estimation.
JV_FALLBACK_NOISE_FLOOR_I = 1e-7  # Amperes


# ---------------------------------------------------------------------------
# Optional memory reporter
# ---------------------------------------------------------------------------
try:
    import psutil

    _PROC = psutil.Process(os.getpid())

    def _mem_mb() -> float:
        return _PROC.memory_info().rss / 1024 ** 2

    def _log_mem(tag: str) -> None:
        logger.info(f"[MEM] {tag}: {_mem_mb():.0f} MB")

except ImportError:
    def _log_mem(tag: str) -> None:  # noqa: D401
        return None


# ==============================================================================
# 0. DEVICE LOADER (single-pass preparation)
# ==============================================================================
def _load_device(parquet_path: Path) -> pd.DataFrame:
    """
    Load one device's raw J-V parquet and prepare it in place.

    Steps (all in-place or single allocation):
      - read parquet
      - parse Timestamp as UTC
      - sort by Timestamp
      - downcast numerics to float32
      - categorical ScanDirection
      - assign 'curve' identifier (Reverse->start transition)
    """
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    df.sort_values("Timestamp", inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)

    for col in ("Voltage_V", "Current_A", "Power_mW"):
        if col in df.columns:
            df[col] = df[col].astype("float32")

    if df["ScanDirection"].dtype == object:
        df["ScanDirection"] = df["ScanDirection"].astype("category")

    is_reverse_start = (df["ScanDirection"] == "Reverse") & (
        df["ScanDirection"].shift(1) != "Reverse"
    )
    is_reverse_start.iloc[0] = True
    df["curve"] = is_reverse_start.cumsum().astype("int32")

    return df


# ==============================================================================
# 1. NOISE FLOOR ESTIMATION (no full-frame copy)
# ==============================================================================
def _estimate_noise_floor_i(df: pd.DataFrame) -> float:
    """
    Estimate instrument current noise floor from night-time curves.
    Operates on column subsets only; never copies the full DataFrame.
    """
    local_hours = df["Timestamp"].dt.tz_convert(DEPLOYMENT_TIMEZONE).dt.hour
    night_mask = ~local_hours.between(OPERATIONAL_HOUR_START, OPERATIONAL_HOUR_END)
    n_night = int(night_mask.sum())

    if n_night < 500:
        logger.warning(
            "Only %d night points; using fallback noise floor %.2e A.",
            n_night, JV_FALLBACK_NOISE_FLOOR_I,
        )
        return JV_FALLBACK_NOISE_FLOOR_I

    night = pd.DataFrame({
        "curve": df.loc[night_mask, "curve"].to_numpy(),
        "I": df.loc[night_mask, "Current_A"].to_numpy(),
    })
    gradient = night.groupby("curve", sort=False)["I"].diff().abs().dropna()
    floor = float(gradient.median())
    del night, gradient
    gc.collect()

    if not np.isfinite(floor) or floor <= 0:
        logger.warning("Noise floor invalid (%.3e); using fallback.", floor)
        return JV_FALLBACK_NOISE_FLOOR_I

    logger.info("Instrument current noise floor estimated: %.3e A", floor)
    return floor


# ==============================================================================
# 2. PER-DEVICE FILTERING TOPOLOGY (in-place, no copies)
# ==============================================================================
def _process_single_cell(
    df: pd.DataFrame,
    name: str,
    cell_id: int,
    noise_floor_i: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the three-layer filter to one device's DataFrame.

    The input df is mutated in place and returned filtered. The caller must
    not reuse the original reference after this call.
    """
    # --- METADATA ------------------------------------------------------
    df["cell_name"] = name
    df["cell_id"] = cell_id
    df["is_reverse"] = (df["ScanDirection"] == "Reverse").astype("int8")

    # --- LAYER 1: INVARIANTS -------------------------------------------
    local_hours = df["Timestamp"].dt.tz_convert(DEPLOYMENT_TIMEZONE).dt.hour
    df["is_night"] = ~local_hours.between(
        OPERATIONAL_HOUR_START, OPERATIONAL_HOUR_END
    )
    df["is_unphysical_point"] = (df["Current_A"] < 0) & (df["Voltage_V"] < 0.5)

    # --- LAYER 2: PER-CURVE SCORES -------------------------------------
    # Point-level gradient magnitude (within each curve).
    df["_dI_abs"] = (
        df.groupby("curve", sort=False)["Current_A"]
        .diff()
        .abs()
        .astype("float32")
    )

    # Consolidated curve-level aggregation.
    curve_stats = df.groupby("curve", sort=False).agg(
        v_span=("Voltage_V", lambda x: float(x.max() - x.min())),
        i_span=("Current_A", lambda x: float(x.max() - x.min())),
        n_points=("Voltage_V", "size"),
        is_night=("is_night", "all"),
        neg_ratio=("is_unphysical_point", "mean"),
    )

    # Voltage-span ratio: relative to the cell's own population.
    v_span_median = float(curve_stats["v_span"].median())
    if v_span_median <= 0:
        v_span_median = 1.0
    curve_stats["v_span_ratio"] = curve_stats["v_span"] / v_span_median

    # Signal-to-noise ratio on current.
    curve_stats["snr_i"] = curve_stats["i_span"] / (noise_floor_i + 1e-15)

    # Spike detection: |dI| / i_span > JV_SPIKE_RELATIVE_THRESHOLD, counted as
    # an absolute number of points per curve.
    df["_curve_i_span"] = (
        df["curve"].map(curve_stats["i_span"]).astype("float32")
    )
    df["_is_spike_point"] = (
        df["_dI_abs"] / (df["_curve_i_span"] + 1e-12)
    ) > JV_SPIKE_RELATIVE_THRESHOLD

    spike_agg = df.groupby("curve", sort=False)["_is_spike_point"].agg(
        spike_count="sum",
        spike_score="mean",
    )
    curve_stats["spike_count"] = spike_agg["spike_count"]
    curve_stats["spike_score"] = spike_agg["spike_score"]
    del spike_agg
    gc.collect()

    # Hysteresis index (FEATURE, not gate): |PCE_rev - PCE_fwd| / PCE_rev.
    df["_power"] = (df["Voltage_V"] * df["Current_A"]).astype("float32")
    mpp = (
        df.groupby(["curve", "is_reverse"], sort=False)["_power"]
        .max()
        .unstack("is_reverse")
    )
    pce_rev = mpp.get(1, pd.Series(np.nan, index=mpp.index))
    pce_fwd = mpp.get(0, pd.Series(np.nan, index=mpp.index))
    denom = pce_rev.abs().clip(lower=1e-12)
    curve_stats["hysteresis_index"] = (
        (pce_rev - pce_fwd).abs() / denom
    ).reindex(curve_stats.index)
    del mpp, pce_rev, pce_fwd, denom
    df.drop(columns=["_power"], inplace=True)

    # --- LAYER 3: DECISION ---------------------------------------------
    curve_stats["rej_night"] = curve_stats["is_night"]
    curve_stats["rej_unphysical"] = curve_stats["neg_ratio"] > JV_UNPHYSICAL_RATIO_TOL
    curve_stats["rej_vspan_low"] = curve_stats["v_span_ratio"] < JV_V_SPAN_RATIO_MIN
    curve_stats["rej_snr_low"] = curve_stats["snr_i"] < JV_SNR_MIN
    curve_stats["rej_spike"] = curve_stats["spike_count"] > JV_SPIKE_MAX_POINTS

    curve_stats["is_curve_valid"] = (
        ~curve_stats["rej_night"]
        & ~curve_stats["rej_unphysical"]
        & ~curve_stats["rej_vspan_low"]
        & ~curve_stats["rej_snr_low"]
        & ~curve_stats["rej_spike"]
    ).astype("int8")

    # --- MAP BACK (no merge) -------------------------------------------
    df["is_curve_valid"] = (
        df["curve"].map(curve_stats["is_curve_valid"]).astype("int8")
    )

    # --- AUDIT TABLE (small) -------------------------------------------
    audit = curve_stats.reset_index()[
        [
            "curve", "n_points", "v_span", "i_span",
            "v_span_ratio", "snr_i", "spike_count", "spike_score",
            "hysteresis_index", "neg_ratio",
            "rej_night", "rej_unphysical", "rej_vspan_low", "rej_snr_low", "rej_spike",
            "is_curve_valid",
        ]
    ].copy()
    audit["cell_name"] = name
    audit["cell_id"] = cell_id
    del curve_stats
    gc.collect()

    # --- POINT-LEVEL CLEANUP -------------------------------------------
    # Drop isolated unphysical points from otherwise-valid curves.
    drop_mask = df["is_unphysical_point"] & (df["is_curve_valid"] == 1)
    if drop_mask.any():
        df.drop(index=df.index[drop_mask], inplace=True)
    del drop_mask
    gc.collect()

    # --- DROP TEMP COLUMNS ---------------------------------------------
    df.drop(
        columns=[
            "is_night", "is_unphysical_point",
            "_dI_abs", "_curve_i_span", "_is_spike_point",
        ],
        inplace=True,
        errors="ignore",
    )
    gc.collect()

    return df, audit


# ==============================================================================
# 3. PHYSICS PARAMETER EXTRACTION
# ==============================================================================
def _interp_zero_crossing(x: np.ndarray, y: np.ndarray, positive_x_only: bool = False) -> float:
    sign_y = np.sign(y)
    sign_changes = np.where(np.diff(sign_y) != 0)[0]
    if positive_x_only:
        sign_changes = sign_changes[x[sign_changes] > 0.1]
    if len(sign_changes) > 0:
        i = sign_changes[0]
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y1 - y0 != 0:
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return float(x[np.argmin(np.abs(y))])


def extract_physics_parameters(df_curve: pd.DataFrame) -> dict:
    df_curve = df_curve.sort_values(by="Voltage_V")
    V = df_curve["Voltage_V"].to_numpy(dtype=float)
    I_A = df_curve["Current_A"].to_numpy(dtype=float)
    J = (I_A * 1000.0) / CELL_AREA_CM2
    P = V * J

    results = {
        "id_curve": int(df_curve["id_curve"].iloc[0]),
        "voc": np.nan, "jsc": np.nan, "ff": np.nan,
        "v_mpp": np.nan, "j_mpp": np.nan, "p_mpp": np.nan,
        "err_mpp": 0, "err_voc": 0, "err_jsc": 0, "err_ff": 0,
    }

    if len(V) < 10:
        results.update({"err_mpp": 1, "err_voc": 1, "err_jsc": 1, "err_ff": 1})
        return results

    try:
        window = int(min(11, len(V)))
        if window % 2 == 0:
            window -= 1
        P_smooth = (
            np.asarray(savgol_filter(P, window_length=window, polyorder=2))
            if window > 3 else P
        )
        mpp_idx = int(np.argmax(P_smooth))
        results["p_mpp"] = float(P_smooth[mpp_idx])
        results["v_mpp"] = float(V[mpp_idx])
        results["j_mpp"] = float(J[mpp_idx])
        if results["p_mpp"] <= 0 or results["v_mpp"] <= 0 or results["j_mpp"] <= 0:
            results["err_mpp"] = 1
    except Exception:
        results["err_mpp"] = 1

    try:
        nearest = np.argsort(np.abs(J))[:5]
        mean_v_near = float(np.mean(V[nearest]))
        results["voc"] = _interp_zero_crossing(V, J, positive_x_only=True)
        if mean_v_near != 0 and (
            results["voc"] / mean_v_near >= 1.5 or results["voc"] <= 0
        ):
            results["voc"] = mean_v_near
    except Exception:
        results["err_voc"] = 1

    try:
        nearest = np.argsort(np.abs(V))[:5]
        mean_j_near = float(np.mean(J[nearest]))
        if V.min() <= 0 <= V.max() and V.min() < V.max():
            results["jsc"] = _interp_zero_crossing(V, J)
        else:
            results["jsc"] = float(J[np.argmin(np.abs(V))])
        if mean_j_near != 0 and (
            results["jsc"] / mean_j_near >= 1.5 or results["jsc"] <= 0
        ):
            results["jsc"] = mean_j_near
    except Exception:
        results["err_jsc"] = 1

    v_mpp, j_mpp = results["v_mpp"], results["j_mpp"]
    voc, jsc = results["voc"], results["jsc"]
    if all(pd.notna(x) for x in (v_mpp, j_mpp, voc, jsc)) and voc > 0 and jsc > 0:
        results["ff"] = float((v_mpp * j_mpp) / (voc * jsc))
        if not (0.0 < results["ff"] < 1.0):
            results["ff"] = np.nan
            results["err_ff"] = 1
    else:
        results["err_ff"] = 1

    return results


def extract_all_physics_parameters(
    jv_dataset: pd.DataFrame, batch_size: int = 500
) -> pd.DataFrame:
    logger.info("Extracting physical parameters in memory-safe batches...")
    gc.collect()

    valid_df = jv_dataset[jv_dataset["is_curve_valid"] == 1]
    total_curves = int(valid_df["id_curve"].nunique())
    logger.info(f"Total valid curves to process: {total_curves:,}")

    if total_curves == 0:
        return pd.DataFrame()

    extracted: list[dict] = []
    grouped = valid_df.groupby("id_curve", sort=False)

    for i, (_, group) in enumerate(
        tqdm(grouped, total=total_curves, desc="Extracting Physics")
    ):
        extracted.append(extract_physics_parameters(group))
        if (i + 1) % batch_size == 0:
            gc.collect()

    return pd.DataFrame(extracted)


# ==============================================================================
# 4. CHRONOMETRIC DIAGNOSTICS
# ==============================================================================
def analyze_curve_timings(jv_df: pd.DataFrame) -> None:
    logger.info("Computing chronometric metadata...")

    all_curves_stats = (
        jv_df.groupby(["cell_name", "id_curve"])
        .agg(
            t_min=("Timestamp", "min"),
            t_max=("Timestamp", "max"),
            is_valid=("is_curve_valid", "max"),
        )
        .sort_values("t_min")
    )

    global_durations: list[float] = []
    global_intervals: list[float] = []
    report_lines = ["\n=== TEMPORAL ANALYSIS PER DEVICE ==="]

    for cell in jv_df["cell_name"].unique():
        cell_stats = all_curves_stats.loc[cell]
        intervals = cell_stats["t_min"].diff().dt.total_seconds()
        intervals_clean = intervals[intervals < 3600]
        avg_interval = intervals_clean.mean()
        global_intervals.extend(intervals_clean.dropna().tolist())

        valid_stats = cell_stats[cell_stats["is_valid"] == 1]
        durations = (valid_stats["t_max"] - valid_stats["t_min"]).dt.total_seconds()
        avg_duration = durations.mean()
        global_durations.extend(durations.dropna().tolist())
        total_valid = len(valid_stats)

        valid_df = jv_df[(jv_df["cell_name"] == cell) & (jv_df["is_curve_valid"] == 1)]
        if not valid_df.empty:
            local_ts = valid_df["Timestamp"].dt.tz_convert(DEPLOYMENT_TIMEZONE)
            valid_local = valid_df.assign(_local=local_ts).set_index("_local")
            max_daily = int(valid_local.resample("D")["id_curve"].nunique().max())
        else:
            max_daily = 0

        report_lines.extend(
            [
                f"[{cell}]",
                f"  -> Valid cycles retained: {total_valid}",
                f"  -> Mean duration (Rev+Fwd): {avg_duration:.2f} s"
                if total_valid > 0 else "  -> Mean duration: N/A",
                f"  -> Mean idle interval: {avg_interval:.2f} s"
                if not np.isnan(avg_interval) else "  -> Mean idle interval: N/A",
                f"  -> Peak daily cycle volume: {max_daily}\n",
            ]
        )

    report_lines.extend(
        [
            "=" * 40,
            "=== AGGREGATED GLOBAL TEMPORAL METRICS ===",
            f"Mean Global Sweep Duration: {np.mean(global_durations):.2f} s"
            if global_durations else "Mean Global Sweep Duration: N/A",
            f"Mean Global Hardware Idle:  {np.mean(global_intervals):.2f} s"
            if global_intervals else "Mean Global Hardware Idle: N/A",
            "=" * 40,
        ]
    )
    logger.info("\n".join(report_lines))


# ==============================================================================
# 5. ORCHESTRATION
# ==============================================================================
def _load_and_filter_devices(
    device_dirs: list[str],
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    all_points: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    noise_floor_i: Optional[float] = None

    _log_mem("start")

    for cell_id, name in enumerate(device_dirs):
        parquet_path = DIR_PROCESSED / name / f"{name}_jv.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet missing for device: {name}")
            continue

        logger.info(f"Loading device: {name}")
        df = _load_device(parquet_path)
        if df.empty:
            logger.warning(f"Empty parquet for device {name}")
            continue

        _log_mem(f"{name} loaded")

        if noise_floor_i is None:
            noise_floor_i = _estimate_noise_floor_i(df)
            _log_mem(f"{name} noise floor done")

        logger.info(f"Filtering device: {name}")
        filtered, audit = _process_single_cell(df, name, cell_id, noise_floor_i)

        del df
        gc.collect()
        _log_mem(f"{name} filtered")

        all_points.append(filtered)
        all_audits.append(audit)

        total = len(audit)
        accepted = int(audit["is_curve_valid"].sum())
        logger.info(
            f"  [{name}] {total:,} curves -> {accepted:,} valid "
            f"({100.0 * accepted / max(total, 1):.1f}%)"
        )

    if not all_points:
        return None, None

    logger.info("Concatenating filtered device chunks...")
    _log_mem("pre-concat")
    consolidated_points = pd.concat(all_points, axis=0, ignore_index=True)
    consolidated_audit = pd.concat(all_audits, axis=0, ignore_index=True)
    del all_points, all_audits
    gc.collect()
    _log_mem("post-concat")

    return consolidated_points, consolidated_audit


def _log_quality_control_report(df: pd.DataFrame, audit: pd.DataFrame) -> None:
    logger.info("Aggregating global diagnostics report...")
    try:
        per_cell = audit.groupby("cell_name").agg(
            total=("is_curve_valid", "count"),
            valid=("is_curve_valid", "sum"),
            rej_night=("rej_night", "sum"),
            rej_unphysical=("rej_unphysical", "sum"),
            rej_vspan_low=("rej_vspan_low", "sum"),
            rej_snr_low=("rej_snr_low", "sum"),
            rej_spike=("rej_spike", "sum"),
        )
        per_cell["yield_pct"] = (
            100.0 * per_cell["valid"] / per_cell["total"]
        ).round(1)
        logger.info(
            f"\n=== QUALITY CONTROL METRICS ===\n"
            f"Total curves evaluated: {len(audit):,}\n"
            f"Total valid retained  : {int(audit['is_curve_valid'].sum()):,}\n\n"
            f"Per-device yield:\n{per_cell.to_string()}"
        )
    except Exception as e:
        logger.warning(f"Diagnostics aggregation failed: {e}")

    logger.info("\n--- PIPELINE RETENTION AUDIT ---")
    logger.info(
        f"Gross Cycle Count per direction:\n"
        f"{df.groupby('is_reverse')['id_curve'].nunique().to_string()}"
    )
    logger.info(
        f"Net Valid Cycle Count (post-filter):\n"
        f"{df[df['is_curve_valid'] == 1].groupby('is_reverse')['id_curve'].nunique().to_string()}"
    )


def main() -> None:
    FILE_JV_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    FILE_FILTERING_META.parent.mkdir(parents=True, exist_ok=True)

    if not DIR_PROCESSED.exists():
        logger.error(f"Target directory missing: {DIR_PROCESSED}.")
        raise FileNotFoundError(f"Directory {DIR_PROCESSED} not found.")

    device_dirs = sorted(d.name for d in DIR_PROCESSED.iterdir() if d.is_dir())
    logger.info(f"Discovered processed datasets: {device_dirs}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        df_filtered, audit = _load_and_filter_devices(device_dirs)
        if df_filtered is None or audit is None:
            logger.error("No valid artifacts loaded.")
            return

        df_filtered["id_curve"] = df_filtered.groupby(
            ["cell_name", "curve"]
        ).ngroup()

        audit_path = FILE_JV_FILTERED.parent / "02_filtering_audit.parquet"
        audit.to_parquet(
            audit_path, engine="pyarrow", compression="snappy", index=False
        )
        logger.info(f"Per-curve audit table serialized to: {audit_path}")

        _log_quality_control_report(df_filtered, audit)
        analyze_curve_timings(df_filtered)

        valid_only_df = df_filtered[df_filtered["is_curve_valid"] == 1]
        valid_counts_dict = (
            valid_only_df.groupby("cell_name")["id_curve"].nunique().to_dict()
        )
        with open(FILE_FILTERING_META, "w") as f:
            json.dump(valid_counts_dict, f, indent=4)
        logger.info(f"Metadata serialized to: {FILE_FILTERING_META}")

        logger.info("Extracting physics parameters for valid curves...")
        df_physics = extract_all_physics_parameters(valid_only_df)

        if df_physics.empty:
            logger.warning("No physics parameters extracted. Skipping merge.")
        else:
            float_cols = df_physics.select_dtypes(include=["float64", "float"]).columns
            df_physics[float_cols] = df_physics[float_cols].astype("float32")
            int_cols = df_physics.select_dtypes(include=["int64", "int"]).columns
            df_physics[int_cols] = df_physics[int_cols].astype("int32")

            del valid_only_df
            gc.collect()

            logger.info("Merging physics parameters into the main dataset...")
            df_filtered = df_filtered.merge(df_physics, on="id_curve", how="left")
            del df_physics
            gc.collect()

        # --- PHYSICS GATE: reject dead cells (Voc≈0, Jsc≈0) ---
        logger.info("Applying physics gate (Voc/Jsc magnitude)...")

        valid_mask = df_filtered["is_curve_valid"] == 1
        valid_curves = df_filtered[valid_mask]
        n_valid_before = int(valid_curves["id_curve"].nunique())

        curve_physics = valid_curves.groupby(
            ["cell_name", "cell_id", "curve"], sort=False
        ).agg(voc=("voc", "first"), jsc=("jsc", "first"))

        curve_physics["rej_dead_cell"] = (
            (curve_physics["voc"].fillna(0) < JV_VOC_MIN_V)
            | (curve_physics["jsc"].fillna(0) < JV_JSC_MIN_MA_CM2)
        )

        n_dead = int(curve_physics["rej_dead_cell"].sum())
        logger.info(
            f"  → Physics gate rejected {n_dead:,} of {n_valid_before:,} "
            f"valid curves ({100.0 * n_dead / max(n_valid_before, 1):.1f}%)"
        )

        # --- Memory-safe mask via numeric encoding ---
        # (cell_id, curve) is a unique key. Encode as a single int64:
        #   key = cell_id * 2**32 + curve
        # This avoids building 28.7M Python strings, which was the OOM cause.
        _K = 2**32
        dead_df = curve_physics[curve_physics["rej_dead_cell"]].reset_index()

        dead_keys = (
            dead_df["cell_id"].to_numpy(dtype=np.int64) * _K
            + dead_df["curve"].to_numpy(dtype=np.int64)
        )
        df_keys = (
            df_filtered["cell_id"].to_numpy(dtype=np.int64) * _K
            + df_filtered["curve"].to_numpy(dtype=np.int64)
        )

        # Hash-based isin — O(N), peak extra memory ~ 500 MB for 28.7M rows.
        mask = pd.Series(df_keys).isin(pd.Series(dead_keys)).to_numpy()
        del df_keys
        gc.collect()

        df_filtered["rej_dead_cell"] = mask
        df_filtered["is_curve_valid"] = np.where(
            mask, 0, df_filtered["is_curve_valid"].to_numpy(dtype=np.int8)
        ).astype(np.int8)
        del mask
        gc.collect()

        # --- Update the audit table (small: 104k rows) ---
        audit_keys = (
            audit["cell_id"].to_numpy(dtype=np.int64) * _K
            + audit["curve"].to_numpy(dtype=np.int64)
        )
        audit_mask = pd.Series(audit_keys).isin(pd.Series(dead_keys)).to_numpy()
        audit["rej_dead_cell"] = audit_mask
        audit["is_curve_valid"] = np.where(
            audit_mask, 0, audit["is_curve_valid"].to_numpy(dtype=np.int8)
        ).astype(np.int8)
        del audit_keys, audit_mask, dead_keys
        gc.collect()

        audit.to_parquet(
            audit_path, engine="pyarrow", compression="snappy", index=False
        )
        logger.info(
            f"Audit table updated: {int(audit['is_curve_valid'].sum()):,} valid curves"
        )

        logger.info(f"Serializing filtered dataset to {FILE_JV_FILTERED}...")
        df_filtered.to_parquet(
            FILE_JV_FILTERED, engine="pyarrow", compression="snappy", index=False
        )
        logger.info("Filtering pipeline successfully terminated.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Execution aborted due to an unhandled exception.")
        raise