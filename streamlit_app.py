"""
ParaSol Dashboard
==================
Meteorological and Photovoltaic Performance Monitoring Panel
with Digital Twin Integration (Early Failure Detection).
"""
import datetime
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# PAGE CONFIGURATION (Must be the first Streamlit command)
# =====================================================================
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="ParaSol Dashboard"
)

LOGO_URL = "https://www.emiliojuarez.es/imgs/logo-oss.jpg"
CREATOR_LINK = "https://linkedin.com/in/cristian-carretero-fernandez"

# Absolute paths to data
FLEET_DATA_PATH = Path("data/aggregated/outdoor/meteo_mppt_10min.parquet")
SURVIVAL_DATA_PATH = Path("data/survival/outdoor/survival_dataset.parquet")
ARTIFACTS_PATH = Path("data/anomaly/artifacts/early_failure_artifacts.joblib")

# Rutas absolutas a los 3 parquets del Gemelo Digital
ANOMALY_DIR = Path("/home/cristian/Documentos/Proyectos/parasol-rag-architecture/data/anomaly/outdoor")
TWIN_DIAGNOSTICS_PATH = ANOMALY_DIR / "anomaly_scored_dataset.parquet"
PCE_NORM_PATH = ANOMALY_DIR / "pce_norm_predictions.parquet"
PFF_PRED_PATH = ANOMALY_DIR / "pff_predictions.parquet"

# Physical constants
CELL_AREA_M2 = 0.64 / 10000.0  # 0.64 cm² converted to m²

# PARCHE 1: umbral unificado con data_aggregation.py / anomaly_detection.py.
# Antes: > 10 W/m^2 en 3 puntos de este archivo (inestabilidad numerica del denominador
# en la division de PCE con irradiancia baja). Los artefactos de supervivencia ya usan 100.0
# como default; ahora las 3 ocurrencias sueltas quedan consistentes con ese mismo valor.
# TODO (Parche 6): mover a src/common.py junto con CELL_AREA_M2 y el resto de constantes
# compartidas, e importarlo en vez de redefinirlo aqui.
PCE_IRRADIANCE_MIN_W_M2 = 100.0
MAX_NATIVE_POINTS = 3000


def get_resolution_rules(duration_days: float, label_stat: str) -> list[tuple[str, str]]:
    """Return the available server-side resolutions for a selected time window."""
    if duration_days <= 1.0:
        return [("native", "Native (10 min)")]
    if duration_days <= 7.0:
        return [
            ("native", "Native (10 min)"),
            ("1h", f"{label_stat} Hourly (1H)"),
            ("1D", f"{label_stat} Daily (1D)"),
        ]
    if duration_days <= 30.0:
        return [
            ("1h", f"{label_stat} Hourly (1H)"),
            ("6h", f"{label_stat} 6-Hourly (6H)"),
            ("1D", f"{label_stat} Daily (1D)"),
            ("3D", f"{label_stat} 3-Day (3D)"),
            ("7D", f"{label_stat} Weekly (7D)"),
        ]
    return [
        ("1D", f"{label_stat} Daily (1D)"),
        ("3D", f"{label_stat} 3-Day (3D)"),
        ("7D", f"{label_stat} Weekly (7D)"),
        ("30D", f"{label_stat} Monthly (30D)"),
    ]

# =====================================================================
# STYLES
# =====================================================================
def load_styles() -> None:
    """Loads external CSS and injects it into the application."""
    try:
        with open("assets/style.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# =====================================================================
# DATA LOADING AND CACHING
# =====================================================================
@st.cache_data(show_spinner=False)
def load_global_data() -> pd.DataFrame:
    try:
        df = pd.read_parquet(FLEET_DATA_PATH)
        if df.index.name == "Timestamp":
            df = df.reset_index()
        df["Timestamp"] = (
            pd.to_datetime(df["Timestamp"], utc=True)
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        df.set_index("Timestamp", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading the fleet dataset: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_survival_data() -> pd.DataFrame:
    try:
        df = pd.read_parquet(SURVIVAL_DATA_PATH)
        df["Timestamp"] = (
            pd.to_datetime(df["Timestamp"], utc=True)
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        df.set_index("Timestamp", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading the survival dataset: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_twin_diagnostics() -> pd.DataFrame:
    """Loads the enriched observations used by the Digital Twin charts."""
    try:
        df = pd.read_parquet(TWIN_DIAGNOSTICS_PATH)
        df["Timestamp"] = (
            pd.to_datetime(df["Timestamp"], utc=True)
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
        )
        df.set_index("Timestamp", inplace=True)
        return df
    except Exception as e:
        st.warning(f"Failed to load Digital Twin diagnostics dataset: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_pce_norm_data() -> pd.DataFrame:
    """Loads the isolated PCE Normalization predictions."""
    try:
        df = pd.read_parquet(PCE_NORM_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
        df.set_index("Timestamp", inplace=True)
        return df
    except Exception as e:
        st.warning(f"Failed to load PCE Norm dataset: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_pff_pred_data() -> pd.DataFrame:
    """Loads the isolated pFF structural predictions."""
    try:
        df = pd.read_parquet(PFF_PRED_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
        df.set_index("Timestamp", inplace=True)
        return df
    except Exception as e:
        st.warning(f"Failed to load pFF dataset: {e}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_ml_artifacts() -> dict:
    """Loads the Digital Twin brain generated by XGBoost."""
    try:
        if ARTIFACTS_PATH.exists():
            return joblib.load(ARTIFACTS_PATH)
    except Exception as e:
        st.warning(f"Failed to load Digital Twin artifacts: {e}")
    return {}


@st.cache_data(show_spinner=False)
def load_xai_rules() -> dict:
    """Loads deterministic surrogate-tree rules exported by early screening."""
    xai_path = ARTIFACTS_PATH.parent.parent / "diagnostics" / "xai_surrogate_rules.json"
    try:
        if xai_path.exists():
            with xai_path.open(encoding="utf-8") as file:
                rules = json.load(file)
            return rules if isinstance(rules, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Failed to load XAI surrogate rules: {exc}")
    return {}


def get_device_options(ml_artifacts: dict, survival_df: pd.DataFrame) -> list[str]:
    """Returns the fleet devices available in the loaded diagnostic datasets."""
    devices = set()
    summary_table = ml_artifacts.get("summary_table", pd.DataFrame())
    if isinstance(summary_table, pd.DataFrame):
        devices.update(str(device) for device in summary_table.index if pd.notna(device))
    if not survival_df.empty and "cell_name" in survival_df.columns:
        devices.update(str(device) for device in survival_df["cell_name"].dropna().unique())
    return sorted(devices)

# =====================================================================
# ADVANCED PLOTLY CHART GENERATION
# =====================================================================
def create_plotly_chart(df, y_cols, color_seq=None, central_metric="Mean", band_min_col=None, band_max_col=None, resolution_rules=None):
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    if color_seq is None:
        color_seq = ["#36B9CC", "#1E293B", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6", "#F43F5E", "#64748B"]

    agg_func = "median" if central_metric == "Median" else "mean"
    label_stat = "Median" if central_metric == "Median" else "Mean"
    
    def hex_to_rgba(hex_color, alpha=0.25):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
        
    has_bands = band_min_col and band_max_col and band_min_col in df.columns and band_max_col in df.columns
    min_dt = df.index.min()
    max_dt = df.index.max()
    duration_days = (max_dt - min_dt).total_seconds() / 86400.0
    resolution_rules = resolution_rules or get_resolution_rules(duration_days, label_stat)
    if len(df) > MAX_NATIVE_POINTS and resolution_rules[0][0] == "native":
        resolution_rules = get_resolution_rules(7.0001, label_stat)
        duration_days = max(duration_days, 7.0001)

    rule_freqs = [freq for freq, _ in resolution_rules]
    if rule_freqs == ["native"]:
        for i, col in enumerate(y_cols):
            c = color_seq[i % len(color_seq)]
            name = str(col)
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=name, legendgroup=name, line=dict(color=c, width=1.5), marker=dict(size=4), visible=True, connectgaps=True))
        updatemenus_config = []

    elif rule_freqs == ["native", "1h", "1D"]:
        df_1h_main = df.resample("1h").agg(agg_func, numeric_only=True)
        df_1h_min = df.resample("1h").min(numeric_only=True)
        df_1h_max = df.resample("1h").max(numeric_only=True)
        df_1d_main = df.resample("1D").agg(agg_func, numeric_only=True)
        df_1d_min = df.resample("1D").min(numeric_only=True)
        df_1d_max = df.resample("1D").max(numeric_only=True)
        for i, col in enumerate(y_cols):
            c = color_seq[i % len(color_seq)]
            c_fill = hex_to_rgba(c, 0.25)
            name = str(col)
            use_band = has_bands and col in ["ModuleTemp_Mean_C", "ModuleTemp_Median_C"]
            b_min = band_min_col if use_band else col
            b_max = band_max_col if use_band else col

            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=name, legendgroup=name, line=dict(color=c, width=1.5), marker=dict(size=4), visible=True, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1h_min.index, y=df_1h_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1h_max.index, y=df_1h_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1h_main.index, y=df_1h_main[col], mode="lines+markers", name=f"{name} ({label_stat} 1H)", legendgroup=name, line=dict(color=c, width=2.0), marker=dict(size=5), visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1d_min.index, y=df_1d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1d_max.index, y=df_1d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1d_main.index, y=df_1d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 1D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=6), visible=False, connectgaps=True))

        n = len(y_cols)
        vis_orig = ([True, False, False, False, False, False, False]) * n
        vis_1h   = ([False, True, True, True, False, False, False]) * n
        vis_1d   = ([False, False, False, False, True, True, True]) * n

        trace_indices = list(range(7 * n))

        buttons_agg = [
            dict(label="Native (10 min)", method="restyle", args=[{"visible": vis_orig}, trace_indices]),
            dict(label=f"{label_stat} Hourly (1H)", method="restyle", args=[{"visible": vis_1h}, trace_indices]),
            dict(label=f"{label_stat} Daily (1D)", method="restyle", args=[{"visible": vis_1d}, trace_indices]),
        ]
        updatemenus_config = [
            dict(type="buttons", direction="right", buttons=buttons_agg, active=0, showactive=True,
                 x=1, xanchor="right", y=-0.15, yanchor="top", font=dict(size=11, color="#1E293B"),
                 bgcolor="#F8FAFC", bordercolor="#E2E8F0")
        ]

    elif rule_freqs == ["1h", "6h", "1D", "3D", "7D"]:
        df_1h_main = df.resample("1h").agg(agg_func, numeric_only=True)
        df_1h_min = df.resample("1h").min(numeric_only=True)
        df_1h_max = df.resample("1h").max(numeric_only=True)
        
        df_6h_main = df.resample("6h").agg(agg_func, numeric_only=True)
        df_6h_min = df.resample("6h").min(numeric_only=True)
        df_6h_max = df.resample("6h").max(numeric_only=True)
        
        df_1d_main = df.resample("1D").agg(agg_func, numeric_only=True)
        df_1d_min = df.resample("1D").min(numeric_only=True)
        df_1d_max = df.resample("1D").max(numeric_only=True)

        df_3d_main = df.resample("3D").agg(agg_func, numeric_only=True)
        df_3d_min = df.resample("3D").min(numeric_only=True)
        df_3d_max = df.resample("3D").max(numeric_only=True)
        
        df_7d_main = df.resample("7D").agg(agg_func, numeric_only=True)
        df_7d_min = df.resample("7D").min(numeric_only=True)
        df_7d_max = df.resample("7D").max(numeric_only=True)
        
        for i, col in enumerate(y_cols):
            c = color_seq[i % len(color_seq)]
            c_fill = hex_to_rgba(c, 0.25)
            name = str(col)
            use_band = has_bands and col in ["ModuleTemp_Mean_C", "ModuleTemp_Median_C"]
            b_min = band_min_col if use_band else col
            b_max = band_max_col if use_band else col

            fig.add_trace(go.Scatter(x=df_1h_min.index, y=df_1h_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=True, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1h_max.index, y=df_1h_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=True, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1h_main.index, y=df_1h_main[col], mode="lines+markers", name=f"{name} ({label_stat} 1H)", legendgroup=name, line=dict(color=c, width=2.0), marker=dict(size=5), visible=True, connectgaps=True))
            
            fig.add_trace(go.Scatter(x=df_6h_min.index, y=df_6h_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_6h_max.index, y=df_6h_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_6h_main.index, y=df_6h_main[col], mode="lines+markers", name=f"{name} ({label_stat} 6H)", legendgroup=name, line=dict(color=c, width=2.0), marker=dict(size=5), visible=False, connectgaps=True))

            fig.add_trace(go.Scatter(x=df_1d_min.index, y=df_1d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1d_max.index, y=df_1d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1d_main.index, y=df_1d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 1D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=6), visible=False, connectgaps=True))

            fig.add_trace(go.Scatter(x=df_3d_min.index, y=df_3d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_3d_max.index, y=df_3d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_3d_main.index, y=df_3d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 3D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=7), visible=False, connectgaps=True))
            
            fig.add_trace(go.Scatter(x=df_7d_min.index, y=df_7d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_7d_max.index, y=df_7d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_7d_main.index, y=df_7d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 7D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=8), visible=False, connectgaps=True))

        n = len(y_cols)
        vis_1h = ([True, True, True, False, False, False, False, False, False, False, False, False, False, False, False]) * n
        vis_6h = ([False, False, False, True, True, True, False, False, False, False, False, False, False, False, False]) * n
        vis_1d = ([False, False, False, False, False, False, True, True, True, False, False, False, False, False, False]) * n
        vis_3d = ([False, False, False, False, False, False, False, False, False, True, True, True, False, False, False]) * n
        vis_7d = ([False, False, False, False, False, False, False, False, False, False, False, False, True, True, True]) * n

        trace_indices = list(range(15 * n))

        buttons_agg = [
            dict(label=f"{label_stat} Hourly (1H)", method="restyle", args=[{"visible": vis_1h}, trace_indices]),
            dict(label=f"{label_stat} 6-Hourly (6H)", method="restyle", args=[{"visible": vis_6h}, trace_indices]),
            dict(label=f"{label_stat} Daily (1D)", method="restyle", args=[{"visible": vis_1d}, trace_indices]),
            dict(label=f"{label_stat} 3-Day (3D)", method="restyle", args=[{"visible": vis_3d}, trace_indices]),
            dict(label=f"{label_stat} Weekly (7D)", method="restyle", args=[{"visible": vis_7d}, trace_indices]),
        ]
        updatemenus_config = [
            dict(type="buttons", direction="right", buttons=buttons_agg, active=0, showactive=True,
                 x=1, xanchor="right", y=-0.15, yanchor="top", font=dict(size=11, color="#1E293B"),
                 bgcolor="#F8FAFC", bordercolor="#E2E8F0")
        ]

    else:
        df_1d_main = df.resample("1D").agg(agg_func, numeric_only=True)
        df_1d_min = df.resample("1D").min(numeric_only=True)
        df_1d_max = df.resample("1D").max(numeric_only=True)

        df_3d_main = df.resample("3D").agg(agg_func, numeric_only=True)
        df_3d_min = df.resample("3D").min(numeric_only=True)
        df_3d_max = df.resample("3D").max(numeric_only=True)
        
        df_7d_main = df.resample("7D").agg(agg_func, numeric_only=True)
        df_7d_min = df.resample("7D").min(numeric_only=True)
        df_7d_max = df.resample("7D").max(numeric_only=True)
        
        df_30d_main = df.resample("30D").agg(agg_func, numeric_only=True)
        df_30d_min = df.resample("30D").min(numeric_only=True)
        df_30d_max = df.resample("30D").max(numeric_only=True)
        
        for i, col in enumerate(y_cols):
            c = color_seq[i % len(color_seq)]
            c_fill = hex_to_rgba(c, 0.25)
            name = str(col)
            use_band = has_bands and col in ["ModuleTemp_Mean_C", "ModuleTemp_Median_C"]
            b_min = band_min_col if use_band else col
            b_max = band_max_col if use_band else col

            fig.add_trace(go.Scatter(x=df_1d_min.index, y=df_1d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=True, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1d_max.index, y=df_1d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=True, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_1d_main.index, y=df_1d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 1D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=6), visible=True, connectgaps=True))

            fig.add_trace(go.Scatter(x=df_3d_min.index, y=df_3d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_3d_max.index, y=df_3d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_3d_main.index, y=df_3d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 3D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=7), visible=False, connectgaps=True))
            
            fig.add_trace(go.Scatter(x=df_7d_min.index, y=df_7d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_7d_max.index, y=df_7d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_7d_main.index, y=df_7d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 7D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=8), visible=False, connectgaps=True))
            
            fig.add_trace(go.Scatter(x=df_30d_min.index, y=df_30d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_30d_max.index, y=df_30d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", legendgroup=name, showlegend=False, visible=False, connectgaps=True))
            fig.add_trace(go.Scatter(x=df_30d_main.index, y=df_30d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 30D)", legendgroup=name, line=dict(color=c, width=2.5), marker=dict(size=10), visible=False, connectgaps=True))

        n = len(y_cols)
        vis_1d = ([True, True, True, False, False, False, False, False, False, False, False, False]) * n
        vis_3d = ([False, False, False, True, True, True, False, False, False, False, False, False]) * n
        vis_7d = ([False, False, False, False, False, False, True, True, True, False, False, False]) * n
        vis_30d = ([False, False, False, False, False, False, False, False, False, True, True, True]) * n

        trace_indices = list(range(12 * n))

        buttons_agg = [
            dict(label=f"{label_stat} Daily (1D)", method="restyle", args=[{"visible": vis_1d}, trace_indices]),
            dict(label=f"{label_stat} 3-Day (3D)", method="restyle", args=[{"visible": vis_3d}, trace_indices]),
            dict(label=f"{label_stat} Weekly (7D)", method="restyle", args=[{"visible": vis_7d}, trace_indices]),
            dict(label=f"{label_stat} Monthly (30D)", method="restyle", args=[{"visible": vis_30d}, trace_indices]),
        ]
        updatemenus_config = [
            dict(type="buttons", direction="right", buttons=buttons_agg, active=0, showactive=True,
                 x=1, xanchor="right", y=-0.15, yanchor="top", font=dict(size=11, color="#1E293B"),
                 bgcolor="#F8FAFC", bordercolor="#E2E8F0")
        ]
        
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=60), plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=updatemenus_config,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig

def create_mppt_with_irradiance_chart(
    df,
    power_cols,
    irr_col="POA_Irradiance_W_m2",
    central_metric="Mean",
    resolution_rules=None,
    secondary_name="POA Irradiance",
    secondary_unit="W/m²",
    secondary_color="#38BDF8",
    primary_axis_title="Power (mW)",
):
    """
    Generates a synchronized dual-axis chart with Min/Max confidence bands 
    for each power/PCE trace, updating coherently across temporal resolutions.
    """
    if df.empty or not power_cols or irr_col not in df.columns:
        return go.Figure()

    fig = go.Figure()
    agg_func = "median" if central_metric == "Median" else "mean"
    label_stat = "Median" if central_metric == "Median" else "Mean"
    color_seq = ["#1E293B", "#2563EB", "#16A34A", "#7C3AED", "#DB2777", "#4F46E5", "#64748B", "#111827"]

    secondary_hex = secondary_color.lstrip("#")
    secondary_rgb = tuple(int(secondary_hex[i:i + 2], 16) for i in (0, 2, 4))
    secondary_fillcolor = f"rgba({secondary_rgb[0]},{secondary_rgb[1]},{secondary_rgb[2]},0.08)"

    def hex_to_rgba(hex_color, alpha=0.20):
        h = hex_color.lstrip("#")
        r, g, b = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    min_dt = df.index.min()
    max_dt = df.index.max()
    duration_days = (max_dt - min_dt).total_seconds() / 86400.0

    rules = resolution_rules or get_resolution_rules(duration_days, label_stat)
    
    # Cada nivel de resolución genera 3 trazas por columna de potencia (Min, Max, Main) + 1 traza ambiental
    traces_per_level = (len(power_cols) * 3) + 1  

    for idx_rule, (freq, label) in enumerate(rules):
        is_active = (idx_rule == 0)

        if freq == "native" and len(df) <= MAX_NATIVE_POINTS:
            df_main = df
            df_min = df
            df_max = df
        else:
            effective_freq = "1h" if freq == "native" else freq
            df_main = df.resample(effective_freq).agg(agg_func, numeric_only=True)
            df_min = df.resample(effective_freq).min(numeric_only=True)
            df_max = df.resample(effective_freq).max(numeric_only=True)

        # 1. Trazas de Potencia / PCE en Eje Y1 con banda Min/Max
        for i, col in enumerate(power_cols):
            c = color_seq[i % len(color_seq)]
            c_fill = hex_to_rgba(c, 0.20)
            name = str(col)
            
            b_min = col if col not in df_min.columns else col
            b_max = col if col not in df_max.columns else col

            # Traza invisible inferior para el relleno
            fig.add_trace(go.Scatter(
                x=df_min.index, y=df_min[b_min], mode="lines",
                line=dict(width=0), name=f"Min {name}", legendgroup=name,
                showlegend=False, visible=is_active, connectgaps=True
            ))
            # Traza superior con relleno contra la anterior
            fig.add_trace(go.Scatter(
                x=df_max.index, y=df_max[b_max], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=c_fill,
                name=f"Max {name}", legendgroup=name, showlegend=False,
                visible=is_active, connectgaps=True
            ))
            # Traza central principal
            fig.add_trace(go.Scatter(
                x=df_main.index, y=df_main[col],
                mode="lines+markers" if freq == "native" or duration_days <= 7.0 else "lines",
                name=f"{name} ({label_stat} {label.split('(')[-1].replace(')', '')})" if freq != "native" else name,
                legendgroup=name, line=dict(color=c, width=1.8),
                marker=dict(size=4), yaxis="y1", visible=is_active, connectgaps=True
            ))

        # 2. Traza ambiental en Eje Y2 a la misma frecuencia
        fig.add_trace(go.Scatter(
            x=df_main.index,
            y=df_main[irr_col] if irr_col in df_main.columns else [0]*len(df_main),
            mode="lines",
            name=secondary_name,
            line=dict(color=secondary_color, width=1.5, dash="dot"),
            marker=dict(color=secondary_color),
            fill="tozeroy",
            fillcolor=secondary_fillcolor,
            yaxis="y2",
            visible=is_active,
            connectgaps=True,
            hovertemplate=f"<b>%{{x}}</b><br>{secondary_name}: %{{y:.1f}} {secondary_unit}<extra></extra>"
        ))

    # Mascaras de visibilidad sincronizadas para los botones de resolución
    n_levels = len(rules)
    buttons = []
    if n_levels > 1:
        for idx in range(n_levels):
            vis = [False] * (n_levels * traces_per_level)
            start = idx * traces_per_level
            end = start + traces_per_level
            for k in range(start, end):
                vis[k] = True
            buttons.append(dict(label=rules[idx][1], method="restyle", args=[{"visible": vis}]))

    updatemenus = [dict(
        type="buttons", direction="right", buttons=buttons, active=0, showactive=True,
        x=1, xanchor="right", y=-0.15, yanchor="top", font=dict(size=11, color="#1E293B"),
        bgcolor="#F8FAFC", bordercolor="#E2E8F0"
    )] if len(buttons) > 1 else []

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=updatemenus,
        yaxis=dict(
            title=dict(text=primary_axis_title, font=dict(color="#1E293B")),
            showgrid=True, gridcolor="#F1F5F9", rangemode="tozero"
        ),
        yaxis2=dict(
            title=dict(text=f"{secondary_name} ({secondary_unit})", font=dict(color=secondary_color)),
            overlaying="y", side="right", showgrid=False,
            tickfont=dict(color=secondary_color), rangemode="tozero"
        ),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9")
    )
    return fig

# =====================================================================
# DUAL-AXIS PV VS ENVIRONMENTAL CHART
# =====================================================================
def create_pv_vs_env_chart(df, power_cols, temp_col, pv_metric, env_metric, central_metric="Mean", resolution_rules=None):
    if df.empty or not power_cols:
        return go.Figure()
        
    df_calc = pd.DataFrame(index=df.index)
    
    # Base PV Calculations (10-min step) -> to kW for conversion
    conv_factor = (10.0 / 60.0) / (1000.0 * CELL_AREA_M2)
    df_calc['Yield'] = df[power_cols].mean(axis=1) * conv_factor
    
    if "POA_Irradiance_W_m2" in df.columns:
        df_calc['Radiation'] = df["POA_Irradiance_W_m2"] * (10.0 / 60.0) / 1000.0
    else:
        df_calc['Radiation'] = 0
        
    # Base Env Calculations
    if env_metric == "Thermal Load (°C·h)" and temp_col in df.columns:
        df_calc['Env'] = df[temp_col] * (10.0 / 60.0)
        env_name, env_unit, env_color = "Thermal Load", "°C·h", "#DC2626"
    elif env_metric == "Absolute Humidity Dose (g/m³·h)" and "AbsoluteHumidity_g_m3" in df.columns:
        df_calc['Env'] = df["AbsoluteHumidity_g_m3"] * (10.0 / 60.0)
        env_name, env_unit, env_color = "Humidity Dose", "g/m³·h", "#0891B2"
    else:
        df_calc['Env'] = df_calc['Radiation']
        env_name, env_unit, env_color = "Radiation Dose", "kWh/m²", "#D97706"

    pv_is_yield = (pv_metric == "Energy Yield (kWh/m²)")
    pv_name = "Energy Yield" if pv_is_yield else "Efficiency"
    pv_unit = "kWh/m²" if pv_is_yield else "%"
    pv_color = "#15803D" if pv_is_yield else "#2563EB"

    min_dt = df_calc.index.min()
    max_dt = df_calc.index.max()
    duration_days = (max_dt - min_dt).total_seconds() / 86400.0

    label_stat = "Median" if central_metric == "Median" else "Mean"
    resolution_rules = resolution_rules or get_resolution_rules(duration_days, label_stat)
    aggregation_labels = {
        "1h": "Hourly (1H)",
        "6h": "6-Hourly (6H)",
        "1D": "Daily (1D)",
        "7D": "Weekly (7D)",
        "30D": "Monthly (30D)",
    }
    normalized_rules = []
    for freq, label in resolution_rules:
        if freq in ("native", "6h"):
            if freq == "6h":
                continue
            freq = "1h"
        if freq not in {rule_freq for rule_freq, _ in normalized_rules}:
            normalized_rules.append((freq, aggregation_labels.get(freq, label)))
    resolution_rules = normalized_rules
    if len(df_calc) > MAX_NATIVE_POINTS:
        resolution_rules = [
            ("1h" if freq == "native" else freq, label)
            for freq, label in resolution_rules
        ]

    fig = go.Figure()

    for i, (freq, label) in enumerate(resolution_rules):
        is_vis = (i == 0)

        if freq == "native":
            res_yield = df_calc["Yield"]
            res_rad = df_calc["Radiation"]
            res_env = df_calc["Env"]
            days = 1 / 144
        else:
            days = pd.Timedelta(freq).total_seconds() / 86400.0
            res_yield = df_calc["Yield"].resample(freq).sum()
            res_rad = df_calc["Radiation"].resample(freq).sum()
            res_env = df_calc["Env"].resample(freq).sum()
        mask = res_rad > 0
        
        if pv_is_yield:
            y_pv = res_yield[mask]
            y_pv = y_pv[y_pv >= 0] # Filter extreme noise
        else:
            aligned_y, aligned_r = res_yield.align(res_rad, join='inner')
            valid = aligned_r > 0.005 # > 5 Wh/m² radiation to compute efficiency
            y_pv = (aligned_y[valid] / aligned_r[valid]) * 100.0
            y_pv = y_pv.replace([np.inf, -np.inf], np.nan).dropna()
            
            # [CRITICAL FIX] Cap aggregated efficiency to physical bounds (0% - 100%) to avoid night division artifacts
            y_pv = y_pv[(y_pv >= 0) & (y_pv <= 100)]
        
        y_env = res_env[mask]
        
        def get_widths_labels(idx):
            labels, widths = [], []
            for date in idx:
                if days < 1:
                    end_date = min(date + pd.Timedelta(hours=1), max_dt)
                    labels.append(f"{date.strftime('%Y-%m-%d %H:%M')}")
                    widths.append(3600000 * 0.95)
                else:
                    end_date = min(date + pd.Timedelta(days=days-1), max_dt)
                    real_days = (end_date.date() - date.date()).days + 1
                    if real_days <= 1:
                        labels.append(date.strftime('%Y-%m-%d'))
                    else:
                        labels.append(f"{date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    widths.append((real_days * 86400000) * 0.95)
            return labels, widths

        custom_pv, widths_pv = get_widths_labels(y_pv.index)
        fig.add_trace(go.Bar(
            x=y_pv.index, y=y_pv.values, customdata=custom_pv,
            name=f"{pv_name}", marker_color=pv_color, opacity=0.8,
            width=widths_pv, offset=0, yaxis="y1",
            hovertemplate=f"<b>%{{customdata}}</b><br>{pv_name}: %{{y:.2f}} {pv_unit}<extra></extra>",
            visible=is_vis
        ))
        
        custom_env, widths_env = get_widths_labels(y_env.index)
        fig.add_trace(go.Scatter(
            x=y_env.index, y=y_env.values, customdata=custom_env,
            name=f"{env_name}", mode="lines+markers",
            line=dict(color=env_color, width=2.5), marker=dict(size=6, color=env_color),
            yaxis="y2",
            hovertemplate=f"<b>%{{customdata}}</b><br>{env_name}: %{{y:.2f}} {env_unit}<extra></extra>",
            visible=is_vis
        ))

    n_rules = len(resolution_rules)
    buttons = []
    if n_rules > 1:
        for i, (_, label) in enumerate(resolution_rules):
            vis = [False] * (n_rules * 2)
            vis[i*2] = True
            vis[i*2+1] = True
            buttons.append(dict(label=label, method="update", args=[{"visible": vis}]))

    updatemenus = [dict(
        type="buttons", direction="right", buttons=buttons, active=0, showactive=True,
        x=1, xanchor="right", y=-0.2, yanchor="top", font=dict(size=11, color="#1E293B"),
        bgcolor="#F8FAFC", bordercolor="#E2E8F0"
    )] if len(buttons) > 1 else []

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=updatemenus,
        yaxis=dict(
            title=dict(text=f"{pv_name} ({pv_unit})", font=dict(color=pv_color)),
            tickfont=dict(color=pv_color),
            showgrid=True, gridcolor="#F1F5F9", rangemode="tozero"
        ),
        yaxis2=dict(
            title=dict(text=f"{env_name} ({env_unit})", font=dict(color=env_color)),
            tickfont=dict(color=env_color),
            overlaying="y", side="right", showgrid=False, rangemode="tozero"
        ),
        xaxis=dict(showgrid=False)
    )
    return fig

def get_trace_color(fig, trace_name: str, default_color: str = "grey") -> str:
    for trace in fig.data:
        if trace.name == trace_name:
            return trace.line.color or trace.marker.color or default_color
    return default_color


def create_digital_twin_chart(df_twin, cell_name, resolution_rules):
    """Plots normalized PCE and pFF against their Digital Twin predictions."""
    from plotly.subplots import make_subplots

    required_cols = {
        "PCE_Relative", "Twin_PCE_Pred_Relative", "pFF", "Twin_pFF_Pred",
        "Alert_PCE", "Alert_pFF", "In_Action_Window"
    }
    if df_twin.empty or not required_cols.issubset(df_twin.columns):
        return go.Figure()

    cell_df = df_twin[df_twin["cell_name"] == cell_name].sort_index()
    if cell_df.empty:
        return go.Figure()

    duration_days = (cell_df.index.max() - cell_df.index.min()).total_seconds() / 86400.0
    rules = resolution_rules or get_resolution_rules(duration_days, "Mean")
    if len(cell_df) > MAX_NATIVE_POINTS and rules[0][0] == "native":
        rules = get_resolution_rules(7.0001, "Mean")

    freq = rules[0][0]
    if freq != "native":
        agg_map = {
            "PCE_Relative": "mean", "Twin_PCE_Pred_Relative": "mean",
            "pFF": "mean", "Twin_pFF_Pred": "mean",
            "Alert_PCE": "max", "Alert_pFF": "max", "In_Action_Window": "max"
        }
        cell_df = cell_df.resample(freq).agg(agg_map).dropna(subset=["PCE_Relative"])

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("PCE Normalizado (relativo al pico inicial)", "Fill Factor (pFF)")
    )
    fig.add_trace(go.Scatter(x=cell_df.index, y=cell_df["PCE_Relative"], mode="lines", name="PCE Real (norm.)", line=dict(color="#10B981", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=cell_df.index, y=cell_df["Twin_PCE_Pred_Relative"], mode="lines", name="Twin PCE (predicho)", line=dict(color="#10B981", width=1.5, dash="dot")), row=1, col=1)
    alert_pce = cell_df[cell_df["Alert_PCE"].astype(bool)]
    fig.add_trace(go.Scatter(x=alert_pce.index, y=alert_pce["PCE_Relative"], mode="markers", name="Alerta PCE", marker=dict(color="#F43F5E", size=7, symbol="x")), row=1, col=1)
    fig.add_trace(go.Scatter(x=cell_df.index, y=cell_df["pFF"], mode="lines", name="pFF Real", line=dict(color="#3B82F6", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=cell_df.index, y=cell_df["Twin_pFF_Pred"], mode="lines", name="Twin pFF (predicho)", line=dict(color="#3B82F6", width=1.5, dash="dot")), row=2, col=1)
    alert_pff = cell_df[cell_df["Alert_pFF"].astype(bool)]
    fig.add_trace(go.Scatter(x=alert_pff.index, y=alert_pff["pFF"], mode="markers", name="Alerta pFF", marker=dict(color="#F59E0B", size=7, symbol="x")), row=2, col=1)

    action_df = cell_df[cell_df["In_Action_Window"].astype(bool)]
    if not action_df.empty:
        fig.add_vrect(x0=action_df.index.min(), x1=action_df.index.max(), fillcolor="#FEF3C7", opacity=0.25, line_width=0, row="all", col="all")

    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="PCE / PCE₀", row=1, col=1)
    fig.update_yaxes(title_text="pFF", row=2, col=1)
    return fig

def build_fleet_pff_pivot(df_twin_diag, active_cells, start_dt, end_dt):
    """Pivots the twin diagnostics dataset into one pFF column per active cell."""
    if df_twin_diag.empty or not active_cells:
        return pd.DataFrame()
    window = df_twin_diag[
        (df_twin_diag.index >= start_dt) &
        (df_twin_diag.index <= end_dt) &
        (df_twin_diag["cell_name"].isin(active_cells))
    ]
    if window.empty or "pFF" not in window.columns:
        return pd.DataFrame()
    pivot = window.pivot_table(index=window.index, columns="cell_name", values="pFF", aggfunc="mean")
    pivot.index.name = "Timestamp"
    return pivot

def build_fleet_pce_pivot(df_twin_diag, active_cells, start_dt, end_dt):
    """Pivots the twin diagnostics dataset into one normalized PCE column per active cell."""
    if df_twin_diag.empty or not active_cells:
        return pd.DataFrame()
    window = df_twin_diag[
        (df_twin_diag.index >= start_dt) &
        (df_twin_diag.index <= end_dt) &
        (df_twin_diag["cell_name"].isin(active_cells))
    ]
    if window.empty or "PCE_Relative" not in window.columns:
        return pd.DataFrame()
    pivot = window.pivot_table(index=window.index, columns="cell_name", values="PCE_Relative", aggfunc="mean")
    pivot.index.name = "Timestamp"
    return pivot

# =====================================================================
# KPI UTILITIES
# =====================================================================
def calculate_delta_pct(actual, previous):
    if pd.isna(actual) or pd.isna(previous) or previous == 0:
        return None
    return ((actual - previous) / previous) * 100
    
def format_kpi(value, unit, decimals=1):
    return f"{value:.{decimals}f} {unit}" if pd.notna(value) else f"-- {unit}"
    
def format_delta(actual, previous):
    delta = calculate_delta_pct(actual, previous)
    return f"{delta:+.1f}%" if delta is not None else "N/A"

# =====================================================================
# TOOLBAR: TIME WINDOW SELECTION
# =====================================================================
def render_toolbar(min_date: datetime.date, max_date: datetime.date):
    with st.container(border=True):
        col_label, col_control = st.columns([1, 4], vertical_alignment="center")
        with col_label:
            st.markdown("**Time Window**")
        with col_control:
            time_window = st.segmented_control(
                "Time Window",
                options=["1D", "3D", "7D", "30D", "Historical", "Custom Dates"],
                default="1D",
                label_visibility="collapsed",
            )
        time_window = time_window or "1D"
        start_date, end_date = min_date, max_date

        if time_window == "Custom Dates":
            col_d1, col_d2, _ = st.columns([1, 1, 2])
            with col_d1:
                start_date = st.date_input("From", value=min_date)
            with col_d2:
                end_date = st.date_input("To", value=max_date)

    return time_window, start_date, end_date

# =====================================================================
# GLOBAL SIDEBAR: LOGO
# =====================================================================
def render_branding() -> None:
    st.logo(LOGO_URL, size="large", link=LOGO_URL)

dates_index = pd.date_range(start="2026-01-01", periods=100, freq="D")
plotly_config = {
    "modeBarButtonsToRemove": ["autoScale2d", "select2d", "lasso2d"],
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "parasol_chart",
        "height": 600,
        "width": 1000,
        "scale": 2
    }
}

# =====================================================================
# VIEW 1: GENERAL OVERVIEW
# =====================================================================
def general_overview():
    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.markdown(
        "<h4 style='font-size: 1.1rem; color: #1E293B; margin-bottom: 0;'>KPI Options</h4>",
        unsafe_allow_html=True
    )
    kpi1_type = st.sidebar.radio("Photovoltaic (KPI 1)", ["Fleet MPPT Power", "Fleet PCE"])
    kpi2_type = st.sidebar.radio("Irradiance (KPI 2)", ["Mean POA Irradiance", "Accumulated POA Dose"])
    kpi3_type = st.sidebar.radio("Temperature (KPI 3)", ["Mean Module Temp.", "Mean Ambient Temp."])
    kpi4_type = st.sidebar.radio("Humidity (KPI 4)", ["Relative Humidity", "Absolute Humidity", "Humidity Dose"])

    st.sidebar.divider()

    st.sidebar.markdown(
        "<h4 style='font-size: 1.1rem; color: #1E293B; margin-bottom: 0;'>Chart Aggregations</h4>",
        unsafe_allow_html=True
    )
    selected_metric = st.sidebar.radio(
        "Central Tendency",
        ["Mean", "Median"],
        index=0,
        help="Statistical metric applied to the temporal resampling of charts.",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"""
        <div style='text-align: center; font-size: 0.85rem; color: #64748B; margin-top: 1rem;'>
            Developed by <br>
            <a href='{CREATOR_LINK}' target='_blank' style='color: #36B9CC; text-decoration: none; font-weight: 600;'>
                Cristian Carretero
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- DATA LOADING ---
    fleet_df = load_global_data()
    ml_artifacts = load_ml_artifacts()
    survival_df = load_survival_data()
    df_twin_diag = load_twin_diagnostics()
    df_pce_norm = load_pce_norm_data()
    df_pff_pred = load_pff_pred_data()
    alert_thresholds = ml_artifacts.get("alert_thresholds", {})
    model_pce = ml_artifacts.get("model_pce")
    model_pff = ml_artifacts.get("model_pff")
    dual_twin_ready = model_pce is not None and model_pff is not None and {
        "pce", "pff"
    }.issubset(alert_thresholds)
    if ml_artifacts and not dual_twin_ready:
        st.warning("Dual Digital Twin artifacts are incomplete: PCE and pFF models and thresholds are required.")

    if not fleet_df.empty:
        max_date = pd.to_datetime(fleet_df.index.max()).date()
        min_date = pd.to_datetime(fleet_df.index.min()).date()
    else:
        max_date = datetime.date.today()
        min_date = max_date - datetime.timedelta(days=30)

    # --- HEADER & TOOLBAR ---
    st.markdown("### ParaSol Dashboard")
    st.caption("Global fleet monitoring, early failure detection, and environmental context.")
    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    
    time_window, start_date, end_date = render_toolbar(min_date, max_date)
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # --- TIME WINDOW FILTERING ---
    plot_df, prev_df = pd.DataFrame(), pd.DataFrame()
    if not fleet_df.empty:
        max_timestamp = fleet_df.index.max()
        if time_window in ("1D", "3D", "7D", "30D"):
            days = {"1D": 1, "3D": 3, "7D": 7, "30D": 30}[time_window]
            anchor_time = max_timestamp
            if anchor_time.hour == 0 and anchor_time.minute == 0 and anchor_time.second == 0:
                anchor_time -= pd.Timedelta(seconds=1)
            delta_start = pd.Timedelta(days=days - 1)
            start_dt = (anchor_time - delta_start).normalize()

            plot_df = fleet_df[fleet_df.index >= start_dt].copy()
            delta_prev = pd.Timedelta(days=days)
            prev_df = fleet_df[(fleet_df.index >= (start_dt - delta_prev)) & (fleet_df.index < start_dt)].copy()
        
        elif time_window == "Historical":
            plot_df = fleet_df.copy()
            prev_df = pd.DataFrame()
        
        elif time_window == "Custom Dates":
            if start_date > end_date:
                st.error("❌ **Selection Error:** The start date cannot be later than the end date.")
                st.stop()
            else:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                plot_df = fleet_df[(fleet_df.index >= start_dt) & (fleet_df.index <= end_dt)].copy()
                if plot_df.empty:
                    st.error("❌ **No Records:** No data available for the selected period.")
                    st.stop()
                delta_dates = end_dt - start_dt
                prev_df = fleet_df[(fleet_df.index >= (start_dt - delta_dates)) & (fleet_df.index < start_dt)].copy()
        
        plot_df = plot_df.dropna(how="all")
        if plot_df.empty:
            st.error(f"❌ **No Records:** No data available for the selected time window ({time_window}).")
            st.stop()

    duration_days = (plot_df.index.max() - plot_df.index.min()).total_seconds() / 86400.0
    resolution_rules = get_resolution_rules(duration_days, selected_metric)

    # -------------------------------------------------------------
    # IDENTIFICATION OF ACTIVE CELLS (Timezone-naive)
    # -------------------------------------------------------------
    min_t, max_t = None, None
    if not plot_df.empty:
        min_t = plot_df.index.min()
        max_t = plot_df.index.max()
        if getattr(min_t, "tzinfo", None) is not None:
            min_t = min_t.replace(tzinfo=None)
        if getattr(max_t, "tzinfo", None) is not None:
            max_t = max_t.replace(tzinfo=None)

    survival_naive_df = pd.DataFrame()
    active_cells_window = []
    
    if not survival_df.empty:
        survival_naive_df = survival_df.copy()
        survival_dt_index = pd.DatetimeIndex(survival_naive_df.index)
        if getattr(survival_dt_index, "tz", None) is not None:
            survival_dt_index = survival_dt_index.tz_localize(None)
        survival_naive_df.index = survival_dt_index
        
        if min_t is not None and max_t is not None:
            mask_surv_window = (survival_naive_df.index >= min_t) & (survival_naive_df.index <= max_t)
            cell_name_col = survival_naive_df["cell_name"]
            active_cells_window = pd.unique(cell_name_col[mask_surv_window])

    active_cells_lower = [str(c).lower() for c in active_cells_window]
    
    all_power_cols = [c for c in plot_df.columns if 'power' in c.lower() or 'p_mpp' in c.lower()]
    
    # Filter power columns to only active cells
    if active_cells_lower:
        active_power_cols = [c for c in all_power_cols if any(ac in c.lower() for ac in active_cells_lower)]
    else:
        active_power_cols = all_power_cols

    def get_valid_cols(columns_list, df_local):
        return [c for c in columns_list if df_local[c].notna().any() and (df_local[c] != 0).any()]

    active_power_cols = get_valid_cols(active_power_cols, plot_df)

    # -------------------------------------------------------------
    # DYNAMIC PHYSICAL PCE & POWER CALCULATION (FOR CHARTS & KPIS)
    # -------------------------------------------------------------
    pce_cols_plot = []
    power_cols_plot = []
    
    if 'POA_Irradiance_W_m2' in plot_df.columns:
        # PARCHE 1: umbral subido de 10 a PCE_IRRADIANCE_MIN_W_M2 (100 W/m^2), consistente con
        # data_aggregation.py y anomaly_detection.py, para evitar inestabilidad numerica del
        # denominador (POA * CELL_AREA_M2) al calcular la PCE instantanea en el dashboard.
        mask_sun_plot = plot_df['POA_Irradiance_W_m2'] > PCE_IRRADIANCE_MIN_W_M2
        mask_sun_prev = prev_df['POA_Irradiance_W_m2'] > PCE_IRRADIANCE_MIN_W_M2 if not prev_df.empty else pd.Series(False, index=prev_df.index)

        for p_col in active_power_cols:
            dev_id = p_col.split('_')[-1]
            pce_col_name = f"PCE_{dev_id} (%)"
            power_col_name = f"Power_{dev_id} (mW)"
            
            pce_cols_plot.append(pce_col_name)
            power_cols_plot.append(power_col_name)
            
            plot_df[power_col_name] = plot_df[p_col] * 1000.0
            if not prev_df.empty and p_col in prev_df.columns:
                prev_df[power_col_name] = prev_df[p_col] * 1000.0
                
            plot_df[pce_col_name] = np.nan
            plot_df.loc[mask_sun_plot, pce_col_name] = (
                plot_df.loc[mask_sun_plot, p_col] / (plot_df.loc[mask_sun_plot, 'POA_Irradiance_W_m2'] * CELL_AREA_M2)
            ) * 100.0
            
            # [CRITICAL FIX] Capping instantaneous PCE to avoid micro-noise spikes
            plot_df[pce_col_name] = plot_df[pce_col_name].clip(lower=0, upper=100)
            
            if not prev_df.empty and p_col in prev_df.columns:
                prev_df[pce_col_name] = np.nan
                prev_df.loc[mask_sun_prev, pce_col_name] = (
                    prev_df.loc[mask_sun_prev, p_col] / (prev_df.loc[mask_sun_prev, 'POA_Irradiance_W_m2'] * CELL_AREA_M2)
                ) * 100.0
                prev_df[pce_col_name] = prev_df[pce_col_name].clip(lower=0, upper=100)

    def calc_physical_pce(df_local, p_cols):
        if not p_cols or df_local.empty or 'POA_Irradiance_W_m2' not in df_local.columns: return np.nan
        # PARCHE 1: umbral subido de 10 a PCE_IRRADIANCE_MIN_W_M2 (100 W/m^2)
        mask = df_local['POA_Irradiance_W_m2'] > PCE_IRRADIANCE_MIN_W_M2
        if not mask.any(): return np.nan
        pce_arrays = []
        for c in p_cols:
            incident_power_w = df_local.loc[mask, 'POA_Irradiance_W_m2'] * CELL_AREA_M2
            pce = df_local.loc[mask, c] / incident_power_w
            pce_arrays.append(pce)
        pce_df = pd.concat(pce_arrays)
        # Cap global KPI to bounds
        pce_df = pce_df[(pce_df >= 0) & (pce_df <= 1)]
        return pce_df.replace([0, np.inf, -np.inf], np.nan).mean() * 100

    mean_pce_act = calc_physical_pce(plot_df, active_power_cols)
    mean_pce_prev = calc_physical_pce(prev_df, active_power_cols)

    # --- KPI 4: Digital Twin Dynamic Status ---
    df_summary_dyn = pd.DataFrame()
    df_summary = pd.DataFrame()
    if ml_artifacts and min_t is not None and max_t is not None and not survival_naive_df.empty:
        df_summary = ml_artifacts.get("summary_table", pd.DataFrame()).copy()

        idx_present = df_summary.index.intersection(active_cells_window)
        df_summary_dyn = df_summary.loc[idx_present].copy()

        if not df_summary_dyn.empty:
            for col in ["t80_failure_date", "ml_alert_date"]:
                df_summary_dyn[col] = pd.to_datetime(df_summary_dyn[col])
                if getattr(df_summary_dyn[col].dt, 'tz', None) is not None:
                    df_summary_dyn[col] = df_summary_dyn[col].dt.tz_localize(None)
                    
            df_summary_dyn["t80_occurred"] = df_summary_dyn["t80_failure_date"].notna() & (df_summary_dyn["t80_failure_date"] <= max_t)
            df_summary_dyn["ml_occurred"] = df_summary_dyn["ml_alert_date"].notna() & (df_summary_dyn["ml_alert_date"] <= max_t)

            mask_future_t80 = df_summary_dyn["t80_failure_date"] > max_t
            df_summary_dyn.loc[mask_future_t80, ["t80_failure_date", "survival_days"]] = [pd.NaT, np.nan]
            mask_future_ml = df_summary_dyn["ml_alert_date"] > max_t
            df_summary_dyn.loc[mask_future_ml, ["ml_alert_date", "threshold_15pct_day"]] = [pd.NaT, np.nan]
            
            def classify_state(row):
                if row["t80_occurred"]: return "Physical Collapse"
                if row["ml_occurred"]: return "ML Predictive Alert"
                return "Healthy"

            df_summary_dyn["Current_State"] = df_summary_dyn.apply(classify_state, axis=1)
            
            n_def_count = int((df_summary_dyn["t80_occurred"] | df_summary_dyn["ml_occurred"]).sum())
            total_ml_cells = len(df_summary_dyn)
            n_val = total_ml_cells - n_def_count
            kpi_p4_val = f"{n_val} / {total_ml_cells}"
            
            if n_def_count == 0:
                kpi_p4_delta = "100% Operational"
            elif n_val == 0:
                kpi_p4_delta = f"-{n_def_count} Anomalies (Critical)"
            else:
                pct_healthy = (n_val / total_ml_cells) * 100.0
                kpi_p4_delta = f"-{n_def_count} Anomalies ({pct_healthy:.0f}% Healthy)"
            delta_col = "normal"
        else:
            kpi_p4_val, kpi_p4_delta, delta_col = "0 / 0", "No devices", "off"
    else:
        kpi_p4_val, kpi_p4_delta, delta_col = "--", "No ML data", "off"

    # --- LIFECYCLE ALERT (>14 DAYS) AND DIAGNOSTIC TABLE ---
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    audit_report = st.expander("View Audit Report (Digital Twin)", expanded=False)
    with audit_report:
        st.markdown("##### Digital Twin Diagnostics")
        st.caption("Machine-learning validation status and lifecycle diagnostics for the monitored cells.")
        st.metric("Validated Cells (ML)", kpi_p4_val, kpi_p4_delta, delta_color=delta_col)
    if not df_summary_dyn.empty and max_t is not None and not survival_naive_df.empty:
        absolute_day_zero = survival_naive_df.groupby('cell_name').apply(lambda x: x.index.min())
        operation_ends = pd.Series(pd.to_datetime(max_t), index=df_summary_dyn.index).astype('datetime64[ns]')

        mask_deaths = df_summary_dyn['t80_occurred']
        if mask_deaths.any():
            death_dates = pd.to_datetime(df_summary_dyn.loc[mask_deaths, 't80_failure_date']).astype('datetime64[ns]')
            operation_ends.loc[mask_deaths] = death_dates

        exposures = (operation_ends - absolute_day_zero.loc[operation_ends.index]).dt.total_seconds() / 86400.0
        mature_cells = exposures[exposures > 14.0]

        if not mature_cells.empty:
            cell_names = ", ".join([f"**{cel}** ({days:.1f} days)" for cel, days in mature_cells.items()])
            with audit_report:
                st.warning(
                    f"⚠️ ***Burn-in* Phase Concluded:** The following cells have exceeded 14 days of outdoor exposure "
                    f"by the selected date: {cell_names}. "
                    "The Digital Twin window of action has concluded for these devices. "
                    "Transition inference to the **Long-Term Predictive Model**."
                )

        with audit_report:
            st.markdown(f"XGBoost Classification adjusted to **{max_t.strftime('%Y-%m-%d %H:%M')}**:")
            df_render = df_summary_dyn.reset_index().copy()

            st.markdown("##### Accumulated Operation Time (Days of Exposure)")
            df_operation = pd.DataFrame({
                'cell_name': exposures.index,
                'Operation_Days': exposures.values,
                'Start': absolute_day_zero.loc[exposures.index].values,
                'End': operation_ends.values
            }).merge(
                df_render[['cell_name', 'Current_State']],
                on='cell_name',
                how='left'
            ).sort_values(by='Operation_Days', ascending=True)
            
            color_map = {'Physical Collapse': '#F43F5E', 'ML Predictive Alert': '#F59E0B', 'Healthy': '#10B981'}
            df_operation['Color'] = df_operation['Current_State'].map(color_map)
            
            fig_uptime = go.Figure(go.Bar(
                x=df_operation['Operation_Days'],
                y=df_operation['cell_name'],
                orientation='h',
                marker_color=df_operation['Color'],
                text=df_operation['Operation_Days'].apply(lambda x: f"{x:.1f} d"),
                textposition='inside',
                insidetextanchor='middle',
                hovertemplate="<b>%{y}</b><br>Start: %{customdata[0]}<br>End (Death/Today): %{customdata[1]}<br>Operation: %{x:.2f} days<br>Status: %{customdata[2]}<extra></extra>",
                customdata=np.stack((df_operation['Start'].dt.strftime('%Y-%m-%d'), df_operation['End'].dt.strftime('%Y-%m-%d %H:%M'), df_operation['Current_State']), axis=-1)
            ))
            fig_uptime.update_layout(
                margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", height=150 + (len(df_operation) * 35),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Real Days of Exposure"), yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_uptime, width="stretch", config=plotly_config)
            st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

            st.markdown("##### Diagnostics and Anomaly Extraction")
            st.caption(
                "Short-term screening uses instantaneous thermodynamic variables and their temporal gradients "
                "(thermal shock and moisture ingress). Accumulated thermal and humidity doses are reserved for "
                "long-term survival analysis. Alerts use the OR gate: PCE power drop or pFF structural mutation."
            )

            base_col_config = {
                "cell_name": "Device",
                "Current_State": st.column_config.TextColumn("Status (Diagnostic)"),
                "alert_freq_pct": st.column_config.ProgressColumn("Alert Freq. (%)", format="%.1f %%", min_value=0, max_value=100),
                "alert_pce_pct": st.column_config.ProgressColumn("Alert PCE (%)", format="%.1f %%", min_value=0, max_value=100),
                "alert_pff_pct": st.column_config.ProgressColumn("Alert pFF (%)", format="%.1f %%", min_value=0, max_value=100),
                "threshold_15pct_day": st.column_config.NumberColumn("15% Crossing Day", format="%.1f"),
                "survival_days": st.column_config.NumberColumn("Survival Days (T80)", format="%.1f"),
                "t80_failure_date": st.column_config.DatetimeColumn("T80 Date", format="YYYY/MM/DD HH:mm"),
                "ml_alert_date": st.column_config.DatetimeColumn("ML Alert Date", format="YYYY/MM/DD HH:mm"),
            }

            def prepare_audit_summary(artifact_key):
                summary = ml_artifacts.get(artifact_key, df_summary).copy()
                summary = summary.loc[summary.index.intersection(active_cells_window)].copy()
                if summary.empty:
                    return summary.reset_index()

                for col in ["t80_failure_date", "ml_alert_date"]:
                    summary[col] = pd.to_datetime(summary[col])
                    if getattr(summary[col].dt, 'tz', None) is not None:
                        summary[col] = summary[col].dt.tz_localize(None)

                summary["t80_occurred"] = summary["t80_failure_date"].notna() & (summary["t80_failure_date"] <= max_t)
                summary["ml_occurred"] = summary["ml_alert_date"].notna() & (summary["ml_alert_date"] <= max_t)

                mask_future_t80 = summary["t80_failure_date"] > max_t
                summary.loc[mask_future_t80, ["t80_failure_date", "survival_days"]] = [pd.NaT, np.nan]
                mask_future_ml = summary["ml_alert_date"] > max_t
                summary.loc[mask_future_ml, ["ml_alert_date", "threshold_15pct_day"]] = [pd.NaT, np.nan]

                summary["Current_State"] = summary.apply(classify_state, axis=1)
                return summary.reset_index()

            df_render_pce = prepare_audit_summary("summary_pce")
            df_render_pff = prepare_audit_summary("summary_pff")

            tab_tbl_all, tab_tbl_pce, tab_tbl_pff = st.tabs([
                " Resumen Global Combinado",
                " Alertas PCE Normalizado",
                " Alertas pFF (Estructural)"
            ])

            with tab_tbl_all:
                cols_all = [
                    "cell_name", "alert_freq_pct", "alert_pce_pct", "alert_pff_pct",
                    "survival_days", "t80_failure_date", "threshold_15pct_day",
                    "ml_alert_date", "Current_State"
                ]
                st.dataframe(
                    df_render[cols_all],
                    column_config=base_col_config,
                    hide_index=True, width="stretch"
                )

            with tab_tbl_pce:
                cols_pce = [
                    "cell_name", "alert_pce_pct", "survival_days", "t80_failure_date",
                    "threshold_15pct_day", "ml_alert_date", "Current_State"
                ]
                st.dataframe(
                    df_render_pce[cols_pce],
                    column_config=base_col_config,
                    hide_index=True, width="stretch"
                )

            with tab_tbl_pff:
                cols_pff = [
                    "cell_name", "alert_pff_pct", "survival_days", "t80_failure_date",
                    "threshold_15pct_day", "ml_alert_date", "Current_State"
                ]
                st.dataframe(
                    df_render_pff[cols_pff],
                    column_config=base_col_config,
                    hide_index=True, width="stretch"
                )

    # -------------------------------------------------------------
    # DIGITAL TWIN: pFF Y PCE NORMALIZADO DE TODAS LAS CELDAS (TABS)
    # -------------------------------------------------------------
    audit_report.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    audit_report.markdown("##### Digital Twin: Indicadores de Fleet por Celda")
    audit_report.caption(
        "PCE normalizado respecto al pico inicial de cada celda y Fill Factor real, superpuestos por "
        "celda dentro de la ventana seleccionada."
    )
    cell_names_sorted = sorted(str(c) for c in active_cells_window)
    tab_fleet_pff, tab_fleet_pce = audit_report.tabs(["Fill Factor (pFF)", "PCE Normalizado"])
    with tab_fleet_pff:
        df_pff_pivot = build_fleet_pff_pivot(df_pff_pred, cell_names_sorted, plot_df.index.min(), plot_df.index.max())
        if df_pff_pivot.empty:
            tab_fleet_pff.info("No hay observaciones de pFF del Gemelo Digital disponibles para esta ventana.")
        else:
            fig_pff_fleet = create_plotly_chart(
                df_pff_pivot,
                list(df_pff_pivot.columns),
                central_metric=selected_metric,
                resolution_rules=resolution_rules
            )
            fig_pff_fleet.update_layout(height=420, margin=dict(b=60))
            fig_pff_fleet.update_yaxes(title_text="pFF")
            tab_fleet_pff.plotly_chart(fig_pff_fleet, width="stretch", config=plotly_config)
    with tab_fleet_pce:
        df_pce_pivot = build_fleet_pce_pivot(df_pce_norm, cell_names_sorted, plot_df.index.min(), plot_df.index.max())
        if df_pce_pivot.empty:
            last_pce_obs = None
            if not df_pce_norm.empty and "PCE_Relative" in df_pce_norm.columns:
                pce_valid = df_pce_norm.loc[df_pce_norm["PCE_Relative"].notna()]
                if not pce_valid.empty:
                    last_pce_obs = pce_valid.index.max()
            message = "No hay observaciones de PCE normalizado del Gemelo Digital disponibles para esta ventana."
            if last_pce_obs is not None:
                message += f" Último dato disponible: {last_pce_obs:%Y-%m-%d %H:%M}."
            tab_fleet_pce.info(message)
        else:
            fig_pce_fleet = create_plotly_chart(
                df_pce_pivot,
                list(df_pce_pivot.columns),
                central_metric=selected_metric,
                resolution_rules=resolution_rules
            )
            fig_pce_fleet.update_layout(height=420, margin=dict(b=60))
            fig_pce_fleet.update_yaxes(title_text="PCE / PCE₀")
            tab_fleet_pce.plotly_chart(fig_pce_fleet, width="stretch", config=plotly_config)

    st.markdown("<hr style='margin: 3rem 0; opacity: 0.5;'>", unsafe_allow_html=True)

    # =================================================================
    # SECTION 2: PHOTOVOLTAIC PERFORMANCE & TELEMETRY
    # =================================================================
    st.markdown(f"### Photovoltaic Performance & Telemetry ({time_window})")
    
    has_prev = not prev_df.empty
    pce_act = mean_pce_act
    pce_prev = mean_pce_prev
        
    if power_cols_plot:
        plot_df['Fleet_Power_Mean'] = plot_df[power_cols_plot].mean(axis=1)
        pwr_act = plot_df['Fleet_Power_Mean'].mean()
        if has_prev and any(c in prev_df.columns for c in power_cols_plot):
            prev_df['Fleet_Power_Mean'] = prev_df[[c for c in power_cols_plot if c in prev_df.columns]].mean(axis=1)
            pwr_prev = prev_df['Fleet_Power_Mean'].mean()
        else:
            pwr_prev = np.nan
    else:
        pwr_act, pwr_prev = np.nan, np.nan

    # KPI 1: PV Performance
    if kpi1_type == "Fleet PCE":
        kpi1_label, kpi1_val, kpi1_delta = "Avg Fleet PCE", format_kpi(pce_act, "%", 2), format_delta(pce_act, pce_prev)
    else:
        kpi1_label, kpi1_val, kpi1_delta = "Avg Fleet MPPT Power", format_kpi(pwr_act, "mW", 2), format_delta(pwr_act, pwr_prev)

    # KPI 2: Irradiance
    if kpi2_type == "Mean POA Irradiance":
        irr_act = plot_df["POA_Irradiance_W_m2"].mean() if "POA_Irradiance_W_m2" in plot_df.columns else np.nan
        irr_prev = prev_df["POA_Irradiance_W_m2"].mean() if has_prev and "POA_Irradiance_W_m2" in prev_df.columns else np.nan
        kpi2_label, kpi2_val, kpi2_delta = "Mean POA Irradiance", format_kpi(irr_act, "W/m²", 0), format_delta(irr_act, irr_prev)
    else:
        dose_act = (plot_df["POA_Irradiance_W_m2"].sum() * (10 / 60)) / 1000 if not plot_df.empty and "POA_Irradiance_W_m2" in plot_df.columns else np.nan
        dose_prev = (prev_df["POA_Irradiance_W_m2"].sum() * (10 / 60)) / 1000 if has_prev and "POA_Irradiance_W_m2" in prev_df.columns else np.nan
        kpi2_label, kpi2_val, kpi2_delta = "Accumulated POA Dose", format_kpi(dose_act, "kWh/m²", 1), format_delta(dose_act, dose_prev)

    # KPI 3: Temperature
    if kpi3_type == "Mean Module Temp.":
        t_act = plot_df["ModuleTemp_Mean_C"].mean() if "ModuleTemp_Mean_C" in plot_df.columns else np.nan
        t_prev = prev_df["ModuleTemp_Mean_C"].mean() if has_prev and "ModuleTemp_Mean_C" in prev_df.columns else np.nan
        kpi3_label = "Mean Module Temp."
    else:
        t_act = plot_df["AmbientTemp_C"].mean() if "AmbientTemp_C" in plot_df.columns else np.nan
        t_prev = prev_df["AmbientTemp_C"].mean() if has_prev and "AmbientTemp_C" in prev_df.columns else np.nan
        kpi3_label = "Mean Ambient Temp."
    kpi3_val, kpi3_delta = format_kpi(t_act, "°C", 1), format_delta(t_act, t_prev)

    # KPI 4: Humidity
    if kpi4_type == "Relative Humidity":
        h_act = plot_df["RelativeHumidity_pct"].mean() if "RelativeHumidity_pct" in plot_df.columns else np.nan
        h_prev = prev_df["RelativeHumidity_pct"].mean() if has_prev and "RelativeHumidity_pct" in prev_df.columns else np.nan
        kpi4_label, kpi4_val, kpi4_delta = "Mean Rel. Humidity", format_kpi(h_act, "%", 1), format_delta(h_act, h_prev)
    elif kpi4_type == "Absolute Humidity":
        h_act = plot_df["AbsoluteHumidity_g_m3"].mean() if "AbsoluteHumidity_g_m3" in plot_df.columns else np.nan
        h_prev = prev_df["AbsoluteHumidity_g_m3"].mean() if has_prev and "AbsoluteHumidity_g_m3" in prev_df.columns else np.nan
        kpi4_label, kpi4_val, kpi4_delta = "Mean Abs. Humidity", format_kpi(h_act, "g/m³", 2), format_delta(h_act, h_prev)
    else:
        h_act = (plot_df["AbsoluteHumidity_g_m3"].sum() * (10 / 60)) if not plot_df.empty and "AbsoluteHumidity_g_m3" in plot_df.columns else np.nan
        h_prev = (prev_df["AbsoluteHumidity_g_m3"].sum() * (10 / 60)) if has_prev and "AbsoluteHumidity_g_m3" in prev_df.columns else np.nan
        kpi4_label, kpi4_val, kpi4_delta = "Accumulated Hum. Dose", format_kpi(h_act, "g/m³·h", 1), format_delta(h_act, h_prev)

    # Renderizado estricto a 4 columnas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(kpi1_label, kpi1_val, kpi1_delta)
    col2.metric(kpi2_label, kpi2_val, kpi2_delta)
    col3.metric(kpi3_label, kpi3_val, kpi3_delta)
    col4.metric(kpi4_label, kpi4_val, kpi4_delta)
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Re-asignar temp_col fijo para que las gráficas inferiores no fallen
    temp_col = "ModuleTemp_Mean_C"

    # --- MPPT POWER WITH DYNAMIC ENVIRONMENTAL AXIS ---
    secondary_metric = st.session_state.get("secondary_axis_metric", "POA Irradiance")
    secondary_options = {
        "POA Irradiance": (
            "POA_Irradiance_W_m2", "POA Irradiance", "W/m²", "#D97706"
        ),
        "Module Temperature": (
            temp_col, "Module Temperature", "°C", "#DC2626"
        ),
        "Absolute Humidity": (
            "AbsoluteHumidity_g_m3", "Absolute Humidity", "g/m³", "#0891B2"
        ),
    }
    secondary_col, secondary_name, secondary_unit, secondary_color = secondary_options[secondary_metric]

    tab_mppt, tab_pce = st.tabs(["MPPT Power (mW)", "PCE (%)"])
    with tab_mppt:
        if power_cols_plot and secondary_col in plot_df.columns:
            fig_pwr = create_mppt_with_irradiance_chart(
                df=plot_df,
                power_cols=power_cols_plot,
                irr_col=secondary_col,
                central_metric=selected_metric,
                resolution_rules=resolution_rules,
                secondary_name=secondary_name,
                secondary_unit=secondary_unit,
                secondary_color=secondary_color,
                primary_axis_title="Power (mW)",
            )
            st.plotly_chart(fig_pwr, width="stretch", config=plotly_config)
        elif not power_cols_plot:
            st.info("No active MPPT Power data recorded for the selected period.")
        else:
            st.info(f"No {secondary_name} data recorded for the selected period.")

    with tab_pce:
        if pce_cols_plot and secondary_col in plot_df.columns:
            fig_pce = create_mppt_with_irradiance_chart(
                df=plot_df,
                power_cols=pce_cols_plot,
                irr_col=secondary_col,
                central_metric=selected_metric,
                resolution_rules=resolution_rules,
                secondary_name=secondary_name,
                secondary_unit=secondary_unit,
                secondary_color=secondary_color,
                primary_axis_title="PCE (%)",
            )
            st.plotly_chart(fig_pce, width="stretch", config=plotly_config)
        elif not pce_cols_plot:
            st.info("No active PCE data recorded for the selected period.")
        else:
            st.info(f"No {secondary_name} data recorded for the selected period.")

    st.selectbox(
        "Secondary Axis (Environmental)",
        ["POA Irradiance", "Module Temperature", "Absolute Humidity"],
        key="secondary_axis_metric",
    )

    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

    # =================================================================
    # COLLAPSIBLE: YIELD VS ENVIRONMENTAL STRESS ANALYSIS
    # =================================================================
    with st.expander(" Yield vs. Environmental Stress Analysis (Fleet Average)", expanded=False):
        st.caption("Compare aggregate photovoltaic response against environmental stressors.")

        col_ui1, col_ui2 = st.columns(2)
        with col_ui1:
            pv_metric = st.selectbox(
                "Primary Axis (Photovoltaic)", 
                ["Energy Yield (kWh/m²)", "Conversion Efficiency (%)"]
            )
        with col_ui2:
            env_metric = st.selectbox(
                "Secondary Axis (Environmental)", 
                ["Radiation Dose (kWh/m²)", "Thermal Load (°C·h)", "Absolute Humidity Dose (g/m³·h)"]
            )

        if plot_df.empty or not active_power_cols:
            st.info("Insufficient active telemetry data.")
        else:
            fig_dual = create_pv_vs_env_chart(
                df=plot_df, 
                power_cols=active_power_cols, 
                temp_col=temp_col, 
                pv_metric=pv_metric, 
                env_metric=env_metric,
                central_metric=selected_metric,
                resolution_rules=resolution_rules
            )
            st.plotly_chart(fig_dual, width="stretch", config=plotly_config)

    st.markdown("<hr style='margin: 2rem 0; opacity: 0.5;'>", unsafe_allow_html=True)
    
    # --- SECONDARY SENSORS: 2x2 GRID (NO TABS) ---
    st.markdown("##### Environmental Context")
    
    # Row 1
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        st.markdown("###### Instantaneous POA Irradiance (W/m²)")
        if "POA_Irradiance_W_m2" in plot_df.columns:
            fig_irr = create_plotly_chart(
                plot_df, "POA_Irradiance_W_m2", ["#36B9CC"], central_metric=selected_metric,
                resolution_rules=resolution_rules
            )
            fig_irr.update_layout(height=320, margin=dict(b=20))
            st.plotly_chart(fig_irr, width="stretch", config=plotly_config)
            
    with r1_c2:
        st.markdown("###### Ambient vs. Module Temperature (°C)")
        if "AmbientTemp_C" in plot_df.columns and temp_col in plot_df.columns:
            fig_temp = create_plotly_chart(
                plot_df,
                ["AmbientTemp_C", temp_col],
                ["#1E293B", "#F59E0B"],
                central_metric=selected_metric,
                resolution_rules=resolution_rules
            )
            fig_temp.update_layout(height=320, margin=dict(b=20))
            st.plotly_chart(fig_temp, width="stretch", config=plotly_config)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Row 2
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        st.markdown("###### Relative Humidity (%)")
        if "RelativeHumidity_pct" in plot_df.columns:
            fig_rh = create_plotly_chart(
                plot_df, "RelativeHumidity_pct", ["#3B82F6"], central_metric=selected_metric,
                resolution_rules=resolution_rules
            )
            fig_rh.update_layout(height=320, margin=dict(b=20))
            st.plotly_chart(fig_rh, width="stretch", config=plotly_config)
            
    with r2_c2:
        st.markdown("###### Absolute Humidity (g/m³)")
        if "AbsoluteHumidity_g_m3" in plot_df.columns:
            fig_ah = create_plotly_chart(
                plot_df, "AbsoluteHumidity_g_m3", ["#8B5CF6"], central_metric=selected_metric,
                resolution_rules=resolution_rules
            )
            fig_ah.update_layout(height=320, margin=dict(b=20))
            st.plotly_chart(fig_ah, width="stretch", config=plotly_config)

# =====================================================================
# VIEW 2: DEVICE ANALYSIS
# =====================================================================
def device_analysis():
    ml_artifacts = load_ml_artifacts()
    survival_df = load_survival_data()
    xai_rules = load_xai_rules()
    device_options = get_device_options(ml_artifacts, survival_df)

    st.sidebar.subheader("Cell Selection")
    if not device_options:
        st.sidebar.warning("No devices found in the diagnostic datasets.")
        st.stop()
    target_device = st.sidebar.selectbox("Target Device", device_options)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Created by [{CREATOR_LINK.split('/')[-1]}]({CREATOR_LINK}).")
    st.markdown(f"### Unitary Diagnostics — Device **{target_device}**")
    st.caption("Detailed evaluation of electrical parameters and degradation indicators.")
    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        col_label, col_control = st.columns([1, 4], vertical_alignment="center")
        with col_label:
            st.markdown("** Time Window**")
        with col_control:
            st.segmented_control("Device Range", options=["Last Sweep", "Weekly", "Historical"], default="Last Sweep", label_visibility="collapsed")
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("State of Health (SoH)", "88.4 %", "-1.2%")
    col2.metric("Current PCE", "16.8 %", "-0.2 %")
    col3.metric("Anomalies (Isolation Forest)", "Normal", "Stable")
    col4.metric("Estimated Lifespan", "1,450 h", "-40 h")
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Efficiency (PCE)", "J-V Kinematics", "Prognostics (XGBoost)", "LLM Diagnostics"])
    with tab1:
        st.markdown("### Historical Evolution of Power Conversion Efficiency (PCE)")
        pce_cell = pd.DataFrame({"PCE": np.linspace(18.5, 14.0, 100) + np.random.normal(0, 0.2, 100)}, index=dates_index)
        fig_t1 = go.Figure(go.Scatter(x=pce_cell.index, y=pce_cell["PCE"], line=dict(color="#10B981")))
        fig_t1.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        st.plotly_chart(fig_t1, width="stretch", config=plotly_config)
    with tab2:
        st.markdown("### J-V Characteristic Curves (Reverse / Forward Scans)")
        voltage = np.linspace(-0.2, 1.2, 100)
        j_forward = 22 * (1 - np.exp(10 * (voltage - 1.05)))
        fig_t2 = go.Figure(go.Scatter(x=voltage, y=j_forward, line=dict(color="#F43F5E")))
        fig_t2.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        st.plotly_chart(fig_t2, width="stretch", config=plotly_config)
    with tab3:
        st.markdown("### Degradation Projection (XGBoost Engine with Monotonicity)")
        projection = np.linspace(0, 100, 50)
        damage = np.log1p(projection) * 20
        fig_t3 = go.Figure(go.Scatter(x=projection, y=damage, line=dict(color="#8B5CF6")))
        fig_t3.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
        st.plotly_chart(fig_t3, width="stretch", config=plotly_config)
    with tab4:
        st.markdown("### Thermodynamic Report & Explainability (XAI + LLM)")
        device_rules = xai_rules.get(target_device)
        if device_rules:
            threshold_day = device_rules.get("Threshold_15pct_Day")
            if threshold_day is not None:
                st.metric("15% Alert Crossing", f"Day {threshold_day:.2f}")
            st.markdown("**Features used by the short-term Digital Twin**")
            st.write(", ".join(device_rules.get("Features_Used", [])))
            st.markdown("**Deterministic surrogate-tree rules**")
            st.code("\n".join(device_rules.get("Extracted_Rules", [])), language="text")
            st.caption(
                "The alert is triggered when either the PCE power twin or the pFF structural twin detects "
                "underperformance. The rules above expose the exact thermodynamic conditions for this cell."
            )
        else:
            summary_table = ml_artifacts.get("summary_table", pd.DataFrame())
            is_defective = (
                isinstance(summary_table, pd.DataFrame)
                and target_device in summary_table.index
                and bool(summary_table.loc[target_device, "extrinsic_failure"])
            )
            if is_defective:
                st.warning("No surrogate-tree rules were exported for this defective cell.")
            else:
                st.info("No surrogate-tree anomaly rules were exported for this validated cell.")

# =====================================================================
# APP STARTUP
# =====================================================================
load_styles()
render_branding()
general_page = st.Page(general_overview, title="General Overview", icon=":material/dashboard:")
device_page = st.Page(device_analysis, title="Device Analysis", icon=":material/troubleshoot:")
pg = st.navigation({"ANALYSIS MODULES": [general_page, device_page]})
pg.run()