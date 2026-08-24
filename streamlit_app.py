"""
ParaSol Dashboard
==================
Panel de monitorización meteorológica y de rendimiento fotovoltaico.
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# CONFIGURACIÓN DE PÁGINA (debe ser el primer comando de Streamlit)
# =====================================================================
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="ParaSol Dashboard"
)

LOGO_URL = "https://www.emiliojuarez.es/imgs/logo-oss.jpg"
CREATOR_LINK = "https://linkedin.com/in/cristian-carretero-fernandez"

# =====================================================================
# ESTILOS
# =====================================================================
def cargar_estilos() -> None:
    """Carga el CSS externo y lo inyecta en la aplicación."""
    try:
        with open("assets/style.css", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ No se ha encontrado el archivo 'assets/style.css'. Asegúrate de que existe.")

# =====================================================================
# CARGA Y CACHÉ DE DATOS METEOROLÓGICOS
# =====================================================================
@st.cache_data(show_spinner=False)
def cargar_datos_meteo() -> pd.DataFrame:
    ruta_archivo = (
        r"C:\Users\crica\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio"
        r"\parasol-rag-architecture-main\data\aggregated\outdoor\meteo_10min.parquet"
    )
    try:
        df = pd.read_parquet(ruta_archivo)
        if df.index.name == "Timestamp":
            df = df.reset_index()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.set_index("Timestamp", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos meteorológicos agregados: {e}")
        return pd.DataFrame()

# =====================================================================
# GRÁFICO PLOTLY AVANZADO
# =====================================================================
def crear_grafico_plotly(
    df, y_cols, color_seq=None, metrica_central="Media", band_min_col=None, band_max_col=None
):
    fig = go.Figure()
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    if color_seq is None:
        color_seq = ["#36B9CC", "#1E293B", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6", "#F43F5E", "#64748B"]

    agg_func = "median" if metrica_central == "Mediana" else "mean"
    label_stat = "Mediana" if metrica_central == "Mediana" else "Media"

    def hex_to_rgba(hex_color, alpha=0.25):
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    has_bands = band_min_col and band_max_col and band_min_col in df.columns and band_max_col in df.columns

    df_1d_main = df.resample("1D").agg(agg_func, numeric_only=True)
    df_1d_min = df.resample("1D").min(numeric_only=True)
    df_1d_max = df.resample("1D").max(numeric_only=True)
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

        fig.add_trace(go.Scatter(x=df.index, y=df[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", showlegend=False, visible=True, connectgaps=True))
        fig.add_trace(go.Scatter(x=df.index, y=df[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=True, connectgaps=True))
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name, line=dict(color=c, width=1.5), visible=True, connectgaps=True))

        fig.add_trace(go.Scatter(x=df_1d_min.index, y=df_1d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_1d_max.index, y=df_1d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_1d_main.index, y=df_1d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 1D)", line=dict(color=c, width=2.5), marker=dict(size=6), visible=False, connectgaps=True))

        fig.add_trace(go.Scatter(x=df_7d_min.index, y=df_7d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_7d_max.index, y=df_7d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_7d_main.index, y=df_7d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 7D)", line=dict(color=c, width=2.5), marker=dict(size=8), visible=False, connectgaps=True))

        fig.add_trace(go.Scatter(x=df_30d_min.index, y=df_30d_min[b_min], mode="lines", line=dict(width=0), name=f"Min {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_30d_max.index, y=df_30d_max[b_max], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_30d_main.index, y=df_30d_main[col], mode="lines+markers", name=f"{name} ({label_stat} 30D)", line=dict(color=c, width=2.5), marker=dict(size=10), visible=False, connectgaps=True))

    n = len(y_cols)
    vis_orig = ([True, True, True, False, False, False, False, False, False, False, False, False]) * n
    vis_1d = ([False, False, False, True, True, True, False, False, False, False, False, False]) * n
    vis_7d = ([False, False, False, False, False, False, True, True, True, False, False, False]) * n
    vis_30d = ([False, False, False, False, False, False, False, False, False, True, True, True]) * n

    botones_agregados = [
        dict(label="Nativo (10 min)", method="update", args=[{"visible": vis_orig}]),
        dict(label=f"{label_stat} Diaria (1D)", method="update", args=[{"visible": vis_1d}]),
        dict(label=f"{label_stat} Semanal (7D)", method="update", args=[{"visible": vis_7d}]),
        dict(label=f"{label_stat} Mensual (30D)", method="update", args=[{"visible": vis_30d}]),
    ]

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[dict(type="buttons", direction="right", buttons=botones_agregados, showactive=True, x=1, xanchor="right", y=-0.15, yanchor="top", font=dict(size=11, color="#1E293B"), bgcolor="#F8FAFC", bordercolor="#E2E8F0")],
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
    return fig

# =====================================================================
# UTILIDADES DE KPI
# =====================================================================
def calcular_delta_pct(actual, previo):
    if pd.isna(actual) or pd.isna(previo) or previo == 0:
        return None
    return ((actual - previo) / previo) * 100

def format_kpi(valor, unidad, decimales=1):
    return f"{valor:.{decimales}f} {unidad}" if pd.notna(valor) else f"-- {unidad}"

def format_delta(actual, previo):
    delta = calcular_delta_pct(actual, previo)
    return f"{delta:+.1f}%" if delta is not None else "N/A"

# =====================================================================
# BARRA DE HERRAMIENTAS: ventana temporal
# =====================================================================
def render_toolbar(min_date: datetime.date, max_date: datetime.date):
    with st.container(border=True):
        col_label, col_control = st.columns([1, 4], vertical_alignment="center")
        with col_label:
            st.markdown("**Ventana temporal**")
        with col_control:
            rango_temporal = st.segmented_control(
                "Ventana temporal",
                options=["1D", "7D", "30D", "Histórico", "Fechas"],
                default="1D", 
                label_visibility="collapsed",
            )
        rango_temporal = rango_temporal or "1D" 

        fecha_inicio, fecha_fin = min_date, max_date
        if rango_temporal == "Fechas":
            col_d1, col_d2, _ = st.columns([1, 1, 2])
            with col_d1:
                fecha_inicio = st.date_input("Desde", value=min_date)
            with col_d2:
                fecha_fin = st.date_input("Hasta", value=max_date)

    return rango_temporal, fecha_inicio, fecha_fin

# =====================================================================
# SIDEBAR GLOBAL: logo
# =====================================================================
def render_branding() -> None:
    st.logo(LOGO_URL, size="large", link=LOGO_URL)

fechas = pd.date_range(start="2026-01-01", periods=100, freq="D")
plotly_config = {
    "modeBarButtonsToRemove": ["autoScale2d", "select2d", "lasso2d"],
    "displaylogo": False,
}

# =====================================================================
# VISTA 1: VISTA GENERAL
# =====================================================================
def vista_general():
    st.sidebar.markdown(
        "<h4 style='font-size: 1.1rem; color: #1E293B; margin-bottom: 0;'>Opciones de KPIs</h4>", 
        unsafe_allow_html=True
    )

    tipo_irradiancia = st.sidebar.radio("Irradiancia (KPI 1)", ["Media", "Pico"])
    tipo_temp_modulo = st.sidebar.radio("Temp. Módulo (KPI 3)", ["Media", "Mediana"])
    tipo_kpi4 = st.sidebar.radio("Métrica Secundaria (KPI 4)", ["Dosis Térmica", "Humedad Relativa"])

    st.sidebar.divider() 

    st.sidebar.markdown(
        "<h4 style='font-size: 1.1rem; color: #1E293B; margin-bottom: 0;'>Agregaciones de Gráficas</h4>", 
        unsafe_allow_html=True
    )
    metrica_seleccionada = st.sidebar.radio(
        "Tendencia Central",
        ["Media", "Mediana"],
        index=0,
        help="Métrica estadística aplicada en los remuestreos temporales de las gráficas.",
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        f"""
        <div style='text-align: center; font-size: 0.85rem; color: #64748B; margin-top: 1rem;'>
            Desarrollado por <br>
            <a href='{CREATOR_LINK}' target='_blank' style='color: #36B9CC; text-decoration: none; font-weight: 600;'>
                Cristian Carretero
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_meteo = cargar_datos_meteo()
    if not df_meteo.empty:
        max_date = pd.to_datetime(df_meteo.index.max()).date()
        min_date = pd.to_datetime(df_meteo.index.min()).date()
    else:
        max_date = datetime.date.today()
        min_date = max_date - datetime.timedelta(days=30)

    st.markdown("### Estación ParaSol")
    st.caption("Monitoreo ambiental histórico y en tiempo real.")
    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)

    rango_temporal, fecha_inicio, fecha_fin = render_toolbar(min_date, max_date)
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    df_plot, df_prev = pd.DataFrame(), pd.DataFrame()

    if not df_meteo.empty:
        max_timestamp = df_meteo.index.max()

        if rango_temporal in ("1D", "7D", "30D"):
            dias = {"1D": 1, "7D": 7, "30D": 30}[rango_temporal]
            delta = pd.Timedelta(days=dias)
            df_plot = df_meteo[df_meteo.index >= (max_timestamp - delta)]
            df_prev = df_meteo[(df_meteo.index >= (max_timestamp - 2 * delta)) & (df_meteo.index < (max_timestamp - delta))]
        
        elif rango_temporal == "Histórico":
            df_plot = df_meteo.copy()
            df_prev = pd.DataFrame()
            
        elif rango_temporal == "Fechas":
            if fecha_inicio > fecha_fin:
                st.error("❌ **Error de selección:** La fecha de inicio no puede ser posterior a la fecha de fin.")
                st.stop()
            else:
                start_dt = pd.to_datetime(fecha_inicio)
                end_dt = pd.to_datetime(fecha_fin) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                
                # Filtramos primero para comprobar si hay registros coincidentes
                df_plot = df_meteo[(df_meteo.index >= start_dt) & (df_meteo.index <= end_dt)]
                
                # Si no hay ningún registro coincidente, lanzamos error rojo y detenemos antes del warning de límites
                if df_plot.empty:
                    st.error(f"❌ **Sin registros:** No se han encontrado datos disponibles para el periodo seleccionado. Los datos disponibles en el sistema solo abarcan desde el "
                        f"**{min_date.strftime('%d/%m/%Y')}** hasta el **{max_date.strftime('%d/%m/%Y')}**.")
                    st.stop()
                
                # Si sí hay registros pero el usuario se salió por los extremos, lanzamos la advertencia informativa
                if fecha_inicio < min_date or fecha_fin > max_date:
                    st.warning(
                        f"⚠️ **Rango fuera de los límites del dataset:** "
                        f"Has seleccionado desde el **{fecha_inicio.strftime('%d/%m/%Y')}** hasta el **{fecha_fin.strftime('%d/%m/%Y')}**. "
                        f"Sin embargo, los datos disponibles en el sistema solo abarcan desde el "
                        f"**{min_date.strftime('%d/%m/%Y')}** hasta el **{max_date.strftime('%d/%m/%Y')}**. "
                        f"Se muestran únicamente los registros que coinciden dentro del periodo disponible."
                    )
                
                delta_fechas = end_dt - start_dt
                df_prev = df_meteo[(df_meteo.index >= (start_dt - delta_fechas)) & (df_meteo.index < start_dt)]

        df_plot = df_plot.dropna(how="all")
        if df_plot.empty:
            st.error(f"❌ **Sin registros:** no hay datos disponibles para la ventana temporal seleccionada ({rango_temporal}).")
            st.stop()

        if "POA_Irradiance_W_m2" in df_plot.columns:
            has_prev = not df_prev.empty
            if tipo_irradiancia == "Media":
                irr_act, kpi1_label = df_plot["POA_Irradiance_W_m2"].mean(), "Irradiancia POA Media"
                irr_prev = df_prev["POA_Irradiance_W_m2"].mean() if has_prev else np.nan
            else:
                irr_act, kpi1_label = df_plot["POA_Irradiance_W_m2"].max(), "Irradiancia POA Pico"
                irr_prev = df_prev["POA_Irradiance_W_m2"].max() if has_prev else np.nan
            kpi1_val, kpi1_delta = format_kpi(irr_act, "W/m²", 0), format_delta(irr_act, irr_prev)

            t_amb_act = df_plot["AmbientTemp_C"].mean()
            t_amb_prev = df_prev["AmbientTemp_C"].mean() if has_prev else np.nan
            kpi2_val, kpi2_delta = format_kpi(t_amb_act, "°C", 1), format_delta(t_amb_act, t_amb_prev)

            temp_col = "ModuleTemp_Mean_C" if tipo_temp_modulo == "Media" else "ModuleTemp_Median_C"
            kpi3_label = f"Temp. Módulo ({tipo_temp_modulo})"
            t_mod_act = df_plot[temp_col].mean()
            t_mod_prev = df_prev[temp_col].mean() if has_prev else np.nan
            kpi3_val, kpi3_delta = format_kpi(t_mod_act, "°C", 1), format_delta(t_mod_act, t_mod_prev)

            if tipo_kpi4 == "Dosis Térmica":
                kpi4_act = (df_plot["POA_Irradiance_W_m2"].sum() * (10 / 60)) / 1000
                kpi4_prev = (df_prev["POA_Irradiance_W_m2"].sum() * (10 / 60)) / 1000 if has_prev else np.nan
                kpi4_label, kpi4_val = "Dosis Radiación Acum.", format_kpi(kpi4_act, "kWh/m²", 1)
                kpi4_delta = format_delta(kpi4_act, kpi4_prev)
            else:
                hum_act = df_plot["RelativeHumidity_pct"].mean()
                hum_prev = df_prev["RelativeHumidity_pct"].mean() if has_prev else np.nan
                kpi4_label, kpi4_val = "Humedad Relativa Media", format_kpi(hum_act, "%", 1)
                kpi4_delta = format_delta(hum_act, hum_prev)

            st.markdown(f"#### Telemetría Ambiental ({rango_temporal})")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(kpi1_label, kpi1_val, kpi1_delta)
            col2.metric("Temp. Ambiente Media", kpi2_val, kpi2_delta)
            col3.metric(kpi3_label, kpi3_val, kpi3_delta)
            col4.metric(kpi4_label, kpi4_val, kpi4_delta)

            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

            tab_irr, tab_temp, tab_hum = st.tabs(["Irradiancia POA", "Temperaturas", "Humedad Relativa"])

            with tab_irr:
                st.markdown("##### Irradiancia POA (W/m²)")
                fig1 = crear_grafico_plotly(df_plot, "POA_Irradiance_W_m2", ["#36B9CC"], metrica_central=metrica_seleccionada)
                st.plotly_chart(fig1, width="stretch", config=plotly_config)

            with tab_temp:
                st.markdown("##### Temperatura Ambiente vs Temperatura del Módulo (°C)")
                mod_temp_col = "ModuleTemp_Mean_C" if tipo_temp_modulo == "Media" else "ModuleTemp_Median_C"
                fig2 = crear_grafico_plotly(
                    df_plot,
                    ["AmbientTemp_C", mod_temp_col],
                    ["#1E293B", "#F59E0B"],
                    metrica_central=metrica_seleccionada,
                    band_min_col="ModuleTemp_Min_C",
                    band_max_col="ModuleTemp_Max_C"
                )
                st.plotly_chart(fig2, width="stretch", config=plotly_config)

            with tab_hum:
                st.markdown("##### Humedad Relativa (%)")
                fig3 = crear_grafico_plotly(df_plot, "RelativeHumidity_pct", ["#3B82F6"], metrica_central=metrica_seleccionada)
                st.plotly_chart(fig3, width="stretch", config=plotly_config)
    else:
        st.warning("No se han podido cargar los datos de meteo. Comprueba la ruta del archivo Parquet.")

    st.markdown(f"#### Rendimiento Global ({rango_temporal})")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Potencia Máx. Flota", "4.2 kW", "+0.1 kW")
    col_p2.metric("Energía Total", "842 kWh", "+5.2%")
    col_p3.metric("PCE Medio", "18.4 %", "-0.2%")
    col_p4.metric("Celdas Activas", "8 / 8", "Operativas")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Evolución Comparativa de la Flota (Pseudo Fill-Factor)")
    st.markdown("Superposición de las 8 celdas bajo las mismas condiciones de estrés exterior.")

    np.random.seed(42)
    df_flota = pd.DataFrame(
        np.linspace(0.75, 0.55, 100).reshape(-1, 1) + np.random.normal(0, 0.015, size=(100, 8)),
        index=fechas,
        columns=["M83", "P12", "A162", "A164", "A167", "A170", "ASLRX", "M0"]
    )
    fig_flota = go.Figure()
    for col in df_flota.columns:
        fig_flota.add_trace(go.Scatter(x=df_flota.index, y=df_flota[col], mode="lines", name=col))
    fig_flota.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    st.plotly_chart(fig_flota, width="stretch", config=plotly_config)

# =====================================================================
# VISTA 2: ANÁLISIS POR DISPOSITIVO
# =====================================================================
def analisis_dispositivo():
    st.sidebar.subheader("Selección de Celda")
    celda_objetivo = st.sidebar.selectbox("Dispositivo Objetivo", ("M83", "P12", "A162", "A164", "A167", "A170", "ASLRX", "M0"))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Creado por [Cristian Carretero]({CREATOR_LINK}).")

    st.markdown(f"### Diagnóstico Unitario — Dispositivo **{celda_objetivo}**")
    st.caption("Evaluación detallada de parámetros eléctricos e indicadores de degradación.")
    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        col_label, col_control = st.columns([1, 4], vertical_alignment="center")
        with col_label:
            st.markdown("** Ventana temporal**")
        with col_control:
            st.segmented_control("Rango dispositivo", options=["Último Barrido", "Semanal", "Histórico"], default="Último Barrido", label_visibility="collapsed")
    
    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estado de Salud (SoH)", "88.4 %", "-1.2%")
    col2.metric("pFF Actual", "0.68", "-0.02")
    col3.metric("Anomalías (Isolation Forest)", "Normal", "Estable")
    col4.metric("Vida Útil Estimada", "1,450 h", "-40 h")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Salud (pFF)", "Cinemática J-V", "Pronóstico (XGBoost)", "Diagnóstico LLM"])

    with tab1:
        st.markdown("### Evolución Histórica del Pseudo Fill-Factor (pFF)")
        pff_celda = pd.DataFrame({"pFF": np.linspace(0.75, 0.55, 100) + np.random.normal(0, 0.02, 100)}, index=fechas)
        fig_t1 = go.Figure(go.Scatter(x=pff_celda.index, y=pff_celda["pFF"], line=dict(color="#10B981")))
        fig_t1.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_t1, width="stretch", config=plotly_config)

    with tab2:
        st.markdown("### Curvas Características J-V (Escaneos Reversa / Directa)")
        voltaje = np.linspace(-0.2, 1.2, 100)
        j_directa = 22 * (1 - np.exp(10 * (voltaje - 1.05)))
        fig_t2 = go.Figure(go.Scatter(x=voltaje, y=j_directa, line=dict(color="#F43F5E")))
        fig_t2.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_t2, width="stretch", config=plotly_config)

    with tab3:
        st.markdown("### Proyección de Degradación (Motor XGBoost con Monotonicidad)")
        proyeccion = np.linspace(0, 100, 50)
        dano = np.log1p(proyeccion) * 20
        fig_t3 = go.Figure(go.Scatter(x=proyeccion, y=dano, line=dict(color="#8B5CF6")))
        fig_t3.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_t3, width="stretch", config=plotly_config)

    with tab4:
        st.markdown("### Reporte Termodinámico y Explicabilidad (XAI + LLM)")
        st.info(f"Reporte de diagnóstico para la celda {celda_objetivo}: los valores SHAP indican que la caída del pFF está altamente condicionada por la acumulación térmica en los picos de irradiancia en Zaragoza. Se aconseja monitorizar la histéresis en el próximo ciclo de barrido.")

# =====================================================================
# ARRANQUE DE LA APP
# =====================================================================
cargar_estilos()
render_branding()

pagina_general = st.Page(vista_general, title="Vista General", icon=":material/dashboard:")
pagina_dispositivo = st.Page(analisis_dispositivo, title="Análisis por Dispositivo", icon=":material/troubleshoot:")

pg = st.navigation({"MÓDULOS DE ANÁLISIS": [pagina_general, pagina_dispositivo]})
pg.run()