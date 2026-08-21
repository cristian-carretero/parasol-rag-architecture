import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title="ParaSol Dashboard")

# Cargar estilos CSS externos
try:
    with open('assets/style.css', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass  # Ignorar si no existe el archivo CSS en desarrollo


# =====================================================================
# FUNCIÓN DE CARGA Y CACHÉ DE DATOS METEOROLÓGICOS (AGREGADOS)
# =====================================================================
@st.cache_data
def cargar_datos_meteo():
    ruta_archivo = r"C:\Users\crica\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\parasol-rag-architecture-main\data\aggregated\outdoor\meteo_10min.parquet"
    try:
        df = pd.read_parquet(ruta_archivo)
        if df.index.name == 'Timestamp':
            df = df.reset_index()
            
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos meteorológicos agregados: {e}")
        return pd.DataFrame()


# =====================================================================
# FUNCIÓN AVANZADA DE PLOTLY (CON BANDAS BLINDADAS Y CONNECTGAPS)
# =====================================================================
def crear_grafico_plotly(df, y_cols, color_seq=None, metrica_central='mean', band_min_col=None, band_max_col=None):
    fig = go.Figure()

    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if color_seq is None:
        color_seq = ['#36B9CC', '#1E293B', '#F59E0B', '#3B82F6', '#10B981', '#8B5CF6', '#F43F5E', '#64748B']

    agg_func = 'median' if metrica_central == 'Mediana' else 'mean'

    def hex_to_rgba(hex_color, alpha=0.25):  
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r},{g},{b},{alpha})'

    has_bands = band_min_col and band_max_col and band_min_col in df.columns and band_max_col in df.columns

    # Resample completo conservando todas las columnas numéricas
    df_1d_main = df.resample('1D').agg(agg_func, numeric_only=True)
    df_1d_min  = df.resample('1D').min(numeric_only=True)
    df_1d_max  = df.resample('1D').max(numeric_only=True)

    df_7d_main = df.resample('7D').agg(agg_func, numeric_only=True)
    df_7d_min  = df.resample('7D').min(numeric_only=True)
    df_7d_max  = df.resample('7D').max(numeric_only=True)

    df_30d_main = df.resample('30D').agg(agg_func, numeric_only=True)
    df_30d_min  = df.resample('30D').min(numeric_only=True)
    df_30d_max  = df.resample('30D').max(numeric_only=True)

    for i, col in enumerate(y_cols):
        c = color_seq[i % len(color_seq)]
        c_fill = hex_to_rgba(c, 0.25)
        name = str(col)

        # Aplicar bandas solo a las columnas de temperatura del módulo
        use_band = has_bands and col in ['ModuleTemp_Mean_C', 'ModuleTemp_Median_C']
        b_min = band_min_col if use_band else col
        b_max = band_max_col if use_band else col

        # 1. Original (10min) - 3 trazas
        fig.add_trace(go.Scatter(x=df.index, y=df[b_min], mode='lines', line=dict(width=0), name=f"Min {name}", showlegend=False, visible=True, connectgaps=True))
        fig.add_trace(go.Scatter(x=df.index, y=df[b_max], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=True, connectgaps=True))
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name, line=dict(color=c, width=1.5), visible=True, connectgaps=True))

        # 2. 1D - 3 trazas
        fig.add_trace(go.Scatter(x=df_1d_min.index, y=df_1d_min[b_min], mode='lines', line=dict(width=0), name=f"Min {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_1d_max.index, y=df_1d_max[b_max], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_1d_main.index, y=df_1d_main[col], mode='lines+markers', name=f"{name} ({metrica_central} 1D)", line=dict(color=c, width=2.5), marker=dict(size=6), visible=False, connectgaps=True))

        # 3. 7D - 3 trazas
        fig.add_trace(go.Scatter(x=df_7d_min.index, y=df_7d_min[b_min], mode='lines', line=dict(width=0), name=f"Min {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_7d_max.index, y=df_7d_max[b_max], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_7d_main.index, y=df_7d_main[col], mode='lines+markers', name=f"{name} ({metrica_central} 7D)", line=dict(color=c, width=2.5), marker=dict(size=8), visible=False, connectgaps=True))

        # 4. 30D - 3 trazas
        fig.add_trace(go.Scatter(x=df_30d_min.index, y=df_30d_min[b_min], mode='lines', line=dict(width=0), name=f"Min {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_30d_max.index, y=df_30d_max[b_max], mode='lines', line=dict(width=0), fill='tonexty', fillcolor=c_fill, name=f"Max {name}", showlegend=False, visible=False, connectgaps=True))
        fig.add_trace(go.Scatter(x=df_30d_main.index, y=df_30d_main[col], mode='lines+markers', name=f"{name} ({metrica_central} 30D)", line=dict(color=c, width=2.5), marker=dict(size=10), visible=False, connectgaps=True))

    n = len(y_cols)
    vis_orig = ([True,  True,  True,  False, False, False, False, False, False, False, False, False]) * n
    vis_1d   = ([False, False, False, True,  True,  True,  False, False, False, False, False, False]) * n
    vis_7d   = ([False, False, False, False, False, False, True,  True,  True,  False, False, False]) * n
    vis_30d  = ([False, False, False, False, False, False, False, False, False, True,  True,  True ]) * n

    botones_agregados = [
        dict(label="Original", method="update", args=[{"visible": vis_orig}]),
        dict(label="1D", method="update", args=[{"visible": vis_1d}]),
        dict(label="7D", method="update", args=[{"visible": vis_7d}]),
        dict(label="30D", method="update", args=[{"visible": vis_30d}])
    ]

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=60), 
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            buttons=botones_agregados,
            showactive=True,
            x=1, xanchor="right",
            y=-0.15, yanchor="top",
            font=dict(size=11, color="#1E293B"),
            bgcolor="#F8FAFC",
            bordercolor="#E2E8F0"
        )]
    )

    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9')

    return fig


# =====================================================================
# 1. SIDEBAR: Logo centrado y navegación
# =====================================================================
st.sidebar.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; width: 100%; margin-top: -2rem; margin-bottom: 1rem; position: relative; z-index: 0;">
        <img src="https://www.emiliojuarez.es/imgs/logo-oss.jpg" width="110" style="margin-bottom: 0.5rem; border-radius: 4px;">
        <span style="font-weight: 700; font-size: 1.2rem; color: #1E293B; line-height: 1.2; letter-spacing: -0.02em;">ParaSol Dashboard</span>
        <span style="color: #64748B; font-size: 0.8rem; font-weight: 500; margin-top: 4px;">v1.0</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

modo_vista = st.sidebar.radio(
    "Navegación del Sistema",
    ("Vista General", "Análisis por Dispositivo")
)
st.sidebar.markdown("---")

if modo_vista == "Análisis por Dispositivo":
    st.sidebar.subheader("Selección de Celda")
    celda_objetivo = st.sidebar.selectbox(
        "Dispositivo Objetivo",
        ('M83', 'P12', 'A162', 'A164', 'A167', 'A170', 'ASLRX', 'M0')
    )
else:
    celda_objetivo = "M83"
    plot_height = 350
    st.sidebar.subheader("Opciones de KPIs (Telemetría)")
    tipo_irradiancia = st.sidebar.radio("Métrica de Irradiancia (KPI 1)", ["Media", "Pico"])
    tipo_temp_modulo = st.sidebar.radio("Métrica Temp. Módulo (KPI 3)", ["Media", "Mediana"])
    tipo_kpi4 = st.sidebar.radio("Métrica Secundaria (KPI 4)", ["Dosis Térmica", "Humedad Relativa"])

st.sidebar.markdown("""
---
Creado por [Cristian Carretero](https://linkedin.com/in/cristian-carretero-fernandez).
""")


# =====================================================================
# RENDERIZADO DEL PANEL PRINCIPAL
# =====================================================================
fechas = pd.date_range(start="2026-01-01", periods=100, freq="D")
plotly_config = {
    'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'lasso2d'],
    'displaylogo': False
}

if modo_vista == "Vista General":

    df_meteo = cargar_datos_meteo()

    if not df_meteo.empty:
        max_date = pd.to_datetime(df_meteo.index.max()).date()
        min_date = pd.to_datetime(df_meteo.index.min()).date()
    else:
        max_date = datetime.date.today()
        min_date = max_date - datetime.timedelta(days=30)
    
    # Por defecto mostrar todo el histórico
    fecha_inicio_default = min_date
    fecha_fin_default = max_date

    col_head, col_filtro = st.columns([2, 3])

    with col_head:
        st.markdown("### Estación ParaSol")
        st.caption("Monitoreo ambiental histórico y en tiempo real.")

    with col_filtro:
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        rango_temporal = st.radio(
            "Ventana Temporal",
            ("1D", "7D", "30D", "Histórico", "Fechas"),
            horizontal=True,
            label_visibility="collapsed"
        )

        if rango_temporal == "Fechas" and not df_meteo.empty:
            with st.container(key="rango_fechas_box"):
                col_d1, col_d2 = st.columns(2, gap="small")
                with col_d1:
                    fecha_inicio = st.date_input(
                        "Desde",
                        value=fecha_inicio_default,
                        label_visibility="collapsed"
                    )
                with col_d2:
                    fecha_fin = st.date_input(
                        "Hasta",
                        value=fecha_fin_default,
                        label_visibility="collapsed"
                    )

    st.markdown("---")

    # =====================================================================
    # LÓGICA MATEMÁTICA PARA FILTRADO Y KPIs DINÁMICOS
    # =====================================================================
    df_plot = pd.DataFrame()
    df_prev = pd.DataFrame()   
    
    if not df_meteo.empty:
        max_timestamp = df_meteo.index.max()
        
        # 1. Definición de ventanas temporales (Actual vs Previa)
        if rango_temporal == "1D":
            delta = pd.Timedelta(days=1)
            df_plot = df_meteo[df_meteo.index >= (max_timestamp - delta)]
            df_prev = df_meteo[(df_meteo.index >= (max_timestamp - 2*delta)) & (df_meteo.index < (max_timestamp - delta))]
        elif rango_temporal == "7D":
            delta = pd.Timedelta(days=7)
            df_plot = df_meteo[df_meteo.index >= (max_timestamp - delta)]
            df_prev = df_meteo[(df_meteo.index >= (max_timestamp - 2*delta)) & (df_meteo.index < (max_timestamp - delta))]
        elif rango_temporal == "30D":
            delta = pd.Timedelta(days=30)
            df_plot = df_meteo[df_meteo.index >= (max_timestamp - delta)]
            df_prev = df_meteo[(df_meteo.index >= (max_timestamp - 2*delta)) & (df_meteo.index < (max_timestamp - delta))]
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
                    st.error(f"❌ **Sin registros:** No se han encontrado datos disponibles para el periodo seleccionado: del **{fecha_inicio.strftime('%d/%m/%Y')}** al **{fecha_fin.strftime('%d/%m/%Y')}**.")
                    st.stop()
                
                # Si sí hay registros pero el usuario se salió por los extremos, lanzamos la advertencia informativa
                if fecha_inicio < min_date or fecha_fin > max_date:
                    st.warning(
                        f"⚠️ **Rango fuera de los límites del dataset:** "
                        f"Has seleccionado desde el **{fecha_inicio.strftime('%d/%m/%Y')}** hasta el **{fecha_fin.strftime('%d/%m/%Y')}**. "
                        f"Sin embargo, los datos reales disponibles en el sistema solo abarcan desde el "
                        f"**{min_date.strftime('%d/%m/%Y')}** hasta el **{max_date.strftime('%d/%m/%Y')}**. "
                        f"Se muestran únicamente los registros que coinciden dentro del periodo disponible."
                    )
                
                delta_fechas = end_dt - start_dt
                df_prev = df_meteo[(df_meteo.index >= (start_dt - delta_fechas)) & (df_meteo.index < start_dt)]
        
        df_plot = df_plot.dropna(how='all')

        # Control de seguridad general por si acaso
        if df_plot.empty:
            st.error(f"❌ **Sin registros:** No se han encontrado datos disponibles para la ventana temporal seleccionada ({rango_temporal}).")
            st.stop()

        # Procesar KPIs y gráficos únicamente si df_plot tiene datos válidos
        if 'POA_Irradiance_W_m2' in df_plot.columns:
            # Funciones auxiliares para formato y deltas
            def calcular_delta_pct(actual, previo):
                if pd.isna(actual) or pd.isna(previo) or previo == 0:
                    return None
                return ((actual - previo) / previo) * 100

            def format_kpi(valor, unidad, decimales=1):
                return f"{valor:.{decimales}f} {unidad}" if pd.notna(valor) else f"-- {unidad}"

            def format_delta(actual, previo):
                delta = calcular_delta_pct(actual, previo)
                return f"{delta:+.1f}%" if delta is not None else "N/A"

            has_prev = not df_prev.empty

            # KPI 1: Irradiancia
            if tipo_irradiancia == "Media":
                irr_act = df_plot['POA_Irradiance_W_m2'].mean()
                irr_prev = df_prev['POA_Irradiance_W_m2'].mean() if has_prev else np.nan
                kpi1_label = "Irradiancia POA Media"
            else:
                irr_act = df_plot['POA_Irradiance_W_m2'].max()
                irr_prev = df_prev['POA_Irradiance_W_m2'].max() if has_prev else np.nan
                kpi1_label = "Irradiancia POA Pico"
                
            kpi1_val = format_kpi(irr_act, "W/m²", 0)
            kpi1_delta = format_delta(irr_act, irr_prev)

            # KPI 2: Temperatura Ambiente
            t_amb_act = df_plot['AmbientTemp_C'].mean()
            t_amb_prev = df_prev['AmbientTemp_C'].mean() if has_prev else np.nan
            kpi2_val = format_kpi(t_amb_act, "°C", 1)
            kpi2_delta = format_delta(t_amb_act, t_amb_prev)

            # KPI 3: Temperatura Módulo (Media o Mediana según Sidebar)
            if tipo_temp_modulo == "Media":
                temp_col = 'ModuleTemp_Mean_C'
                kpi3_label = "Temp. Módulo (Media)"
            else:
                temp_col = 'ModuleTemp_Median_C'
                kpi3_label = "Temp. Módulo (Mediana)"

            t_mod_act = df_plot[temp_col].mean() 
            t_mod_prev = df_prev[temp_col].mean() if has_prev else np.nan
            kpi3_val = format_kpi(t_mod_act, "°C", 1)
            kpi3_delta = format_delta(t_mod_act, t_mod_prev)

            # KPI 4: Dosis / Humedad
            if tipo_kpi4 == "Dosis Térmica":
                kpi4_act = (df_plot['POA_Irradiance_W_m2'].sum() * (10/60)) / 1000
                kpi4_prev = (df_prev['POA_Irradiance_W_m2'].sum() * (10/60)) / 1000 if has_prev else np.nan
                kpi4_label = "Dosis Radiación Acum."
                kpi4_val = format_kpi(kpi4_act, "kWh/m²", 1)
                kpi4_delta = format_delta(kpi4_act, kpi4_prev)
            else:
                hum_act = df_plot['RelativeHumidity_pct'].mean()
                hum_prev = df_prev['RelativeHumidity_pct'].mean() if has_prev else np.nan
                kpi4_label = "Humedad Relativa Media"
                kpi4_val = format_kpi(hum_act, "%", 1)
                kpi4_delta = format_delta(hum_act, hum_prev)

            # Renderizado de KPIs
            st.markdown(f"####  Telemetría Ambiental ({rango_temporal})")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(kpi1_label, kpi1_val, kpi1_delta)
            col2.metric("Temp. Ambiente Media", kpi2_val, kpi2_delta)
            col3.metric(kpi3_label, kpi3_val, kpi3_delta)
            col4.metric(kpi4_label, kpi4_val, kpi4_delta)
            
            # =====================================================================
            # GRÁFICOS TEMPORALES
            # =====================================================================
            st.markdown("<br>", unsafe_allow_html=True)
            col_metrica, col_info = st.columns([1.5, 3.5])
            with col_metrica:
                metrica_seleccionada = st.selectbox(
                    "Tendencia Central (Agregados)", 
                    ["Media", "Mediana"], 
                    index=0,
                    help="La Media calcula el promedio. La Mediana ignora picos extremos o errores del sensor."
                )
            with col_info:
                st.info("💡 Haz clic en los botones **1D, 7D o 30D** bajo cada gráfica para ver la tendencia con su **Banda de Confianza (Max - Min)**.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab_irr, tab_temp, tab_hum = st.tabs([
                "Irradiancia POA", 
                "Temperaturas", 
                "Humedad Relativa"
            ])

            with tab_irr:
                st.markdown("##### Irradiancia POA (W/m²)")
                fig1 = crear_grafico_plotly(df_plot, 'POA_Irradiance_W_m2', ['#36B9CC'], metrica_central=metrica_seleccionada)
                st.plotly_chart(fig1, width='stretch', config=plotly_config)

            with tab_temp:
                st.markdown("##### Temperatura Ambiente vs Temperatura del Módulo (°C)")
                mod_temp_col = 'ModuleTemp_Mean_C' if tipo_temp_modulo == "Media" else 'ModuleTemp_Median_C'
                fig2 = crear_grafico_plotly(
                    df_plot, 
                    ['AmbientTemp_C', mod_temp_col], 
                    ['#1E293B', '#F59E0B'], 
                    metrica_central=metrica_seleccionada,
                    band_min_col='ModuleTemp_Min_C',
                    band_max_col='ModuleTemp_Max_C'
                )
                st.plotly_chart(fig2, width='stretch', config=plotly_config)

            with tab_hum:
                st.markdown("##### Humedad Relativa (%)")
                fig3 = crear_grafico_plotly(df_plot, 'RelativeHumidity_pct', ['#3B82F6'], metrica_central=metrica_seleccionada)
                st.plotly_chart(fig3, width='stretch', config=plotly_config)

    else:
        st.warning("No se han podido cargar los datos de meteo. Comprueba la ruta del archivo Parquet.")

    # =====================================================================
    # RENDIMIENTO GLOBAL DE PLANTA Y FLOTA
    # =====================================================================
    st.markdown("---")
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
        columns=['M83', 'P12', 'A162', 'A164', 'A167', 'A170', 'ASLRX', 'M0']
    )
    fig_flota = go.Figure()
    for col in df_flota.columns:
        fig_flota.add_trace(go.Scatter(x=df_flota.index, y=df_flota[col], mode='lines', name=col))
    fig_flota.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified")
    st.plotly_chart(fig_flota, width='stretch', config=plotly_config)

else:
    # VISTA DE DISPOSITIVOS
    col_head_disp, col_filtro_disp = st.columns([3, 2])

    with col_head_disp:
        st.markdown(f"### Diagnóstico Unitario — Dispositivo <b>{celda_objetivo}</b>", unsafe_allow_html=True)
        st.caption("Evaluación detallada de parámetros eléctricos e indicadores de degradación.")

    with col_filtro_disp:
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        rango_disp = st.radio(
            "Rango Dispositivo",
            ("Último Barrido", "Semanal", "Histórico"),
            horizontal=True,
            label_visibility="collapsed"
        )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estado de Salud (SoH)", "88.4 %", "-1.2%")
    col2.metric("pFF Actual", "0.68", "-0.02")
    col3.metric("Anomalías (Isolation Forest)", "Normal", "Estable")
    col4.metric("Vida Útil Estimada", "1,450 h", "-40 h")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Salud (pFF)",
        "Cinemática J-V",
        "Pronóstico (XGBoost)",
        "Diagnóstico LLM"
    ])

    with tab1:
        st.markdown('### Evolución Histórica del Pseudo Fill-Factor (pFF)')
        pff_celda = pd.DataFrame({'pFF': np.linspace(0.75, 0.55, 100) + np.random.normal(0, 0.02, 100)}, index=fechas)
        fig_t1 = go.Figure(go.Scatter(x=pff_celda.index, y=pff_celda['pFF'], line=dict(color='#10B981')))
        fig_t1.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t1, width='stretch', config=plotly_config)

    with tab2:
        st.markdown('### Curvas Características J-V (Escaneos Reversa / Directa)')
        voltaje = np.linspace(-0.2, 1.2, 100)
        j_directa = 22 * (1 - np.exp(10 * (voltaje - 1.05)))
        fig_t2 = go.Figure(go.Scatter(x=voltaje, y=j_directa, line=dict(color='#F43F5E')))
        fig_t2.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t2, width='stretch', config=plotly_config)

    with tab3:
        st.markdown('### Proyección de Degradación (Motor XGBoost con Monotonicidad)')
        proyeccion = np.linspace(0, 100, 50)
        dano = np.log1p(proyeccion) * 20
        fig_t3 = go.Figure(go.Scatter(x=proyeccion, y=dano, line=dict(color='#8B5CF6')))
        fig_t3.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_t3, width='stretch', config=plotly_config)

    with tab4:
        st.markdown('### Reporte Termodinámico y Explicabilidad (XAI + LLM)')
        st.info(f"Reporte de diagnóstico para la celda {celda_objetivo}: Los valores SHAP indican que la caída del pFF está altamente condicionada por la acumulación térmica en los picos de irradiancia en Zaragoza. Se aconseja monitorizar la histéresis en el próximo ciclo de barrido.")