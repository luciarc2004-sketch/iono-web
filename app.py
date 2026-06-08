import streamlit as st
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim
import matplotlib.dates as mdates

# =====================================================================
# CONFIGURACIÓN PROFESIONAL DE LA PLATAFORMA WEB
# =====================================================================
st.set_page_config(
    page_title="Iono-Explorer Pro GNSS",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# INTERFAZ FORMAL: Deep Space Command (Negro Absoluto y Azul)
st.markdown("""
    <style>
    /* Fondo de la página principal y textos */
    .main { background-color: #000000; color: #cbd5e1; }
    
    /* Títulos en Azul Aeroespacial */
    .stHeading h1, .stHeading h2, .stHeading h3 { color: #3b82f6 !important; }
    
    /* Barra lateral estilizada en gris oscuro/negro */
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1e3a8a; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    
    /* Métricas (Números en Cian de Alta Visibilidad) */
    div[data-testid="stMetricValue"] { color: #06b6d4 !important; font-family: monospace; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    
    /* Personalización de Botones de Comando */
    .stButton>button {
        background-color: #1e3a8a;
        color: #ffffff;
        border-radius: 6px;
        border: 1px solid #3b82f6;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        border-color: #60a5fa;
        color: #ffffff;
    }
    
    /* Alertas y avisos */
    .stAlert { background-color: #0b1329 !important; color: #60a5fa !important; border: 1px solid #1e3a8a !important; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# MOTOR CORE DE DATOS CON COMPROBACIÓN ESTRICTA ANTI-DATOS FALSOS
# =====================================================================
def generar_enlace_dlr(fecha):
    str_anio = fecha.strftime("%Y")
    str_doy = fecha.strftime("%j")
    str_hora = fecha.strftime("%H")
    f_inicio = fecha - datetime.timedelta(minutes=4, seconds=30)
    ts_inicio = f_inicio.strftime("%Y-%m-%dT%H-%M-%S")
    ts_fin = fecha.strftime("%Y-%m-%dT%H-%M-%S")
    
    base = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
    return f"{base}/{str_anio}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{ts_inicio}_{ts_fin}_{str_doy}_D.json"

@st.cache_data(show_spinner=False, ttl=3600)
def descargar_json_dlr(fecha):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        f_intento = fecha.replace(minute=m)
        url = generar_enlace_dlr(f_intento)
        try:
            r = requests.get(url, headers=headers, timeout=4)
            if r.status_code == 200:
                data = r.json()
                # COMPROBACIÓN CRÍTICA: Validar que el archivo contenga datos reales del DLR
                if 'data' in data and 'grid' in data['data'] and 'features' in data['data']['grid']:
                    vtec_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    
                    # Filtro anti-mallas falsas (Malla Europa requiere 43x81 = 3483 puntos exactos)
                    if len(vtec_list) == 3483: 
                        return np.array(vtec_list).reshape(43, 81), f_intento
        except: 
            continue
    return None, None

# =====================================================================
# INTERFAZ DE USUARIO (SIDEBAR DE CONTROL)
# =====================================================================
st.sidebar.title("🛰️ Control Operacional")
st.sidebar.markdown("---")

version_seleccionada = st.sidebar.selectbox(
    "Selecciona la Herramienta",
    ["1. Versión por Horas (Mapas 24h)", "2. Versión Matemática (Predicción vs Realidad)", "3. Versión por Constelaciones (Error Local)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Parámetros Geográficos")
ciudad_user = st.sidebar.text_input("Introduce Localidad", "Madrid")

lats_vector = np.arange(30, 73, 1)
lons_vector = np.arange(-30, 51, 1)
grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

geolocator = Nominatim(user_agent="iono_explorer_pro_command_v4")
try:
    location = geolocator.geocode(ciudad_user, timeout=10)
except:
    location = None

if location:
    lat_idx = (np.abs(lats_vector - location.latitude)).argmin()
    lon_idx = (np.abs(lons_vector - location.longitude)).argmin()
    st.sidebar.success(f"Malla: Lat {lats_vector[lat_idx]}°N | Lon {lons_vector[lon_idx]}°E")
else:
    # Coordenadas de respaldo para asegurar continuidad operacional (Madrid)
    lat_idx = (np.abs(lats_vector - 40.4167)).argmin()
    lon_idx = (np.abs(lons_vector - -3.7037)).argmin()
    st.sidebar.warning("⚠️ Modo seguro: Forzando coordenadas de respaldo (Madrid).")
    location = True 

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Parámetros Temporales")
fecha_user = st.sidebar.date_input("Fecha Base de Estudio", datetime.date(2026, 1, 24))

FREQS_GNSS = {"GPS": 1575.42 * 1e6, "Galileo": 1575.42 * 1e6, "GLONASS": 1602.00 * 1e6, "BeiDou": 1561.10 * 1e6}

# =====================================================================
# ÁREA PRINCIPAL DE PRESENTACIÓN DE DATOS
# =====================================================================
st.title("🖥️ Iono-Explorer Pro: Terminal de Auditoría Atmosférica")
st.markdown("Plataforma analítica de refracción ionosférica para sistemas globales de navegación.")

if not location:
    st.warning("⚠️ Introduce una ciudad válida en la barra lateral para inicializar el sistema.")
else:
    # -----------------------------------------------------------------
    # HERRAMIENTA 1: VERSIÓN POR HORAS (MAPAS DE RETRASO EN METROS)
    # -----------------------------------------------------------------
    if version_seleccionada == "1. Versión por Horas (Mapas 24h)":
        st.header("🗺️ Distribución Espacial del Retraso de Grupo (Europa)")
        hora_mapa = st.slider("Hora de Observación (UTC)", 0, 23, 13)
        constelacion_select = st.selectbox("Evaluar Mapa en Metros para la señal de:", ["GPS", "Galileo", "GLONASS", "BeiDou"])
        
        f_c = FREQS_GNSS[constelacion_select]
        factor_m = (40.3 / (f_c ** 2)) * 1e16
        
        with st.spinner("Descargando matriz y proyectando espacio métrico..."):
            fecha_h = datetime.datetime.combine(fecha_user, datetime.time(hora_mapa, 0))
            matriz_tecu, _ = descargar_json_dlr(fecha_h)
            
            if matriz_tecu is not None:
                matriz_metros = matriz_tecu * factor_m
                v_min = max(0.0, float(np.floor(np.min(matriz_metros) - 0.5)))
                v_max = float(np.ceil(np.max(matriz_metros) + 0.5))
                
                try:
                    # Configuración de mapa estilo panel oscuro de control
                    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100, facecolor='#000000')
                    ax.set_facecolor('#000000')
                    ax.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    
                    ax.add_feature(cfeature.LAND, facecolor='#0a0a0a', zorder=1)
                    ax.add_feature(cfeature.OCEAN, facecolor='#020205', zorder=1)
                    ax.add_feature(cfeature.COASTLINE, edgecolor='#1e3a8a', linewidth=0.8, zorder=3)
                    
                    gl = ax.gridlines(draw_labels=True, color='#1e3a8a', alpha=0.15, linestyle='--')
                    gl.top_labels, gl.right_labels = False, False
                    gl.xlabel_style, gl.ylabel_style = {'color': '#3b82f6'}, {'color': '#3b82f6'}
                    
                    mesh = ax.pcolormesh(grid_lon, grid_lat, matriz_metros, transform=ccrs.PlateCarree(),
                                         cmap='jet', alpha=0.8, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
                    
                    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.03)
                    cbar.set_label(f"Retraso de la Onda (Metros)", color='#3b82f6', weight='bold')
                    cbar.ax.tick_params(labelscolor='#3b82f6')
                    
                    st.pyplot(fig)
                except Exception:
                    # Modo seguro en negro absoluto si Cartopy no compila en el host
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#000000')
                    ax.set_facecolor('#000000')
                    mesh = ax.pcolormesh(grid_lon, grid_lat, matriz_metros, cmap='jet', shading='gouraud', vmin=v_min, vmax=v_max)
                    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7)
                    cbar.set_label("Retraso (Metros)", color='#3b82f6')
                    cbar.ax.tick_params(labelscolor='#3b82f6')
                    ax.tick_params(colors='#3b82f6')
                    st.pyplot(fig)
                
                c1, c2 = st.columns(2)
                c1.metric(f"Máximo Retraso en Europa ({constelacion_select})", f"{np.max(matriz_metros):.2f} metros")
                c2.metric(f"Impacto Estimado en tu Localidad", f"{matriz_metros[lat_idx, lon_idx]:.2f} metros")
            else:
                st.error("Enlaces del DLR no disponibles o corruptos para este bloque horario.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 2: VERSIÓN MATEMÁTICA (PREDICCIÓN REAL VS REALIDAD)
    # -----------------------------------------------------------------
    elif version_seleccionada == "2. Versión Matemática (Predicción vs Realidad)":
        st.header(f"📈 Auditoría de Tendencia y Comprobación de Enlaces")
        rango_dias = st.slider("Ventana de días para el entrenamiento de la serie", 5, 15, 7)
        st.info("El sistema entrenará las ecuaciones de persistencia estacional y contrastará los resultados abriendo los links reales de las siguientes 6 horas.")
        
        if st.button("🧠 Iniciar Auditoría de Enlaces"):
            with st.spinner("Ejecutando escáner de red bihorario..."):
                cronologia_tecu = []
                fechas_list = []
                
                total_pasos = rango_dias * 12
                barra_progreso = st.progress(0)
                
                for i in range(total_pasos):
                    f_calc = datetime.datetime.combine(fecha_user, datetime.time(0,0)) - datetime.timedelta(hours=(total_pasos-i)*2)
                    m_tecu, _ = descargar_json_dlr(f_calc)
                    if m_tecu is not None:
                        cronologia_tecu.append(m_tecu[lat_idx, lon_idx])
                        fechas_list.append(f_calc)
                    barra_progreso.progress((i+1) / total_pasos)
                
                if len(cronologia_tecu) > 24:
                    vector_serie = np.array(cronologia_tecu)
                    
                    # Modelo Matemático Autorregresivo Estacional
                    periodo = 12
                    perfil_estacional = np.zeros(periodo)
                    for i in range(periodo):
                        perfil_estacional[i] = np.mean(vector_serie[i::periodo])
                        
                    ultimo_val = vector_serie[-1]
                    ultimo_slot = (len(vector_serie) - 1) % periodo
                    anomalia = ultimo_val - perfil_estacional[ultimo_slot]
                    
                    predicciones_futuras = []
                    fechas_futuras = []
                    alpha = 0.85
                    for k in range(1, 4):
                        slot_futuro = (ultimo_slot + k) % periodo
                        val_pred = perfil_estacional[slot_futuro] + anomalia * (alpha ** k)
                        predicciones_futuras.append(val_pred)
                        fechas_futuras.append(fechas_list[-1] + datetime.timedelta(hours=k*2))
                    
                    # DESCARGA COMPLEMENTARIA: Extraer curvas reales para romper datos falsos
                    realidad_futura = []
                    fechas_reales_futuras = []
                    
                    for f_fut in fechas_futuras:
                        m_real, _ = descargar_json_dlr(f_fut)
                        if m_real is not None:
                            realidad_futura.append(m_real[lat_idx, lon_idx])
                            fechas_reales_futuras.append(f_fut)
                    
                    # Construcción del Lienzo Formal (Negro y Azul)
                    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='#000000')
                    ax.set_facecolor('#000000')
                    
                    # Curvas en gama de azules y blanco tecnológico
                    ax.plot(fechas_list[-12:], vector_serie[-12:], color='#60a5fa', linewidth=2, label="Historial Real Verificado", marker='o')
                    ax.plot(fechas_futuras, predicciones_futuras, color='#2563eb', linewidth=2.5, linestyle='--', label="Algoritmo Matemático (Modelo)", marker='x')
                    
                    if realidad_futura:
                        ax.plot(fechas_reales_futuras, realidad_futura, color='#ffffff', linewidth=2.5, label="Validación Real (Datos del Servidor)", marker='s')
                        
                    # Configuración estricta del marco geométrico del gráfico
                    todos_los_valores = list(vector_serie[-12:]) + list(predicciones_futuras) + list(realidad_futura)
                    y_min = max(0.0, float(np.floor(min(todos_los_valores) - 2)))
                    y_max = float(np.ceil(max(todos_los_valores) + 2))
                    ax.set_ylim(y_min, y_max)
                    
                    # Configuración de colores de rejilla y fuentes de los ejes
                    ax.tick_params(colors='#3b82f6', labelsize=9)
                    ax.xaxis.label.set_color('#3b82f6')
                    ax.yaxis.label.set_color('#3b82f6')
                    ax.spines['bottom'].set_color('#1e3a8a')
                    ax.spines['left'].set_color('#1e3a8a')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, linestyle='--', alpha=0.1, color='#3b82f6')
                    ax.set_ylabel("Densidad de Electrones (TECU)", weight='bold')
                    
                    legend = ax.legend(facecolor='#000000', edgecolor='#1e3a8a')
                    for text in legend.get_texts():
                        text.set_color('#60a5fa')
                        
                    st.pyplot(fig)
                    
                    if len(realidad_futura) == len(predicciones_futuras):
                        error_medio = np.mean(np.abs(np.array(realidad_futura) - np.array(predicciones_futuras)))
                        st.success(f"🎯 Análisis de desviación cerrado. El error absoluto medio del algoritmo fue de: **{error_medio:.3f} TECU**.")
                    else:
                        st.warning("⚠️ Operación parcial: Los últimos enlaces del futuro en tiempo real aún no han sido liberados por el DLR.")
                else:
                    st.error("Fallo de red: No se detectaron suficientes mallas reales válidas para computar la serie.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 3: VERSIÓN POR CONSTELACIONES (ESPECTRO MULTI-FRECUENCIA)
    # -----------------------------------------------------------------
    elif version_seleccionada == "3. Versión por Constelaciones (Error)":
        st.header(f"📡 Desviación Métrica de Pseudodistancia en {ciudad_user}")
        st.markdown("Comparación absoluta directa del error inducido en las bandas portadoras primarias.")
        
        with st.spinner("Calculando desfasajes bihorarios..."):
            perfiles_tecu_24h = []
            horas_validas = []
            
            for h in range(0, 24, 2):
                fecha_h = datetime.datetime.combine(fecha_user, datetime.time(h, 0))
                m_tecu, _ = descargar_json_dlr(fecha_h)
                if m_tecu is not None:
                    perfiles_tecu_24h.append(m_tecu[lat_idx, lon_idx])
                    horas_validas.append(f"{h:02d}:00")
            
            if perfiles_tecu_24h:
                vector_tecu = np.array(perfiles_tecu_24h)
                
                fig, ax = plt.subplots(figsize=(11, 5), facecolor='#000000')
                ax.set_facecolor('#000000')
                
                # Paleta técnica de alto contraste sobre fondo negro
                colores = {"GLONASS": "#3b82f6", "GPS": "#06b6d4", "Galileo": "#ffffff", "BeiDou": "#ef4444"}
                estilos = {"GLONASS": "-", "GPS": "-", "Galileo": "--", "BeiDou": "-"}
                
                todos_los_metros = []
                for name_const, f_const in FREQS_GNSS.items():
                    factor = (40.3 / (f_const ** 2)) * 1e16
                    metros_error = vector_tecu * factor
                    todos_los_metros.extend(metros_error)
                    
                    ax.plot(horas_validas, metros_error, 
                            color=colores[name_const], linestyle=estilos[name_const],
                            linewidth=2.5 if name_const == "Galileo" else 2,
                            label=f"{name_const} ({f_const/1e6:.1f} MHz)", marker='o')
                
                ax.set_ylim(max(0.0, float(np.floor(min(todos_los_metros) - 0.5))), float(np.ceil(max(todos_los_metros) + 0.5)))
                
                ax.tick_params(colors='#3b82f6', labelsize=9)
                ax.xaxis.label.set_color('#3b82f6')
                ax.yaxis.label.set_color('#3b82f6')
                ax.spines['bottom'].set_color('#1e3a8a')
                ax.spines['left'].set_color('#1e3a8a')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, linestyle='--', alpha=0.1, color='#3b82f6')
                
                ax.set_ylabel("Retraso en Metros", weight='bold')
                ax.set_xlabel("Hora de Observación (UTC)", weight='bold')
                
                legend = ax.legend(facecolor='#000000', edgecolor='#1e3a8a')
                for text in legend.get_texts():
                    text.set_color('#60a5fa')
                    
                st.pyplot(fig)
            else:
                st.error("El servidor del DLR no devolvió datos limpios para esta fecha.")
