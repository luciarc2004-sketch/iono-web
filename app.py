import streamlit as st
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import RegularGridInterpolator
from geopy.geocoders import Nominatim

# =====================================================================
# CONFIGURACIÓN DE LA PLATAFORMA WEB
# =====================================================================
st.set_page_config(
    page_title="Iono-Explorer Pro GNSS",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# INYECCIÓN CSS: Tema Formal "Deep Space Command"
st.markdown("""
    <style>
    /* Fondo de la pantalla principal y textos generales */
    .main { background-color: #000000; color: #cbd5e1; }
    
    /* Configuración de títulos y subtítulos */
    .stHeading h1, .stHeading h2, .stHeading h3 { color: #3b82f6 !important; }
    
    /* Panel Lateral (Sidebar) */
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1e3a8a; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    
    /* Cuadro de Métricas (Valores en Cian) */
    div[data-testid="stMetricValue"] { color: #06b6d4 !important; font-family: monospace; font-size: 2rem !important; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    
    /* Botones Interactivos */
    .stButton>button {
        background-color: #1e3a8a; color: #ffffff; border-radius: 6px;
        border: 1px solid #3b82f6; font-weight: bold; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #2563eb; border-color: #60a5fa; color: #ffffff; }
    
    /* Cajas de Alerta y Consolas */
    .stAlert { background-color: #0b1329 !important; color: #60a5fa !important; border: 1px solid #1e3a8a !important; }
    </style>
    """, unsafe_allow_html=True)

# Excepción personalizada para el control de errores en la nube
class AlarmaDatosFalsosError(Exception):
    pass

# =====================================================================
# MOTORES CORE DE DESCARGA DE ENLACES (DLR)
# =====================================================================
def generar_enlace_dlr_europa(fecha):
    str_anio = fecha.strftime("%Y")
    str_doy = fecha.strftime("%j")
    str_hora = fecha.strftime("%H")
    f_inicio = fecha - datetime.timedelta(minutes=4, seconds=30)
    ts_inicio = f_inicio.strftime("%Y-%m-%dT%H-%M-%S")
    ts_fin = fecha.strftime("%Y-%m-%dT%H-%M-%S")
    base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
    nombre_archivo = f"DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{ts_inicio}_{ts_fin}_{str_doy}_D.json"
    return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/{nombre_archivo}"

@st.cache_data(show_spinner=False, ttl=1800)
def descargar_datos_sistemas(fecha):
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Descarga e inspección de la Malla Regional de Europa
    matriz_eur, f_valida_eur = None, None
    for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        f_intento = fecha.replace(minute=m)
        url_europa = generar_enlace_dlr_europa(f_intento)
        try:
            r = requests.get(url_europa, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if 'data' in data and 'grid' in data['data'] and 'features' in data['data']['grid']:
                    vtec_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    if len(vtec_list) == 3483: # 43 filas x 81 columnas
                        matriz_eur = np.array(vtec_list).reshape(43, 81)
                        f_valida_eur = f_intento
                        break
        except: continue

    # 2. Descarga e inspección de la Malla Planetaria Global (Latest)
    matriz_glb = None
    url_global = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL/v2.0.0/latest/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL_latest_D.json"
    try:
        r = requests.get(url_global, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if 'data' in data and 'grid' in data['data'] and 'features' in data['data']['grid']:
                vtec_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                if len(vtec_list) == 5329: # 73 x 73 puntos
                    matriz_glb = np.array(vtec_list).reshape(73, 73)
    except: pass

    return matriz_eur, matriz_glb, f_valida_eur

# =====================================================================
# INTERFAZ DE USUARIO: BARRA LATERAL (CONTROL DE TIEMPO)
# =====================================================================
st.sidebar.title("🛰️ Centro de Control")
st.sidebar.markdown("---")
st.sidebar.subheader("⏰ Temporizador de Consulta")
fecha_seleccionada = st.sidebar.date_input("Fecha Base (Europa)", datetime.date(2026, 1, 24))
hora_seleccionada = st.sidebar.slider("Hora de Observación (UTC)", 0, 23, 4)

fecha_combinada = datetime.datetime.combine(fecha_seleccionada, datetime.time(hora_seleccionada, 0))

# Ejecutar descargas automáticas protegidas en memoria caché
matriz_eur, matriz_glb, fecha_real_eur = descargar_datos_sistemas(fecha_combinada)

# =====================================================================
# PANEL PRINCIPAL: INTRODUCCIÓN CIENTÍFICA e INTERFAZ DE INICIO
# =====================================================================
st.title("🖥️ Plataforma de Monitoreo Ionosférico Global Pro")
st.markdown("Auditoría espacial y análisis de retraso de grupo en tiempo real.")

# Sección Informativa Básica
with st.expander("📖 Glosario Técnico: ¿Qué estamos midiendo? (Conceptos Clave)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🌌 La Ionosfera")
        st.write("Es una capa de la atmósfera terrestre (situada entre los 80 y los 600 km de altitud) que se encuentra permanentemente ionizada debido a la radiación solar. Contiene una gran cantidad de electrones libres que interactúan directamente con las ondas de radio.")
        st.markdown("### 📡 Contenido Total de Electrones (TEC)")
        st.write("El TEC (Total Electron Content) mide la cantidad total de electrones libres presentes en un cilindro de sección transversal de 1 $m^2$ a lo largo de la trayectoria que recorre la señal desde el satélite hasta el receptor en tierra.")
    with c2:
        st.markdown("### 🧮 La Unidad TECU")
        st.write("Es la unidad de medida estándar en la física de la alta atmósfera. **1 TECU** equivale exactamente a $10^{16}$ electrones libres por metro cuadrado ($e^-/m^2$).")
        st.markdown("### 🛠️ ¿Para qué sirve esta auditoría?")
        st.write("Cuando las señales de los satélites (GPS, Galileo...) cruzan la ionosfera, sufren un retraso de grupo proporcional a la densidad de electrones. Este retraso deforma la medición de distancia introduciendo errores métricos en la geolocalización. Monitorear el TECU permite calcular y mitigar este desfase.")

st.markdown("---")

# =====================================================================
# SECCIÓN: CONSOLA DE CONSULTA LOCAL DE TECU
# =====================================================================
st.subheader("📌 Consola de Diagnóstico Local")
localidad_user = st.text_input("Introduce cualquier municipio o ciudad del planeta (Ej: Madrid, Ciudad de México, Tokio):", "Madrid")

if matriz_eur is not None and matriz_glb is not None:
    # Vectores fijos reglamentarios de las mallas del DLR
    lons_eur = np.arange(-30, 51, 1)
    lats_eur = np.arange(30, 73, 1)
    lons_glb = np.linspace(-180, 180, 73)
    lats_glb = np.linspace(-90, 90, 73)

    # Inicializar interpoladores oficiales de Scipy
    interp_europa = RegularGridInterpolator((lats_eur, lons_eur), matriz_eur, method='linear', bounds_error=False, fill_value=None)
    interp_global = RegularGridInterpolator((lats_glb, lons_glb), matriz_glb, method='linear', bounds_error=False, fill_value=None)

    # Geolocalización de la consulta
    geolocator = Nominatim(user_agent="iono_explorer_v6_init")
    try:
        loc = geolocator.geocode(localidad_user, timeout=4)
    except:
        loc = None

    if loc:
        lat, lon = loc.latitude, loc.longitude
        dentro_europa = (30 <= lat <= 72) and (-30 <= lon <= 50)
        punto_matematico = np.array([[lat, lon]])

        # Lógica de decisión de malla por precisión
        if dentro_europa:
            valor_tecu = float(interp_europa(punto_matematico)[0])
            fuente_txt = "Malla Regional de Europa (Alta Resolución: 1°)"
            estado_tipo = "success"
        else:
            valor_tecu = float(interp_global(punto_matematico)[0])
            fuente_txt = "Malla Planetaria Global (Resolución Estándar: 73x73)"
            estado_tipo = "info"

        # Mostrar Resultados Estructurados
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Localidad Detectada", f"{loc.name.split(',')[0]}")
        col_m2.metric("Coordenadas Radiales", f"{lat:.3f}°N, {lon:.3f}°E")
        col_m3.metric("Densidad Ionosférica", f"{valor_tecu:.3f} TECU")
        
        st.markdown(f"**Fuente del dato:** `{fuente_txt}`")
    else:
        st.warning("⚠️ Localidad no reconocida por el servidor geográfico de respaldo. Introduce otro término.")
else:
    st.error("❌ Error de comunicación: Los repositorios de datos del DLR no están disponibles en este momento.")

st.markdown("---")

# =====================================================================
# SECCIÓN: RENDERIZADO GRÁFICO SIMULTÁNEO (EUROPA VS GLOBAL)
# =====================================================================
st.subheader("🗺️ Cartografía Espacial Unificada (Visualización Perpendicular)")

if matriz_eur is not None and matriz_glb is not None:
    with st.spinner("Generando matrices cartográficas en entorno oscuro..."):
        # Crear lienzo dual integrado con el fondo negro de la web
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=100, facecolor='#000000',
                                       subplot_kw={'projection': ccrs.PlateCarree()})

        # --- MAPA 1: ENTORNO REGIONAL EUROPA ---
        ax1.set_facecolor('#000000')
        ax1.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='#0a0a0a', zorder=1)
        ax1.add_feature(cfeature.OCEAN, facecolor='#020205', zorder=1)
        ax1.add_feature(cfeature.COASTLINE, edgecolor='#1e3a8a', linewidth=0.9, zorder=3)
        ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#1e3a8a', alpha=0.5, zorder=3)

        gl1 = ax1.gridlines(draw_labels=True, color='#1e3a8a', alpha=0.15, linestyle='--')
        gl1.top_labels, gl1.right_labels = False, False
        gl1.xlabel_style, gl1.ylabel_style = {'color': '#3b82f6', 'size': 8}, {'color': '#3b82f6', 'size': 8}

        grid_lon_eur, grid_lat_eur = np.meshgrid(lons_eur, lats_eur)
        map_eur = ax1.pcolormesh(grid_lon_eur, grid_lat_eur, matriz_eur, transform=ccrs.PlateCarree(),
                                 cmap='jet', alpha=0.8, shading='gouraud', zorder=2)
        
        cbar1 = fig.colorbar(map_eur, ax=ax1, orientation='horizontal', pad=0.06, shrink=0.7)
        cbar1.set_label('Densidad Vertical de Electrones (TECU)', color='#3b82f6', fontsize=9)
        cbar1.ax.tick_params(labelcolor='#3b82f6', labelsize=8)
        ax1.set_title(f"REPOSITORIO REGIONAL EUROPA\nVentana: {fecha_real_eur.strftime('%Y-%m-%d %H:%M')} UTC", color='#3b82f6', weight='bold', size=10)

        # --- MAPA 2: ENTORNO PLANETARIO GLOBAL ---
        ax2.set_facecolor('#000000')
        ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='#0a0a0a', zorder=1)
        ax2.add_feature(cfeature.OCEAN, facecolor='#020205', zorder=1)
        ax2.add_feature(cfeature.COASTLINE, edgecolor='#1e3a8a', linewidth=0.8, zorder=3)

        gl2 = ax2.gridlines(draw_labels=True, color='#1e3a8a', alpha=0.12, linestyle='--')
        gl2.top_labels, gl2.right_labels = False, False
        gl2.xlabel_style, gl2.ylabel_style = {'color': '#3b82f6', 'size': 8}, {'color': '#3b82f6', 'size': 8}

        grid_lon_glb, grid_lat_glb = np.meshgrid(lons_glb, lats_glb)
        map_glb = ax2.pcolormesh(grid_lon_glb, grid_lat_glb, matriz_glb, transform=ccrs.PlateCarree(),
                                 cmap='jet', alpha=0.75, shading='gouraud', zorder=2)
        
        cbar2 = fig.colorbar(map_glb, ax=ax2, orientation='horizontal', pad=0.06, shrink=0.7)
        cbar2.set_label('Densidad Vertical de Electrones (TECU)', color='#3b82f6', fontsize=9)
        cbar2.ax.tick_params(labelcolor='#3b82f6', labelsize=8)
        ax2.set_title("REPOSITORIO PLANETARIO GLOBAL\nEstado: LATEST (Tiempo Real Absoluto)", color='#3b82f6', weight='bold', size=10)

        st.pyplot(fig)
else:
    st.info("⌛ Conectando con los servidores del DLR alemán para inicializar la cartografía...")
