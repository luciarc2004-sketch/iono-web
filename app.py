
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
    .main { background-color: #000000; color: #cbd5e1; }
    .stHeading h1, .stHeading h2, .stHeading h3 { color: #3b82f6 !important; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #1e3a8a; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    div[data-testid="stMetricValue"] { color: #06b6d4 !important; font-family: monospace; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    .stButton>button {
        background-color: #1e3a8a; color: #ffffff; border-radius: 6px;
        border: 1px solid #3b82f6; font-weight: bold; width: 100%; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #2563eb; border-color: #60a5fa; }
    .stAlert { background-color: #0b1329 !important; color: #60a5fa !important; border: 1px solid #1e3a8a !important; }
    </style>
    """, unsafe_allow_html=True)

# Base de datos interna de respaldo para evitar bloqueos del geolocalizador en la nube
CAPITALES_BACKUP = {
    "madrid": (40.4167, -3.7037), "barcelona": (41.3851, 2.1734), "sevilla": (37.3891, -5.9845),
    "berlin": (52.5200, 13.4050), "paris": (48.8566, 2.3522), "londres": (51.5074, -0.1278),
    "roma": (41.9028, 12.4964), "bruselas": (50.8503, 4.3517), "lisboa": (38.7223, -9.1393)
}

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
                if 'data' in data and 'grid' in data['data'] and 'features' in data['data']['grid']:
                    vtec_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    if len(vtec_list) == 3483: # Malla Europa Certificada (43x81)
                        return np.array(vtec_list).reshape(43, 81), f_intento
        except: continue
    return None, None

# =====================================================================
# INTERFAZ DE USUARIO (BARRA LATERAL DE CONTROL OPERACIONAL)
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

# RESOLUCIÓN GEOGRÁFICA HÍBRIDA EVITA-BLOQUEOS
c_lat, c_lon = None, None
geolocator = Nominatim(user_agent="iono_explorer_pro_command_v5")
try:
    location = geolocator.geocode(ciudad_user, timeout=5)
    if location:
        c_lat, c_lon = location.latitude, location.longitude
except:
    pass

if c_lat is None:
    ciudad_clean = ciudad_user.lower().strip()
    if ciudad_clean in CAPITALES_BACKUP:
        c_lat, c_lon = CAPITALES_BACKUP[ciudad_clean]
    else:
        c_lat, c_lon = 40.4167, -3.7037 # Respaldo absoluto si no se encuentra nada

# Comprobación de límites de cobertura de la Malla Europa
if not (30 <= c_lat <= 72) or not (-30 <= c_lon <= 50):
    st.sidebar.error("❌ Fuera de cobertura del radar de Europa (Malla A). Alertas suspendidas.")
    cobertura_valida = False
else:
    lat_idx = (np.abs(lats_vector - c_lat)).argmin()
    lon_idx = (np.abs(lons_vector - c_lon)).argmin()
    st.sidebar.success(f"Malla fija: Lat {lats_vector[lat_idx]}°N | Lon {lons_vector[lon_idx]}°E")
    cobertura_valida = True

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Parámetros Temporales")
fecha_user = st.sidebar.date_input("Fecha de Análisis", datetime.date(2026, 1, 24))
hora_user = st.sidebar.slider("Hora de Sincronización Base (UTC)", 0, 23, 12)

# MAPAMUNDI GLOBAL DE INTERÉS ORBITAL EN LA SIDEBAR
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Ubicación Orbital de la Estación")
fig_glob, ax_glob = plt.subplots(figsize=(4, 2.5), facecolor='#050505')
ax_glob.set_facecolor('#000000')
ax_glob.plot([c_lon], [c_lat], color='#06b6d4', marker='o', markersize=8, markeredgecolor='white', zorder=5)
# Límites mundiales simples
ax_glob.set_xlim(-180, 180)
ax_glob.set_ylim(-90, 90)
ax_glob.axis('off')
ax_glob.text(c_lon + 10, c_lat + 5, "STATION", color='#06b6d4', fontsize=7, weight='bold')
# Dibujar una cuadrícula de referencia espacial simple
for lon_line in range(-180, 180, 60): ax_glob.axvline(lon_line, color='#1e3a8a', alpha=0.1, linewidth=0.5)
for lat_line in range(-90, 90, 30): ax_glob.axhline(lat_line, color='#1e3a8a', alpha=0.1, linewidth=0.5)
st.sidebar.pyplot(fig_glob)

FREQS_GNSS = {"GPS": 1575.42 * 1e6, "Galileo": 1575.42 * 1e6, "GLONASS": 1602.00 * 1e6, "BeiDou": 1561.10 * 1e6}

# =====================================================================
# ÁREA PRINCIPAL DE PRESENTACIÓN DE DATOS
# =====================================================================
st.title("🖥️ Iono-Explorer Pro: Terminal de Auditoría Atmosférica")
st.markdown("Plataforma analítica de refracción ionosférica para sistemas globales de navegación.")

if not cobertura_valida:
    st.warning("⚠️ Introduce una localidad válida dentro del espacio euroatlántico en la barra lateral.")
else:
    # -----------------------------------------------------------------
    # HERRAMIENTA 1: VERSIÓN POR HORAS (MAPAS DE RETRASO EN METROS)
    # -----------------------------------------------------------------
    if version_seleccionada == "1. Versión por Horas (Mapas 24h)":
        st.header("🗺️ Distribución Espacial del Retraso de Grupo (Europa)")
        constelacion_select = st.selectbox("Evaluar Mapa en Metros para la señal de:", ["GPS", "Galileo", "GLONASS", "BeiDou"])
        
        f_c = FREQS_GNSS[constelacion_select]
        factor_m = (40.3 / (f_c ** 2)) * 1e16
        
        with st.spinner("Descargando matriz y proyectando espacio métrico..."):
            fecha_h = datetime.datetime.combine(fecha_user, datetime.time(hora_user, 0))
            matriz_tecu, _ = descargar_json_dlr(fecha_h)
            
            if matriz_tecu is not None:
                matriz_metros = matriz_tecu * factor_m
                v_min = max(0.0, float(np.floor(np.min(matriz_metros) - 0.5)))
                v_max = float(np.ceil(np.max(matriz_metros) + 0.5))
                
                try:
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
                    cbar.ax.tick_params(labelcolor='#3b82f6') # FIJADO: labelcolor sin 's'
                    
                    st.pyplot(fig)
                except Exception:
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#000000')
                    ax.set_facecolor('#000000')
                    mesh = ax.pcolormesh(grid_lon, grid_lat, matriz_metros, cmap='jet', shading='gouraud', vmin=v_min, vmax=v_max)
                    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7)
                    cbar.set_label("Retraso (Metros)", color='#3b82f6')
                    cbar.ax.tick_params(labelcolor='#3b82f6') # FIJADO: labelcolor sin 's'
                    ax.tick_params(colors='#3b82f6')
                    st.pyplot(fig)
                
                c1, c2 = st.columns(2)
                c1.metric(f"Máximo Retraso en Europa ({constelacion_select})", f"{np.max(matriz_metros):.2f} m")
                c2.metric(f"Impacto Estimado en Coordenada Seleccionada", f"{matriz_metros[lat_idx, lon_idx]:.2f} m")
            else:
                st.error("Datos reales no disponibles para este bloque horario en el servidor del DLR.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 2: VERSIÓN MATEMÁTICA (PREDICCIÓN VS REALIDAD COMPLETA)
    # -----------------------------------------------------------------
    elif version_seleccionada == "2. Versión Matemática (Predicción vs Realidad)":
        st.header(f"📈 Auditoría de Tendencia Centrada a las {hora_user:02d}:00 UTC")
        rango_dias = st.slider("Ventana de días para el entrenamiento de la serie", 5, 15, 7)
        
        if st.button("🧠 Iniciar Auditoría de Enlaces"):
            with st.spinner("Ejecutando escáner de red bihorario..."):
                cronologia_tecu = []
                fechas_list = []
                
                total_pasos = rango_dias * 12
                barra_progreso = st.progress(0)
                
                # Sincronización horaria exacta solicitada por el usuario
                fecha_base_calculo = datetime.datetime.combine(fecha_user, datetime.time(hora_user, 0))
                
                for i in range(total_pasos):
                    f_calc = fecha_base_calculo - datetime.timedelta(hours=(total_pasos-i)*2)
                    m_tecu, _ = descargar_json_dlr(f_calc)
                    if m_tecu is not None:
                        cronologia_tecu.append(m_tecu[lat_idx, lon_idx])
                        fechas_list.append(f_calc)
                    barra_progreso.progress((i+1) / total_pasos)
                
                if len(cronologia_tecu) > 24:
                    vector_serie = np.array(cronologia_tecu)
                    
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
                    for k in range(1, 4): # Próximas 6 horas
                        slot_futuro = (ultimo_slot + k) % periodo
                        val_pred = perfil_estacional[slot_futuro] + anomalia * (alpha ** k)
                        predicciones_futuras.append(val_pred)
                        fechas_futuras.append(fechas_list[-1] + datetime.timedelta(hours=k*2))
                    
                    realidad_futura = []
                    fechas_reales_futuras = []
                    for f_fut in fechas_futuras:
                        m_real, _ = descargar_json_dlr(f_fut)
                        if m_real is not None:
                            realidad_futura.append(m_real[lat_idx, lon_idx])
                            fechas_reales_futuras.append(f_fut)
                    
                    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='#000000')
                    ax.set_facecolor('#000000')
                    
                    ax.plot(fechas_list[-12:], vector_serie[-12:], color='#60a5fa', linewidth=2, label="Historial Real Verificado", marker='o')
                    ax.plot(fechas_futuras, predicciones_futuras, color='#2563eb', linewidth=2.5, linestyle='--', label="Algoritmo Matemático (Modelo)", marker='x')
                    
                    if realidad_futura:
                        ax.plot(fechas_reales_futuras, realidad_futura, color='#ffffff', linewidth=2.5, label="Validación Real (Datos del Servidor)", marker='s')
                        
                    todos_los_valores = list(vector_serie[-12:]) + list(predicciones_futuras) + list(realidad_futura)
                    y_min = max(0.0, float(np.floor(min(todos_los_valores) - 2)))
                    y_max = float(np.ceil(max(todos_los_valores) + 2))
                    ax.set_ylim(y_min, y_max)
                    
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
                    for text in legend.get_texts(): text.set_color('#60a5fa')
                        
                    st.pyplot(fig)
                    
                    if len(realidad_futura) == len(predicciones_futuras):
                        error_medio = np.mean(np.abs(np.array(realidad_futura) - np.array(predicciones_futuras)))
                        st.success(f"🎯 Análisis cerrado. El error absoluto medio de la proyección a las {hora_user:02d}:00 fue de: **{error_medio:.3f} TECU**.")
                    else:
                        st.warning("⚠️ Operación parcial: Los enlaces del tiempo real exacto aún están siendo procesados por el DLR.")
                else:
                    st.error("Fallo de red: No se detectaron suficientes mallas reales válidas.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 3: VERSIÓN POR CONSTELACIONES (ESPECTRO MULTI-FRECUENCIA)
    # -----------------------------------------------------------------
    elif version_seleccionada == "3. Versión por Constelaciones (Error)":
        st.header(f"📡 Ventana Diaria de Desviación Métrica Absoluta")
        
        with st.spinner("Calculando retrasos de grupo por frecuencia..."):
            perfiles_tecu_24h = []
            horas_validas = []
            
            # Sincronización de un ciclo completo de 24h partiendo de la hora base
            for h_offset in range(0, 24, 2):
                fecha_h = datetime.datetime.combine(fecha_user, datetime.time(hora_user, 0)) + datetime.timedelta(hours=h_offset)
                m_tecu, _ = descargar_json_dlr(fecha_h)
                if m_tecu is not None:
                    perfiles_tecu_24h.append(m_tecu[lat_idx, lon_idx])
                    horas_validas.append(fecha_h.strftime("%H:%M"))
            
            if perfiles_tecu_24h:
                vector_tecu = np.array(perfiles_tecu_24h)
                
                fig, ax = plt.subplots(figsize=(11, 5), facecolor='#000000')
                ax.set_facecolor('#000000')
                
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
                ax.set_xlabel("Línea del Tiempo desde Sincronización Base (UTC)", weight='bold')
                
                legend = ax.legend(facecolor='#000000', edgecolor='#1e3a8a')
                for text in legend.get_texts(): text.set_color('#60a5fa')
                    
                st.pyplot(fig)
            else:
                st.error("El servidor del DLR no devolvió datos limpios para esta ventana de tiempo.")
