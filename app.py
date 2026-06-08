import streamlit as st
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from geopy.geocoders import Nominatim
import matplotlib.dates as mdates

# Configuración profesional de la plataforma web
st.set_page_config(
    page_title="Iono-Explorer Pro GNSS",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para adaptar la web al modo oscuro científico
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stHeading h1, .stHeading h2, .stHeading h3 { color: #deff9a !important; }
    div[data-testid="stMetricValue"] { color: #4ade80 !important; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# MOTOR CORE DE DATOS (DLR CONNECTOR v2.0.0)
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

@st.cache_data(show_spinner=False, ttl=3600) # Cache de 1 hora por seguridad
def descargar_json_dlr(fecha):
    headers = {"User-Agent": "Mozilla/5.0"}
    for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        f_intento = fecha.replace(minute=m)
        url = generar_enlace_dlr(f_intento)
        try:
            r = requests.get(url, headers=headers, timeout=4)
            if r.status_code == 200:
                data = r.json()
                vtec_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                if len(vtec_list) == 3483: # Malla A (43x81)
                    return np.array(vtec_list).reshape(43, 81), m
        except: continue
    return None, None

# =====================================================================
# INTERFAZ DE USUARIO (SIDEBAR MULTI-OPCIÓN)
# =====================================================================
st.sidebar.title("🛰️ Iono-Explorer Pro")
st.sidebar.markdown("---")

# Opción PRINCIPAL: Selección de la Versión/Herramienta
version_seleccionada = st.sidebar.selectbox(
    "Selecciona la Herramienta Web",
    ["1. Versión por Horas (Mapas 24h)", "2. Versión Matemática (Predicción)", "3. Versión por Constelaciones (Error Local)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Parámetros Geográficos")
ciudad_user = st.sidebar.text_input("Introduce Localidad", "Madrid")

# Vectores fijos de la Cuadrícula Europa (Versión A)
lats_vector = np.arange(30, 73, 1)
lons_vector = np.arange(-30, 51, 1)
grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

# Resolver geolocalización en la nube
geolocator = Nominatim(user_agent="iono_explorer_pro_web")
try:
    location = geolocator.geocode(ciudad_user)
except:
    location = None

if location:
    lat_idx = (np.abs(lats_vector - location.latitude)).argmin()
    lon_idx = (np.abs(lons_vector - location.longitude)).argmin()
    st.sidebar.success(f"Malla: Lat {lats_vector[lat_idx]}°N | Lon {lons_vector[lon_idx]}°E")
else:
    st.sidebar.error("Error: Ciudad no localizada en la malla.")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Parámetros Temporales")
fecha_user = st.sidebar.date_input("Fecha Base", datetime.date(2026, 1, 24))

# Frecuencias fijas de tu tabla técnica para los cálculos de error en metros
FREQS_GNSS = {"GPS": 1575.42 * 1e6, "Galileo": 1575.42 * 1e6, "GLONASS": 1602.00 * 1e6, "BeiDou": 1561.10 * 1e6}

# =====================================================================
# ÁREA DE RENDERIZADO PRINCIPAL
# =====================================================================
st.title("📊 Panel de Control e Impacto Ionosférico GNSS")
st.markdown("Auditoría espacial y matemática con datos validados del Centro Aeroespacial Alemán (DLR).")

if not location:
    st.warning("⚠️ Introduce una ciudad válida en la barra lateral para activar el motor de cálculo.")
else:
    # -----------------------------------------------------------------
    # HERRAMIENTA 1: VERSIÓN POR HORAS (MAPAS DINÁMICOS)
    # -----------------------------------------------------------------
    if version_seleccionada == "1. Versión por Horas (Mapas 24h)":
        st.header("🗺️ Análisis Espacial de Retraso de Grupo (24 Horas)")
        hora_mapa = st.slider("Selecciona la Hora de Observación (UTC)", 0, 23, 13)
        constelacion_select = st.selectbox("Selecciona Constelación para evaluar el mapa en Metros", ["GPS", "Galileo", "GLONASS", "BeiDou"])
        
        f_c = FREQS_GNSS[constelacion_select]
        factor_m = (40.3 / (f_c ** 2)) * 1e16
        
        with st.spinner("Procesando mapas horarios..."):
            fecha_h = datetime.datetime.combine(fecha_user, datetime.time(hora_mapa, 0))
            matriz_tecu, min_real = descargar_json_dlr(fecha_h)
            
            if matriz_tecu is not None:
                matriz_metros = matriz_tecu * factor_m
                
                # Regla del +-2 en metros (ajustada proporcionalmente para metros)
                v_min = max(0.0, float(np.floor(np.min(matriz_metros) - 0.5)))
                v_max = float(np.ceil(np.max(matriz_metros) + 0.5))
                
                # Renderizar Cartopy en la Web
                fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                ax.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, facecolor='#1e293b', zorder=1)
                ax.add_feature(cfeature.OCEAN, facecolor='#0f172a', zorder=1)
                ax.add_feature(cfeature.COASTLINE, edgecolor='#f8fafc', linewidth=1, zorder=3)
                
                gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.1, linestyle='--')
                gl.top_labels, gl.right_labels = False, False
                
                mesh = ax.pcolormesh(grid_lon, grid_lat, matriz_metros, transform=ccrs.PlateCarree(),
                                     cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
                
                cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.03)
                cbar.set_label(f"Retraso de Grupo estimado para {constelacion_select} (Metros)", color="black")
                
                st.pyplot(fig)
                
                # Métricas Rápidas en Pantalla
                c1, c2 = st.columns(2)
                c1.metric(f"Peor Retraso en Europa ({constelacion_select})", f"{np.max(matriz_metros):.2f} metros")
                c2.metric("Valor del pixel en la ciudad seleccionada", f"{matriz_metros[lat_idx, lon_idx]:.2f} metros")
            else:
                st.error("Servidor DLR fuera de línea o datos no disponibles para esta hora específica.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 2: VERSIÓN MATEMÁTICA (PREDICCIÓN HISTÓRICA)
    # -----------------------------------------------------------------
    elif version_seleccionada == "2. Versión Matemática (Predicción)":
        st.header(f"📈 Modelo Autorregresivo de Inercia Estacional: {ciudad_user}")
        rango_dias = st.slider("Días del pasado para entrenar el modelo matemático", 5, 15, 7)
        st.info(f"El sistema analizará {rango_dias * 12} muestras del pasado (Paso bihorario) para predecir las siguientes 6 horas.")
        
        if st.button("🧠 Ejecutar Modelo Matemático"):
            with st.spinner("Compilando serie temporal y ejecutando ecuaciones de persistencia..."):
                cronologia_tecu = []
                fechas_list = []
                
                # Bucle de descarga bihoraria acelerada para web
                total_pasos = rango_dias * 12
                barra_progreso = st.progress(0)
                
                for i in range(total_pasos):
                    f_calc = datetime.datetime.combine(fecha_user, datetime.time(0,0)) - datetime.timedelta(hours=(total_pasos-i)*2)
                    m_tecu, _ = descargar_json_dlr(f_calc)
                    if m_tecu is not None:
                        cronologia_tecu.append(m_tecu[lat_idx, lon_idx])
                        fechas_list.append(f_calc)
                    barra_progreso.progress((i+1)/total_pasos)
                
                if len(cronologia_tecu) > 24:
                    vector_serie = np.array(cronologia_tecu)
                    
                    # Ejecución del Modelo Matemático AR-Seasonal (Tu motor de inercia)
                    periodo = 12
                    perfil_estacional = np.zeros(periodo)
                    for i in range(periodo):
                        perfil_estacional[i] = np.mean(vector_serie[i::periodo])
                        
                    ultimo_val = vector_serie[-1]
                    ultimo_slot = (len(vector_serie) - 1) % periodo
                    anomalia = ultimo_val - perfil_estacional[ultimo_slot]
                    
                    # Proyectar 3 puntos adelante (6 horas en pasos de 2h)
                    predicciones = []
                    fechas_futuras = []
                    alpha = 0.85
                    for k in range(1, 4):
                        slot_futuro = (ultimo_slot + k) % periodo
                        val_pred = perfil_estacional[slot_futuro] + anomalia * (alpha ** k)
                        predicciones.append(val_pred)
                        fechas_futuras.append(fechas_list[-1] + datetime.timedelta(hours=k*2))
                    
                    # Gráfica de Predicción con Regla del +-2
                    fig, ax = plt.subplots(figsize=(11, 4.5))
                    ax.plot(fechas_list[-24:], vector_serie[-24:], color='#2979ff', linewidth=2.5, label="Pasado Real Registrado", marker='o')
                    ax.plot(fechas_futuras, predicciones, color='#ff3d00', linewidth=2.5, linestyle='--', label="Proyección Matemática (Futuro 6h)", marker='x')
                    
                    # Aplicar tus límites fijos de seguridad estrictos
                    y_min = max(0.0, float(np.floor(min(np.min(vector_serie[-24:]), min(predicciones)) - 2)))
                    y_max = float(np.ceil(max(np.max(vector_serie[-24:]), max(predicciones)) + 2))
                    ax.set_ylim(y_min, y_max)
                    
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_ylabel("Densidad de Electrones (TECU)", weight='bold')
                    ax.set_title(f"Predicción Ionosférica Local en {ciudad_user.upper()}", weight='bold')
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.success("🤖 Modelo matemático ejecutado con éxito. La línea discontinua roja representa la proyección más probable basada en la inercia actual de la alta atmósfera.")
                else:
                    st.error("No se pudieron recolectar suficientes puntos reales del servidor del DLR para entrenar el modelo.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 3: VERSIÓN POR CONSTELACIONES (COMPARATIVA DE METROS)
    # -----------------------------------------------------------------
    elif version_seleccionada == "3. Versión por Constelaciones (Error)":
        st.header(f"📡 Comparativa Multi-Constelación Absoluta en {ciudad_user}")
        st.markdown("Evaluación simultánea de las 4 bandas L1 principales del espectro orbital.")
        
        with st.spinner("Calculando retrasos de grupo por frecuencia..."):
            perfiles_tecu_24h = []
            horas_validas = []
            
            for h in range(0, 24, 2): # Descarga bihoraria del día para optimizar la carga web
                fecha_h = datetime.datetime.combine(fecha_user, datetime.time(h, 0))
                m_tecu, _ = descargar_json_dlr(fecha_h)
                if m_tecu is not None:
                    perfiles_tecu_24h.append(m_tecu[lat_idx, lon_idx])
                    horas_validas.append(f"{h:02d}:00")
            
            if perfiles_tecu_24h:
                vector_tecu = np.array(perfiles_tecu_24h)
                
                # Construcción del lienzo multi-línea
                fig, ax = plt.subplots(figsize=(11, 5))
                colores = {"GLONASS": "#0d47a1", "GPS": "#00c853", "Galileo": "#ffd600", "BeiDou": "#d50000"}
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
                
                # Ajuste simétrico estricto de ejes vertical con la regla del +-2 en metros
                ax.set_ylim(max(0.0, float(np.floor(min(todos_los_metros) - 0.5))), float(np.ceil(max(todos_los_metros) + 0.5)))
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_ylabel("Retraso de Grupo en Pseudodistancia (Metros)", weight='bold')
                ax.set_xlabel("Hora del Día (UTC)", weight='bold')
                ax.set_title(f"Desviación Absoluta de la Señal el {fecha_user.strftime('%d/%m/%Y')}", weight='bold')
                ax.legend()
                st.pyplot(fig)
                
                st.info("💡 **Nota física:** Observa cómo las líneas de GPS y Galileo se solapan de forma perfecta. Esto confirma visualmente su diseño interoperable, mientras que BeiDou experimenta el peor escenario de retraso métrico por operar en el espectro de frecuencia más bajo.")
            else:
                st.error("No se pudieron recuperar datos para la fecha seleccionada.")
