import streamlit as st
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
    # Escaneo riguroso por minutos contiguos para asegurar datos reales válidos
    for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        f_intento = fecha.replace(minute=m)
        url = generar_enlace_dlr(f_intento)
        try:
            r = requests.get(url, headers=headers, timeout=4)
            if r.status_code == 200:
                data = r.json()
                # COMPROBACIÓN CRÍTICA: Validar estructura y contenido real
                if 'data' in data and 'grid' in data['data'] and 'features' in data['data']['grid']:
                    vtec_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    
                    # Filtro anti-mallas falsas o corruptas (Malla Europa Versión A requiere 43x81 = 3483 puntos)
                    if len(vtec_list) == 3483: 
                        return np.array(vtec_list).reshape(43, 81), f_intento
        except: 
            continue
    return None, None

# =====================================================================
# INTERFAZ DE USUARIO (SIDEBAR MULTI-OPCIÓN)
# =====================================================================
st.sidebar.title("🛰️ Iono-Explorer Pro")
st.sidebar.markdown("---")

version_seleccionada = st.sidebar.selectbox(
    "Selecciona la Herramienta Web",
    ["1. Versión por Horas (Mapas 24h)", "2. Versión Matemática (Predicción vs Realidad)", "3. Versión por Constelaciones (Error Local)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Parámetros Geográficos")
ciudad_user = st.sidebar.text_input("Introduce Localidad", "Madrid")

lats_vector = np.arange(30, 73, 1)
lons_vector = np.arange(-30, 51, 1)
grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

geolocator = Nominatim(user_agent="iono_explorer_pro_web_app_v3")
try:
    location = geolocator.geocode(ciudad_user, timeout=10)
except:
    location = None

if location:
    lat_idx = (np.abs(lats_vector - location.latitude)).argmin()
    lon_idx = (np.abs(lons_vector - location.longitude)).argmin()
    st.sidebar.success(f"Malla: Lat {lats_vector[lat_idx]}°N | Lon {lons_vector[lon_idx]}°E")
else:
    # Backup estático automático para Madrid en caso de fallo de geolocalización externa
    lat_idx = (np.abs(lats_vector - 40.4167)).argmin()
    lon_idx = (np.abs(lons_vector - -3.7037)).argmin()
    st.sidebar.warning("⚠️ Usando coordenadas por defecto (Madrid) por latencia del buscador.")
    location = True 

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Parámetros Temporales")
fecha_user = st.sidebar.date_input("Fecha Base de Estudio", datetime.date(2026, 1, 24))

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
        
        with st.spinner("Procesando mapas horarios reales..."):
            fecha_h = datetime.datetime.combine(fecha_user, datetime.time(hora_mapa, 0))
            matriz_tecu, _ = descargar_json_dlr(fecha_h)
            
            if matriz_tecu is not None:
                matriz_metros = matriz_tecu * factor_m
                v_min = max(0.0, float(np.floor(np.min(matriz_metros) - 0.5)))
                v_max = float(np.ceil(np.max(matriz_metros) + 0.5))
                
                try:
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
                    cbar.set_label(f"Retraso de Grupo estimado para {constelacion_select} (Metros)")
                    st.pyplot(fig)
                except Exception:
                    st.info("ℹ️ Renderizando en modo seguro de alta velocidad (sin mapa de fondo por restricciones del host).")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mesh = ax.pcolormesh(grid_lon, grid_lat, matriz_metros, cmap='jet', shading='gouraud', vmin=v_min, vmax=v_max)
                    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7)
                    st.pyplot(fig)
                
                c1, c2 = st.columns(2)
                c1.metric(f"Peor Retraso en Europa ({constelacion_select})", f"{np.max(matriz_metros):.2f} metros")
                c2.metric(f"Retraso estimado en tu coordenada", f"{matriz_metros[lat_idx, lon_idx]:.2f} metros")
            else:
                st.error("Datos reales no disponibles para este bloque horario en el servidor del DLR.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 2: VERSIÓN MATEMÁTICA (PREDICCIÓN VS REALIDAD CORREGIDA)
    # -----------------------------------------------------------------
    elif version_seleccionada == "2. Versión Matemática (Predicción vs Realidad)":
        st.header(f"📈 Auditoría de Predicción Matemática vs. Realidad: {ciudad_user}")
        rango_dias = st.slider("Días del pasado para entrenamiento del modelo", 5, 15, 7)
        st.info("El sistema entrenará el algoritmo con el pasado y descargará los links reales de las siguientes 6 horas para medir el error exacto del modelo.")
        
        if st.button("🧠 Ejecutar Comparativa Rigurosa"):
            with st.spinner("1/2: Recolectando serie histórica real certificada..."):
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
                    
                    # Algoritmo de Inercia Estacional (Predicción)
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
                    for k in range(1, 4): # Próximas 6 horas (Pasos de 2h)
                        slot_futuro = (ultimo_slot + k) % periodo
                        val_pred = perfil_estacional[slot_futuro] + anomalia * (alpha ** k)
                        predicciones_futuras.append(val_pred)
                        fechas_futuras.append(fechas_list[-1] + datetime.timedelta(hours=k*2))
                    
                    # EXTRA NUEVO: DESCARGAR Y VERIFICAR LOS LINKS REALES DEL FUTURO/PRESENTE
                    st.text("2/2: Verificando links del DLR para extraer la curva de la realidad...")
                    realidad_futura = []
                    fechas_reales_futuras = []
                    
                    for f_fut in fechas_futuras:
                        m_real, _ = descargar_json_dlr(f_fut)
                        if m_real is not None:
                            realidad_futura.append(m_real[lat_idx, lon_idx])
                            fechas_reales_futuras.append(f_fut)
                    
                    # Renderizado de la gráfica comparativa
                    fig, ax = plt.subplots(figsize=(11, 4.5))
                    # Mostrar las últimas 24 horas del pasado
                    ax.plot(fechas_list[-12:], vector_serie[-12:], color='#2979ff', linewidth=2, label="Pasado Real Registrado", marker='o')
                    # Mostrar la curva matemática estimada
                    ax.plot(fechas_futuras, predicciones_futuras, color='#ff3d00', linewidth=2.5, linestyle='--', label="Proyección Matemática (Modelo)", marker='x')
                    
                    # Si existen datos reales en esos links, pintamos la línea de la verdad para comparar
                    if realidad_futura:
                        ax.plot(fechas_reales_futuras, realidad_futura, color='#00e676', linewidth=2.5, label="Realidad Absoluta (Datos DLR Verificados)", marker='s')
                        
                    # Regla estricta +-2 para asegurar que no se corten las líneas
                    todos_los_valores = list(vector_serie[-12:]) + list(predicciones_futuras) + list(realidad_futura)
                    y_min = max(0.0, float(np.floor(min(todos_los_valores) - 2)))
                    y_max = float(np.ceil(max(todos_los_valores) + 2))
                    ax.set_ylim(y_min, y_max)
                    
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_ylabel("Densidad de Electrones (TECU)", weight='bold')
                    ax.set_title(f"Auditoría del Modelo: Predicción vs Realidad en {ciudad_user.upper()}", weight='bold')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Cuadro de mandos estadístico del error
                    if len(realidad_futura) == len(predicciones_futuras):
                        error_medio = np.mean(np.abs(np.array(realidad_futura) - np.array(predicciones_futuras)))
                        st.success(f"🎯 Comparativa completada. El error medio absoluto del modelo en esta ventana fue de **{error_medio:.3f} TECU**.")
                    else:
                        st.warning("⚠️ Gráfica generada parcialmente. Algunos links futuros del DLR aún no han sido publicados en el servidor en tiempo real (Clima espacial en curso).")
                else:
                    st.error("No se pudieron recopilar suficientes enlaces reales limpios para inicializar el algoritmo.")

    # -----------------------------------------------------------------
    # HERRAMIENTA 3: VERSIÓN POR CONSTELACIONES (COMPARATIVA DE METROS)
    # -----------------------------------------------------------------
    elif version_seleccionada == "3. Versión por Constelaciones (Error)":
        st.header(f"📡 Comparativa Multi-Constelación Absoluta")
        st.markdown("Se evalúan simultáneamente las 4 bandas L1 principales libres de falsificaciones de datos.")
        
        with st.spinner("Calculando retrasos de grupo por frecuencia..."):
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
                
                ax.set_ylim(max(0.0, float(np.floor(min(todos_los_metros) - 0.5))), float(np.ceil(max(todos_los_metros) + 0.5)))
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_ylabel("Retraso de Grupo en Pseudodistancia (Metros)", weight='bold')
                ax.set_xlabel("Hora del Día (UTC)", weight='bold')
                ax.legend()
                st.pyplot(fig)
            else:
                st.error("Error al leer los links de la fecha seleccionada. Servidor no disponible.")
