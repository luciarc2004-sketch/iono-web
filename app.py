import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import RegularGridInterpolator
import streamlit as st

# Configuración de la página web
st.set_page_config(page_title="Portal de Monitoreo Ionosférico", layout="wide")

# =====================================================================
# CONFIGURACIÓN DE PESTAÑAS PRINCIPALES
# =====================================================================
tab1, tab2, tab3 = st.tabs(["🌍 Inicio y Monitoreo Real", "📊 Análisis en el pasado", "🛠️ Herramientas GNSS"])

# FUNCION COMPARTIDA DE GEOCODIFICACIÓN
def geocodificar_localidad(nombre_lugar):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={nombre_lugar}&format=json&limit=1"
        headers = {"User-Agent": "Streamlit_TEC_Monitor_App"}
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon']), data[0]['display_name']
    except Exception:
        pass
    return None, None, None

# =====================================================================
# PESTAÑA 1: INICIO Y MONITOREO EN TIEMPO REAL
# =====================================================================
with tab1:
    st.title("🛰️ Sistema Unificado de Monitoreo Ionosférico (TEC/TECU)")

    st.markdown("""
    ### ¿Cómo afectan el TEC y el TECU a las señales GNSS?
    El **Contenido Total de Electrones (TEC)** es la cantidad integrada de electrones atrapados en la ionosfera a lo largo de la trayectoria de una señal de satélite. Se mide en unidades **TECU** (1 TECU = $10^{16}$ electrones por metro cuadrado). 

    La presencia de estos electrones libres interactúa de forma directa con las señales emitidas por sistemas globales de navegación por satélite (**GNSS**), tales como GPS, Galileo, GLONASS o BeiDou, causando los siguientes efectos principales:
    * **Retardo Ionosférico:** Desacelera la velocidad de grupo de la señal de radio (y acelera la fase), lo que se traduce en un error de distancia calculado por el receptor. 10 TECU equivalen a aproximadamente 1.6 metros de error de rango en la frecuencia L1.
    * **Cintilación Ionosférica:** Fluctuaciones rápidas en la amplitud y fase de la señal que pueden provocar la pérdida de enganche (loss-of-lock) del satélite por parte del receptor.
    * **Variabilidad Espacio-Temporal:** Durante tormentas solares, el TEC aumenta drásticamente de forma impredecible, afectando la precisión de servicios de alta precisión (como RTK o navegación aérea guiada por satélite).
    """)

    st.divider()

    def generar_enlace_dlr_europa_actual():
        ahora = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
        anio, mes, dia, hora = ahora.year, ahora.month, ahora.day, ahora.hour
        minuto = (ahora.minute // 15) * 15
        
        fecha_fin = datetime.datetime(anio, mes, dia, hora, minuto, 0)
        str_anio = fecha_fin.strftime("%Y")
        str_doy = fecha_fin.strftime("%j")
        str_hora = fecha_fin.strftime("%H")

        fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
        timestamp_inicio = fecha_inicio.strftime("%Y-%m-%dT%H-%M-%S")
        timestamp_fin = fecha_fin.strftime("%Y-%m-%dT%H-%M-%S")

        base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
        nombre_archivo = f"DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"
        return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/{nombre_archivo}"

    url_europa = generar_enlace_dlr_europa_actual()
    url_global = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL/v2.0.0/latest/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL_latest_D.json"
    headers = {"User-Agent": "Mozilla/5.0"}

    @st.cache_data(ttl=900)
    def cargar_datos_vtec():
        res_eur = requests.get(url_europa, headers=headers, timeout=15)
        res_eur.raise_for_status()
        data_eur = res_eur.json()
        vtec_eur_list = [f['properties']['vtec_assimilated_tecu'] for f in data_eur['data']['grid']['features']]
        matriz_vtec_eur = np.array(vtec_eur_list).reshape(43, 81)
        
        res_glb = requests.get(url_global, headers=headers, timeout=15)
        res_glb.raise_for_status()
        data_glb = res_glb.json()
        vtec_glb_list = [f['properties']['vtec_assimilated_tecu'] for f in data_glb['data']['grid']['features']]
        matriz_vtec_glb = np.array(vtec_glb_list).reshape(73, 73)
        
        return matriz_vtec_eur, matriz_vtec_glb

    try:
        matriz_vtec_eur, matriz_vtec_glb = cargar_datos_vtec()
        lons_eur, lats_eur = np.arange(-30, 51, 1), np.arange(30, 73, 1)
        lons_glb, lats_glb = np.linspace(-180, 180, 73), np.linspace(-90, 90, 73)
        
        interp_europa = RegularGridInterpolator((lats_eur, lons_eur), matriz_vtec_eur, method='linear', bounds_error=False, fill_value=None)
        interp_global = RegularGridInterpolator((lats_glb, lons_glb), matriz_vtec_glb, method='linear', bounds_error=False, fill_value=None)

        st.subheader("🔍 Consulta de TECU por Localidad (Tiempo Real)")
        localidad_usuario = st.text_input("Escribe el nombre de una ciudad o región:", key="loc_actual")

        if localidad_usuario:
            lat, lon, nombre_completo = geocodificar_localidad(localidad_usuario)
            if lat is not None:
                dentro_europa = (30 <= lat <= 72) and (-30 <= lon <= 50)
                punto_consulta = np.array([[lat, lon]])
                if dentro_europa:
                    valor_tecu = float(interp_europa(punto_consulta)[0])
                    fuente = "Malla Regional de Europa (Alta Precisión)"
                else:
                    valor_tecu = float(interp_global(punto_consulta)[0])
                    fuente = "Malla Planetaria Global"
                    
                col1, col2, col3 = st.columns(3)
                col1.metric(label="📍 Ubicación", value=localidad_usuario.capitalize())
                col2.metric(label="📡 Valor VTEC", value=f"{valor_tecu:.3f} TECU")
                col3.info(f"**Coordenadas:** {lat:.2f}°N, {lon:.2f}°E\n\n**Fuente:** {fuente}")
            else:
                st.error("No se pudo encontrar la localización.")

        st.divider()

        st.subheader("🗺️ Mapas de Contenido Total de Electrones (VTEC) en Tiempo Real")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})

        ax1.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=1)
        ax1.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
        ax1.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
        ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#666666', zorder=3)
        g1 = ax1.gridlines(draw_labels=True, color='gray', alpha=0.25, linestyle='--')
        g1.top_labels, g1.right_labels = False, False
        grid_lon_eur, grid_lat_eur = np.meshgrid(lons_eur, lats_eur)
        map_eur = ax1.pcolormesh(grid_lon_eur, grid_lat_eur, matriz_vtec_eur, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', zorder=2)
        fig.colorbar(map_eur, ax=ax1, orientation='horizontal', pad=0.07, shrink=0.7).set_label('VTEC EUROPA (TECU)', weight='bold')

        ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=1)
        ax2.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
        ax2.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.0, zorder=3)
        ax2.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
        g2 = ax2.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
        g2.top_labels, g2.right_labels = False, False
        grid_lon_glb, grid_lat_glb = np.meshgrid(lons_glb, lats_glb)
        map_glb = ax2.pcolormesh(grid_lon_glb, grid_lat_glb, matriz_vtec_glb, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.8, shading='gouraud', zorder=2)
        fig.colorbar(map_glb, ax=ax2, orientation='horizontal', pad=0.07, shrink=0.7).set_label('VTEC GLOBAL (TECU)', weight='bold')

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error en Tiempo Real: {e}")

# =====================================================================
# PESTAÑA 2: ANÁLISIS EN EL PASADO (REDISEÑADA COMPLETAMENTE)
# =====================================================================
with tab2:
    st.title("📊 Análisis Histórico: Mapas e Interpolar en el Pasado")
    st.write("Selecciona una fecha y una hora histórica. El sistema buscará en los repositorios oficiales del DLR.")

    # Selectores de fecha y hora interactivos
    col_f1, col_f2, col_f3 = st.columns(3)
    fecha_sel = col_f1.date_input("Selecciona la Fecha:", datetime.date(2026, 1, 24))
    hora_sel = col_f2.slider("Hora (UTC):", 0, 23, 4)
    minuto_sel = col_f3.slider("Minuto:", 0, 55, 0, step=5)

    # Redondeo forzado automático para evitar links rotos (Bloques válidos de 15 minutos en el DLR)
    minuto_ajustado = (minuto_sel // 15) * 15
    if minuto_ajustado != minuto_sel:
        st.caption(f"ℹ️ Los datos se aproximaron automáticamente al bloque válido más cercano del DLR: **{hora_sel:02d}:{minuto_ajustado:02d} UTC**.")

    # Generación dinámica del link histórico
    def generar_enlace_dlr_pasado(anio, mes, dia, hora, minuto):
        fecha_fin = datetime.datetime(anio, mes, dia, hora, minuto, 0)
        str_anio = fecha_fin.strftime("%Y")
        str_doy = fecha_fin.strftime("%j")
        str_hora = fecha_fin.strftime("%H")

        fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
        timestamp_inicio = fecha_inicio.strftime("%Y-%m-%dT%H-%M-%S")
        timestamp_fin = fecha_fin.strftime("%Y-%m-%dT%H-%M-%S")

        base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
        nombre_archivo = f"DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"
        return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/{nombre_archivo}"

    url_pasado = generar_enlace_dlr_pasado(fecha_sel.year, fecha_sel.month, fecha_sel.day, hora_sel, minuto_ajustado)

    # Botón ejecutor para evitar que se sature la red pidiendo datos con cada click del slider
    if st.button("🚀 Cargar Mapa Histórico de Europa"):
        headers = {"User-Agent": "Mozilla/5.0"}
        
        with st.spinner("Conectando y validando base de datos histórica del DLR..."):
            try:
                response = requests.get(url_pasado, headers=headers, timeout=12)
                response.raise_for_status() # Lanza error si el archivo no existe (Evita datos falsos)
                data_p = response.json()

                vtec_p_list = []
                if 'data' in data_p and 'grid' in data_p['data']:
                    for feature in data_p['data']['grid']['features']:
                        vtec_p_list.append(feature['properties']['vtec_assimilated_tecu'])

                # Protección estricta contra archivos corruptos/vacíos
                if len(vtec_p_list) != 3483:
                    st.error("🚨 Alarma: El archivo de esta fecha no cumple con la estructura reglamentaria de 3483 puntos.")
                else:
                    matriz_pasado = np.array(vtec_p_list).reshape(43, 81)
                    lons_p, lats_p = np.arange(-30, 51, 1), np.arange(30, 73, 1)

                    st.success(f"📌 Archivo validado y descargado con éxito.")

                    # --- SECCIÓN CONSULTA LOCALIDAD PASADA ---
                    st.markdown("### 🔍 Consulta de Localidad en esta Fecha")
                    localidad_p_usuario = st.text_input("Ingresa una ciudad para conocer su TECU histórico:", "Madrid", key="loc_pasada")
                    
                    if localidad_p_usuario:
                        lat_p, lon_p, _ = geocodificar_localidad(localidad_p_usuario)
                        if lat_p is not None and (30 <= lat_p <= 7
