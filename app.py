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

# FUNCIÓN DE GEOCODIFICACIÓN MEJORADA CON COORDENADAS DE RESPALDO (ANTI-BLOQUEOS)
def geocodificar_localidad(nombre_lugar):
    nombre_clean = nombre_lugar.strip().lower()
    
    # Diccionario local para asegurar que las ciudades principales siempre funcionen al instante
    ciudades_respaldo = {
        "madrid": (40.4167, -3.7037),
        "barcelona": (41.3851, 2.1734),
        "valencia": (39.4699, -0.3763),
        "sevilla": (37.3891, -5.9845),
        "zaragoza": (41.6488, -0.8891),
        "malaga": (36.7212, -4.4214),
        "paris": (48.8566, 2.3522),
        "berlin": (52.5200, 13.4050),
        "roma": (41.9028, 12.4964),
        "londres": (51.5074, -0.1278),
        "lisboa": (38.7223, -9.1393)
    }
    
    # Si la ciudad está en nuestra lista local, la devolvemos de inmediato (100% fiable)
    if nombre_clean in ciudades_respaldo:
        lat, lon = ciudades_respaldo[nombre_clean]
        return lat, lon, nombre_lugar.capitalize()
        
    # Si no está en el diccionario, intentamos buscarla en la API externa
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={nombre_lugar}&format=json&limit=1"
        headers = {"User-Agent": "Streamlit_Ionosphere_App_Final_v3"}
        res = requests.get(url, headers=headers, timeout=5)
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
        
        cbar_eur = fig.colorbar(map_eur, ax=ax1, orientation='horizontal', pad=0.07, shrink=0.7)
        cbar_eur.set_label('VTEC EUROPA (TECU)', weight='bold')

        ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=1)
        ax2.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
        ax2.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.0, zorder=3)
        ax2.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
        g2 = ax2.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--')
        g2.top_labels, g2.right_labels = False, False
        grid_lon_glb, grid_lat_glb = np.meshgrid(lons_glb, lats_glb)
        map_glb = ax2.pcolormesh(grid_lon_glb, grid_lat_glb, matriz_vtec_glb, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.8, shading='gouraud', zorder=2)
        
        cbar_glb = fig.colorbar(map_glb, ax=ax2, orientation='horizontal', pad=0.07, shrink=0.7)
        cbar_glb.set_label('VTEC GLOBAL (TECU)', weight='bold')

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error en Tiempo Real: {e}")

# =====================================================================
# PESTAÑA 2: ANÁLISIS EN EL PASADO
# =====================================================================
with tab2:
    st.title("📊 Análisis Histórico: Mapas e Interpolar en el Pasado")
    st.write("Selecciona una fecha y una hora histórica. El sistema buscará en los repositorios oficiales del DLR.")

    if 'matriz_pasado' not in st.session_state:
        st.session_state.matriz_pasado = None
    if 'fecha_mapa' not in st.session_state:
        st.session_state.fecha_mapa = ""

    col_f1, col_f2, col_f3 = st.columns(3)
    fecha_sel = col_f1.date_input("Selecciona la Fecha:", datetime.date(2026, 1, 24))
    hora_sel = col_f2.slider("Hora (UTC):", 0, 23, 4)
    minuto_sel = col_f3.slider("Minuto:", 0, 55, 0, step=5)

    minuto_ajustado = (minuto_sel // 15) * 15
    if minuto_ajustado != minuto_sel:
        st.caption(f"ℹ️ Los datos se aproximaron automáticamente al bloque de 15 min: **{hora_sel:02d}:{minuto_ajustado:02d} UTC**.")

    def generar_enlace_dlr_pasado(anio, mes, dia, hora, minuto):
        fecha_fin = datetime.datetime(anio, mes, dia, hora, minuto, 0)
        str_anio = fecha_fin.strftime("%Y")
        str_doy = fecha_fin.strftime("%j")
        str_hora = fecha_fin.strftime("%H")
        fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
        timestamp_inicio = fecha_inicio.strftime("%Y-%m-%dT%H-%M-%S")
        timestamp_fin = fecha_fin.strftime("%Y-%m-%dT%H-%M-%S")
        base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
        return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"

    url_pasado = generar_enlace_dlr_pasado(fecha_sel.year, fecha_sel.month, fecha_sel.day, hora_sel, minuto_ajustado)

    if st.button("🚀 Cargar Mapa Histórico de Europa"):
        headers = {"User-Agent": "Mozilla/5.0"}
        with st.spinner("Descargando base de datos histórica del DLR..."):
            try:
                response = requests.get(url_pasado, headers=headers, timeout=12)
                response.raise_for_status() 
                data_p = response.json()

                vtec_p_list = []
                if 'data' in data_p and 'grid' in data_p['data']:
                    for feature in data_p['data']['grid']['features']:
                        vtec_p_list.append(feature['properties']['vtec_assimilated_tecu'])

                if len(vtec_p_list) != 3483:
                    st.error("🚨 El archivo de esta fecha está corrupto o incompleto.")
                else:
                    st.session_state.matriz_pasado = np.array(vtec_p_list).reshape(43, 81)
                    st.session_state.fecha_mapa = f"{fecha_sel.strftime('%d/%m/%Y')} - {hora_sel:02d}:{minuto_ajustado:02d} UTC"
                    st.success(f"📌 Archivo de la fecha {st.session_state.fecha_mapa} cargado correctamente.")
            except Exception:
                st.error(f"❌ No existen registros en el DLR para la fecha/hora {hora_sel:02d}:{minuto_ajustado:02d} del {fecha_sel.strftime('%d/%m/%Y')}.")

    # AREA DE TRABAJO HISTORICA SECUENCIAL
    if st.session_state.matriz_pasado is not None:
        st.divider()
        st.markdown(f"### 🔍 Consulta de Localidad en el Pasado ({st.session_state.fecha_mapa})")
        
        with st.form("formulario_consulta_pasado"):
            localidad_p_usuario = st.text_input("Ingresa una ciudad para conocer su TECU histórico (Ej: madrid, barcelona, sevilla...):", "madrid")
            boton_consultar_ciudad = st.form_submit_button("Calcular TECU")

        lons_p, lats_p = np.arange(-30, 51, 1), np.arange(30, 73, 1)

        if boton_consultar_ciudad and localidad_p_usuario:
            lat_p, lon_p, nombre_completo_p = geocodificar_localidad(localidad_p_usuario)
            
            if lat_p is not None and (30 <= lat_p <= 72) and (-30 <= lon_p <= 50):
                interp_p = RegularGridInterpolator((lats_p, lons_p), st.session_state.matriz_pasado, method='linear', bounds_error=False, fill_value=None)
                val_tecu_p = float(interp_p(np.array([[lat_p, lon_p]]))[0])
                
                # Muestra el resultado de forma súper vistosa
                st.metric(label=f"📡 Valor VTEC obtenido en {localidad_p_usuario.capitalize()}", value=f"{val_tecu_p:.3f} TECU")
                st.info(f"📌 **Datos de localización usados:** Latitud: {lat_p:.3f}° | Longitud: {lon_p:.3f}°")
            else:
                st.warning("La localidad indicada está fuera del rango de cobertura de Europa o el buscador externo no respondió. Prueba con otra ciudad cercana.")

        # RENDERIZADO DEL MAPA HISTÓRICO
        st.markdown("### 🗺️ Malla Regional de Europa Reconstruida")
        fig_p = plt.figure(figsize=(11, 6), dpi=100)
        ax_p = plt.axes(projection=ccrs.PlateCarree())
        ax_p.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
        
        ax_p.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=1)
        ax_p.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
        ax_p.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.2, zorder=3)
        ax_p.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#666666', zorder=3)
        
        grid_p = ax_p.gridlines(draw_labels=True, color='gray', alpha=0.3, linestyle='--')
        grid_p.top_labels, grid_p.right_labels = False, False
        
        grid_lon_p, grid_lat_p = np.meshgrid(lons_p, lats_p)
        mapa_p = ax_p.pcolormesh(grid_lon_p, grid_lat_p, st.session_state.matriz_pasado, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', zorder=2)
        
        cbar_p = plt.colorbar(mapa_p, ax=ax_p, orientation='vertical', pad=0.02, shrink=0.8)
        cbar_p.set_label('VTEC ASSIMILATED (TECU)', weight='bold')
        
        plt.title(f"MAPA DE TEC RECONSTRUIDO\nFECHA: {st.session_state.fecha_mapa}", fontsize=11, weight='bold')
        st.pyplot(fig_p)

# =====================================================================
# PESTAÑA 3: HERRAMIENTAS GNSS
# =====================================================================
with tab3:
    st.title("🛠️ Calculadoras y Conversores GNSS")
    st.write("Espacio listo para implementar algoritmos adicionales de procesamiento de retraso de onda ionosférica.")
