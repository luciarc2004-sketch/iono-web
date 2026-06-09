import datetime
import requests
import time
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
# CONFIGURACIÓN GLOBAL
# =====================================================================
MINUTOS_CONTIGUOS_GLOBAL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

# Definición global de las pestañas obligatorias
tab1, tab2, tab3 = st.tabs(["🌍 Inicio y Monitoreo Real", "📊 Análisis en el pasado", "📈 Evolución TECU"])

# FUNCIÓN COMPARTIDA DE GEOCODIFICACIÓN AMPLIADA A TODA LA MALLA COBIERTA
def geocodificar_localidad(nombre_lugar):
    nombre_clean = nombre_lugar.strip().lower()
    
    # Diccionario ampliado con puntos estratégicos dentro de los límites de la cuadrícula (-30 a 50 Lon, 30 a 72 Lat)
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
        "lisboa": (38.7223, -9.1393),
        # Localidades no europeas dentro de la malla geográfica del DLR:
        "rabat": (34.0209, -6.8416),
        "el cairo": (30.0444, 31.2357),
        "tunez": (36.8065, 10.1815),
        "argel": (36.7538, 3.0588),
        "reikiavik": (64.1466, -21.9426),
        "ankara": (39.9334, 32.8597)
    }
    
    if nombre_clean in ciudades_respaldo:
        lat, lon = ciudades_respaldo[nombre_clean]
        return lat, lon, nombre_lugar.capitalize()
        
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={nombre_lugar}&format=json&limit=1"
        headers = {"User-Agent": "Streamlit_Ionosphere_App_v3"}
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
    * **Variabilidad Espacio-Temporal:** Durante tormentas solares, el TEC aumenta drácticamente de forma impredecible, afectando la precisión de servicios de alta precisión (como RTK o navegación aérea guiada por satélite).
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
                    fuente = "Malla Regional (Alta Precisión - Res: 1°)"
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
        cbar_eur.set_label('VTEC MALLA REGIONAL (TECU)', weight='bold')

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

    if st.button("🚀 Cargar Mapa Histórico"):
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

    if st.session_state.matriz_pasado is not None:
        st.divider()
        st.markdown(f"### 🔍 Consulta de Localidad en el Pasado ({st.session_state.fecha_mapa})")
        
        with st.form("formulario_consulta_pasado"):
            localidad_p_usuario = st.text_input("Ingresa cualquier localidad dentro del recuadro del mapa:", "Madrid")
            boton_consultar_ciudad = st.form_submit_button("Calcular TECU")

        lons_p, lats_p = np.arange(-30, 51, 1), np.arange(30, 73, 1)

        if boton_consultar_ciudad and localidad_p_usuario:
            lat_p, lon_p, nombre_completo_p = geocodificar_localidad(localidad_p_usuario)
            
            # CORREGIDO: El filtro valida estrictamente los límites cartográficos
            if lat_p is not None and (30 <= lat_p <= 72) and (-30 <= lon_p <= 50):
                interp_p = RegularGridInterpolator((lats_p, lons_p), st.session_state.matriz_pasado, method='linear', bounds_error=False, fill_value=None)
                val_tecu_p = float(interp_p(np.array([[lat_p, lon_p]]))[0])
                
                st.metric(label=f"Valor en {localidad_p_usuario.capitalize()}", value=f"{val_tecu_p:.3f} TECU")
                st.caption(f"📍 Coordenadas utilizadas para el cálculo: {lat_p:.3f}°N, {lon_p:.3f}°E")
            else:
                st.warning("La localidad indicada está fuera del área de cobertura de la malla (-30° a 50° Lon, 30° a 72° Lat) o no fue encontrada.")

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
        
        cbar_p = plt.colorbar(mapa_p, ax=ax_p, orientation='horizontal', pad=0.08, shrink=0.7)
        cbar_p.set_label('VTEC ASSIMILATED (TECU)', weight='bold')
        
        plt.title(f"MAPA DE TEC RECONSTRUIDO\nFECHA: {st.session_state.fecha_mapa}", fontsize=11, weight='bold')
        st.pyplot(fig_p)

# =====================================================================
# PESTAÑA 3: EVOLUCIÓN TECU
# =====================================================================
with tab3:
    st.title("📈 Estudio de Evolución Temporal del TECU")
    
    modo_evolucion = st.radio("Selecciona el tipo de análisis temporal:", ["Por Días", "Por Horas"], horizontal=True)

    # -----------------------------------------------------------------
    # MÓDULO: POR DÍAS
    # -----------------------------------------------------------------
    if modo_evolucion == "Por Días":
        st.subheader("📆 Análisis de Evolución Interdiaria (Hora Fija)")
        
        if 'historial_vtec_3d' not in st.session_state:
            st.session_state.historial_vtec_3d = None
            st.session_state.etiquetas_fechas_reales = []
            st.session_state.limites_globales = (0, 15)
            st.session_state.matriz_maximos = None
            st.session_state.ciudades_lista = []

        col_c1, col_c2, col_c3 = st.columns(3)
        fecha_inicial = col_c1.date_input("Fecha Inicial:", datetime.date(2026, 2, 19), key="ev_fecha_ini")
        hora_fija_sel = col_c2.slider("Hora fija de observación (UTC):", 0, 23, 15)
        num_dias_sel = col_c3.slider("Número de días a evaluar:", 2, 15, 10)

        def generar_enlace_dlr_rango(anio, mes, dia, hora, minuto):
            fecha_fin = datetime.datetime(anio, mes, dia, hora, minuto, 0)
            str_anio = fecha_fin.strftime("%Y")
            str_doy = fecha_fin.strftime("%j")
            str_hora = fecha_fin.strftime("%H")
            fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
            timestamp_inicio = fecha_inicio.strftime("%Y-%m-%dT%H-%M-%S")
            timestamp_fin = fecha_fin.strftime("%Y-%m-%dT%H-%M-%S")
            base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
            return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"

        if st.button("🚀 Procesar Rango de Días"):
            headers = {"User-Agent": "Mozilla/5.0"}
            
            temp_etiquetas = []
            temp_3d = np.zeros((num_dias_sel, 43, 81))
            progreso = st.progress(0.0)
            exito_total = True

            for d in range(num_dias_sel):
                fecha_actual = datetime.datetime(fecha_inicial.year, fecha_inicial.month, fecha_inicial.day) + datetime.timedelta(days=d)
                link_exitoso = False
                data = None

                for m in MINUTOS_CONTIGUOS_GLOBAL:
                    url_intento = generar_enlace_dlr_rango(fecha_actual.year, fecha_actual.month, fecha_actual.day, hora_fija_sel, m)
                    try:
                        response = requests.get(url_intento, headers=headers, timeout=4)
                        if response.status_code == 200:
                            data = response.json()
                            link_exitoso = True
                            minuto_exitoso = m
                            break
                    except Exception:
                        pass

                if not link_exitoso:
                    st.error(f"❌ Error: No hay datos disponibles para el día {fecha_actual.strftime('%d/%m/%Y')}. Operación cancelada.")
                    exito_total = False
                    break

                vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                if len(vtec_values_list) == 3483:
                    temp_3d[d, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} ({hora_fija_sel:02d}:{minuto_exitoso:02d})")
                
                progreso.progress((d + 1) / num_dias_sel)

            if exito_total:
                st.session_state.historial_vtec_3d = temp_3d
                st.session_state.etiquetas_fechas_reales = temp_etiquetas
                
                max_r = np.max(temp_3d)
                min_r = np.max([0.0, np.min(temp_3d)])
                st.session_state.limites_globales = (max(0.0, float(np.floor(min_r - 2))), float(np.ceil(max_r + 2)))
                st.session_state.matriz_maximos = np.max(temp_3d, axis=0)
                st.success("📊 Rango temporal procesado con éxito.")

        if st.session_state.historial_vtec_3d is not None:
            st.divider()
            
            v_min, v_max = st.session_state.limites_globales
            lons_vector = np.arange(-30, 51, 1)
            lats_vector = np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            st.subheader("📌 Mapa Fijo de Máximos Absolutos Registrados")
            fig_max, ax_mx = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mx.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_mx.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            ax_mx.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
            ax_mx.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False

            mapa_maximos = ax_mx.pcolormesh(grid_lon, grid_lat, st.session_state.matriz_maximos, 
                                            transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
            
            fig_max.colorbar(mapa_maximos, ax=ax_mx, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO (TECU)', weight='bold')
            ax_mx.set_title("Distribución de Intensidades Máximas Observadas", weight='bold')
            st.pyplot(fig_max)
            plt.close(fig_max)

            st.divider()

            st.subheader("🎬 Reproductor de Video: Evolución Diaria del TEC (0.5s por Frame)")
            col_b1, _, _ = st.columns([1, 1, 4])
            play_video = col_b1.button("▶️ Reproducir Video", key="play_dias")
            contenedor_video_mapa = st.empty()

            if play_video:
                num_frames = len(st.session_state.etiquetas_fechas_reales)
                for f in range(num_frames):
                    fig_video, ax_ev = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_ev.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_ev.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_ev.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    ax_ev.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
                    ax_ev.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
                    
                    matriz_frame = st.session_state.historial_vtec_3d[f, :, :]
                    mapa_dinamico = ax_ev.pcolormesh(grid_lon, grid_lat, matriz_frame, 
                                                     transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
                    
                    fig_video.colorbar(mapa_dinamico, ax=ax_ev, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    ax_ev.set_title(f"Video en Curso ➔ Fecha: {st.session_state.etiquetas_fechas_reales[f]} UTC", weight='bold', color='#1976d2')
                    
                    contenedor_video_mapa.pyplot(fig_video)
                    plt.close(fig_video)
                    time.sleep(0.5)
            else:
                fig_video, ax_ev = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                ax_ev.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                ax_ev.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                ax_ev.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                ax_ev.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
                ax_ev.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
                
                mapa_dinamico = ax_ev.pcolormesh(grid_lon, grid_lat, st.session_state.historial_vtec_3d[0, :, :], 
                                                 transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
                fig_video.colorbar(mapa_dinamico, ax=ax_ev, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                ax_ev.set_title("Video Listo ➔ Presiona Play para iniciar la simulación", weight='bold')
                contenedor_video_mapa.pyplot(fig_video)
                plt.close(fig_video)

            st.divider()

            st.subheader("📊 3. Gráfica Comparativa de Localidades Acumuladas")
            with st.form("formulario_acumulador_ciudades"):
                nueva_ciudad = st.text_input("Ingresa cualquier localidad del mapa para agregarla a la gráfica:", "madrid")
                boton_agregar = st.form_submit_button("➕ Añadir Localidad")

            if boton_agregar and nueva_ciudad:
                lat_c, lon_c, _ = geocodificar_localidad(nueva_ciudad)
                # CORREGIDO: Malla libre de rango para admitir cualquier punto geográfico analizado
                if lat_c is not None and (30 <= lat_c <= 72) and (-30 <= lon_c <= 50):
                    if nueva_ciudad.capitalize() not in [c['name'] for c in st.session_state.ciudades_lista]:
                        st.session_state.ciudades_lista.append({'name': nueva_ciudad.capitalize(), 'lat': lat_c, 'lon': lon_c})
                        st.success(f"Añadida {nueva_ciudad.capitalize()} al histórico.")
                else:
                    st.error("Ubicación no encontrada o fuera del área cartográfica (-30° a 50° Lon, 30° a 72° Lat).")

            if st.session_state.ciudades_lista:
                fig_lineas, ax_lineas = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.ciudades_lista:
                    idx_lat = (np.abs(lats_vector - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(lons_vector - ciudad_obj['lon'])).argmin()
                    perfil_temporal = st.session_state.historial_vtec_3d[:, idx_lat, idx_lon]
                    ax_lineas.plot(range(len(st.session_state.etiquetas_fechas_reales)), perfil_temporal, marker='s', linestyle='-', linewidth=2, label=ciudad_obj['name'])

                ax_lineas.grid(True, linestyle='--', alpha=0.6)
                ax_lineas.set_ylim(v_min, v_max)
                ax_lineas.set_xticks(range(len(st.session_state.etiquetas_fechas_reales)))
                ax_lineas.set_xticklabels(st.session_state.etiquetas_fechas_reales, rotation=25)
                ax_lineas.set_ylabel("VTEC (TECU)", weight='bold')
                ax_lineas.set_title(f"Evolución Comparativa (Eje Y: {int(v_min)}-{int(v_max)} TECU)", weight='bold')
                ax_lineas.legend(loc="upper right")
                st.pyplot(fig_lineas)
                plt.close(fig_lineas)
                
                if st.button("🗑️ Limpiar todas las localidades", key="clear_dias"):
                    st.session_state.ciudades_lista = []
                    st.rerun()

    # -----------------------------------------------------------------
    # MÓDULO: POR HORAS
    # -----------------------------------------------------------------
    elif modo_evolucion == "Por Horas":
        st.subheader("⏱️ Análisis de Evolución Intradía (Hora por Hora - 24h)")
        
        if 'h_historial_vtec_3d' not in st.session_state:
            st.session_state.h_historial_vtec_3d = None
            st.session_state.h_etiquetas_reales = []
            st.session_state.h_limites_globales = (0, 15)
            st.session_state.h_matriz_maximos = None
            st.session_state.h_ciudades_lista = []

        col_h1, _ = st.columns([1, 2])
        fecha_analisis_h = col_h1.date_input("Selecciona el día a analizar:", datetime.date(2026, 1, 24), key="ev_fecha_hor")

        def generar_enlace_dlr_horas(anio, mes, dia, hora, minuto):
            fecha_fin = datetime.datetime(anio, mes, dia, hora, minuto, 0)
            str_anio = fecha_fin.strftime("%Y")
            str_doy = fecha_fin.strftime("%j")
            str_hora = fecha_fin.strftime("%H")
            fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
            timestamp_inicio = fecha_inicio.strftime("%Y-%m-%dT%H-%M-%S")
            timestamp_fin = fecha_fin.strftime("%Y-%m-%dT%H-%M-%S")
            base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
            return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"

        if st.button("🚀 Procesar las 24 Horas"):
            headers = {"User-Agent": "Mozilla/5.0"}
            
            h_temp_etiquetas = []
            h_temp_3d = np.zeros((24, 43, 81))
            progreso_h = st.progress(0.0)
            h_exito_total = True

            for h in range(24):
                link_exitoso = False
                data = None
                minuto_exitoso = 0

                for m in MINUTOS_CONTIGUOS_GLOBAL:
                    url_intento = generar_enlace_dlr_horas(fecha_analisis_h.year, fecha_analisis_h.month, fecha_analisis_h.day, h, m)
                    try:
                        response = requests.get(url_intento, headers=headers, timeout=4)
                        if response.status_code == 200:
                            data = response.json()
                            link_exitoso = True
                            minuto_exitoso = m
                            break
                    except Exception:
                        pass

                if not link_exitoso:
                    st.error(f"❌ Error: Archivos no encontrados en el servidor DLR para la hora {h:02d}:00 UTC. Proceso cancelado.")
                    h_exito_total = False
                    break

                vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                if len(vtec_values_list) == 3483:
                    h_temp_3d[h, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    h_temp_etiquetas.append(f"{h:02d}:{minuto_exitoso:02d}")
                
                progreso_h.progress((h + 1) / 24)

            if h_exito_total:
                st.session_state.h_historial_vtec_3d = h_temp_3d
                st.session_state.h_etiquetas_reales = h_temp_etiquetas
                
                max_r = np.max(h_temp_3d)
                min_r = np.max([0.0, np.min(h_temp_3d)])
                st.session_state.h_limites_globales = (max(0.0, float(np.floor(min_r - 2))), float(np.ceil(max_r + 2)))
                st.session_state.h_matriz_maximos = np.max(h_temp_3d, axis=0)
                st.success("📊 Las 24 horas del día han sido procesadas con éxito.")

        if st.session_state.h_historial_vtec_3d is not None:
            st.divider()
            
            vh_min, vh_max = st.session_state.h_limites_globales
            lons_vector = np.arange(-30, 51, 1)
            lats_vector = np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            st.subheader("📌 Mapa Fijo de Máximos Absolutos del Día")
            fig_max_h, ax_mxh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mxh.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mxh.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_mxh.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_mxh.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            ax_mxh.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
            ax_mxh.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False

            mapa_maximos_h = ax_mxh.pcolormesh(grid_lon, grid_lat, st.session_state.h_matriz_maximos, 
                                               transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vh_min, vmax=vh_max, zorder=2)
            
            fig_max_h.colorbar(mapa_maximos_h, ax=ax_mxh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO HORARIO (TECU)', weight='bold')
            ax_mxh.set_title(f"Intensidad Máxima Alcanzada por Coordenada en las 24h ({fecha_analisis_h.strftime('%d/%m/%Y')})", weight='bold')
            st.pyplot(fig_max_h)
            plt.close(fig_max_h)

            st.divider()

            st.subheader("🎬 Reproductor Horario: Evolución Intradía (0.5s por Frame)")
            col_bh1, _, _ = st.columns([1, 1, 4])
            play_video_h = col_bh1.button("▶️ Reproducir Video Horario", key="play_horas")
            contenedor_video_horas = st.empty()

            if play_video_h:
                for f in range(24):
                    fig_vid_h, ax_evh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_evh.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    ax_evh.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_evh.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_evh.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    ax_evh.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
                    ax_evh.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
                    
                    matriz_frame_h = st.session_state.h_historial_vtec_3d[f, :, :]
                    mapa_dinamico_h = ax_evh.pcolormesh(grid_lon, grid_lat, matriz_frame_h, 
                                                        transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vh_min, vmax=vh_max, zorder=2)
                    
                    fig_vid_h.colorbar(mapa_dinamico_h, ax=ax_evh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    ax_evh.set_title(f"Video Horario ➔ Hora: {st.session_state.h_etiquetas_reales[f]} UTC", weight='bold', color='#1976d2')
                    
                    contenedor_video_horas.pyplot(fig_vid_h)
                    plt.close(fig_vid_h)
                    time.sleep(0.5)
            else:
                fig_vid_h, ax_evh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                ax_evh.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                ax_evh.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                ax_evh.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                ax_evh.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                ax_evh.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
                ax_evh.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
                
                mapa_dinamico_h = ax_evh.pcolormesh(grid_lon, grid_lat, st.session_state.h_historial_vtec_3d[0, :, :], 
                                                    transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vh_min, vmax=vh_max, zorder=2)
                fig_vid_h.colorbar(mapa_dinamico_h, ax=ax_evh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                ax_evh.set_title("Video Listo ➔ Presiona Play para iniciar la línea de tiempo horaria", weight='bold')
                contenedor_video_horas.pyplot(fig_vid_h)
                plt.close(fig_vid_h)

            st.divider()

            # --- GRÁFICA COMPARATIVA ACUMULATIVA HORARIA ---
            st.subheader("📊 3. Gráfica Comparativa de Localidades Acumuladas (24 Horas)")
            with st.form("formulario_acumulador_ciudades_horas"):
                nueva_ciudad_h = st.text_input("Ingresa cualquier localidad del mapa para el análisis de 24h:", "madrid")
                boton_agregar_h = st.form_submit_button("➕ Añadir Localidad")

            if boton_agregar_h and nueva_ciudad_h:
                lat_ch, lon_ch, _ = geocodificar_localidad(nueva_ciudad_h)
                # CORREGIDO: Malla libre de rango horario
                if lat_ch is not None and (30 <= lat_ch <= 72) and (-30 <= lon_ch <= 50):
                    if nueva_ciudad_h.capitalize() not in [c['name'] for c in st.session_state.h_ciudades_lista]:
                        st.session_state.h_ciudades_lista.append({'name': nueva_ciudad_h.capitalize(), 'lat': lat_ch, 'lon': lon_ch})
                        st.success(f"Añadida {nueva_ciudad_h.capitalize()} al análisis horario.")
                else:
                    st.error("Ubicación no encontrada o fuera del área cartográfica (-30° a 50° Lon, 30° a 72° Lat).")

            if st.session_state.h_ciudades_lista:
                fig_lineas_h, ax_lineas_h = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.h_ciudades_lista:
                    idx_lat = (np.abs(lats_vector - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(lons_vector - ciudad_obj['lon'])).argmin()
                    perfil_temporal_h = st.session_state.h_historial_vtec_3d[:, idx_lat, idx_lon]
                    ax_lineas_h.plot(range(24), perfil_temporal_h, marker='o', linestyle='-', linewidth=2, label=ciudad_obj['name'])

                ax_lineas_h.grid(True, linestyle='--', alpha=0.6)
                ax_lineas_h.set_ylim(vh_min, vh_max)
                ax_lineas_h.set_xticks(range(24))
                ax_lineas_h.set_xticklabels([f"{h:02d}h" for h in range(24)], rotation=45)
                ax_lineas_h.set_ylabel("VTEC (TECU)", weight='bold')
                ax_lineas_h.set_xlabel("Línea Temporal de las 24 Horas (UTC)", weight='bold')
                ax_lineas_h.set_title(f"Evolución Intradía Comparativa (Eje Y: {int(vh_min)}-{int(vh_max)} TECU)", weight='bold')
                ax_lineas_h.legend(loc="upper right")
                st.pyplot(fig_lineas_h)
                plt.close(fig_lineas_h)
                
                if st.button("🗑️ Limpiar todas las localidades", key="clear_horas"):
                    st.session_state.h_ciudades_lista = []
                    st.rerun()
