import datetime
import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import RegularGridInterpolator
import streamlit as st

# Configuración de la página web limpia
st.set_page_config(page_title="Portal de Monitoreo Ionosférico", layout="wide")

# =====================================================================
# CONFIGURACIÓN GLOBAL ESTRICTA (REGLA 0-55 TECU BASE Y COORDENADAS)
# =====================================================================
MINUTOS_CONTIGUOS_GLOBAL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
VMIN_TECU_FIJO = 0.0
VMAX_TECU_FIJO = 55.0

# Definición de la mallas estricta de Europa (Versión A)
LAT_MIN, LAT_MAX, DELTA_LAT = 30, 72, 1
LON_MIN, LON_MAX, DELTA_LON = -30, 50, 1

LATS_EUROPA = np.arange(LAT_MIN, LAT_MAX + DELTA_LAT, DELTA_LAT)
LONS_EUROPA = np.arange(LON_MIN, LON_MAX + DELTA_LON, DELTA_LON)
GRID_LON_EUR, GRID_LAT_GRID = np.meshgrid(LONS_EUROPA, LATS_EUROPA)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Inicio y Monitoreo Real", 
    "📊 Análisis en el pasado", 
    "📈 Evolución TECU", 
    "🔮 Pronóstico", 
    "📉 Desviaciones del Modelo"
])

def geocodificar_localidad(nombre_lugar):
    nombre_clean = nombre_lugar.strip().lower()
    nombre_clean = nombre_clean.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    
    ciudades_respaldo = {
        "madrid": (40.4167, -3.7037), "barcelona": (41.3851, 2.1734), "puertollano": (38.6871, -4.1086),
        "valencia": (39.4699, -0.3763), "sevilla": (37.3891, -5.9845), "zaragoza": (41.6488, -0.8891),
        "malaga": (36.7212, -4.4214), "murcia": (37.9922, -1.1307), "palma": (39.5696, 2.6502),
        "las palmas": (28.1235, -15.4363), "bilbao": (43.2630, -2.9350), "alicante": (38.3452, -0.4810),
        "valladolid": (41.6523, -4.7245), "vigo": (42.2406, -8.7207), "gijon": (43.5357, -5.6615),
        "hospitalet": (41.3597, 2.0997), "coruña": (43.3623, -8.4115), "granada": (37.1773, -3.5986),
        "oviedo": (43.3603, -5.8448), "albacete": (38.9943, -1.8585), "santander": (43.4623, -3.8099),
        "toledo": (39.8628, -4.0273), "ciudad real": (38.9861, -3.9275), "palermo": (38.1157, 13.3614),
        "roma": (41.9028, 12.4964), "paris": (48.8566, 2.3522), "berlin": (52.5200, 13.4050),
        "londres": (51.5074, -0.1278), "lisboa": (38.7223, -9.1393), "rabat": (34.0209, -6.8416),
        "el cairo": (30.0444, 31.2357), "tunez": (36.8065, 10.1815), "argel": (36.7538, 3.0588),
        "reikiavik": (64.1466, -21.9426), "ankara": (39.9334, 32.8597)
    }
    
    if nombre_clean in ciudades_respaldo:
        lat, lon = ciudades_respaldo[nombre_clean]
        return lat, lon, nombre_lugar.capitalize()
        
    try:
        url = f"http://api.geonames.org/searchJSON?q={nombre_lugar}&maxRows=1&username=demo"
        res = requests.get(url, timeout=5)
        data = res.json()
        if 'geonames' in data and len(data['geonames']) > 0:
            item = data['geonames'][0]
            return float(item['lat']), float(item['lng']), item['name']
    except Exception: pass
    return None, None, None

def generar_enlace_dlr_seguro(anio, mes, dia, hora, minuto):
    fecha_fin = datetime.datetime(anio, mes, dia, hora, minuto, 0)
    str_anio = fecha_fin.strftime("%Y")
    str_doy = fecha_fin.strftime("%j")
    str_hora = fecha_fin.strftime("%H")
    fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
    return f"https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0/{str_anio}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{fecha_inicio.strftime('%Y-%m-%dT%H-%M-%S')}_{fecha_fin.strftime('%Y-%m-%dT%H-%M-%S')}_{str_doy}_D.json"

# =====================================================================
# PESTAÑA 1: INICIO Y MONITOREO EN TIEMPO REAL
# =====================================================================
with tab1:
    st.title("🛰️ Sistema Unificado de Monitoreo Ionosférico (TEC/TECU)")
    st.markdown("### ¿Cómo afectan el TEC y el TECU a las señales GNSS?")
    st.markdown("El **Contenido Total de Electrones (TEC)** es la cantidad integrada de electrones atrapados en la ionosfera a lo largo de la trayectoria de una señal de satélite. Se mide en unidades **TECU** (1 TECU = $10^{16}$ electrones por metro cuadrado).")
    st.divider()

    url_europa = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0/latest/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_latest_D.json"
    url_global = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL/v2.0.0/latest/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL_latest_D.json"
    headers = {"User-Agent": "Mozilla/5.0"}

    @st.cache_data(ttl=900)
    def cargar_datos_vtec():
        res_eur = requests.get(url_europa, headers=headers, timeout=15)
        res_eur.raise_for_status()
        matriz_vtec_eur = np.array([f['properties']['vtec_assimilated_tecu'] for f in res_eur.json()['data']['grid']['features']]).reshape(43, 81)
        res_glb = requests.get(url_global, headers=headers, timeout=15)
        res_glb.raise_for_status()
        matriz_vtec_glb = np.array([f['properties']['vtec_assimilated_tecu'] for f in res_glb.json()['data']['grid']['features']]).reshape(73, 73)
        return matriz_vtec_eur, matriz_vtec_glb

    try:
        matriz_vtec_eur, matriz_vtec_glb = cargar_datos_vtec()
        lons_glb, lats_glb = np.linspace(-180, 180, 73), np.linspace(-90, 90, 73)
        
        interp_europa = RegularGridInterpolator((LATS_EUROPA, LONS_EUROPA), matriz_vtec_eur, method='linear', bounds_error=False, fill_value=None)
        interp_global = RegularGridInterpolator((lats_glb, lons_glb), matriz_vtec_glb, method='linear', bounds_error=False, fill_value=None)

        st.subheader("🔍 Consulta de TECU por Localidad o Coordenadas (Tiempo Real)")
        
        # INTERFAZ DUAL DE LOCALIZACIÓN (Puntos de consulta 1)
        tipo_busqueda_t1 = st.radio("Elige el método de posicionamiento:", ["Buscar por Ciudad/Región", "Introducir Coordenadas Manuales (Lat/Lon)"], horizontal=True, key="radio_t1")
        
        lat, lon, label_punto = None, None, ""
        
        if tipo_busqueda_t1 == "Buscar por Ciudad/Región":
            localidad_usuario = st.text_input("Escribe el nombre de una ciudad o región:", "Madrid", key="txt_t1")
            if localidad_usuario:
                lat, lon, label_punto = geocodificar_localidad(localidad_usuario)
        else:
            col_l1, col_l2 = st.columns(2)
            lat_manual = col_l1.number_input("Latitud (°N):", min_value=-90.0, max_value=90.0, value=40.41, step=0.01, key="num_lat_t1")
            lon_manual = col_l2.number_input("Longitud (°E):", min_value=-180.0, max_value=180.0, value=-3.70, step=0.01, key="num_lon_t1")
            lat, lon, label_punto = lat_manual, lon_manual, f"Coordenadas Puras"

        if lat is not None and lon is not None:
            dentro_europa = (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX)
            punto_consulta = np.array([[lat, lon]])
            valor_tecu = float(interp_europa(punto_consulta)[0]) if dentro_europa else float(interp_global(punto_consulta)[0])
            fuente = "Malla Regional Europa" if dentro_europa else "Malla Planetaria Global"
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="📍 Punto de Entrada", value=label_punto)
            col2.metric(label="📡 Valor VTEC", value=f"{valor_tecu:.3f} TECU")
            col3.info(f"**Coordenadas de Análisis:** {lat:.4f}°N, {lon:.4f}°E\n\n**Fuente del Dato:** {fuente}")
        else:
            if tipo_busqueda_t1 == "Buscar por Ciudad/Región": st.error("No se pudo mapear la ciudad.")

        st.divider()
        
        ajuste_local_t1 = st.toggle("🔍 Optimizar rango de color al Máx/Mín local de este mapa", key="toggle_t1")
        
        if ajuste_local_t1:
            vmin_eur, vmax_eur = float(np.min(matriz_vtec_eur)), float(np.max(matriz_vtec_eur))
            vmin_glb, vmax_glb = float(np.min(matriz_vtec_glb)), float(np.max(matriz_vtec_glb))
            lbl_status = "Rango de Color Adaptado Localmente"
        else:
            vmin_eur, vmax_eur = VMIN_TECU_FIJO, VMAX_TECU_FIJO
            vmin_glb, vmax_glb = VMIN_TECU_FIJO, VMAX_TECU_FIJO
            lbl_status = "Escala Fija Universal (0-55 TECU)"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
        ax1.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax1.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax1.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1)
        
        map_eur = ax1.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, matriz_vtec_eur, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_eur, vmax=vmax_eur)
        fig.colorbar(map_eur, ax=ax1, orientation='horizontal', pad=0.07, shrink=0.7).set_label(f'VTEC MALLA REGIONAL (TECU) [{lbl_status}]', weight='bold')

        ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax2.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax2.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.0)
        grid_lon_glb, grid_lat_glb = np.meshgrid(lons_glb, lats_glb)
        
        map_glb = ax2.pcolormesh(grid_lon_glb, grid_lat_glb, matriz_vtec_glb, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.8, shading='gouraud', vmin=vmin_glb, vmax=vmax_glb)
        fig.colorbar(map_glb, ax=ax2, orientation='horizontal', pad=0.07, shrink=0.7).set_label(f'VTEC GLOBAL (TECU) [{lbl_status}]', weight='bold')
        st.pyplot(fig)
    except Exception as e: st.error(f"Error en Tiempo Real: {e}")

# =====================================================================
# PESTAÑA 2: ANÁLISIS EN EL PASADO
# =====================================================================
with tab2:
    st.title("📊 Análisis Histórico: Mapas e Interpolar en el Pasado")
    if 'matriz_pasado' not in st.session_state:
        st.session_state.matriz_pasado = None
    if 'fecha_mapa' not in st.session_state:
        st.session_state.fecha_mapa = ""

    col_f1, col_f2, col_f3 = st.columns(3)
    fecha_sel = col_f1.date_input("Selecciona la Fecha:", datetime.date(2026, 1, 24), key="past_date")
    hora_sel = col_f2.slider("Hora (UTC):", 0, 23, 4, key="past_hour")
    minuto_sel = col_f3.slider("Minuto:", 0, 55, 0, step=5, key="past_min")

    minuto_ajustado = (minuto_sel // 15) * 15
    url_pasado = generar_enlace_dlr_seguro(fecha_sel.year, fecha_sel.month, fecha_sel.day, hora_sel, minuto_ajustado)

    if st.button("🚀 Cargar Mapa Histórico"):
        with st.spinner("Sincronizando Malla Geomagnética Histórica con el DLR..."):
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                response = requests.get(url_pasado, headers=headers, timeout=12)
                response.raise_for_status() 
                vtec_p_list = [f['properties']['vtec_assimilated_tecu'] for f in response.json()['data']['grid']['features']]
                st.session_state.matriz_pasado = np.array(vtec_p_list).reshape(43, 81)
                st.session_state.fecha_mapa = f"{fecha_sel.strftime('%d/%m/%Y')} - {hora_sel:02d}:{minuto_ajustado:02d} UTC"
                st.success("📌 Archivo cargado correctamente.")
            except Exception: st.error("❌ No existen registros en el DLR para la fecha/hora solicitada.")

    if st.session_state.matriz_pasado is not None:
        st.divider()
        st.subheader("📍 Interpolar Píxel Histórico Específico")
        
        # INTERFAZ DUAL DE LOCALIZACIÓN (Puntos de consulta 2)
        tipo_busqueda_t2 = st.radio("Método de entrada de localización histórica:", ["Buscar por Nombre", "Coordenadas directas (Lat/Lon)"], horizontal=True, key="radio_t2")
        
        lat_p, lon_p, label_p = None, None, ""
        
        if tipo_busqueda_t2 == "Buscar por Nombre":
            localidad_p_usuario = st.text_input("Ingresa cualquier localidad dentro de la cuadrícula:", "Madrid", key="txt_t2")
            if localidad_p_usuario:
                lat_p, lon_p, label_p = geocodificar_localidad(localidad_p_usuario)
        else:
            col_lp1, col_lp2 = st.columns(2)
            lat_p_manual = col_lp1.number_input("Latitud exacta (°N):", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.01, key="num_lat_t2")
            lon_p_manual = col_lp2.number_input("Longitud exacta (°E):", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.01, key="num_lon_t2")
            lat_p, lon_p, label_p = lat_p_manual, lon_p_manual, f"Coordenadas fijas"

        if lat_p is not None and lon_p is not None:
            if (LAT_MIN <= lat_p <= LAT_MAX) and (LON_MIN <= lon_p <= LON_MAX):
                interp_p = RegularGridInterpolator((LATS_EUROPA, LONS_EUROPA), st.session_state.matriz_pasado, method='linear', bounds_error=False, fill_value=None)
                val_tecu_p = float(interp_p(np.array([[lat_p, lon_p]]))[0])
                st.metric(label=f"Intensidad Calculada ({label_p})", value=f"{val_tecu_p:.3f} TECU")
                st.caption(f"Coordenadas evaluadas: {lat_p:.3f}°N, {lon_p:.3f}°E")
            else: st.warning("Las coordenadas introducidas están fuera de la cuadrícula de Europa.")

        st.divider()
        ajuste_local_t2 = st.toggle("🔍 Optimizar rango de color al Máx/Mín local de este mapa pasado", key="toggle_t2")
        
        if ajuste_local_t2:
            vmin_p, vmax_p = float(np.min(st.session_state.matriz_pasado)), float(np.max(st.session_state.matriz_pasado))
            lbl_status_p = "Rango de Color Adaptado Localmente"
        else:
            vmin_p, vmax_p = VMIN_TECU_FIJO, VMAX_TECU_FIJO
            lbl_status_p = "Escala Fija Universal (0-55 TECU)"

        fig_p = plt.figure(figsize=(11, 6), dpi=100)
        ax_p = plt.axes(projection=ccrs.PlateCarree())
        ax_p.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax_p.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax_p.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax_p.add_feature(cfeature.COASTLINE, edgecolor='#222222')
        
        mapa_p = ax_p.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.matriz_pasado, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_p, vmax=vmax_p)
        plt.colorbar(mapa_p, ax=ax_p, orientation='horizontal', pad=0.08, shrink=0.7).set_label(f'VTEC ASSIMILATED (TECU) [{lbl_status_p}]', weight='bold')
        st.pyplot(fig_p)
        plt.close(fig_p)

# =====================================================================
# PESTAÑA 3: EVOLUCIÓN TECU
# =====================================================================
with tab3:
    st.title("📈 Estudio de Evolución Temporal del TECU")
    modo_evolucion = st.radio("Selecciona el tipo de análisis temporal:", ["Por Días", "Por Horas"], horizontal=True)

    # SUB-PESTAÑA 1: EVOLUCIÓN POR DÍAS
    if modo_evolucion == "Por Días":
        st.subheader("📆 Análisis de Evolución Interdiaria (Hora Fija)")
        if 'historial_vtec_3d' not in st.session_state:
            st.session_state.historial_vtec_3d, st.session_state.etiquetas_fechas_reales = None, []
            st.session_state.matriz_maximos, st.session_state.ciudades_lista = None, []

        col_c1, col_c2, col_c3 = st.columns(3)
        fecha_inicial = col_c1.date_input("Fecha Inicial:", datetime.date(2026, 2, 19), key="ev_fecha_ini")
        hora_fija_sel = col_c2.slider("Hora fija de observación (UTC):", 0, 23, 15, key="ev_hour_dias")
        num_dias_sel = col_c3.slider("Número de días a evaluar:", 2, 15, 10, key="ev_num_dias")

        if st.button("🚀 Procesar Rango de Días"):
            with st.spinner("Extrayendo Bloques Temporales del Servidor DLR..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                temp_etiquetas, temp_3d = [], np.zeros((num_dias_sel, 43, 81))
                exito_total = True

                for d in range(num_dias_sel):
                    fecha_actual = datetime.datetime(fecha_inicial.year, fecha_inicial.month, fecha_inicial.day) + datetime.timedelta(days=d)
                    link_exitoso = False
                    for m in MINUTOS_CONTIGUOS_GLOBAL:
                        url_intento = generar_enlace_dlr_seguro(fecha_actual.year, fecha_actual.month, fecha_actual.day, hora_fija_sel, m)
                        try:
                            response = requests.get(url_intento, headers=headers, timeout=4)
                            if response.status_code == 200:
                                data = response.json()
                                link_exitoso, minuto_exitoso = True, m
                                break
                        except Exception: pass

                    if not link_exitoso:
                        st.error(f"❌ Sin datos para el día {fecha_actual.strftime('%d/%m/%Y')}.")
                        exito_total = False
                        break

                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    temp_3d[d, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} ({hora_fija_sel:02d}:{minuto_exitoso:02d})")

                if exito_total:
                    st.session_state.historial_vtec_3d = temp_3d
                    st.session_state.etiquetas_fechas_reales = temp_etiquetas
                    st.session_state.matriz_maximos = np.max(temp_3d, axis=0)
                    st.success("📊 Rango temporal procesado.")

        if st.session_state.historial_vtec_3d is not None:
            ajuste_local_t3_dias = st.toggle("🔍 Optimizar rango de color al Máx/Mín de este bloque de días", key="toggle_t3_dias")
            
            if ajuste_local_t3_dias:
                vmin_d = max(0.0, float(np.floor(np.min(st.session_state.historial_vtec_3d) - 2)))
                vmax_d = float(np.ceil(np.max(st.session_state.historial_vtec_3d) + 2))
                lbl_status_d = f"Escala Local Ajustada ({int(vmin_d)}-{int(vmax_d)} TECU)"
            else:
                vmin_d, vmax_d = VMIN_TECU_FIJO, VMAX_TECU_FIJO
                lbl_status_d = "Escala Fija Universal (0-55 TECU)"

            st.subheader("📌 Mapa Fijo de Máximos Absolutos Registrados")
            fig_max, ax_mx = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mx.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
            ax_mx.add_feature(cfeature.COASTLINE, edgecolor='#222222')
            
            mapa_maximos = ax_mx.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_d, vmax=vmax_d)
            fig_max.colorbar(mapa_maximos, ax=ax_mx, orientation='horizontal', pad=0.08, shrink=0.7).set_label(f'PICO MÁXIMO (TECU) [{lbl_status_d}]', weight='bold')
            st.pyplot(fig_max)
            plt.close(fig_max)

            st.subheader("🎬 Reproductor de Video: Evolución Diaria (0.5s por Frame)")
            if st.button("▶️ Reproducir Video", key="play_dias"):
                contenedor_video_mapa = st.empty()
                for f in range(len(st.session_state.etiquetas_fechas_reales)):
                    fig_video, ax_ev = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_ev.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6')
                    ax_ev.add_feature(cfeature.COASTLINE, edgecolor='#222222')
                    
                    mapa_dinamico = ax_ev.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.historial_vtec_3d[f, :, :], transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_d, vmax=vmax_d)
                    fig_video.colorbar(mapa_dinamico, ax=ax_ev, orientation='horizontal', pad=0.08, shrink=0.7).set_label(f'VTEC (TECU) [{lbl_status_d}]', weight='bold')
                    contenedor_video_mapa.pyplot(fig_video)
                    plt.close(fig_video)
                    time.sleep(0.5)

            st.subheader("📊 Gráfica Comparativa de Localidades Acumuladas")
            
            # INTERFAZ DUAL DE LOCALIZACIÓN (Puntos de consulta 3 - Lote Días)
            tipo_busqueda_t3d = st.radio("Formato de inserción de localidad:", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3d")
            
            lat_c, lon_c, name_c = None, None, ""
            
            if tipo_busqueda_t3d == "Por Nombre de Ciudad":
                nueva_ciudad = st.text_input("Ingresa cualquier localidad del mapa:", "Madrid", key="txt_t3d")
                if nueva_ciudad:
                    lat_c, lon_c, name_c = geocodificar_localidad(nueva_ciudad)
            else:
                col_lc1, col_lc2 = st.columns(2)
                lat_c_man = col_lc1.number_input("Latitud punto:", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.1, key="num_lat_t3d")
                lon_c_man = col_lc2.number_input("Longitud punto:", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.1, key="num_lon_t3d")
                lat_c, lon_c, name_c = lat_c_man, lon_c_man, f"Punto ({lat_c_man:.1f}, {lon_c_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico", key="btn_t3d"):
                if lat_c is not None and lon_c is not None:
                    if (LAT_MIN <= lat_c <= LAT_MAX) and (LON_MIN <= lon_c <= LON_MAX):
                        if name_c not in [c['name'] for c in st.session_state.ciudades_lista]:
                            st.session_state.ciudades_lista.append({'name': name_c, 'lat': lat_c, 'lon': lon_c})
                            st.success(f"Añadido: {name_c}")
                    else: st.error("Fuera de la malla de Europa.")

            if st.session_state.ciudades_lista:
                fig_lineas, ax_lineas = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.ciudades_lista:
                    idx_lat = (np.abs(LATS_EUROPA - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(LONS_EUROPA - ciudad_obj['lon'])).argmin()
                    ax_lineas.plot(range(len(st.session_state.etiquetas_fechas_reales)), st.session_state.historial_vtec_3d[:, idx_lat, idx_lon], marker='s', linewidth=2, label=ciudad_obj['name'])
                ax_lineas.grid(True, linestyle='--')
                
                ax_lineas.set_ylim(vmin_d, vmax_d)
                ax_lineas.set_xticks(range(len(st.session_state.etiquetas_fechas_reales)))
                ax_lineas.set_xticklabels(st.session_state.etiquetas_fechas_reales, rotation=25)
                ax_lineas.legend(loc="upper right")
                st.pyplot(fig_lineas)
                plt.close(fig_lineas)

    # SUB-PESTAÑA 2: EVOLUCIÓN POR HORAS (24H)
    elif modo_evolucion == "Por Horas":
        st.subheader("⏱️ Análisis de Evolución Intradía (Hora por Hora - 24h)")
        if 'h_historial_vtec_3d' not in st.session_state:
            st.session_state.h_historial_vtec_3d, st.session_state.h_etiquetas_reales = None, []
            st.session_state.h_matriz_maximos, st.session_state.h_ciudades_lista = None, []

        fecha_analisis_h = st.date_input("Selecciona el día a analizar:", datetime.date(2026, 1, 24), key="ev_fecha_hor")

        if st.button("🚀 Procesar las 24 Horas"):
            with st.spinner("Escaneando Ciclos Diurnos Horarios (24 Frames)..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                h_temp_etiquetas, h_temp_3d = [], np.zeros((24, 43, 81))
                h_exito_total = True

                for h in range(24):
                    link_exitoso = False
                    for m in MINUTOS_CONTIGUOS_GLOBAL:
                        url_intento = generar_enlace_dlr_seguro(fecha_analisis_h.year, fecha_analisis_h.month, fecha_analisis_h.day, h, m)
                        try:
                            response = requests.get(url_intento, headers=headers, timeout=4)
                            if response.status_code == 200:
                                data = response.json()
                                link_exitoso, minuto_exitoso = True, m
                                break
                        except Exception: pass

                    if not link_exitoso:
                        st.error(f"❌ Error en hora {h:02d}. Cancelado.")
                        h_exito_total = False
                        break

                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    h_temp_3d[h, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    h_temp_etiquetas.append(f"{h:02d}:{minuto_exitoso:02d}")

                if h_exito_total:
                    st.session_state.h_historial_vtec_3d = h_temp_3d
                    st.session_state.h_etiquetas_reales = h_temp_etiquetas
                    st.session_state.h_matriz_maximos = np.max(h_temp_3d, axis=0)
                    st.success("📊 Completado.")

        if st.session_state.h_historial_vtec_3d is not None:
            ajuste_local_t3_horas = st.toggle("🔍 Optimizar rango de color al Máx/Mín real de estas 24 horas", key="toggle_t3_horas")
            
            if ajuste_local_t3_horas:
                vmin_h = max(0.0, float(np.floor(np.min(st.session_state.h_historial_vtec_3d) - 2)))
                vmax_h = float(np.ceil(np.max(st.session_state.h_historial_vtec_3d) + 2))
                lbl_status_h = f"Escala Local Ajustada ({int(vmin_h)}-{int(vmax_h)} TECU)"
            else:
                vmin_h, vmax_h = VMIN_TECU_FIJO, VMAX_TECU_FIJO
                lbl_status_h = "Escala Fija Universal (0-55 TECU)"

            st.subheader("📌 Mapa Fijo de Máximos Absolutos del Día")
            fig_max_h, ax_mxh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mxh.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax_mxh.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            ax_mxh.add_feature(cfeature.COASTLINE, edgecolor='#222222')
            
            mapa_maximos_h = ax_mxh.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.h_matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_h, vmax=vmax_h)
            fig_max_h.colorbar(mapa_maximos_h, ax=ax_mxh, orientation='horizontal', pad=0.08, shrink=0.7).set_label(f'PICO MÁXIMO HORARIO (TECU) [{lbl_status_h}]', weight='bold')
            st.pyplot(fig_max_h)
            plt.close(fig_max_h)

            st.subheader("🎬 Reproductor Horario (0.5s por Frame)")
            if st.button("▶️ Reproducir Video Horario", key="play_horas"):
                contenedor_video_horas = st.empty()
                for f in range(24):
                    fig_vid_h, ax_evh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_evh.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_evh.add_feature(cfeature.LAND, facecolor='#f6f6f6')
                    ax_evh.add_feature(cfeature.COASTLINE, edgecolor='#222222')
                    
                    mapa_dinamico_h = ax_evh.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.h_historial_vtec_3d[f, :, :], transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_h, vmax=vmax_h)
                    fig_vid_h.colorbar(mapa_dinamico_h, ax=ax_evh, orientation='horizontal', pad=0.08, shrink=0.7).set_label(f'VTEC (TECU) [{lbl_status_h}]', weight='bold')
                    contenedor_video_horas.pyplot(fig_vid_h)
                    plt.close(fig_vid_h)
                    time.sleep(0.5)

            st.subheader("📊 Gráfica Comparativa de Localidades Acumuladas (24 Horas)")
            
            # INTERFAZ DUAL DE LOCALIZACIÓN (Puntos de consulta 4 - Ciclo Horas)
            tipo_busqueda_t3h = st.radio("Formato de inserción de localidad (24h):", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3h")
            
            lat_ch, lon_ch, name_ch = None, None, ""
            
            if tipo_busqueda_t3h == "Por Nombre de Ciudad":
                nueva_ciudad_h = st.text_input("Nombre de la ciudad:", "Madrid", key="txt_t3h")
                if nueva_ciudad_h:
                    lat_ch, lon_ch, name_ch = geocodificar_localidad(nueva_ciudad_h)
            else:
                col_lch1, col_lch2 = st.columns(2)
                lat_ch_man = col_lch1.number_input("Latitud nodo:", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.1, key="num_lat_t3h")
                lon_ch_man = col_lch2.number_input("Longitud nodo:", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.1, key="num_lon_t3h")
                lat_ch, lon_ch, name_ch = lat_ch_man, lon_ch_man, f"Punto ({lat_ch_man:.1f}, {lon_ch_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico Horario", key="btn_t3h"):
                if lat_ch is not None and lon_ch is not None:
                    if (LAT_MIN <= lat_ch <= LAT_MAX) and (LON_MIN <= lon_ch <= LON_MAX):
                        if name_ch not in [c['name'] for c in st.session_state.h_ciudades_lista]:
                            st.session_state.h_ciudades_lista.append({'name': name_ch, 'lat': lat_ch, 'lon': lon_ch})
                            st.success(f"Añadido: {name_ch}")
                    else: st.error("Fuera de la malla de Europa.")

            if st.session_state.h_ciudades_lista:
                fig_lineas_h, ax_lineas_h = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.h_ciudades_lista:
                    idx_lat = (np.abs(LATS_EUROPA - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(LONS_EUROPA - ciudad_obj['lon'])).argmin()
                    ax_lineas_h.plot(range(24), st.session_state.h_historial_vtec_3d[:, idx_lat, idx_lon], marker='o', linewidth=2, label=ciudad_obj['name'])
                ax_lineas_h.grid(True, linestyle='--')
                
                ax_lineas_h.set_ylim(vmin_h, vmax_h)
                ax_lineas_h.set_xticks(range(24))
                ax_lineas_h.set_xticklabels([f"{h:02d}h" for h in range(24)], rotation=45)
                ax_lineas_h.legend(loc="upper right")
                st.pyplot(fig_lineas_h)
                plt.close(fig_lineas_h)

# =====================================================================
# PESTAÑA 4 Y PESTAÑA 5: EN ESPERA (VACÍAS EN REPOSO)
# =====================================================================
# =====================================================================
# PESTAÑA 4: PRONÓSTICO (REESTRUCTURACIÓN COMPLETA BASADA EN TU SCRIPT)
# =====================================================================
with tab4:
    st.title("🔮 Predicción Científica del VTEC Ionosférico")
    st.markdown("### Modelo de Climatología + Persistencia Amortiguada (Validación a 3 Horas)")
    st.divider()

    # Inicialización de la memoria de estado exclusiva para el pronóstico de la pestaña 4
    if 'p4_vector_pasado' not in st.session_state:
        st.session_state.p4_vector_pasado = None
        st.session_state.p4_fechas_pasado = []
        st.session_state.p4_vector_futuro_real = None
        st.session_state.p4_vector_futuro_calc = None
        st.session_state.p4_fechas_futuro = []
        st.session_state.p4_mae = 0.0
        st.session_state.p4_acierto = 0.0
        st.session_state.p4_info_punto = ""

    # FORMULARIO DE ENTRADA DE CONFIGURACIÓN
    st.subheader("⚙️ Configuración del Escenario de Análisis")
    
    col_p1, col_p2 = st.columns(2)
    p4_fecha_base = col_p1.date_input("Selecciona la fecha base del historial (24 Horas):", datetime.date(2026, 1, 1), key="p4_date_sel")
    
    # Sistema dual alternativo de localización solicitado
    p4_tipo_pos = col_p2.radio("Método de posicionamiento para la predicción:", ["Por Nombre de Ciudad/Región", "Por Coordenadas Manuales (Lat/Lon)"], horizontal=True, key="p4_radio_pos")
    
    lat_p4, lon_p4, label_p4 = None, None, ""
    
    if p4_tipo_pos == "Por Nombre de Ciudad/Región":
        ciudad_p4_txt = st.text_input("Escribe el nombre de la localidad objetivo:", "Madrid", key="p4_txt_city")
        if ciudad_p4_txt:
            lat_p4, lon_p4, label_p4 = geocodificar_localidad(ciudad_p4_txt)
    else:
        col_p4_lat, col_p4_lon = st.columns(2)
        lat_p4_num = col_p4_lat.number_input("Latitud del receptor (°N):", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.01, key="p4_num_lat")
        lon_p4_num = col_p4_lon.number_input("Longitud del receptor (°E):", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.01, key="p4_num_lon")
        lat_p4, lon_p4, label_p4 = lat_p4_num, lon_p4_num, f"Coordenadas Manuales"

    # BOTÓN DE EJECUCIÓN DEL PRONÓSTICO
    if st.button("🚀 Ejecutar Modelo Matemático", key="p4_btn_run"):
        if lat_p4 is not None and lon_p4 is not None:
            if (LAT_MIN <= lat_p4 <= LAT_MAX) and (LON_MIN <= lon_p4 <= LON_MAX):
                
                with st.spinner("Descargando ciclo diurno y computando matriz estacional de inercia..."):
                    idx_lat_p4 = (np.abs(LATS_EUROPA - lat_p4)).argmin()
                    idx_lon_p4 = (np.abs(LONS_EUROPA - lon_p4)).argmin()
                    
                    headers = {"User-Agent": "Mozilla/5.0"}
                    minutos_objetivo = [0]
                    
                    temp_cronologia_pasado = []
                    temp_fechas_pasado = []
                    exito_descarga_p1 = True
                    
                    # --- FASE 1: EXTRACCIÓN DEL PASADO (24 HORAS CONSECUTIVAS) ---
                    for hora_actual in range(24):
                        for min_obj in minutos_objetivo:
                            link_exitoso = False
                            # Escáner de contingencia en pasos de 5 min si falla la hora en punto
                            minutos_contingencia = range(0, 30, 5)
                            
                            for m in minutos_contingencia:
                                url_intento = generar_enlace_dlr_seguro(p4_fecha_base.year, p4_fecha_base.month, p4_fecha_base.day, hora_actual, m)
                                try:
                                    response = requests.get(url_intento, headers=headers, timeout=3)
                                    if response.status_code == 200:
                                        data = response.json()
                                        link_exitoso = True
                                        break
                                except Exception: pass
                                
                            if not link_exitoso:
                                st.error(f"❌ Enlace roto o ausente en el servidor para la hora {hora_actual:02d}:00 UTC.")
                                exito_descarga_p1 = False
                                break
                        if not exito_descarga_p1: break
                        
                        vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                        matriz_instante = np.array(vtec_values_list).reshape(43, 81)
                        temp_cronologia_pasado.append(matriz_instante[idx_lat_p4, idx_lon_p4])
                        temp_fechas_pasado.append(datetime.datetime(p4_fecha_base.year, p4_fecha_base.month, p4_fecha_base.day) + datetime.timedelta(hours=hora_actual))

                    if exito_descarga_p1:
                        vector_vtec_serie = np.array(temp_cronologia_pasado)
                        
                        # --- FASE 2: DESCARGA DE LOS 3 PUNTOS REALES DEL FUTURO PARA VALIDACIÓN ---
                        temp_cronologia_futuro_real = []
                        temp_fechas_futuro = []
                        p4_fecha_futuro_base = datetime.datetime(p4_fecha_base.year, p4_fecha_base.month, p4_fecha_base.day) + datetime.timedelta(days=1)
                        exito_descarga_p2 = True
                        
                        for hora_val in range(3): # 3 horas fijas hacia adelante
                            link_exitoso = False
                            for m in minutos_contingencia:
                                url_intento = generar_enlace_dlr_seguro(p4_fecha_futuro_base.year, p4_fecha_futuro_base.month, p4_fecha_futuro_base.day, hora_val, m)
                                try:
                                    response = requests.get(url_intento, headers=headers, timeout=3)
                                    if response.status_code == 200:
                                        data_val = response.json()
                                        link_exitoso = True
                                        break
                                except Exception: pass
                                
                            if not link_exitoso:
                                st.error(f"❌ No se encontraron datos de validación real para el día siguiente a las {hora_val:02d}:00 UTC.")
                                exito_descarga_p2 = False
                                break
                                
                            vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data_val['data']['grid']['features']]
                            matriz_instante = np.array(vtec_values_list).reshape(43, 81)
                            temp_cronologia_futuro_real.append(matriz_instante[idx_lat_p4, idx_lon_p4])
                            temp_fechas_futuro.append(p4_fecha_futuro_base + datetime.timedelta(hours=hora_val))

                        if exito_descarga_p2:
                            vector_real_futuro = np.array(temp_cronologia_futuro_real)
                            
                            # --- FASE 3: EJECUCIÓN DEL MODELO MATEMÁTICO AUTORREGRESIVO ---
                            periodo = 24            # 24 muestras por día
                            puntos_prediccion = 3   # 3 puntos adelante
                            
                            perfil_estacional = np.zeros(periodo)
                            for i in range(periodo):
                                # Al ser un escenario intradía de 24h, la media estacional equivale al valor directo
                                perfil_estacional[i] = vector_vtec_serie[i]
                                
                            ultimo_valor_real = vector_vtec_serie[-1]
                            ultimo_slot_horario = (len(vector_vtec_serie) - 1) % periodo
                            anomalia_inicial = ultimo_valor_real - perfil_estacional[ultimo_slot_horario]
                            
                            vector_prediccion_futura = []
                            alpha = 0.85 # Coeficiente estricto de inercia ionosférica
                            
                            for k in range(1, puntos_prediccion + 1):
                                slot_futuro = (ultimo_slot_horario + k) % periodo
                                valor_predicho = perfil_estacional[slot_futuro] + anomalia_inicial * (alpha ** k)
                                vector_prediccion_futura.append(valor_predicho)
                                
                            vector_prediccion_futura = np.array(vector_prediccion_futura)
                            
                            # Cómputo de la precisión analítica en consola/pantalla
                            errores_punto_a_punto = np.abs(vector_real_futuro - vector_prediccion_futura)
                            mae_calculado = float(np.mean(errores_punto_a_punto))
                            acierto_calculado = max(0.0, 100 - (mae_calculado / np.mean(vector_real_futuro)) * 100)
                            
                            # Volcar todas las variables calculadas de forma segura a la memoria de Streamlit
                            st.session_state.p4_vector_pasado = vector_vtec_serie
                            st.session_state.p4_fechas_pasado = temp_fechas_pasado
                            st.session_state.p4_vector_futuro_real = vector_real_futuro
                            st.session_state.p4_vector_futuro_calc = vector_prediccion_futura
                            st.session_state.p4_fechas_futuro = temp_fechas_futuro
                            st.session_state.p4_mae = mae_calculado
                            st.session_state.p4_acierto = acierto_calculado
                            st.session_state.p4_info_punto = f"{label_p4} | Coordenadas: {lat_p4:.3f}°N, {lon_p4:.3f}°E"
                            
                            st.success("🎯 Modelo ejecutado y pronóstico verificado correctamente.")
            else:
                st.error("❌ Las coordenadas ingresadas se encuentran fuera de la cuadrícula de Europa.")
        else:
            st.error("❌ Especifica una ciudad válida o un par de coordenadas Lat/Lon numéricas.")

    # DESPLIEGUE GRÁFICO (REPLICA EXACTA DE TU DISEÑO INYECTADO)
    if st.session_state.p4_vector_pasado is not None:
        st.divider()
        
        # Cuadro de KPIS de control analítico de fiabilidad
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric(label="📊 Error Absoluto Medio (MAE)", value=f"{st.session_state.p4_mae:.3f} TECU")
        col_k2.metric(label="🎯 Porcentaje Estimado de Acierto", value=f"{st.session_state.p4_acierto:.1f} %")
        col_k3.info(f"**Ubicación de Análisis:**\n{st.session_state.p4_info_punto}")
        
        # Selector interactivo de rango vertical solicitado
        p4_toggle_ejes = st.toggle("🔍 Activar regla +-2 local en el Eje Y (Optimizar visualización)", key="toggle_p4_ejes")
        
        fig_p4, ax_p4 = plt.subplots(figsize=(15, 6), dpi=100)
        
        # 1. Pasado inmediato real (Línea Azul con marcadores redondos)
        ax_p4.plot(st.session_state.p4_fechas_pasado, st.session_state.p4_vector_pasado, 
                   color='#2979ff', linewidth=2, label='Pasado Inmediato Real (DLR)', marker='o', markersize=4)
        
        # 2. Predicción matemática (Línea discontinua naranja con marcadores 'x')
        ax_p4.plot(st.session_state.p4_fechas_futuro, st.session_state.p4_vector_futuro_calc, 
                   color='#ff3d00', linewidth=2.5, linestyle='--', label='Predicción Matemática', marker='x', zorder=4)
        
        # 3. Datos reales de validación (Línea verde continua con marcadores cuadrados)
        ax_p4.plot(st.session_state.p4_fechas_futuro, st.session_state.p4_vector_futuro_real, 
                   color='#00e676', linewidth=2.5, label='Datos Reales de Validación', marker='s', markersize=5, zorder=3)
        
        # Sombreado del margen de error translúcido entre la realidad y la predicción
        ax_p4.fill_between(st.session_state.p4_fechas_futuro, st.session_state.p4_vector_futuro_calc, st.session_state.p4_vector_futuro_real, 
                           color='#ff3d00', alpha=0.1, label='Margen de Error')
        
        ax_p4.grid(True, linestyle='--', alpha=0.5)
        
        # Control estricto de los límites verticales del eje Y
        if p4_toggle_ejes:
            valores_totales = np.concatenate([st.session_state.p4_vector_pasado, st.session_state.p4_vector_futuro_real, st.session_state.p4_vector_futuro_calc])
            Y_MIN_VAL = max(0.0, float(np.floor(np.min(valores_totales) - 2)))
            Y_MAX_VAL = float(np.ceil(np.max(valores_totales) + 2))
            ax_p4.set_ylim(Y_MIN_VAL, Y_MAX_VAL)
            str_escala = f"Regla +-2 Local: {int(Y_MIN_VAL)}-{int(Y_MAX_VAL)} TECU"
        else:
            ax_p4.set_ylim(VMIN_TECU_FIJO, VMAX_TECU_FIJO)
            str_escala = f"Escala Universal Fija: {int(VMIN_TECU_FIJO)}-{int(VMAX_TECU_FIJO)} TECU"
            
        # Formateador horario elegante para el eje temporal X en formato DD/MM y HH:MM
        ax_p4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_p4.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
        
        plt.xlabel("Línea Temporal de Control (UTC)", fontsize=11, weight='bold')
        plt.ylabel("Intensidad VTEC (TECU)", fontsize=11, weight='bold')
        plt.title(f"TEST DE VELOCIDAD: PREDICCIÓN A 3 HORAS\n[Eje Y Controlado con {str_escala}]", fontsize=12, weight='bold', pad=15)
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        st.pyplot(fig_p4)
        plt.close(fig_p4)
