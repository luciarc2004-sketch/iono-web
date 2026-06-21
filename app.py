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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 Inicio", 
    "📊 Análisis en el pasado", 
    "📈 Evolución TECU", 
    " Pronóstico", 
    "📉 Desviaciones del Modelo",
    "💬 Comentarios "
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
# PESTAÑA 1: INICIO Y MONITOREO EN TIEMPO REAL (ACTUALIZADA)
# =====================================================================
with tab1:
    st.title("🛰️ Sistema en Tiempo Real de Monitoreo Ionosférico (TEC/TECU)")
    st.markdown("### ¿Cómo afectan el TEC y el TECU a las señales GNSS?")
    st.markdown("El **Contenido Total de Electrones (TEC)** es la cantidad integrada de electrones atrapados en la ionosfera a lo largo de la trayectoria de una señal de satélite. Se mide en unidades **TECU** ($1\\text{ TECU} = 10^{16}$ electrones por metro cuadrado).")
    st.divider()

    # Enlaces de los servidores del DLR para Tiempo Real
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
        
        # Generación de interpoladores espaciales lineales en tiempo real
        interp_europa = RegularGridInterpolator((LATS_EUROPA, LONS_EUROPA), matriz_vtec_eur, method='linear', bounds_error=False, fill_value=None)
        interp_global = RegularGridInterpolator((lats_glb, lats_glb), matriz_vtec_glb, method='linear', bounds_error=False, fill_value=None)


     # Interruptor de control para el ajuste de escala local solicitado
        ajuste_local_t1 = st.toggle("🔍 Optimizar rango de color al Máx/Mín local de este mapa", key="toggle_t1")
        
        if ajuste_local_t1:
            vmin_eur, vmax_eur = float(np.min(matriz_vtec_eur)), float(np.max(matriz_vtec_eur))
            vmin_glb, vmax_glb = float(np.min(matriz_vtec_glb)), float(np.max(matriz_vtec_glb))
            lbl_status = "Rango de Color Adaptado Localmente"
        else:
            vmin_eur, vmax_eur = VMIN_TECU_FIJO, VMAX_TECU_FIJO
            vmin_glb, vmax_glb = VMIN_TECU_FIJO, VMAX_TECU_FIJO
            lbl_status = "Escala Fija Universal (0-55 TECU)"

        # Construcción y renderizado de la figura dual
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Sub-mapa 1: Malla Regional Europa
        ax1.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax1.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax1.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1)
        
        map_eur = ax1.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, matriz_vtec_eur, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_eur, vmax=vmax_eur)
        fig.colorbar(map_eur, ax=ax1, orientation='horizontal', pad=0.07, shrink=0.7).set_label(f'VTEC MALLA REGIONAL (TECU) [{lbl_status}]', weight='bold')

        # Sub-mapa 2: Malla Planetaria Global
        ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax2.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax2.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.0)
        grid_lon_glb, grid_lat_glb = np.meshgrid(lons_glb, lats_glb)
        
        map_glb = ax2.pcolormesh(grid_lon_glb, grid_lat_glb, matriz_vtec_glb, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.8, shading='gouraud', vmin=vmin_glb, vmax=vmax_glb)
        fig.colorbar(map_glb, ax=ax2, orientation='horizontal', pad=0.07, shrink=0.7).set_label(f'VTEC GLOBAL (TECU) [{lbl_status}]', weight='bold')


        st.pyplot(fig)
        st.divider()
        
   
        st.subheader("🔍 Consulta de TECU por Localidad o Coordenadas")
        
        # Sistema dual alternativo de entrada de localización
        tipo_busqueda_t1 = st.radio("Elige el método de posicionamiento:", ["Buscar por localidad", "Introducir Coordenadas Manuales (Lat/Lon)"], horizontal=True, key="radio_t1")
        
        lat, lon, label_punto = None, None, ""
        
        if tipo_busqueda_t1 == "Buscar por localidad":
            localidad_usuario = st.text_input("Escribe el nombre de una ciudad o región:", "Toledo", key="txt_t1")
            if localidad_usuario:
                lat, lon, label_punto = geocodificar_localidad(localidad_usuario)
        else:
            col_l1, col_l2 = st.columns(2)
            lat_manual = col_l1.number_input("Latitud (°N):", min_value=-90.0, max_value=90.0, value=40.41, step=0.01, key="num_lat_t1")
            lon_manual = col_l2.number_input("Longitud (°E):", min_value=-180.0, max_value=180.0, value=-3.70, step=0.01, key="num_lon_t1")
            lat, lon, label_punto = lat_manual, lon_manual, f"Coordenadas Puras"

        # Procesamiento dinámico del píxel consultado
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

        
    

        # -----------------------------------------------------------------
        #  ENLACES DE INTERÉS Y RECURSOS 
        # -----------------------------------------------------------------
        st.divider()
        st.subheader("🔗 Enlaces de Interés y Recursos")
        
        col_lnk1, col_lnk2, col_lnk3 = st.columns(3)
        
        with col_lnk1:
            st.markdown("#### 🛰️ Proveedores de Datos")
            st.markdown("- [DLR IMPC Portal](https://impc.dlr.de/) - Centro Alemán de Operaciones Espaciales.")
            st.markdown("- [IGS Iono](https://igs.org/wg/ionosphere/) - International GNSS Service.")
            
        with col_lnk2:
            st.markdown("#### 📚 Documentación y Ciencia")
            st.markdown("- [ESA Navipedia - Ionosphere](https://gssc.esa.int/navipedia/) - Retrasos ionosféricos en GNSS.")
            st.markdown("- [NOAA Space Weather](https://www.swpc.noaa.gov/) - Predicción del clima espacial.")
            
        with col_lnk3:
            st.markdown("#### 🛠️ Herramientas Complementarias")
            st.markdown("- [CDDIS NASA](https://cddis.nasa.gov/) - Archivo de datos de geodesia espacial.")
            st.markdown("- [Códigos Web](https://github.com/luciarc2004-sketch) - Código público.")

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

    if st.button("🚀 Cargar Mapa"):
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

        st.divider()
        st.subheader("📍 Valor VTEC de un punto")
        
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

# =====================================================================
# PESTAÑA 3: EVOLUCIÓN TECU (REPRODUCTOR DINÁMICO DE "FLAMES" INYECTADO)
# =====================================================================
with tab3:
    st.title("📈 Estudio de Evolución Temporal del TECU")
    
    # Selector unificado en la cabecera con las tres opciones oficiales
    modo_evolucion = st.radio(
        "Selecciona el tipo de análisis temporal:", 
        ["Por Días (Hora Fija)", "Por Horas (24h Único Día)", "Días Completos (Rango Continuo)"], 
        horizontal=True, 
        key="radio_modo_evolucion_global"
    )

    # =====================================================================
    # BLOQUE 1: POR DÍAS (HORA FIJA)
    # =====================================================================
    if modo_evolucion == "Por Días (Hora Fija)":
        st.subheader("📆 Análisis de Evolución Interdiaria (Hora Fija)")
        if 'historial_vtec_3d' not in st.session_state:
            st.session_state.historial_vtec_3d, st.session_state.etiquetas_fechas_reales = None, []
            st.session_state.matriz_maximos, st.session_state.ciudades_lista = None, []

        col_c1, col_c2, col_c3 = st.columns(3)
        fecha_inicial = col_c1.date_input("Fecha Inicial:", datetime.date(2026, 2, 19), key="ev_fecha_ini")
        hora_fija_sel = col_c2.slider("Hora fija de observación (UTC):", 0, 23, 15, key="ev_hour_dias")
        num_dias_sel = col_c3.slider("Número de días a evaluar:", 2, 15, 10, key="ev_num_dias")

        if st.button("🚀 Procesar Rango de Días", key="btn_ev_dias"):
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
                                link_exitoso = True
                                break
                        except Exception: pass

                    if not link_exitoso:
                        st.error(f"❌ Sin datos para el día {fecha_actual.strftime('%d/%m/%Y')}.")
                        exito_total = False
                        break

                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    temp_3d[d, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} ({hora_fija_sel:02d}:00)")

                if exito_total:
                    st.session_state.historial_vtec_3d = temp_3d
                    st.session_state.etiquetas_fechas_reales = temp_etiquetas
                    st.session_state.matriz_maximos = np.max(temp_3d, axis=0)
                    st.success("📊 Rango temporal pasado procesado.")

        if st.session_state.historial_vtec_3d is not None:
            ajuste_local_t3_dias = st.toggle("🔍 Optimizar rango de color al Máx/Mín de este bloque de días", key="toggle_t3_dias")
            vmin_d, vmax_d = (max(0.0, float(np.floor(np.min(st.session_state.historial_vtec_3d) - 2))), float(np.ceil(np.max(st.session_state.historial_vtec_3d) + 2))) if ajuste_local_t3_dias else (VMIN_TECU_FIJO, VMAX_TECU_FIJO)
            
            # --- REPRODUCTOR DINÁMICO (MAPA DE FLAMES) ---
            st.subheader("🎬 Reproductor Dinámico de Evolución (Por Días)")
            if st.button("▶️ Reproducir Serie Diaria Frame por Frame", key="btn_play_dias_frames"):
                contenedor_dias_anim = st.empty()
                for f in range(len(st.session_state.etiquetas_fechas_reales)):
                    fig_a = plt.figure(figsize=(10, 6), dpi=100)
                    ax_a = plt.axes(projection=ccrs.PlateCarree())
                    ax_a.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_a.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_a.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_a.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    
                    mapa_a = ax_a.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.historial_vtec_3d[f, :, :], 
                                          transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_d, vmax=vmax_d, zorder=2)
                    plt.colorbar(mapa_a, ax=ax_a, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    ax_a.set_title(f"FRAME DIARIO: {st.session_state.etiquetas_fechas_reales[f]} UTC", fontsize=10, weight='bold')
                    
                    contenedor_dias_anim.pyplot(fig_a)
                    plt.close(fig_a)
                    time.sleep(0.50) # Cambia cada 0.5 segundos

            # --- MAPA DE MÁXIMOS ABSOLUTOS ---
            st.subheader("📌 Mapa Fijo de Máximos Absolutos Registrados")
            fig_max, ax_mx = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mx.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6'); ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd'); ax_mx.add_feature(cfeature.COASTLINE, edgecolor='#222222')
            mapa_maximos = ax_mx.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_d, vmax=vmax_d)
            fig_max.colorbar(mapa_maximos, ax=ax_mx, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO (TECU)', weight='bold')
            st.pyplot(fig_max); plt.close(fig_max)

            # --- GRÁFICAS POR CIUDAD ---
            st.subheader("📊 Gráfica Comparativa de Localidades Acumuladas")
            tipo_busqueda_t3d = st.radio("Formato de inserción de localidad:", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3d")
            lat_c, lon_c, name_c = None, None, ""
            if tipo_busqueda_t3d == "Por Nombre de Ciudad":
                nueva_ciudad = st.text_input("Ingresa cualquier localidad del mapa:", "Madrid", key="txt_t3d")
                if nueva_ciudad: lat_c, lon_c, name_c = geocodificar_localidad(nueva_ciudad)
            else:
                col_lc1, col_lc2 = st.columns(2)
                lat_c_man = col_lc1.number_input("Latitud punto:", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.1, key="num_lat_t3d")
                lon_c_man = col_lc2.number_input("Longitud punto:", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.1, key="num_lon_t3d")
                lat_c, lon_c, name_c = lat_c_man, lon_c_man, f"Punto ({lat_c_man:.1f}, {lon_c_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico", key="btn_t3d"):
                if lat_c is not None and lon_c is not None and (LAT_MIN <= lat_c <= LAT_MAX) and (LON_MIN <= lon_c <= LON_MAX):
                    if name_c not in [c['name'] for c in st.session_state.ciudades_lista]: st.session_state.ciudades_lista.append({'name': name_c, 'lat': lat_c, 'lon': lon_c})
            if st.session_state.ciudades_lista:
                fig_lineas, ax_lineas = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.ciudades_lista:
                    idx_lat = (np.abs(LATS_EUROPA - ciudad_obj['lat'])).argmin(); idx_lon = (np.abs(LONS_EUROPA - ciudad_obj['lon'])).argmin()
                    ax_lineas.plot(range(len(st.session_state.etiquetas_fechas_reales)), st.session_state.historial_vtec_3d[:, idx_lat, idx_lon], marker='s', linewidth=2, label=ciudad_obj['name'])
                ax_lineas.grid(True, linestyle='--'); ax_lineas.set_ylim(vmin_d, vmax_d); ax_lineas.set_xticks(range(len(st.session_state.etiquetas_fechas_reales))); ax_lineas.set_xticklabels(st.session_state.etiquetas_fechas_reales, rotation=25); ax_lineas.legend(loc="upper right")
                st.pyplot(fig_lineas); plt.close(fig_lineas)

    # =====================================================================
    # BLOQUE 2: POR HORAS (24H ÚNICO DÍA)
    # =====================================================================
    elif modo_evolucion == "Por Horas (24h Único Día)":
        st.subheader("⏱️ Análisis de Evolución Intradía (Hora por Hora - 24h)")
        if 'h_historial_vtec_3d' not in st.session_state:
            st.session_state.h_historial_vtec_3d, st.session_state.h_etiquetas_reales = None, []
            st.session_state.h_matriz_maximos, st.session_state.h_ciudades_lista = None, []

        fecha_analisis_h = st.date_input("Selecciona el día a analizar:", datetime.date(2026, 1, 24), key="ev_fecha_hor")

        if st.button("🚀 Procesar las 24 Horas", key="btn_ev_horas"):
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
                                link_exitoso = True
                                break
                        except Exception: pass

                    if not link_exitoso:
                        st.error(f"❌ Error en hora {h:02d}. Cancelado.")
                        h_exito_total = False
                        break

                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    h_temp_3d[h, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    h_temp_etiquetas.append(f"{h:02d}:00")

                if h_exito_total:
                    st.session_state.h_historial_vtec_3d = h_temp_3d
                    st.session_state.h_etiquetas_reales = h_temp_etiquetas
                    st.session_state.h_matriz_maximos = np.max(h_temp_3d, axis=0)
                    st.success("📊 Completado.")

        if st.session_state.h_historial_vtec_3d is not None:
            ajuste_local_t3_horas = st.toggle("🔍 Optimizar rango de color al Máx/Mín real de estas 24 horas", key="toggle_t3_horas")
            vmin_h, vmax_h = (max(0.0, float(np.floor(np.min(st.session_state.h_historial_vtec_3d) - 2))), float(np.ceil(np.max(st.session_state.h_historial_vtec_3d) + 2))) if ajuste_local_t3_horas else (VMIN_TECU_FIJO, VMAX_TECU_FIJO)
            
            # --- REPRODUCTOR DINÁMICO (MAPA DE FLAMES) ---
            st.subheader("🎬 Reproductor Dinámico de Evolución (Por Horas)")
            if st.button("▶️ Reproducir Serie Horaria Frame por Frame", key="btn_play_horas_frames"):
                contenedor_horas_anim = st.empty()
                for f in range(len(st.session_state.h_etiquetas_reales)):
                    fig_a = plt.figure(figsize=(10, 6), dpi=100)
                    ax_a = plt.axes(projection=ccrs.PlateCarree())
                    ax_a.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_a.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_a.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_a.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    
                    mapa_a = ax_a.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.h_historial_vtec_3d[f, :, :], 
                                          transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_h, vmax=vmax_h, zorder=2)
                    plt.colorbar(mapa_a, ax=ax_a, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    ax_a.set_title(f"FRAME HORARIO: {st.session_state.h_etiquetas_reales[f]} UTC", fontsize=10, weight='bold')
                    
                    contenedor_horas_anim.pyplot(fig_a)
                    plt.close(fig_a)
                    time.sleep(0.50) # Cambia cada 0.5 segundos

            # --- MAPA DE MÁXIMOS ABSOLUTOS ---
            st.subheader("📌 Mapa Fijo de Máximos Absolutos del Día")
            fig_max_h, ax_mxh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mxh.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax_mxh.add_feature(cfeature.LAND, facecolor='#f6f6f6'); ax_mxh.add_feature(cfeature.COASTLINE, edgecolor='#222222')
            mapa_maximos_h = ax_mxh.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.h_matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_h, vmax=vmax_h)
            fig_max_h.colorbar(mapa_maximos_h, ax=ax_mxh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO HORARIO (TECU)', weight='bold')
            st.pyplot(fig_max_h); plt.close(fig_max_h)

            # --- GRÁFICAS POR CIUDAD ---
            st.subheader("📊 Gráfica Comparativa de Localidades Acumuladas (24 Horas)")
            tipo_busqueda_t3h = st.radio("Formato de inserción de localidad (24h):", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3h")
            lat_ch, lon_ch, name_ch = None, None, ""
            if tipo_busqueda_t3h == "Por Nombre de Ciudad":
                nueva_ciudad_h = st.text_input("Nombre de la ciudad:", "Madrid", key="txt_t3h")
                if nueva_ciudad_h: lat_ch, lon_ch, name_ch = geocodificar_localidad(nueva_ciudad_h)
            else:
                col_lch1, col_lch2 = st.columns(2)
                lat_ch_man = col_lch1.number_input("Latitud nodo:", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.1, key="num_lat_t3h")
                lon_ch_man = col_lch2.number_input("Longitud nodo:", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.1, key="num_lon_t3h")
                lat_ch, lon_ch, name_ch = lat_ch_man, lon_ch_man, f"Punto ({lat_ch_man:.1f}, {lon_ch_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico Horario", key="btn_t3h"):
                if lat_ch is not None and lon_ch is not None and (LAT_MIN <= lat_ch <= LAT_MAX) and (LON_MIN <= lon_ch <= LON_MAX):
                    if name_ch not in [c['name'] for c in st.session_state.h_ciudades_lista]: st.session_state.h_ciudades_lista.append({'name': name_ch, 'lat': lat_ch, 'lon': lon_ch})
            if st.session_state.h_ciudades_lista:
                fig_lineas_h, ax_lineas_h = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.h_ciudades_lista:
                    idx_lat = (np.abs(LATS_EUROPA - ciudad_obj['lat'])).argmin(); idx_lon = (np.abs(LONS_EUROPA - ciudad_obj['lon'])).argmin()
                    ax_lineas_h.plot(range(24), st.session_state.h_historial_vtec_3d[:, idx_lat, idx_lon], marker='o', linewidth=2, label=ciudad_obj['name'])
                ax_lineas_h.grid(True, linestyle='--'); ax_lineas_h.set_ylim(vmin_h, vmax_h); ax_lineas_h.set_xticks(range(24)); ax_lineas_h.set_xticklabels([f"{h:02d}h" for h in range(24)], rotation=45); ax_lineas_h.legend(loc="upper right")
                st.pyplot(fig_lineas_h); plt.close(fig_lineas_h)

    # =====================================================================
    # BLOQUE 3: DÍAS COMPLETOS (RANGO CONTINUO HORARIO ENCADENADO)
    # =====================================================================
    elif modo_evolucion == "Días Completos (Rango Continuo)":
        st.subheader("📆 Análisis de Evolución Temporal Continua (24h x N Días)")
        
        if 'dc_historial_vtec_3d' not in st.session_state:
            st.session_state.dc_historial_vtec_3d = None
            st.session_state.dc_etiquetas_reales = []
            st.session_state.dc_matriz_maximos = None
            st.session_state.dc_ciudades_lista = []

        col_dc1, col_dc2 = st.columns(2)
        dc_fecha_inicial = col_dc1.date_input("Fecha Inicial del rango:", datetime.date(2026, 1, 20), key="dc_fecha_ini")
        dc_num_dias = col_dc2.slider("Número de días completos a encadenar:", 2, 7, 3, key="dc_num_dias_slider")

        total_horas_rango = dc_num_dias * 24

        if st.button("🚀 Procesar Días Completos (Línea Continua)", key="btn_ev_dc"):
            with st.spinner(f"Descargando y encadenando las {total_horas_rango} horas consecutivas del rango..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                dc_temp_etiquetas = []
                dc_temp_3d = np.zeros((total_horas_rango, 43, 81))
                dc_exito_total = True
                contador_hora_global = 0

                for d in range(dc_num_dias):
                    fecha_actual = datetime.datetime(dc_fecha_inicial.year, dc_fecha_inicial.month, dc_fecha_inicial.day) + datetime.timedelta(days=d)
                    
                    for h in range(24):
                        link_exitoso = False
                        for m in MINUTOS_CONTIGUOS_GLOBAL:
                            url_intento = generar_enlace_dlr_seguro(fecha_actual.year, fecha_actual.month, fecha_actual.day, h, m)
                            try:
                                response = requests.get(url_intento, headers=headers, timeout=4)
                                if response.status_code == 200:
                                    data = response.json()
                                    link_exitoso = True
                                    break
                            except Exception: pass

                        if not link_exitoso:
                            st.error(f"❌ Corte en la cadena de datos. Sin registros el día {fecha_actual.strftime('%d/%m')} a las {h:02d}:00 UTC.")
                            dc_exito_total = False
                            break

                        vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                        dc_temp_3d[contador_hora_global, :, :] = np.array(vtec_values_list).reshape(43, 81)
                        dc_temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} - {h:02d}h")
                        contador_hora_global += 1
                        
                    if not dc_exito_total: break

                if dc_exito_total:
                    st.session_state.dc_historial_vtec_3d = dc_temp_3d
                    st.session_state.dc_etiquetas_reales = dc_temp_etiquetas
                    st.session_state.dc_matriz_maximos = np.max(dc_temp_3d, axis=0)
                    st.success(f"📊 Línea temporal unificada de {total_horas_rango} puntos reales completada.")

        if st.session_state.dc_historial_vtec_3d is not None:
            ajuste_local_dc = st.toggle("🔍 Optimizar rango vertical al Máx/Mín local de esta serie masiva", key="toggle_dc_ejes")
            vmin_dc, vmax_dc = (max(0.0, float(np.floor(np.min(st.session_state.dc_historial_vtec_3d) - 2))), float(np.ceil(np.max(st.session_state.dc_historial_vtec_3d) + 2))) if ajuste_local_dc else (VMIN_TECU_FIJO, VMAX_TECU_FIJO)
            
            # --- REPRODUCTOR DINÁMICO (MAPA DE FLAMES) ---
            st.subheader("🎬 Reproductor Dinámico de la Evolución Continuada")
            if st.button("▶️ Reproducir Serie Completa Frame por Frame", key="btn_play_dc_frames"):
                contenedor_dc_anim = st.empty()
                for f in range(len(st.session_state.dc_etiquetas_reales)):
                    fig_a = plt.figure(figsize=(11, 6), dpi=100)
                    ax_a = plt.axes(projection=ccrs.PlateCarree())
                    ax_a.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_a.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_a.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_a.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    
                    mapa_a = ax_a.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.dc_historial_vtec_3d[f, :, :], 
                                          transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_dc, vmax=vmax_dc, zorder=2)
                    plt.colorbar(mapa_a, ax=ax_a, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    ax_a.set_title(f"FRAME HORARIO CONTINUO: {st.session_state.dc_etiquetas_reales[f]} UTC", fontsize=10, weight='bold')
                    
                    contenedor_dc_anim.pyplot(fig_a)
                    plt.close(fig_a)
                    time.sleep(0.50) # Cambia cada 0.5 segundos

            # --- MAPA DE MÁXIMOS ABSOLUTOS ---
            st.subheader("📌 Mapa Fijo de Máximos Absolutos del Rango Completo")
            fig_max_dc, ax_mxdc = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mxdc.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax_mxdc.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            ax_mxdc.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
            ax_mxdc.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1)
            
            mapa_maximos_dc = ax_mxdc.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.dc_matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_dc, vmax=vmax_dc)
            fig_max_dc.colorbar(mapa_maximos_dc, ax=ax_mxdc, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO DEL PERIODO (TECU)', weight='bold')
            st.pyplot(fig_max_dc); plt.close(fig_max_dc)

            # --- GRÁFICAS POR CIUDAD ---
            st.subheader("📊 Gráfica Continua del Ciclo de Días Completos Encadenados")
            tipo_busqueda_t3dc = st.radio("Formato de inserción de localidad (Modo Continuo):", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3dc")
            lat_dcl, lon_dcl, name_dcl = None, None, ""
            
            if tipo_busqueda_t3dc == "Por Nombre de Ciudad":
                nueva_ciudad_dc = st.text_input("Nombre del municipio:", "Madrid", key="txt_t3dc")
                if nueva_ciudad_dc: lat_dcl, lon_dcl, name_dcl = geocodificar_localidad(nueva_ciudad_dc)
            else:
                col_ldc1, col_ldc2 = st.columns(2)
                lat_dc_man = col_ldc1.number_input("Latitud nodo de análisis:", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.1, key="num_lat_t3dc")
                lon_dc_man = col_ldc2.number_input("Longitud nodo de análisis:", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.1, key="num_lon_t3dc")
                lat_dcl, lon_dcl, name_dcl = lat_dc_man, lon_dc_man, f"Nodo ({lat_dc_man:.1f}, {lon_dc_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico Continuo", key="btn_t3dc"):
                if lat_dcl is not None and lon_dcl is not None and (LAT_MIN <= lat_dcl <= LAT_MAX) and (LON_MIN <= lon_dcl <= LON_MAX):
                    if name_dcl not in [c['name'] for c in st.session_state.dc_ciudades_lista]: st.session_state.dc_ciudades_lista.append({'name': name_dcl, 'lat': lat_dcl, 'lon': lon_dcl})
            
            if st.session_state.dc_ciudades_lista:
                fig_lineas_dc, ax_lineas_dc = plt.subplots(figsize=(15, 5.5))
                for ciudad_obj in st.session_state.dc_ciudades_lista:
                    idx_lat = (np.abs(LATS_EUROPA - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(LONS_EUROPA - ciudad_obj['lon'])).argmin()
                    ax_lineas_dc.plot(range(len(st.session_state.dc_etiquetas_reales)), st.session_state.dc_historial_vtec_3d[:, idx_lat, idx_lon], linewidth=2, label=ciudad_obj['name'])
                
                ax_lineas_dc.grid(True, linestyle='--')
                ax_lineas_dc.set_ylim(vmin_dc, vmax_dc)
                ax_lineas_dc.set_xticks(range(len(st.session_state.dc_etiquetas_reales)))
                ax_lineas_dc.set_ylabel("TECU", weight='bold')
                ax_lineas_dc.set_xticklabels([st.session_state.dc_etiquetas_reales[k] if k % 6 == 0 else "" for k in range(len(st.session_state.dc_etiquetas_reales))], rotation=45, fontsize=8)
                ax_lineas_dc.legend(loc="upper right")
                st.pyplot(fig_lineas_dc); plt.close(fig_lineas_dc)
# =====================================================================
# PESTAÑA 4: PRONÓSTICO (MODO DUAL: HISTÓRICO / TIEMPO REAL OPERATIVO)
# =====================================================================
with tab4:
    st.title("Predicción del VTEC Ionosférico")
    st.markdown("### Motor Predictivo Localizado (Climatología + Persistencia Amortiguada)")
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
        st.session_state.p4_modo_activo = ""

    st.subheader("⚙️ Configuración del Escenario Predictivo")
    
    col_p1, col_p2 = st.columns(2)
    
    # 1. SELECTOR INTERACTIVO DE MODO DE TRABAJO (HISTÓRICO O PRESENTE)
    p4_modo_trabajo = col_p1.radio(
        "Elige el escenario temporal de ejecución:", 
        ["Simulación Histórica (Validación Cruzada)", "Pronóstico Operativo en Tiempo Real (Presente Actual)"], 
        horizontal=False, 
        key="p4_radio_modo_global"
    )
    
    # 2. SELECTOR DUAL ALTERNATIVO DE LOCALIZACIÓN SOLICITADO
    p4_tipo_pos = col_p2.radio(
        "Método de posicionamiento para el receptor:", 
        ["Por Nombre de Ciudad/Región", "Por Coordenadas Manuales (Lat/Lon)"], 
        horizontal=True, 
        key="p4_radio_pos"
    )
    
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

    st.divider()

    # CONTROL CONDICIONAL DE ENTRADA DE FECHA SEGÚN EL MODO ELEGIDO
    if p4_modo_trabajo == "Simulación Histórica (Validación Cruzada)":
        st.markdown("ℹ️ *Este modo descarga un día del pasado completo (pasos de 1h) y lo compara con los datos reales que ocurrieron al día siguiente.*")
        p4_fecha_base = st.date_input("Selecciona la fecha base del historial pasado (24 Horas):", datetime.date(2026, 1, 1), key="p4_date_sel_hist")
    else:
        # Modo Presente Operativo: Calculamos el reloj UTC en tiempo real automáticamente
        ahora_utc = datetime.datetime.utcnow()
        minuto_redondeado = (ahora_utc.minute // 5) * 5
        fecha_base_produccion = ahora_utc.replace(minute=minuto_redondeado, second=0, microsecond=0)
        
        st.info(f"🛰️ **Modo Operativo Activo:** El sistema sincronizará el reloj con el servidor del DLR a la última hora de refresco: **{fecha_base_produccion.strftime('%d/%m/%Y a las %H:%M')} UTC**.")

    # =====================================================================
    # BOTÓN DE EJECUCIÓN DEL PRONÓSTICO DUAL
    # =====================================================================
    if st.button("🚀 Calcular Pronóstico Ionosférico", key="p4_btn_run"):
        if lat_p4 is not None and lon_p4 is not None:
            if (LAT_MIN <= lat_p4 <= LAT_MAX) and (LON_MIN <= lon_p4 <= LON_MAX):
                
                idx_lat_p4 = (np.abs(LATS_EUROPA - lat_p4)).argmin()
                idx_lon_p4 = (np.abs(LONS_EUROPA - lon_p4)).argmin()
                headers = {"User-Agent": "Mozilla/5.0"}
                
                # -----------------------------------------------------------------
                # RAMAL A: EJECUCIÓN EN MODO SIMULACIÓN HISTÓRICA
                # -----------------------------------------------------------------
                if p4_modo_trabajo == "Simulación Histórica (Validación Cruzada)":
                    with st.spinner("Descargando ciclo diurno histórico para validación cruzada..."):
                        temp_cronologia_pasado = []
                        temp_fechas_pasado = []
                        exito_descarga_p1 = True
                        
                        for hora_actual in range(24):
                            link_exitoso = False
                            for m in range(0, 30, 5):
                                url_intento = generar_enlace_dlr_seguro(p4_fecha_base.year, p4_fecha_base.month, p4_fecha_base.day, hora_actual, m)
                                try:
                                    response = requests.get(url_intento, headers=headers, timeout=3)
                                    if response.status_code == 200:
                                        data = response.json()
                                        link_exitoso = True
                                        break
                                except Exception: pass
                            if not link_exitoso:
                                st.error(f"❌ Datos no encontrados para la hora {hora_actual:02d}:00 UTC en el historial.")
                                exito_descarga_p1 = False
                                break
                            
                            vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                            matriz_instante = np.array(vtec_values_list).reshape(43, 81)
                            temp_cronologia_pasado.append(matriz_instante[idx_lat_p4, idx_lon_p4])
                            temp_fechas_pasado.append(datetime.datetime(p4_fecha_base.year, p4_fecha_base.month, p4_fecha_base.day) + datetime.timedelta(hours=hora_actual))

                        if exito_descarga_p1:
                            vector_vtec_serie = np.array(temp_cronologia_pasado)
                            temp_cronologia_futuro_real = []
                            temp_fechas_futuro = []
                            p4_fecha_futuro_base = datetime.datetime(p4_fecha_base.year, p4_fecha_base.month, p4_fecha_base.day) + datetime.timedelta(days=1)
                            exito_descarga_p2 = True
                            
                            for hora_val in range(3):
                                link_exitoso = False
                                for m in range(0, 30, 5):
                                    url_intento = generar_enlace_dlr_seguro(p4_fecha_futuro_base.year, p4_fecha_futuro_base.month, p4_fecha_futuro_base.day, hora_val, m)
                                    try:
                                        response = requests.get(url_intento, headers=headers, timeout=3)
                                        if response.status_code == 200:
                                            data_val = response.json()
                                            link_exitoso = True
                                            break
                                    except Exception: pass
                                if not link_exitoso:
                                    st.error(f"❌ Sin datos de validación para las {hora_val:02d}:00 UTC del día siguiente.")
                                    exito_descarga_p2 = False
                                    break
                                    
                                vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data_val['data']['grid']['features']]
                                matriz_instante = np.array(vtec_values_list).reshape(43, 81)
                                temp_cronologia_futuro_real.append(matriz_instante[idx_lat_p4, idx_lon_p4])
                                temp_fechas_futuro.append(p4_fecha_futuro_base + datetime.timedelta(hours=hora_val))

                            if exito_descarga_p2:
                                vector_real_futuro = np.array(temp_cronologia_futuro_real)
                                perfil_estacional = np.copy(vector_vtec_serie)
                                ultimo_valor_real = vector_vtec_serie[-1]
                                anomalia_inicial = ultimo_valor_real - perfil_estacional[-1]
                                
                                vector_prediccion_futura = []
                                alpha = 0.85
                                for k in range(1, 4):
                                    slot_futuro = k % 24
                                    valor_predicho = perfil_estacional[slot_futuro] + anomalia_inicial * (alpha ** k)
                                    vector_prediccion_futura.append(valor_predicho)
                                
                                vector_prediccion_futura = np.array(vector_prediccion_futura)
                                mae_calc = float(np.mean(np.abs(vector_real_futuro - vector_prediccion_futura)))
                                acierto_calc = max(0.0, 100 - (mae_calc / np.mean(vector_real_futuro)) * 100)
                                
                                # Volcado a memoria de Streamlit (Modo Histórico)
                                st.session_state.p4_vector_pasado = vector_vtec_serie
                                st.session_state.p4_fechas_pasado = temp_fechas_pasado
                                st.session_state.p4_vector_futuro_real = vector_real_futuro
                                st.session_state.p4_vector_futuro_calc = vector_prediccion_futura
                                st.session_state.p4_fechas_futuro = temp_fechas_futuro
                                st.session_state.p4_mae = mae_calc
                                st.session_state.p4_acierto = acierto_calc
                                st.session_state.p4_modo_activo = "HISTÓRICO"
                                st.session_state.p4_info_punto = f"{label_p4} | Coordenadas: {lat_p4:.3f}°N, {lon_p4:.3f}°E"
                                st.success("🎯 Simulación histórica calculada con éxito.")

                # -----------------------------------------------------------------
                # RAMAL B: EJECUCIÓN EN MODO TIEMPO REAL OPERATIVO (PRESENTE)
                # -----------------------------------------------------------------
                else:
                    with st.spinner("Sincronizando con el reloj del DLR y extrayendo las últimas 24 horas reales..."):
                        temp_cronologia_presente = []
                        temp_fechas_presente = []
                        exito_descarga_rt = True
                        
                        # Descargamos las últimas 12 muestras bihorarias (24 horas hacia atrás desde hoy)
                        for i in range(11, -1, -1):
                            fecha_muestra = fecha_base_produccion - datetime.timedelta(hours=i * 2)
                            link_exitoso = False
                            minutos_contingencia = [fecha_muestra.minute, 0, 5, 10, 15, 20, 25, 30]
                            
                            for m in minutos_contingencia:
                                url_intento = generar_enlace_dlr_seguro(fecha_muestra.year, fecha_muestra.month, fecha_muestra.day, fecha_muestra.hour, m)
                                try:
                                    response = requests.get(url_intento, headers=headers, timeout=3)
                                    if response.status_code == 200:
                                        data_rt = response.json()
                                        link_exitoso = True
                                        break
                                except Exception: pass
                                
                            if not link_exitoso:
                                st.error(f"❌ Imposible construir el pasado reciente. Datos ausentes del día {fecha_muestra.strftime('%d/%m')} a las {fecha_muestra.hour:02d}h UTC.")
                                exito_descarga_rt = False
                                break
                                
                            vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data_rt['data']['grid']['features']]
                            matriz_instante = np.array(vtec_values_list).reshape(43, 81)
                            temp_cronologia_presente.append(matriz_instante[idx_lat_p4, idx_lon_p4])
                            temp_fechas_presente.append(fecha_muestra)

                        if exito_descarga_rt:
                            vector_vtec_serie_rt = np.array(temp_cronologia_presente)
                            
                            # Proyección hacia el futuro desconocido e inexistente (+6 horas adelante)
                            perfil_estacional_rt = np.copy(vector_vtec_serie_rt)
                            ultimo_valor_real_rt = vector_vtec_serie_rt[-1]
                            ultimo_slot_horario_rt = len(vector_vtec_serie_rt) - 1
                            
                            vector_prediccion_futura_rt = []
                            temp_fechas_futuro_rt = []
                            alpha = 0.85
                            
                            for k in range(1, 4):
                                fecha_fut_inst = fecha_base_produccion + datetime.timedelta(hours=k * 2)
                                temp_fechas_futuro_rt.append(fecha_fut_inst)
                                slot_futuro = (ultimo_slot_horario_rt + k) % 12
                                valor_predicho_rt = perfil_estacional_rt[slot_futuro] + (ultimo_valor_real_rt - perfil_estacional_rt[ultimo_slot_horario_rt]) * (alpha ** k)
                                vector_prediccion_futura_rt.append(valor_predicho_rt)
                                
                            # Volcar a memoria de Streamlit (Modo Tiempo Real)
                            st.session_state.p4_vector_pasado = vector_vtec_serie_rt
                            st.session_state.p4_fechas_pasado = temp_fechas_presente
                            st.session_state.p4_vector_futuro_real = None  # No existe en el presente actual
                            st.session_state.p4_vector_futuro_calc = np.array(vector_prediccion_futura_rt)
                            st.session_state.p4_fechas_futuro = temp_fechas_futuro_rt
                            st.session_state.p4_modo_activo = "TIEMPO_REAL"
                            st.session_state.p4_info_punto = f"{label_p4} | Coordenadas: {lat_p4:.3f}°N, {lon_p4:.3f}°E"
                            st.success("🔮 Modelo operativo en tiempo real ejecutado. Proyección a +6 horas completada.")
            else:
                st.error("❌ Las coordenadas ingresadas se encuentran fuera de la cuadrícula de Europa.")

    # =====================================================================
    # 3. DESPLIEGUE GRÁFICO INTELIGENTE ADAPTADO SEGÚN EL MODO ACTIVO
    # =====================================================================
    if st.session_state.p4_vector_pasado is not None:
        st.divider()
        
        # Renderizado condicional de los KPI en la cabecera del gráfico
        if st.session_state.p4_modo_activo == "HISTÓRICO":
            col_k1, col_k2, col_k3 = st.columns(3)
            col_k1.metric(label="📊 Error Medio Absoluto (MAE)", value=f"{st.session_state.p4_mae:.3f} TECU")
            col_k2.metric(label="🎯 Porcentaje Estimado de Acierto", value=f"{st.session_state.p4_acierto:.1f} %")
            col_k3.info(f"**Ubicación de Análisis:**\n{st.session_state.p4_info_punto}")
        else:
            col_k1, col_k2 = st.columns([1, 2])
            col_k1.metric(label="📡 Estado del Transmisor DLR", value="ONLINE (Operativo)")
            col_k2.info(f"**Modo de Pronóstico Directo:** Lanzado en Vivo para:\n{st.session_state.p4_info_punto}")

        p4_toggle_ejes = st.toggle("🔍 Activar regla +-2 local en el Eje Y (Optimizar visualización)", key="toggle_p4_ejes")
        
        fig_p4, ax_p4 = plt.subplots(figsize=(15, 6), dpi=100)
        
        # 1. Pasado Real Registrado (Línea Azul con marcadores redondos)
        ax_p4.plot(st.session_state.p4_fechas_pasado, st.session_state.p4_vector_pasado, 
                   color='#1565c0', linewidth=2.5, label='Pasado Registrado (Datos Reales DLR)', marker='o', markersize=5)
        
        # Conectamos visualmente la última muestra real con el inicio de la línea de predicción
        fechas_linea_prediccion = [st.session_state.p4_fechas_pasado[-1]] + st.session_state.p4_fechas_futuro
        valores_linea_prediccion = [st.session_state.p4_vector_pasado[-1]] + list(st.session_state.p4_vector_futuro_calc)

        # 2. Línea Roja Discontinua con la Predicción Matemática
        lbl_roja = 'Predicción Matemática (Horizonte 3h)' if st.session_state.p4_modo_activo == "HISTÓRICO" else 'Predicción Matemática Operativa (+6 Horas Futuras)'
        ax_p4.plot(fechas_linea_prediccion, valores_linea_prediccion, 
                   color='#d50000', linewidth=2.5, linestyle='--', label=lbl_roja, marker='x', markersize=6, zorder=4)
        
        # 3. Renderizado condicional del ramal elegido
        if st.session_state.p4_modo_activo == "HISTÓRICO":
            # Si es histórico pintamos la línea verde de validación real
            ax_p4.plot(st.session_state.p4_fechas_futuro, st.session_state.p4_vector_futuro_real, 
                       color='#00e676', linewidth=2.5, label='Datos Reales de Validación', marker='s', markersize=5, zorder=3)
            ax_p4.fill_between(st.session_state.p4_fechas_futuro, st.session_state.p4_vector_futuro_calc, st.session_state.p4_vector_futuro_real, 
                               color='#ff3d00', alpha=0.1, label='Margen de Error')
            str_titulo_modo = "SIMULACIÓN HISTÓRICA CON VALIDACIÓN"
        else:
            # Si es tiempo real pintamos el sombreado rosa de "zona de pronóstico hacia el futuro" de tu script
            ax_p4.axvspan(st.session_state.p4_fechas_pasado[-1], st.session_state.p4_fechas_futuro[-1], 
                          color='#ffebee', alpha=0.5, label='Zona de Pronóstico Ionosférico')
            str_titulo_modo = f"TIEMPO REAL OPERATIVO (ACTIVO HASTA LAS {st.session_state.p4_fechas_futuro[-1].strftime('%H:%M')} UTC)"

        ax_p4.grid(True, linestyle='--', alpha=0.5)
        
        # Control estricto de los límites verticales del eje Y
        if p4_toggle_ejes:
            valores_totales = np.concatenate([st.session_state.p4_vector_pasado, st.session_state.p4_vector_futuro_calc])
            if st.session_state.p4_vector_futuro_real is not None:
                valores_totales = np.concatenate([valores_totales, st.session_state.p4_vector_futuro_real])
            Y_MIN_VAL = max(0.0, float(np.floor(np.min(valores_totales) - 2)))
            Y_MAX_VAL = float(np.ceil(np.max(valores_totales) + 2))
            ax_p4.set_ylim(Y_MIN_VAL, Y_MAX_VAL)
            str_escala = f"Regla +-2 Local: {int(Y_MIN_VAL)}-{int(Y_MAX_VAL)} TECU"
        else:
            ax_p4.set_ylim(VMIN_TECU_FIJO, VMAX_TECU_FIJO)
            str_escala = f"Escala Universal Fija: {int(VMIN_TECU_FIJO)}-{int(VMAX_TECU_FIJO)} TECU"
            
        # Formateador de tiempo del eje X
        ax_p4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_p4.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
        
        plt.xlabel("Escala Temporal Unificada (UTC)", fontsize=11, weight='bold')
        plt.ylabel("Contenido Total de Electrones - VTEC (TECU)", fontsize=11, weight='bold')
        plt.title(f"PRONÓSTICO DE TEC EN EUROPA [{str_titulo_modo}]\n[Eje Y Controlado con {str_escala}]", fontsize=12, weight='bold', pad=15)
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        st.pyplot(fig_p4)
        plt.close(fig_p4)

        # Imprime la telemetría textual abajo del mapa si es el modo operativo real
        if st.session_state.p4_modo_activo == "TIEMPO_REAL":
            st.info("### 📡 Telemetría de Proyección de Contenido Electrónico Futuro:")
            for idx, f_fut in enumerate(st.session_state.p4_fechas_futuro):
                st.markdown(f"* ⏱️ **Pronóstico para las {f_fut.strftime('%H:%M')} UTC** del {f_fut.strftime('%d/%m')}: `{st.session_state.p4_vector_futuro_calc[idx]:.3f} TECU`")
# =====================================================================
# PESTAÑA 5: DESVIACIONES DEL MODELO 
# =====================================================================
with tab5:
    st.title("📉 Desviaciones e Incertidumbre")
    st.divider()

    # Variables de estado de sesión persistentes para evitar descargas duplicadas
    if 'p5_historial_real_3d' not in st.session_state:
        st.session_state.p5_historial_real_3d = None
        st.session_state.p5_historial_rms_3d = None
        st.session_state.p5_historial_model_3d = None
        st.session_state.p5_historial_desviacion_3d = None
        st.session_state.p5_etiquetas_fechas_reales = []
        st.session_state.p5_fecha_info = ""

    # Tabla física de frecuencias de las señales base
    FRECUENCIAS_GNSS_P5 = {
        "GPS (L1 - 1575.42 MHz)": 1575.42 * 1e6,
        "Galileo (E1 - 1575.42 MHz)": 1575.42 * 1e6,
        "GLONASS (G1 - 1602.00 MHz)": 1602.00 * 1e6,
        "BeiDou (B1I - 1561.10 MHz)": 1561.10 * 1e6
    }

    # CONTROLES PRINCIPALES DE DESCARGA
    st.subheader("📥 Extracción y Configuración de Datos")
    col_v1, col_v2, col_v3 = st.columns(3)
    p5_fecha = col_v1.date_input("Selecciona la Fecha a Auditar:", datetime.date(2026, 1, 24), key="p5_date_calendar")
    p5_constelacion = col_v2.selectbox("📡 Señal GNSS de Referencia:", list(FRECUENCIAS_GNSS_P5.keys()), key="p5_select_freq")
    p5_hora_vista = col_v3.slider("Hora de Observación Estática (UTC):", 0, 23, 13, key="p5_hour_static")

    # Factor de conversión unificado de tu ecuación física de refracción
    f_hz_p5 = FRECUENCIAS_GNSS_P5[p5_constelacion]
    FACTOR_METROS_P5 = (40.3 / (f_hz_p5 ** 2)) * 1e16

    if st.button("🚀 Procesar Bloque Completo (24 Horas Consecutivas)", key="p5_btn_process"):
        with st.spinner("Sincronizando claves multi-variable con el servidor del DLR..."):
            headers = {"User-Agent": "Mozilla/5.0"}
            
            temp_real_3d = np.zeros((24, 43, 81))
            temp_rms_3d = np.zeros((24, 43, 81))
            temp_model_3d = np.zeros((24, 43, 81))
            temp_etiquetas = []
            exito_total_p5 = True

            for h in range(24):
                link_exitoso = False
                minuto_exitoso = 0
                data_instante = None
                
                # Bucle de contingencia temporal en pasos de 5 min
                for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                    url_intento = generar_enlace_dlr_seguro(p5_fecha.year, p5_fecha.month, p5_fecha.day, h, m)
                    try:
                        response = requests.get(url_intento, headers=headers, timeout=4)
                        if response.status_code == 200:
                            data_instante = response.json()
                            link_exitoso = True
                            minuto_exitoso = m
                            break
                    except Exception: pass

                if not link_exitoso:
                    st.error(f"❌ Imposible consolidar la auditoría. Faltan datos a las {h:02d}:00 UTC.")
                    exito_total_p5 = False
                    break

                real_list, rms_list, model_list = [], [], []
                if 'data' in data_instante and 'grid' in data_instante['data'] and 'features' in data_instante['data']['grid']:
                    for feature in data_instante['data']['grid']['features']:
                        if 'properties' in feature:
                            props = feature['properties']
                            if 'vtec_assimilated_tecu' in props and 'vtec_rms_tecu' in props and 'vtec_model_tecu' in props:
                                real_list.append(props['vtec_assimilated_tecu'])
                                rms_list.append(props['vtec_rms_tecu'])
                                model_list.append(props['vtec_model_tecu'])

                if len(real_list) != 43 * 81:
                    st.error(f"❌ Estructura de cuadrícula asimétrica o incompleta en la hora {h:02d}.")
                    exito_total_p5 = False
                    break

                # Conversión y almacenamiento directo en metros reales
                temp_real_3d[h, :, :] = np.array(real_list).reshape(43, 81) * FACTOR_METROS_P5
                temp_rms_3d[h, :, :] = np.array(rms_list).reshape(43, 81) * FACTOR_METROS_P5
                temp_model_3d[h, :, :] = np.array(model_list).reshape(43, 81) * FACTOR_METROS_P5
                temp_etiquetas.append(f"{h:02d}:{minuto_exitoso:02d}")

            if exito_total_p5:
                st.session_state.p5_historial_real_3d = temp_real_3d
                st.session_state.p5_historial_rms_3d = temp_rms_3d
                st.session_state.p5_historial_model_3d = temp_model_3d
                st.session_state.p5_historial_desviacion_3d = temp_model_3d - temp_real_3d
                st.session_state.p5_etiquetas_fechas_reales = temp_etiquetas
                st.session_state.p5_fecha_info = f"{p5_fecha.strftime('%d/%m/%Y')} - {p5_constelacion}"
                st.success("📊 Repositorio multi-variable cargado correctamente en memoria.")

    # MENÚ SECUNDARIO INTERNO PARA SEPARAR LOS PILARES SOLICITADOS
    if st.session_state.p5_historial_real_3d is not None:
        st.divider()
        st.subheader("🎯 Panel de Control de Auditoría Avanzada")
        
        sub_seccion_p5 = st.radio(
            "Selecciona el pilar de estudio científico:",
            ["Pilar 1: Evolución de la Desviación del Modelo (Residuos Teóricos)", 
             "Pilar 2: Incertidumbre del Dato (Margen de Confianza RMS)"],
            horizontal=True, key="radio_sub_p5"
        )
        
        # Sistema dual alternativo de entrada de localización común para la pestaña
        st.markdown("#### 📍 Punto de Control e Intermediación")
        p5_tipo_pos = st.radio("Método de entrada de localización para control:", ["Buscar por Nombre", "Introducir Coordenadas Manuales"], horizontal=True, key="p5_pos_radio")
        
        lat_v, lon_v, label_v = None, None, ""
        if p5_tipo_pos == "Buscar por Nombre":
            ciudad_p5_txt = st.text_input("Ingresa la ciudad o nodo a auditar:", "Madrid", key="p5_txt_loc")
            if ciudad_p5_txt:
                lat_v, lon_v, label_v = geocodificar_localidad(ciudad_p5_txt)
        else:
            col_lv1, col_lv2 = st.columns(2)
            lat_v_man = col_lv1.number_input("Latitud Nodo exacto (°N):", min_value=float(LAT_MIN), max_value=float(LAT_MAX), value=40.41, step=0.01, key="num_lat_p5")
            lon_v_man = col_lv2.number_input("Longitud Nodo exacto (°E):", min_value=float(LON_MIN), max_value=float(LON_MAX), value=-3.70, step=0.01, key="num_lon_p5")
            lat_v, lon_v, label_v = lat_v_man, lon_v_man, "Coordenadas fijas"

        # CÁLCULOS PUNTUALES EN METROS
        if lat_v is not None and lon_v is not None:
            if (LAT_MIN <= lat_v <= LAT_MAX) and (LON_MIN <= lon_v <= LON_MAX):
                idx_lat = (np.abs(LATS_EUROPA - lat_v)).argmin()
                idx_lon = (np.abs(LONS_EUROPA - lon_v)).argmin()
                
                val_real_p = st.session_state.p5_historial_real_3d[p5_hora_vista, idx_lat, idx_lon]
                val_model_p = st.session_state.p5_historial_model_3d[p5_hora_vista, idx_lat, idx_lon]
                val_dev_p = st.session_state.p5_historial_desviacion_3d[p5_hora_vista, idx_lat, idx_lon]
                val_rms_p = st.session_state.p5_historial_rms_3d[p5_hora_vista, idx_lat, idx_lon]
                
                c_1, c_2, c_3 = st.columns(3)
                if "Pilar 1" in sub_seccion_p5:
                    c_1.metric(label="🛰️ Retraso Real", value=f"{val_real_p:.3f} m")
                    c_2.metric(label="🔮 Retraso Teórico", value=f"{val_model_p:.3f} m")
                    c_3.metric(label="📉 Desviación Residual", value=f"{val_dev_p:+.3f} m", delta=f"{val_dev_p:.3f} m", delta_color="inverse")
                else:
                    c_1.metric(label="🛡️ Incertidumbre de Confianza", value=f"± {val_rms_p:.3f} m")
                    c_2.metric(label="📍 Ubicación de Control", value=label_v)
                    c_3.info(f"**Coordenadas:**\nLat {lat_v:.2f}°N | Lon {lon_v:.2f}°E")
            else: st.error("Fuera de la malla de Europa.")

        st.divider()
        p5_toggle_color = st.toggle("🔍 Optimizar rango cromático al Máx/Mín de este mapa específico", key="toggle_p5_scale")

        # =====================================================================
        # DESPLIEGUE DEL PILAR 1: EVOLUCIÓN DE LA DESVIACIÓN
        # =====================================================================
        if "Pilar 1" in sub_seccion_p5:
            st.write(f"### 📉 Análisis de Residuos: Modelo Teórico menos Valor Real")
            
            # Configuración de límites fijos simétricos para la paleta divergente 'seismic'
            if p5_toggle_color:
                lim_abs = float(np.max(np.abs(st.session_state.p5_historial_desviacion_3d[p5_hora_vista])))
                vmin_p5, vmax_p5 = -lim_abs, lim_abs
                str_status = "Ajuste Simétrico Local"
            else:
                lim_abs_global = float(np.ceil(np.max(np.abs(st.session_state.p5_historial_desviacion_3d))))
                vmin_p5, vmax_p5 = -lim_abs_global, lim_abs_global
                str_status = "Rango Universal Centrado"

            fig_p5_d = plt.figure(figsize=(11, 6), dpi=100)
            ax_p5_d = plt.axes(projection=ccrs.PlateCarree())
            ax_p5_d.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax_p5_d.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_p5_d.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_p5_d.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            
            grid_d = ax_p5_d.gridlines(draw_labels=True, color='gray', alpha=0.15, linestyle='--')
            grid_d.top_labels, grid_d.right_labels = False, False
            grid_d.xformatter, grid_d.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER

            mapa_d = ax_p5_d.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.p5_historial_desviacion_3d[p5_hora_vista], 
                                         transform=ccrs.PlateCarree(), cmap='seismic', alpha=0.85, shading='gouraud', vmin=vmin_p5, vmax=vmax_p5, zorder=2)
            plt.colorbar(mapa_d, ax=ax_p5_d, orientation='horizontal', pad=0.08, shrink=0.7).set_label(f'DESVIACIÓN DEL MODELO DE FONDO (METROS) [{str_status}]', weight='bold')
            ax_p5_d.set_title(f"MAPA ESTÁTICO DE RESIDUOS A LAS {p5_hora_vista:02d}:00 UTC\n[Blanco = Coincidencia | Rojo = Sobreestima | Azul = Subestima]", fontsize=10, weight='bold')
            st.pyplot(fig_p5_d)
            plt.close(fig_p5_d)

            # Reproductor dinámico - CORREGIDO CON PROYECCIÓN CARTOPY INTRADÍA
            st.subheader("🎬 Reproductor Dinámico de la Desviación (24 Horas)")
            if st.button("▶️ Reproducir Evolución de Errores", key="btn_play_p5_dev"):
                contenedor_p5_anim = st.empty()
                for f in range(24):
                    fig_a = plt.figure(figsize=(11, 6), dpi=100)
                    ax_a = plt.axes(projection=ccrs.PlateCarree())
                    ax_a.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_a.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_a.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_a.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    
                    grid_a = ax_a.gridlines(draw_labels=True, color='gray', alpha=0.15, linestyle='--')
                    grid_a.top_labels, grid_a.right_labels = False, False
                    grid_a.xformatter, grid_a.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
                    
                    mapa_a = ax_a.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.p5_historial_desviacion_3d[f, :, :], 
                                              transform=ccrs.PlateCarree(), cmap='seismic', alpha=0.85, shading='gouraud', vmin=vmin_p5, vmax=vmax_p5, zorder=2)
                    plt.colorbar(mapa_a, ax=ax_a, orientation='horizontal', pad=0.08, shrink=0.7).set_label('DESVIACIÓN EN METROS', weight='bold')
                    ax_a.set_title(f"FRAME HORARIO DE CONTROL: {st.session_state.p5_etiquetas_fechas_reales[f]} UTC", fontsize=10, weight='bold')
                    
                    contenedor_p5_anim.pyplot(fig_a)
                    plt.close(fig_a)
                    time.sleep(0.4)

        # =====================================================================
        # DESPLIEGUE DEL PILAR 2: INCERTIDUMBRE DEL DATO
        # =====================================================================
        else:
            st.write(f"### 🛡️ Tolerancia Geodésica: Análisis de Confianza de Medida (RMS)")
            
            if p5_toggle_color:
                vmin_rms, vmax_rms = float(np.min(st.session_state.p5_historial_rms_3d[p5_hora_vista])), float(np.max(st.session_state.p5_historial_rms_3d[p5_hora_vista]))
                str_status_r = "Rango Optimizado Local"
            else:
                vmin_rms, vmax_rms = 0.0, float(np.ceil(np.max(st.session_state.p5_historial_rms_3d)))
                str_status_r = "Escala Universal Fija"

            fig_p5_r = plt.figure(figsize=(11, 6), dpi=100)
            ax_p5_r = plt.axes(projection=ccrs.PlateCarree())
            ax_p5_r.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
            ax_p5_r.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=1)
            ax_p5_r.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_p5_r.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            
            grid_r = ax_p5_r.gridlines(draw_labels=True, color='gray', alpha=0.15, linestyle='--')
            grid_r.top_labels, grid_r.right_labels = False, False
            grid_r.xformatter, grid_r.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER

            mapa_r = ax_p5_r.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.p5_historial_rms_3d[p5_hora_vista], 
                                         transform=ccrs.PlateCarree(), cmap='YlOrRd', alpha=0.85, shading='gouraud', vmin=vmin_rms, vmax=vmax_rms, zorder=2)
            plt.colorbar(mapa_r, ax=ax_p5_r, orientation='horizontal', pad=0.08, shrink=0.7).set_label(f'INCERTIDUMBRE DEL DATO (METROS RMS) [{str_status_r}]', weight='bold')
            ax_p5_r.set_title(f"MAPA DE MARGEN DE TOLERANCIA RMS A LAS {p5_hora_vista:02d}:00 UTC\n[Muestra la calidad métrica intrínseca de los datos asimilados por los satélites]", fontsize=10, weight='bold')
            st.pyplot(fig_p5_r)
            plt.close(fig_p5_r)

# =====================================================================
# PESTAÑA 6: COMENTARIOS Y FEEDBACK (TU BASE PERSONALIZADA CORREGIDA)
# =====================================================================
with tab6:
    st.title("💬 Buzón de Sugerencias y Feedback")
    st.markdown("""
    Tu opinión es fundamental para seguir mejorando. 
    Utiliza este formulario para reportar errores, proponer nuevas herramientas o comentar tu experiencia.
    """)
    st.divider()

    # Formulario estético de recogida de información
    with st.form("formulario_feedback", clear_on_submit=False):
        nombre = st.text_input("👤 Tu Nombre / Institución:", placeholder="Ej. Laboratorio INAIA UCLM")
        email_usuario = st.text_input("📧 Correo electrónico de contacto:", placeholder="ejemplo@correo.com")
        
        tipo_comentario = st.selectbox(
            "📌 Tipo de aportación:",
            ["Sugerencia de mejora", "Reportar un fallo", "Opinión", "Petición de código"]
        )
        
        comentario = st.text_area("✍️ Escribe aquí tus comentarios o detalles del error:", height=150)
        
        # Botón de validación del formulario
        procesar_envio = st.form_submit_button("✉️ Preparar Correo de Feedback")

    if procesar_envio:
        if not comentario.strip():
            st.error("⚠️ Por favor, escribe un comentario antes de continuar.")
        else:
            # Configuramos tus datos de recepción (Mantenemos tu correo real)
            TU_EMAIL = "luciarc2004@gmail.com"  
            asunto_correo = f"[FEEDBACK PORTAL IONOSFERA] - {tipo_comentario}"
            
            # Formateamos el cuerpo del texto para el Mailto
            cuerpo_correo = (
                f"Hola,\n\n"
                f"Has recibido un nuevo comentario desde el Portal Ionosférico:\n\n"
                f"Remitente: {nombre if nombre else 'Anónimo'}\n"
                f"Contacto: {email_usuario if email_usuario else 'No facilitado'}\n"
                f"Categoría: {tipo_comentario}\n\n"
                f"Mensaje:\n{comentario}\n\n"
                f"--- Enviado desde la aplicación web ---"
            )
            
            # Codificación segura de caracteres para URLs de correo (evita romper acentos y espacios)
            import urllib.parse
            asunto_codificado = urllib.parse.quote(asunto_correo)
            cuerpo_codificado = urllib.parse.quote(cuerpo_correo)
            
            # Construcción del enlace Mailto dinámico
            mailto_url = f"mailto:{TU_EMAIL}?subject={asunto_codificado}&body={cuerpo_codificado}"
            
            # Notificación y botón de acción para abrir el gestor de correo
            st.success("🎉 ¡Estructura de feedback generada con éxito!")
            
            # REPARADO: Cambiado 'unsafe_allow_value=True' por 'unsafe_allow_html=True'
            st.markdown(
                f'<a href="{mailto_url}" target="_blank" style="text-decoration:none;">'
                f'<div style="padding:12px; background-color:#ff3d00; color:white; text-align:center; '
                f'border-radius:6px; font-weight:bold; font-size:16px; cursor:pointer;">'
                f'🚀 Haz clic aquí para enviar el correo electrónico'
                f'</div></a>', 
                unsafe_allow_html=True
            )
            st.caption("Nota: Al hacer clic en el botón rojo, se abrirá tu aplicación de correo local (Gmail, Outlook...) para enviar la información de manera segura.")






