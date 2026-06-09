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

# Configuración de la página web
st.set_page_config(page_title="Portal de Monitoreo Ionosférico", layout="wide")

# =====================================================================
# CONFIGURACIÓN GLOBAL
# =====================================================================
MINUTOS_CONTIGUOS_GLOBAL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

# Definición global de las pestañas obligatorias (Añadido: Pronóstico)
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Inicio y Monitoreo Real", 
    "📊 Análisis en el pasado", 
    "📈 Evolución TECU",
    "🔮 Pronóstico"
])

# FUNCIÓN COMPARTIDA DE GEOCODIFICACIÓN BLINDADA (SISTEMA HÍBRIDO EXTENDIDO)
def geocodificar_localidad(nombre_lugar):
    nombre_clean = nombre_lugar.strip().lower()
    nombre_clean = nombre_clean.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    
    ciudades_respaldo = {
        "madrid": (40.4167, -3.7037),
        "barcelona": (41.3851, 2.1734),
        "puertollano": (38.6871, -4.1086),
        "valencia": (39.4699, -0.3763),
        "sevilla": (37.3891, -5.9845),
        "zaragoza": (41.6488, -0.8891),
        "malaga": (36.7212, -4.4214),
        "murcia": (37.9922, -1.1307),
        "palma": (39.5696, 2.6502),
        "las palmas": (28.1235, -15.4363),
        "bilbao": (43.2630, -2.9350),
        "alicante": (38.3452, -0.4810),
        "valladolid": (41.6523, -4.7245),
        "vigo": (42.2406, -8.7207),
        "gijon": (43.5357, -5.6615),
        "hospitalet": (41.3597, 2.0997),
        "coruña": (43.3623, -8.4115),
        "granada": (37.1773, -3.5986),
        "oviedo": (43.3603, -5.8448),
        "albacete": (38.9943, -1.8585),
        "santander": (43.4623, -3.8099),
        "toledo": (39.8628, -4.0273),
        "ciudad real": (38.9861, -3.9275),
        "palermo": (38.1157, 13.3614),
        "roma": (41.9028, 12.4964),
        "paris": (48.8566, 2.3522),
        "berlin": (52.5200, 13.4050),
        "londres": (51.5074, -0.1278),
        "lisboa": (38.7223, -9.1393),
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
        url = f"http://api.geonames.org/searchJSON?q={nombre_lugar}&maxRows=1&username=demo"
        res = requests.get(url, timeout=5)
        data = res.json()
        if 'geonames' in data and len(data['geonames']) > 0:
            item = data['geonames'][0]
            return float(item['lat']), float(item['lng']), item['name']
    except Exception:
        pass
        
    return None, None, None

# =====================================================================
# PESTAÑA 1: INICIO Y MONITOREO EN TIEMPO REAL
# =====================================================================
with tab1:
    st.title("🛰️ Sistema Unificado de Monitoreo Ionosférico (TEC/TECU)")
    st.markdown("### ¿Cómo afectan el TEC y el TECU a las señales GNSS?")
    st.markdown("El **Contenido Total de Electrones (TEC)** es la cantidad integrada de electrones atrapados en la ionosfera a lo largo de la trayectoria de una señal de satélite. Se mide en unidades **TECU** (1 TECU = $10^{16}$ electrones por metro cuadrado).")
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
        return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"

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
        with st.form("form_inicio_loc"):
            localidad_usuario = st.text_input("Escribe el nombre de una ciudad o región:", "Madrid")
            boton_inicio_loc = st.form_submit_button("Consultar TECU Real")

        if boton_inicio_loc and localidad_usuario:
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
                st.error("Error de sincronización con el servidor de mapas. Inténtalo de nuevo en unos segundos.")

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
        fig.colorbar(map_eur, ax=ax1, orientation='horizontal', pad=0.07, shrink=0.7).set_label('VTEC MALLA REGIONAL (TECU)', weight='bold')

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
    fecha_sel = col_f1.date_input("Selecciona la Fecha:", datetime.date(2026, 1, 24), key="past_date")
    hora_sel = col_f2.slider("Hora (UTC):", 0, 23, 4, key="past_hour")
    minuto_sel = col_f3.slider("Minuto:", 0, 55, 0, step=5, key="past_min")

    minuto_ajustado = (minuto_sel // 15) * 15
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
                vtec_p_list = [f['properties']['vtec_assimilated_tecu'] for f in data_p['data']['grid']['features']]
                st.session_state.matriz_pasado = np.array(vtec_p_list).reshape(43, 81)
                st.session_state.fecha_mapa = f"{fecha_sel.strftime('%d/%m/%Y')} - {hora_sel:02d}:{minuto_ajustado:02d} UTC"
                st.success(f"📌 Archivo cargado correctamente.")
            except Exception:
                st.error("❌ No existen registros en el DLR para la fecha/hora solicitada.")

    if st.session_state.matriz_pasado is not None:
        st.divider()
        with st.form("formulario_consulta_pasado"):
            localidad_p_usuario = st.text_input("Ingresa cualquier localidad dentro del recuadro del mapa:", "Madrid")
            boton_consultar_ciudad = st.form_submit_button("Calcular TECU")

        lons_p, lats_p = np.arange(-30, 51, 1), np.arange(30, 73, 1)
        if boton_consultar_ciudad and localidad_p_usuario:
            lat_p, lon_p, _ = geocodificar_localidad(localidad_p_usuario)
            if lat_p is not None and (30 <= lat_p <= 72) and (-30 <= lon_p <= 50):
                interp_p = RegularGridInterpolator((lats_p, lons_p), st.session_state.matriz_pasado, method='linear', bounds_error=False, fill_value=None)
                val_tecu_p = float(interp_p(np.array([[lat_p, lon_p]]))[0])
                st.metric(label=f"Valor en {localidad_p_usuario.capitalize()}", value=f"{val_tecu_p:.3f} TECU")
            else:
                st.warning("La localidad indicada está fuera de cobertura o no fue encontrada.")

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
        plt.colorbar(mapa_p, ax=ax_p, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC ASSIMILATED (TECU)', weight='bold')
        plt.title(f"MAPA DE TEC RECONSTRUIDO\nFECHA: {st.session_state.fecha_mapa}", fontsize=11, weight='bold')
        st.pyplot(fig_p)

# =====================================================================
# PESTAÑA 3: EVOLUCIÓN TECU
# =====================================================================
with tab3:
    st.title("📈 Estudio de Evolución Temporal del TECU")
    modo_evolucion = st.radio("Selecciona el tipo de análisis temporal:", ["Por Días", "Por Horas"], horizontal=True)

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
        hora_fija_sel = col_c2.slider("Hora fija de observación (UTC):", 0, 23, 15, key="ev_hour_dias")
        num_dias_sel = col_c3.slider("Número de días a evaluar:", 2, 15, 10, key="ev_num_dias")

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
            temp_etiquetas, temp_3d = [], np.zeros((num_dias_sel, 43, 81))
            progreso = st.progress(0.0)
            exito_total = True

            for d in range(num_dias_sel):
                fecha_actual = datetime.datetime(fecha_inicial.year, fecha_inicial.month, fecha_inicial.day) + datetime.timedelta(days=d)
                link_exitoso = False
                for m in MINUTOS_CONTIGUOS_GLOBAL:
                    url_intento = generar_enlace_dlr_rango(fecha_actual.year, fecha_actual.month, fecha_actual.day, hora_fija_sel, m)
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
                progreso.progress((d + 1) / num_dias_sel)

            if exito_total:
                st.session_state.historial_vtec_3d = temp_3d
                st.session_state.etiquetas_fechas_reales = temp_etiquetas
                st.session_state.limites_globales = (max(0.0, float(np.floor(np.min(temp_3d) - 2))), float(np.ceil(np.max(temp_3d) + 2)))
                st.session_state.matriz_maximos = np.max(temp_3d, axis=0)
                st.success("📊 Procesado con éxito.")

        if st.session_state.historial_vtec_3d is not None:
            v_min, v_max = st.session_state.limites_globales
            lons_vector, lats_vector = np.arange(-30, 51, 1), np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            st.subheader("📌 Mapa Fijo de Máximos Absolutos Registrados")
            fig_max, ax_mx = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mx.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_mx.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            ax_mx.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
            ax_mx.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
            mapa_maximos = ax_mx.pcolormesh(grid_lon, grid_lat, st.session_state.matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
            fig_max.colorbar(mapa_maximos, ax=ax_mx, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO (TECU)', weight='bold')
            st.pyplot(fig_max)
            plt.close(fig_max)

            st.subheader("🎬 Reproductor de Video: Evolución Diaria (0.5s por Frame)")
            if st.button("▶️ Reproducir Video", key="play_dias"):
                contenedor_video_mapa = st.empty()
                for f in range(len(st.session_state.etiquetas_fechas_reales)):
                    fig_video, ax_ev = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_ev.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_ev.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_ev.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    ax_ev.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
                    mapa_dinamico = ax_ev.pcolormesh(grid_lon, grid_lat, st.session_state.historial_vtec_3d[f, :, :], transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
                    fig_video.colorbar(mapa_dinamico, ax=ax_ev, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    ax_ev.set_title(f"Fecha: {st.session_state.etiquetas_fechas_reales[f]} UTC", weight='bold', color='#1976d2')
                    contenedor_video_mapa.pyplot(fig_video)
                    plt.close(fig_video)
                    time.sleep(0.5)

            st.subheader("📊 3. Gráfica Comparativa de Localidades Acumuladas")
            with st.form("formulario_acumulador_ciudades"):
                nueva_ciudad = st.text_input("Ingresa cualquier localidad del mapa:", "madrid")
                boton_agregar = st.form_submit_button("➕ Añadir Localidad")

            if boton_agregar and nueva_ciudad:
                lat_c, lon_c, _ = geocodificar_localidad(nueva_ciudad)
                if lat_c is not None and (30 <= lat_c <= 72) and (-30 <= lon_c <= 50):
                    if nueva_ciudad.capitalize() not in [c['name'] for c in st.session_state.ciudades_lista]:
                        st.session_state.ciudades_lista.append({'name': nueva_ciudad.capitalize(), 'lat': lat_c, 'lon': lon_c})
                else: st.error("Fuera de rango o no encontrada.")

            if st.session_state.ciudades_lista:
                fig_lineas, ax_lineas = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.ciudades_lista:
                    idx_lat = (np.abs(lats_vector - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(lons_vector - ciudad_obj['lon'])).argmin()
                    ax_lineas.plot(range(len(st.session_state.etiquetas_fechas_reales)), st.session_state.historial_vtec_3d[:, idx_lat, idx_lon], marker='s', linewidth=2, label=ciudad_obj['name'])
                ax_lineas.grid(True, linestyle='--')
                ax_lineas.set_ylim(v_min, v_max)
                ax_lineas.set_xticks(range(len(st.session_state.etiquetas_fechas_reales)))
                ax_lineas.set_xticklabels(st.session_state.etiquetas_fechas_reales, rotation=25)
                ax_lineas.legend(loc="upper right")
                st.pyplot(fig_lineas)
                plt.close(fig_lineas)

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

        fecha_analisis_h = st.date_input("Selecciona el día a analizar:", datetime.date(2026, 1, 24), key="ev_fecha_hor")

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
            h_temp_etiquetas, h_temp_3d = [], np.zeros((24, 43, 81))
            progreso_h = st.progress(0.0)
            h_exito_total = True

            for h in range(24):
                link_exitoso = False
                for m in MINUTOS_CONTIGUOS_GLOBAL:
                    url_intento = generar_enlace_dlr_horas(fecha_analisis_h.year, fecha_analisis_h.month, fecha_analisis_h.day, h, m)
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
                progreso_h.progress((h + 1) / 24)

            if h_exito_total:
                st.session_state.h_historial_vtec_3d = h_temp_3d
                st.session_state.h_etiquetas_reales = h_temp_etiquetas
                st.session_state.h_limites_globales = (max(0.0, float(np.floor(np.min(h_temp_3d) - 2))), float(np.ceil(np.max(h_temp_3d) + 2)))
                st.session_state.h_matriz_maximos = np.max(h_temp_3d, axis=0)
                st.success("📊 Completado.")

        if st.session_state.h_historial_vtec_3d is not None:
            vh_min, vh_max = st.session_state.h_limites_globales
            lons_vector, lats_vector = np.arange(-30, 51, 1), np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            st.subheader("📌 Mapa Fijo de Máximos Absolutos del Día")
            fig_max_h, ax_mxh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mxh.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mxh.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_mxh.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_mxh.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            mapa_maximos_h = ax_mxh.pcolormesh(grid_lon, grid_lat, st.session_state.h_matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vh_min, vmax=vh_max, zorder=2)
            fig_max_h.colorbar(mapa_maximos_h, ax=ax_mxh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO HORARIO (TECU)', weight='bold')
            st.pyplot(fig_max_h)
            plt.close(fig_max_h)

            st.subheader("🎬 Reproductor Horario (0.5s por Frame)")
            if st.button("▶️ Reproducir Video Horario", key="play_horas"):
                contenedor_video_horas = st.empty()
                for f in range(24):
                    fig_vid_h, ax_evh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_evh.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    ax_evh.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    mapa_dinamico_h = ax_evh.pcolormesh(grid_lon, grid_lat, st.session_state.h_historial_vtec_3d[f, :, :], transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vh_min, vmax=vh_max, zorder=2)
                    fig_vid_h.colorbar(mapa_dinamico_h, ax=ax_evh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    contenedor_video_horas.pyplot(fig_vid_h)
                    plt.close(fig_vid_h)
                    time.sleep(0.5)

            st.subheader("📊 Gráfica Comparativa de Localidades Acumuladas (24 Horas)")
            with st.form("formulario_acumulador_ciudades_horas"):
                nueva_ciudad_h = st.text_input("Nombre de la ciudad:", "madrid")
                boton_agregar_h = st.form_submit_button("➕ Añadir Localidad")

            if boton_agregar_h and nueva_ciudad_h:
                lat_ch, lon_ch, _ = geocodificar_localidad(nueva_ciudad_h)
                if lat_ch is not None and (30 <= lat_ch <= 72) and (-30 <= lon_ch <= 50):
                    if nueva_ciudad_h.capitalize() not in [c['name'] for c in st.session_state.h_ciudades_lista]:
                        st.session_state.h_ciudades_lista.append({'name': nueva_ciudad_h.capitalize(), 'lat': lat_ch, 'lon': lon_ch})
                else: st.error("Fuera de rango o no encontrada.")

            if st.session_state.h_ciudades_lista:
                fig_lineas_h, ax_lineas_h = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.h_ciudades_lista:
                    idx_lat = (np.abs(lats_vector - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(lons_vector - ciudad_obj['lon'])).argmin()
                    ax_lineas_h.plot(range(24), st.session_state.h_historial_vtec_3d[:, idx_lat, idx_lon], marker='o', linewidth=2, label=ciudad_obj['name'])
                ax_lineas_h.grid(True, linestyle='--')
                ax_lineas_h.set_ylim(vh_min, vh_max)
                ax_lineas_h.set_xticks(range(24))
                ax_lineas_h.set_xticklabels([f"{h:02d}h" for h in range(24)], rotation=45)
                ax_lineas_h.legend(loc="upper right")
                st.pyplot(fig_lineas_h)
                plt.close(fig_lineas_h)

# =====================================================================
# PESTAÑA 4: PRONÓSTICO (NUEVO DESARROLLO VERSIÓN 5.0)
# =====================================================================
with tab4:
    st.title("🔮 Predicción Científica del VTEC Ionosférico")
    st.write("Establece una fecha base para calcular las tendencias estacionales y estimar el comportamiento futuro del TECU.")

    # Selectores de parámetros para el modelo matemático
    col_p1, col_p2, col_p3 = st.columns(3)
    fecha_base_pr = col_p1.date_input("Fecha Base del Historial:", datetime.date(2026, 1, 1), key="pr_date")
    ventana_hist_pr = col_p2.slider("Ventana de historial previo (Días):", 5, 20, 15, key="pr_hist_days")
    horizonte_pr = col_p3.radio("Horizonte de Predicción a calcular:", ["1 Hora", "3 Horas", "6 Horas"], index=2, horizontal=True)

    # Buscador obligatorio de localidad
    with st.form("form_pronostico_ciudad"):
        ciudad_pronostico = st.text_input("Ingresa la ciudad o coordenadas a estudiar:", "Madrid")
        boton_ejecutar_pr = st.form_submit_button("🚀 Calcular Pronóstico y Error Asociado")

    def generar_enlace_dlr_pronostico(anio, mes, dia, hora, minuto):
        fecha_fin = datetime.datetime(anio, mes, dia, hora, minuto, 0)
        str_anio = fecha_fin.strftime("%Y")
        str_doy = fecha_fin.strftime("%j")
        str_hora = fecha_fin.strftime("%H")
        fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
        timestamp_inicio = fecha_inicio.strftime("%Y-%m-%dT%H-%M-%S")
        timestamp_fin = fecha_fin.strftime("%Y-%m-%dT%H-%M-%S")
        base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
        return f"{base_url}/{str_anio}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"

    if boton_ejecutar_pr and ciudad_pronostico:
        lons_vector, lats_vector = np.arange(-30, 51, 1), np.arange(30, 73, 1)
        lat_pr, lon_pr, _ = geocodificar_localidad(ciudad_pronostico)

        if lat_pr is None or not (30 <= lat_pr <= 72) or not (-30 <= lon_pr <= 50):
            st.error("Ubicación fuera de los límites de la malla o inválida. Inténtalo de nuevo.")
        else:
            idx_lat = (np.abs(lats_vector - lat_pr)).argmin()
            idx_lon = (np.abs(lons_vector - lon_pr)).argmin()

            headers = {"User-Agent": "Mozilla/5.0"}
            horas_escanear = range(0, 24, 2) # Pasos bihorarios obligatorios
            cronologia_vtec = []
            fechas_eje_datetime = []
            fecha_base_dt = datetime.datetime(fecha_base_pr.year, fecha_base_pr.month, fecha_base_pr.day)

            # Fase 1: Descarga del Pasado Histórico
            with st.spinner("Descargando historial bihorario..."):
                exito_past = True
                for dia_offset in range(ventana_hist_pr):
                    fecha_actual = fecha_base_dt + datetime.timedelta(days=dia_offset)
                    for hora_actual in horas_escanear:
                        link_exitoso = False
                        for m in [0, 5, 10, 15]: # Escáner de seguridad integrado
                            url_intento = generar_enlace_dlr_pronostico(fecha_actual.year, fecha_actual.month, fecha_actual.day, hora_actual, m)
                            try:
                                response = requests.get(url_intento, headers=headers, timeout=3)
                                if response.status_code == 200:
                                    data = response.json()
                                    link_exitoso = True
                                    break
                            except Exception: pass
                        
                        if not link_exitoso:
                            st.error(f"❌ Error: Datos históricos faltantes para el día {fecha_actual.strftime('%d/%m')} {hora_actual:02d}:00.")
                            exito_past = False
                            break
                    
                    if not exito_past: break
                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    matriz_instante = np.array(vtec_values_list).reshape(43, 81)
                    cronologia_vtec.append(matriz_instante[idx_lat, idx_lon])
                    fechas_eje_datetime.append(fecha_actual + datetime.timedelta(hours=hora_actual))

            if exito_past:
                vector_vtec_serie = np.array(cronologia_vtec)
                
                # Fase 2: Descarga del Futuro Real para Validación (Ventana fija de 6 horas)
                cronologia_real_future = []
                fechas_real_future = []
                fecha_validacion_base = fecha_base_dt + datetime.timedelta(days=ventana_hist_pr)

                with st.spinner("Sincronizando datos de validación..."):
                    for hora_val in [0, 2, 4]:
                        link_exitoso = False
                        for m in [0, 5, 10, 15]:
                            url_intento = generar_enlace_dlr_pronostico(fecha_validacion_base.year, fecha_validacion_base.month, fecha_validacion_base.day, hora_val, m)
                            try:
                                response = requests.get(url_intento, headers=headers, timeout=3)
                                if response.status_code == 200:
                                    data_val = response.json()
                                    link_exitoso = True
                                    break
                            except Exception: pass
                        
                        vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data_val['data']['grid']['features']]
                        matriz_instante = np.array(vtec_values_list).reshape(43, 81)
                        cronologia_real_future.append(matriz_instante[idx_lat, idx_lon])
                        fechas_real_future.append(fecha_validacion_base + datetime.timedelta(hours=hora_val))
                
                vector_real_futuro = np.array(cronologia_real_future)

                # Fase 3: Procesamiento del Modelo Matemático
                periodo = 12 # 12 muestras al día (cada 2 horas)
                
                # Traducir selección a muestras necesarias
                if horizonte_pr == "1 Hora": puntos_prediccion = 1
                elif horizonte_pr == "3 Horas": puntos_prediccion = 2
                else: puntos_prediccion = 3

                perfil_estacional = np.zeros(periodo)
                for i in range(periodo):
                    perfil_estacional[i] = np.mean(vector_vtec_serie[i::periodo])

                ultimo_valor_real = vector_vtec_serie[-1]
                ultimo_slot_horario = (len(vector_vtec_serie) - 1) % periodo
                anomalia_inicial = ultimo_valor_real - perfil_estacional[ultimo_slot_horario]

                vector_prediccion_futura = []
                alpha = 0.85 # Coeficiente de inercia ionosférica

                for k in range(1, puntos_prediccion + 1):
                    slot_futuro = (ultimo_slot_horario + k) % periodo
                    valor_predicho = perfil_estacional[slot_futuro] + anomalia_inicial * (alpha ** k)
                    vector_prediccion_futura = np.append(vector_prediccion_futura, valor_predicho)

                # Cálculos de Ejes Verticales con Regla +-2
                valores_ventana_grafica = np.concatenate([vector_vtec_serie[-12:], vector_real_futuro[:puntos_prediccion], vector_prediccion_futura])
                Y_MIN_VAL = max(0.0, float(np.floor(np.min(valores_ventana_grafica) - 2)))
                Y_MAX_VAL = float(np.ceil(np.max(valores_ventana_grafica) + 2))

                # Fase 4: Renderizado de la Gráfica Predictiva
                st.subheader(f"📊 Gráfica de Rendimiento y Contención Ionosférica en {ciudad_pronostico.capitalize()}")
                
                fig_pr, ax_pr = plt.subplots(figsize=(14, 5.5), dpi=100)
                
                # Pasado (Últimas 24 horas del rango)
                ax_pr.plot(fechas_eje_datetime[-12:], vector_vtec_serie[-12:],
                           color='#2979ff', linewidth=2, label='Historial Real del Pasado (DLR)', marker='o', markersize=4)
                
                # Predicción matemática
                ax_pr.plot(fechas_real_future[:puntos_prediccion], vector_prediccion_futura,
                           color='#ff3d00', linewidth=2.5, linestyle='--', label=f'Predicción de Tendencia ({horizonte_pr})', marker='x', zorder=4)
                
                # Validación real
                ax_pr.plot(fechas_real_future[:puntos_prediccion], vector_real_futuro[:puntos_prediccion],
                           color='#00e676', linewidth=2.5, label='Datos Reales Obtenidos (Validación)', marker='s', markersize=5, zorder=3)

                # Sombreado de error o divergencia
                ax_pr.fill_between(fechas_real_future[:puntos_prediccion], vector_prediccion_futura, vector_real_futuro[:puntos_prediccion], 
                                   color='#ff3d00', alpha=0.1, label='Divergencia Matemática / Margen de Error')

                ax_pr.grid(True, linestyle='--', alpha=0.5)
                ax_pr.set_ylim(Y_MIN_VAL, Y_MAX_VAL)
                ax_pr.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax_pr.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
                ax_pr.set_ylabel("Intensidad VTEC (TECU)", weight='bold')
                ax_pr.set_xlabel("Línea Temporal de Control de Radio (UTC)", weight='bold')
                ax_pr.set_title(f"Evaluación Autoregresiva: Escala Vertical locked [{int(Y_MIN_VAL)}-{int(Y_MAX_VAL)} TECU]", weight='bold')
                ax_pr.legend(loc='upper left')
                
                st.pyplot(fig_pr)
                plt.close(fig_pr)

                # Fase 5: Informe Estadístico de Error Cometido
                st.divider()
                st.subheader("📊 Informe Métrico de Precisión del Modelo")
                
                # Cálculo de desviaciones
                errores_punto_a_punto = np.abs(vector_real_futuro[:puntos_prediccion] - vector_prediccion_futura)
                mae_calculado = np.mean(errores_punto_a_punto)
                porcentaje_acierto = max(0.0, 100 - (mae_calculado / np.mean(vector_real_futuro[:puntos_prediccion])) * 100)

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric(label="📉 Error Absoluto Medio (MAE)", value=f"{mae_calculado:.3f} TECU", delta=f"{mae_calculado:.2f} Error", delta_color="inverse")
                col_m2.metric(label="🎯 Fiabilidad Matemática del Modelo", value=f"{porcentaje_acierto:.1f} %")
                col_m3.info(f"**Análisis de Desviación:**\n\nEl error máximo puntual cometido en la ventana temporal fue de **{np.max(errores_punto_a_punto):.3f} TECU**.")
