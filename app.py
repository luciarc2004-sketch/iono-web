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
# INYECCIÓN DE ESTILO FORMAL (Times New Roman & Detalles Rosa)
# =====================================================================
st.markdown("""
<style>
    /* Configuración de Tipografía Formal (Serif) para toda la aplicación */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Times New Roman', Times, Georgia, serif !important;
    }
    
    /* Títulos principales y subtítulos */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Times New Roman', Times, Georgia, serif !important;
        color: #1a237e !important; /* Azul marino formal para la estructura */
    }
    
    /* Títulos de secciones específicas en Rosa Elegante */
    .rosa-titulo {
        color: #d81b60 !important;
        font-family: 'Times New Roman', Times, Georgia, serif !important;
        font-weight: bold;
    }
    
    /* Ajuste de pestañas (Tabs) */
    button[data-testid="stMarkdownContainer"] p {
        font-size: 16px !important;
    }
    
    /* Bordes de los formularios y cajas de texto con sutiles toques rosa */
    div[data-testid="stForm"] {
        border: 1px solid #f48fb1 !important;
        border-radius: 6px !important;
        background-color: #fcf8f9 !important;
    }
    
    /* Estilo de los botones */
    div.stButton > button {
        background-color: #ffffff !important;
        color: #d81b60 !important;
        border: 1px solid #d81b60 !important;
        font-family: 'Times New Roman', Times, Georgia, serif !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #d81b60 !important;
        color: #ffffff !important;
        border: 1px solid #d81b60 !important;
    }
    
    /* Líneas divisorias en rosa degradado */
    hr {
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
        border: 0 !important;
        height: 1px !important;
        background-image: linear-gradient(to right, rgba(216, 27, 96, 0), rgba(216, 27, 96, 0.75), rgba(216, 27, 96, 0)) !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# CONFIGURACIÓN GLOBAL
# =====================================================================
MINUTOS_CONTIGUOS_GLOBAL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Inicio y Monitoreo Real", 
    "Análisis en el pasado", 
    "Evolución TECU", 
    "Pronóstico", 
    "Desviaciones del Modelo"
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

if 'global_mae' not in st.session_state:
    st.session_state.global_mae = None
    st.session_state.global_prec = None
    st.session_state.global_ciudad_analizada = ""

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
    st.markdown('<h1 class="rosa-titulo">Sistema Unificado de Monitoreo Ionosférico (TEC/TECU)</h1>', unsafe_allow_html=True)
    st.markdown("### Importancia del Contenido Total de Electrones en Sistemas GNSS")
    st.markdown("El Contenido Total de Electrones (TEC) representa la densidad integrada de electrones libres a lo largo del trayecto orbital de una señal de satélite. Esta magnitud física se expresa formalmente en unidades TECU (donde 1 TECU equivale a $10^{16}$ electrones por metro cuadrado).")
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
        lons_eur, lats_eur = np.arange(-30, 51, 1), np.arange(30, 73, 1)
        lons_glb, lats_glb = np.linspace(-180, 180, 73), np.linspace(-90, 90, 73)
        
        interp_europa = RegularGridInterpolator((lats_eur, lons_eur), matriz_vtec_eur, method='linear', bounds_error=False, fill_value=None)
        interp_global = RegularGridInterpolator((lats_glb, lons_glb), matriz_vtec_glb, method='linear', bounds_error=False, fill_value=None)

        st.markdown('<h3 class="rosa-titulo">Consulta de Intensidad por Localidad</h3>', unsafe_allow_html=True)
        with st.form("form_inicio_loc"):
            localidad_usuario = st.text_input("Introduzca el nombre de la ciudad o región:", "Madrid")
            boton_inicio_loc = st.form_submit_button("Consultar Registros")

        if boton_inicio_loc and localidad_usuario:
            lat, lon, _ = geocodificar_localidad(localidad_usuario)
            if lat is not None:
                dentro_europa = (30 <= lat <= 72) and (-30 <= lon <= 50)
                punto_consulta = np.array([[lat, lon]])
                valor_tecu = float(interp_europa(punto_consulta)[0]) if dentro_europa else float(interp_global(punto_consulta)[0])
                fuente = "Malla Regional Europa" if dentro_europa else "Malla Planetaria Global"
                
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Ubicación Analizada", value=localidad_usuario.capitalize())
                col2.metric(label="Magnitud VTEC", value=f"{valor_tecu:.3f} TECU")
                col3.info(f"**Coordenadas:** {lat:.2f}°N, {lon:.2f}°E\n\n**Origen:** {fuente}")
            else: st.error("Error en la resolución geográfica de la localidad.")

        st.divider()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
        ax1.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax1.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax1.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1)
        grid_lon_eur, grid_lat_eur = np.meshgrid(lons_eur, lats_eur)
        map_eur = ax1.pcolormesh(grid_lon_eur, grid_lat_eur, matriz_vtec_eur, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud')
        fig.colorbar(map_eur, ax=ax1, orientation='horizontal', pad=0.07, shrink=0.7).set_label('VTEC MALLA REGIONAL (TECU)', weight='bold')

        ax2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax2.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax2.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.0)
        grid_lon_glb, grid_lat_glb = np.meshgrid(lons_glb, lats_glb)
        map_glb = ax2.pcolormesh(grid_lon_glb, grid_lat_glb, matriz_vtec_glb, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.8, shading='gouraud')
        fig.colorbar(map_glb, ax=ax2, orientation='horizontal', pad=0.07, shrink=0.7).set_label('VTEC GLOBAL (TECU)', weight='bold')
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e: st.error(f"Error técnico en el módulo: {e}")

# =====================================================================
# PESTAÑA 2: ANÁLISIS EN EL PASADO
# =====================================================================
with tab2:
    st.markdown('<h1 class="rosa-titulo">Análisis Histórico de Archivos Reconstruidos</h1>', unsafe_allow_html=True)
    st.write("Módulo de extracción analítica para mallas de datos consolidadas en el repositorio del DLR.")

    if 'matriz_pasado' not in st.session_state:
        st.session_state.matriz_pasado = None
    if 'fecha_mapa' not in st.session_state:
        st.session_state.fecha_mapa = ""

    col_f1, col_f2, col_f3 = st.columns(3)
    fecha_sel = col_f1.date_input("Fecha de Observación:", datetime.date(2026, 1, 24), key="past_date")
    hora_sel = col_f2.slider("Hora de Observación (UTC):", 0, 23, 4, key="past_hour")
    minuto_sel = col_f3.slider("Fracción de Minuto:", 0, 55, 0, step=5, key="past_min")

    minuto_ajustado = (minuto_sel // 15) * 15
    url_pasado = generar_enlace_dlr_seguro(fecha_sel.year, fecha_sel.month, fecha_sel.day, hora_sel, minuto_ajustado)

    if st.button("Cargar Registro Histórico"):
        with st.spinner("Sincronizando Malla Geomagnética Histórica con el DLR..."):
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                response = requests.get(url_pasado, headers=headers, timeout=12)
                response.raise_for_status() 
                vtec_p_list = [f['properties']['vtec_assimilated_tecu'] for f in response.json()['data']['grid']['features']]
                st.session_state.matriz_pasado = np.array(vtec_p_list).reshape(43, 81)
                st.session_state.fecha_mapa = f"{fecha_sel.strftime('%d/%m/%Y')} - {hora_sel:02d}:{minuto_ajustado:02d} UTC"
                st.success("Archivo sincronizado con éxito.")
            except Exception: st.error("No se localizó un registro consolidado para la fecha u hora especificada.")

    if st.session_state.matriz_pasado is not None:
        st.divider()
        with st.form("formulario_consulta_pasado"):
            localidad_p_usuario = st.text_input("Introduzca coordenada o localidad de control:", "Madrid")
            boton_consultar_ciudad = st.form_submit_button("Calcular Parámetros")

        lons_p, lats_p = np.arange(-30, 51, 1), np.arange(30, 73, 1)
        if boton_consultar_ciudad and localidad_p_usuario:
            lat_p, lon_p, _ = geocodificar_localidad(localidad_p_usuario)
            if lat_p is not None and (30 <= lat_p <= 72) and (-30 <= lon_p <= 50):
                interp_p = RegularGridInterpolator((lats_p, lons_p), st.session_state.matriz_pasado, method='linear', bounds_error=False, fill_value=None)
                val_tecu_p = float(interp_p(np.array([[lat_p, lon_p]]))[0])
                st.metric(label=f"Magnitud en {localidad_p_usuario.capitalize()}", value=f"{val_tecu_p:.3f} TECU")
            else: st.warning("La región indicada se encuentra fuera del rango instrumental cartográfico.")

        fig_p = plt.figure(figsize=(11, 6), dpi=100)
        ax_p = plt.axes(projection=ccrs.PlateCarree())
        ax_p.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
        ax_p.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax_p.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax_p.add_feature(cfeature.COASTLINE, edgecolor='#222222')
        grid_lon_p, grid_lat_p = np.meshgrid(lons_p, lats_p)
        mapa_p = ax_p.pcolormesh(grid_lon_p, grid_lat_p, st.session_state.matriz_pasado, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud')
        plt.colorbar(mapa_p, ax=ax_p, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC ASSIMILATED (TECU)', weight='bold')
        st.pyplot(fig_p)
        plt.close(fig_p)

# =====================================================================
# PESTAÑA 3: EVOLUCIÓN TECU
# =====================================================================
with tab3:
    st.markdown('<h1 class="rosa-titulo">Estudio de Evolución Temporal y Variabilidad</h1>', unsafe_allow_html=True)
    modo_evolucion = st.radio("Criterio de segmentación temporal:", ["Por Días", "Por Horas"], horizontal=True)

    if modo_evolucion == "Por Días":
        st.markdown('<h3 class="rosa-titulo">Análisis de Evolución Interdiaria</h3>', unsafe_allow_html=True)
        if 'historial_vtec_3d' not in st.session_state:
            st.session_state.historial_vtec_3d, st.session_state.etiquetas_fechas_reales = None, []
            st.session_state.limites_globales, st.session_state.matriz_maximos, st.session_state.ciudades_lista = (0, 15), None, []

        col_c1, col_c2, col_c3 = st.columns(3)
        fecha_inicial = col_c1.date_input("Fecha Inicial del Periodo:", datetime.date(2026, 2, 19), key="ev_fecha_ini")
        hora_fija_sel = col_c2.slider("Hora Fija de Sincronización (UTC):", 0, 23, 15, key="ev_hour_dias")
        num_dias_sel = col_c3.slider("Rango de Días a Evaluar:", 2, 15, 10, key="ev_num_dias")

        if st.button("Procesar Rango Temporal"):
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
                        st.error(f"Falta de continuidad en el registro para el día: {fecha_actual.strftime('%d/%m/%Y')}.")
                        exito_total = False
                        break

                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    temp_3d[d, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} ({hora_fija_sel:02d}:{minuto_exitoso:02d})")

                if exito_total:
                    st.session_state.historial_vtec_3d = temp_3d
                    st.session_state.etiquetas_fechas_reales = temp_etiquetas
                    st.session_state.limites_globales = (max(0.0, float(np.floor(np.min(temp_3d) - 2))), float(np.ceil(np.max(temp_3d) + 2)))
                    st.session_state.matriz_maximos = np.max(temp_3d, axis=0)
                    st.success("Mallas temporales procesadas correctamente.")

        if st.session_state.historial_vtec_3d is not None:
            v_min, v_max = st.session_state.limites_globales
            lons_vector, lats_vector = np.arange(-30, 51, 1), np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            st.subheader("Distribución Cartográfica de Valores Máximos")
            fig_max, ax_mx = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mx.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
            mapa_maximos = ax_mx.pcolormesh(grid_lon, grid_lat, st.session_state.matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max)
            fig_max.colorbar(mapa_maximos, ax=ax_mx, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO REGISTRADO (TECU)', weight='bold')
            st.pyplot(fig_max)
            plt.close(fig_max)

            st.subheader("Secuencia Dinámica de Fotogramas de Evolución")
            if st.button("Ejecutar Secuencia"):
                contenedor_video_mapa = st.empty()
                for f in range(len(st.session_state.etiquetas_fechas_reales)):
                    fig_video, ax_ev = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_ev.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6')
                    mapa_dinamico = ax_ev.pcolormesh(grid_lon, grid_lat, st.session_state.historial_vtec_3d[f, :, :], transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max)
                    fig_video.colorbar(mapa_dinamico, ax=ax_ev, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    contenedor_video_mapa.pyplot(fig_video)
                    plt.close(fig_video)
                    time.sleep(0.5)

            st.markdown('<h3 class="rosa-titulo">Gráfica Analítica de Estaciones de Control Acumuladas</h3>', unsafe_allow_html=True)
            with st.form("formulario_acumulador_ciudades"):
                nueva_ciudad = st.text_input("Añadir localidad a la serie comparativa:", "madrid")
                boton_agregar = st.form_submit_button("Registrar Localidad")

            if boton_agregar and nueva_ciudad:
                lat_c, lon_c, _ = geocodificar_localidad(nueva_ciudad)
                if lat_c is not None and (30 <= lat_c <= 72) and (-30 <= lon_c <= 50):
                    if nueva_ciudad.capitalize() not in [c['name'] for c in st.session_state.ciudades_lista]:
                        st.session_state.ciudades_lista.append({'name': nueva_ciudad.capitalize(), 'lat': lat_c, 'lon': lon_c})
                else: st.error("Coordenadas fuera del límite cartográfico.")

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

    elif modo_evolucion == "Por Horas":
        st.markdown('<h3 class="rosa-titulo">Análisis de Evolución Intradía (24 Horas)</h3>', unsafe_allow_html=True)
        if 'h_historial_vtec_3d' not in st.session_state:
            st.session_state.h_historial_vtec_3d, st.session_state.h_etiquetas_reales = None, []
            st.session_state.h_limites_globales, st.session_state.h_matriz_maximos, st.session_state.h_ciudades_lista = (0, 15), None, []

        fecha_analisis_h = st.date_input("Día de Estudio Horario:", datetime.date(2026, 1, 24), key="ev_fecha_hor")

        if st.button("Procesar Perfil Diario"):
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
                        st.error(f"Incompletitud de serie instrumental en la hora {h:02d}:00 UTC.")
                        h_exito_total = False
                        break

                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                    h_temp_3d[h, :, :] = np.array(vtec_values_list).reshape(43, 81)
                    h_temp_etiquetas.append(f"{h:02d}:{minuto_exitoso:02d}")

                if h_exito_total:
                    st.session_state.h_historial_vtec_3d = h_temp_3d
                    st.session_state.h_etiquetas_reales = h_temp_etiquetas
                    st.session_state.h_limites_globales = (max(0.0, float(np.floor(np.min(h_temp_3d) - 2))), float(np.ceil(np.max(h_temp_3d) + 2)))
                    st.session_state.h_matriz_maximos = np.max(h_temp_3d, axis=0)
                    st.success("Perfil horario completado.")

        if st.session_state.h_historial_vtec_3d is not None:
            vh_min, vh_max = st.session_state.h_limites_globales
            lons_vector, lats_vector = np.arange(-30, 51, 1), np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            st.subheader("Distribución Cartográfica Estática Diaria")
            fig_max_h, ax_mxh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mxh.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mxh.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            mapa_maximos_h = ax_mxh.pcolormesh(grid_lon, grid_lat, st.session_state.h_matriz_maximos, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vh_min, vmax=vh_max)
            fig_max_h.colorbar(mapa_maximos_h, ax=ax_mxh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('PICO MÁXIMO DIARIO (TECU)', weight='bold')
            st.pyplot(fig_max_h)
            plt.close(fig_max_h)

            st.subheader("Secuencia Dinámica Horaria Continua")
            if st.button("Ejecutar Video Horario"):
                contenedor_video_horas = st.empty()
                for f in range(24):
                    fig_vid_h, ax_evh = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_evh.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    ax_evh.add_feature(cfeature.LAND, facecolor='#f6f6f6')
                    mapa_dinamico_h = ax_evh.pcolormesh(grid_lon, grid_lat, st.session_state.h_historial_vtec_3d[f, :, :], transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vh_min, vmax=vh_max)
                    fig_vid_h.colorbar(mapa_dinamico_h, ax=ax_evh, orientation='horizontal', pad=0.08, shrink=0.7).set_label('VTEC (TECU)', weight='bold')
                    contenedor_video_horas.pyplot(fig_vid_h)
                    plt.close(fig_vid_h)
                    time.sleep(0.5)

            st.markdown('<h3 class="rosa-titulo">Serie Temporal de Estaciones de Control (Diario)</h3>', unsafe_allow_html=True)
            with st.form("formulario_acumulador_ciudades_horas"):
                nueva_ciudad_h = st.text_input("Añadir localidad a la serie:", "madrid")
                boton_agregar_h = st.form_submit_button("Registrar Localidad")

            if boton_agregar_h and nueva_ciudad_h:
                lat_ch, lon_ch, _ = geocodificar_localidad(nueva_ciudad_h)
                if lat_ch is not None and (30 <= lat_ch <= 72) and (-30 <= lon_ch <= 50):
                    if nueva_ciudad_h.capitalize() not in [c['name'] for c in st.session_state.h_ciudades_lista]:
                        st.session_state.h_ciudades_lista.append({'name': nueva_ciudad_h.capitalize(), 'lat': lat_ch, 'lon': lon_ch})
                else: st.error("Región fuera del área instrumental de cobertura.")

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
# PESTAÑA 4 Y PESTAÑA 5: EN ESPERA (VACÍAS EN REPOSO)
# =====================================================================
with tab4:
    st.markdown('<h1 class="rosa-titulo">Predicción Científica del VTEC Ionosférico</h1>', unsafe_allow_html=True)
    st.info("Este módulo se encuentra temporalmente vacío y en reposo para tareas de mantenimiento estructural del algoritmo. Volveremos a implementar el predictor próximamente.")

with tab5:
    st.markdown('<h1 class="rosa-titulo">Informe Analítico de Desviaciones</h1>', unsafe_allow_html=True)
    st.info("El panel estadístico de desviaciones se activará de forma sincronizada en cuanto se reanuden las tareas de desarrollo del bloque de predicción.")
