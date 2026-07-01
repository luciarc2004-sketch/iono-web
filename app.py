import datetime
import json
import requests
import time
import numpy as np
import pandas as pd
import urllib.request
import gzip
import shutil
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import RegularGridInterpolator
import streamlit as st
from skyfield.api import Topos, load, EarthSatellite  
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Configuración de la página web limpia
st.set_page_config(page_title="Portal de Monitoreo Ionosférico", layout="wide")

# =====================================================================
# CONFIGURACIÓN GLOBAL ESTRICTA (REGLA 0-55 TECU BASE Y COORDENADAS)
# =====================================================================
MINUTOS_CONTIGUOS_GLOBAL = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
VMIN_TECU_FIJO = 0.0
VMAX_TECU_FIJO = 60.0

# Definición de la mallas estricta de Europa (Versión A)
LAT_MIN, LAT_MAX, DELTA_LAT = 30, 72, 1
LON_MIN, LON_MAX, DELTA_LON = -30, 50, 1

LATS_EUROPA = np.arange(LAT_MIN, LAT_MAX + DELTA_LAT, DELTA_LAT)
LONS_EUROPA = np.arange(LON_MIN, LON_MAX + DELTA_LON, DELTA_LON)
GRID_LON_EUR, GRID_LAT_GRID = np.meshgrid(LONS_EUROPA, LATS_EUROPA)

# SE AÑADE LA PESTAÑA EXTRA 'AVIACIÓN' EN LA PENÚLTIMA POSICIÓN
tab1, tab2, tab3, tab4, tab5, tab_aviacion, tab6 = st.tabs([
    "🌍 Inicio", 
    "📊 Análisis en el pasado", 
    "📈 Evolución TECU", 
    "🔮 Pronóstico", 
    "📉 Desviaciones del Modelo",
    "🛩️ Aviación",
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

def generar_enlace_dlr_seguro(año, mes, dia, hora, minuto):
    fecha_fin = datetime.datetime(año, mes, dia, hora, minuto, 0)
    str_año = fecha_fin.strftime("%Y")
    str_doy = fecha_fin.strftime("%j")
    str_hora = fecha_fin.strftime("%H")
    fecha_inicio = fecha_fin - datetime.timedelta(minutes=4, seconds=30)
    return f"https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0/{str_año}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{fecha_inicio.strftime('%Y-%m-%dT%H-%M-%S')}_{fecha_fin.strftime('%Y-%m-%dT%H-%M-%S')}_{str_doy}_D.json"
# =====================================================================
# PESTAÑA 1: INICIO Y MONITOREO EN TIEMPO REAL (VERSIÓN AERONÁUTICA PRO)
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

    # MOTOR DE CARGA DE ALTA VELOCIDAD: Filtra en memoria y prepara vectores masivos
    @st.cache_data
 # MOTOR DE CARGA CORREGIDO PARA LA ESTRUCTURA REAL DEL ARCHIVO JSON
    @st.cache_data
    def cargar_aeropuertos_optimizado():
        try:
            with open("aeropuertos_registrados.json", "r", encoding="utf-8") as f:
                datos_dict = json.load(f)
            
            aeropuertos_europa = []
            lats_globales = []
            lons_globales = []
            diccionario_por_oaci = {} # Aquí indexaremos por OACI real
            
            # Recorremos la estructura secuencial ("0", "1", "2"...)
            for aero in datos_dict.values():
                lat_a = aero.get("lat")
                lon_a = aero.get("lon")
                
                # Extraemos el código OACI/ICAO y lo guardamos indexado
                codigo_icao = aero.get("icao")
                if codigo_icao:
                    diccionario_por_oaci[str(codigo_icao).strip().upper()] = aero
                
                if lat_a is not None and lon_a is not None:
                    lats_globales.append(lat_a)
                    lons_globales.append(lon_a)
                    
                    if (LAT_MIN <= lat_a <= LAT_MAX) and (LON_MIN <= lon_a <= LON_MAX):
                        aeropuertos_europa.append(aero)
                        
            return aeropuertos_europa, lons_globales, lats_globales, diccionario_por_oaci
        except Exception as e:
            # Si hay un error de carga, lo capturamos para que no caiga la app
            st.error(f"Error crítico leyendo el JSON de aeropuertos: {e}")
            return [], [], [], {}

    try:
        matriz_vtec_eur, matriz_vtec_glb = cargar_datos_vtec()
        lista_aero_eur, lons_glb_aero, lats_glb_aero, diccionario_completo_aero = cargar_aeropuertos_optimizado()
        
        lons_glb, lats_glb = np.linspace(-180, 180, 73), np.linspace(-90, 90, 73)
        
        # Generación de interpoladores espaciales lineales en tiempo real
        interp_europa = RegularGridInterpolator((LATS_EUROPA, LONS_EUROPA), matriz_vtec_eur, method='linear', bounds_error=False, fill_value=None)
        interp_global = RegularGridInterpolator((lats_glb, lons_glb), matriz_vtec_glb, method='linear', bounds_error=False, fill_value=None)

        # Interruptor de control para el ajuste de escala local
        ajuste_local_t1 = st.toggle("🔍 Optimizar rango de color al Máx/Mín local de este mapa", key="toggle_t1")
        
        if ajuste_local_t1:
            vmin_eur, vmax_eur = float(np.min(matriz_vtec_eur)), float(np.max(matriz_vtec_eur))
            vmin_glb, vmax_glb = float(np.min(matriz_vtec_glb)), float(np.max(matriz_vtec_glb))
            lbl_status = "Rango de Color Adaptado Localmente"
        else:
            vmin_eur, vmax_eur = VMIN_TECU_FIJO, VMAX_TECU_FIJO
            vmin_glb, vmax_glb = VMIN_TECU_FIJO, VMAX_TECU_FIJO
            lbl_status = "Escala Fija Universal (0-55 TECU)"

        # -----------------------------------------------------------------
        # 1. MAPAS PUROS DE IONOSFERA (MAPAS DEL PUNTO)
        # -----------------------------------------------------------------
        fig_tecu, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(18, 7), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Mapa Regional Europa TECU
        ax_t1.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax_t1.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax_t1.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax_t1.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1)
        map_eur = ax_t1.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, matriz_vtec_eur, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_eur, vmax=vmax_eur)
        fig_tecu.colorbar(map_eur, ax=ax_t1, orientation='horizontal', pad=0.07, shrink=0.7).set_label(f'VTEC REGIONAL (TECU) [{lbl_status}]', weight='bold')

        # Mapa Global TECU
        ax_t2.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax_t2.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax_t2.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
        ax_t2.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.0)
        grid_lon_glb, grid_lat_glb = np.meshgrid(lons_glb, lats_glb)
        map_glb = ax_t2.pcolormesh(grid_lon_glb, grid_lat_glb, matriz_vtec_glb, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.8, shading='gouraud', vmin=vmin_glb, vmax=vmax_glb)
        fig_tecu.colorbar(map_glb, ax=ax_t2, orientation='horizontal', pad=0.07, shrink=0.7).set_label(f'VTEC GLOBAL (TECU) [{lbl_status}]', weight='bold')

        st.pyplot(fig_tecu)
        plt.close(fig_tecu)
        st.divider()
        
        # -----------------------------------------------------------------
        # 2. SISTEMA DE CONSULTA POR COORDENADAS O LOCALIDAD GENERAL
        # -----------------------------------------------------------------
        st.subheader("🔍 Consulta de TECU por Localidad o Coordenadas")
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

        if lat is not None and lon is not None:
            dentro_europa = (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX)
            punto_consulta = np.array([[lat, lon]])
            valor_tecu = float(interp_europa(punto_consulta)[0]) if dentro_europa else float(interp_global(punto_consulta)[0])
            fuente = "Malla Regional Europa" if dentro_europa else "Malla Planetaria Global"
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="📍 Punto de Entrada", value=label_punto)
            col2.metric(label="📡 Valor VTEC", value=f"{valor_tecu:.3f} TECU")
            col3.info(f"**Coordenadas de Análisis:** {lat:.4f}°N, {lon:.4f}°E\n\n**Fuente del Dato:** {fuente}")

       

        # -----------------------------------------------------------------
        # 5. ENLACES DE INTERÉS Y RECURSOS (AL FINAL ABSOLUTO)
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
            st.markdown("- [ESA](https://swe.ssa.esa.int/) - ESA Space Weather Service Network.")
            st.markdown("- [Códigos Web](https://github.com/luciarc2004-sketch) - Código público.")

    except Exception as e: 
        st.error(f"Error en Tiempo Real: {e}")


# =====================================================================
# PESTAÑA 2: ANÁLISIS EN EL PASADO (HÍBRIDO DLR / IONEX)
# =====================================================================
with tab2:
    st.title("📊 Análisis Histórico Ionosférico")
    
    # --- 0. INICIALIZACIÓN DE MEMORIA (Para evitar errores de Session State) ---
    if 'matriz_pasado' not in st.session_state:
        st.session_state.matriz_pasado = None
    if 'fecha_mapa' not in st.session_state:
        st.session_state.fecha_mapa = ""
    if 'matriz_ionex' not in st.session_state:
        st.session_state.matriz_ionex = None
    if 'lats_ionex' not in st.session_state:
        st.session_state.lats_ionex = None
    if 'lons_ionex' not in st.session_state:
        st.session_state.lons_ionex = None
    if 'label_fecha_ionex' not in st.session_state:
        st.session_state.label_fecha_ionex = ""

    # =================================================================
    # --- SUB-SECCIÓN A: MOTOR REGIONAL DLR ---
    # =================================================================
    st.header("🇪🇺 Malla Regional Europa (Fuente: DLR)")
    st.markdown("Consulta el registro histórico rápido del Centro Aeroespacial Alemán (DLR). Válido para los últimos días.")

    col_f1, col_f2, col_f3 = st.columns(3)
    fecha_sel = col_f1.date_input("Selecciona la Fecha (DLR):", datetime.date.today() - datetime.timedelta(days=1), key="past_date")
    hora_sel = col_f2.slider("Hora (UTC):", 0, 23, 4, key="past_hour_dlr")
    minuto_sel = col_f3.slider("Minuto:", 0, 55, 0, step=5, key="past_min_dlr")

    minuto_ajustado = (minuto_sel // 15) * 15
    
    # Asumimos que generar_enlace_dlr_seguro ya está definida arriba en tu código base
    url_pasado = generar_enlace_dlr_seguro(fecha_sel.year, fecha_sel.month, fecha_sel.day, hora_sel, minuto_ajustado)

    if st.button("🚀 Cargar Mapa DLR", key="btn_load_dlr"):
        with st.spinner("Sincronizando Malla Geomagnética Histórica con el DLR..."):
            headers = {"User-Agent": "Mozilla/5.0"}
            try:
                response = requests.get(url_pasado, headers=headers, timeout=12)
                response.raise_for_status() 
                vtec_p_list = [f['properties']['vtec_assimilated_tecu'] for f in response.json()['data']['grid']['features']]
                st.session_state.matriz_pasado = np.array(vtec_p_list).reshape(43, 81)
                st.session_state.fecha_mapa = f"{fecha_sel.strftime('%d/%m/%Y')} - {hora_sel:02d}:{minuto_ajustado:02d} UTC"
                st.success("📌 Archivo DLR cargado correctamente.")
            except Exception: 
                st.error("❌ No existen registros en el DLR para la fecha/hora solicitada (caducan en pocos días). Usa el buscador IONEX abajo.")

    if st.session_state.matriz_pasado is not None:
        st.divider()
        ajuste_local_t2 = st.toggle("🔍 Optimizar rango de color al Máx/Mín local", key="toggle_t2")
        
        if ajuste_local_t2:
            vmin_p, vmax_p = float(np.min(st.session_state.matriz_pasado)), float(np.max(st.session_state.matriz_pasado))
            lbl_status_p = "Rango de Color Adaptado Localmente"
        else:
            vmin_p, vmax_p = 0, 55 # O tus variables VMIN_TECU_FIJO si las prefieres
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
        st.subheader("📍 Valor VTEC de un punto (Datos DLR)")
        
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
            else: 
                st.warning("Las coordenadas introducidas están fuera de la cuadrícula de Europa.")

    # =================================================================
    # --- SUB-SECCIÓN B: ANÁLISIS HISTÓRICO EXTENDIDO (IONEX) ---
    # =================================================================
    
    st.write("\n" * 2)
    st.divider()
    st.header("🌍 DATOS IONEX (Archivo Histórico Universal)")
    st.markdown("""
    Descarga automáticamente mallas ionosféricas globales en formato estándar **IONEX (.INX)** directamente desde los servidores científicos de Suiza para auditar cualquier fecha del pasado.
    **(Fuente: CODE / Universidad de Berna)**.
    """)
    
    # 1. Asegurar inicialización de variables en Session State (Evita el AttributeError)
    if 'matriz_ionex' not in st.session_state:
        st.session_state.matriz_ionex = None
    if 'lats_ionex' not in st.session_state:
        st.session_state.lats_ionex = None
    if 'lons_ionex' not in st.session_state:
        st.session_state.lons_ionex = None
    if 'label_fecha_ionex' not in st.session_state:
        st.session_state.label_fecha_ionex = ""

    # 2. Motor de Descarga y Parseo de Archivos IONEX de la Universidad de Berna
    def calcular_url_ionex(fecha_obj):
        year = fecha_obj.strftime("%Y")
        doy = fecha_obj.strftime("%j") # Día del año de 001 a 366
        # Formato moderno oficial de alta resolución del IGS
        nombre_file = f"COD0OPSFIN_{year}{doy}0000_01D_01H_GIM.INX.gz"
        return f"http://ftp.aiub.unibe.ch/CODE/{year}/{nombre_file}"

    def procesar_archivo_ionex(fecha_obj, hora_target):
        url = calcular_url_ionex(fecha_obj)
        tmp_gz = "ionex_download.INX.gz"
        tmp_txt = "ionex_extracted.inx"

        # Descarga y descompresión con librerías nativas del sistema
        urllib.request.urlretrieve(url, tmp_gz)
        with gzip.open(tmp_gz, 'rb') as f_in:
            with open(tmp_txt, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Configuración nativa de la cuadrícula estándar IONEX
        eje_lats = np.arange(87.5, -87.6, -2.5)  # 71 celdas de Norte a Sur
        eje_lons = np.arange(-180.0, 180.1, 5.0)  # 73 celdas de Oeste a Este
        grid_datos = np.zeros((len(eje_lats), len(eje_lons)))
        
        exponente = -1
        mapa_encontrado = False
        idx_fila_lat = 0
        
        try:
            with open(tmp_txt, 'r') as f:
                lineas = f.readlines()
                
            for i, linea in enumerate(lineas):
                # Detectar el exponente multiplicador matemático del archivo
                if "EXPONENT" in linea:
                    numeros = re.findall(r'-?\d+', linea)
                    if numeros: 
                        exponente = int(numeros[0])
                
                # Identificar el mapa horario correcto dentro del archivo diario
                if "EPOCH OF CURRENT MAP" in linea:
                    partes = linea.split()
                    hora_mapa = int(partes[3])
                    if hora_mapa == hora_target:
                        mapa_encontrado = True
                    else:
                        mapa_encontrado = False
                
                # Si termina el mapa de nuestra hora, dejamos de leer el archivo plano
                if "END OF TEC MAP" in linea and mapa_encontrado:
                    break 
                    
                # Extraer los bloques numéricos de la sub-malla de datos
                if mapa_encontrado and "LAT/LON1/LON2/DLON/H" in linea:
                    valores_bloque = []
                    lineas_extra = 1
                    # Cada fila latitudinal tiene exactamente 73 columnas (valores de longitud)
                    while len(valores_bloque) < 73 and i + lineas_extra < len(lineas):
                        linea_datos = lineas[i + lineas_extra].rstrip('\n')
                        # IONEX agrupa los datos en bloques rígidos de 5 caracteres
                        for j in range(0, len(linea_datos), 5):
                            fragmento = linea_datos[j:j+5].strip()
                            if fragmento:
                                valores_bloque.append(int(fragmento))
                        lineas_extra += 1
                    
                    # Guardamos la fila aplicando el factor de escala decimal (habitualmente 10^-1)
                    grid_datos[idx_fila_lat, :] = np.array(valores_bloque) * (10 ** exponente)
                    idx_fila_lat += 1

            if not mapa_encontrado and idx_fila_lat == 0:
                raise ValueError(f"No se localizó el registro de las {hora_target}:00 UTC para esa fecha.")

            return eje_lats, eje_lons, grid_datos

        finally:
            # Limpieza estricta de archivos en el disco del servidor
            if os.path.exists(tmp_gz): os.remove(tmp_gz)
            if os.path.exists(tmp_txt): os.remove(tmp_txt)

    # 3. Interfaz de Configuración del usuario
    col_inp1, col_inp2 = st.columns(2)
    fecha_query = col_inp1.date_input("Calendario Histórico (IONEX):", datetime.date(2024, 1, 24), key="date_ionex_v2")
    hora_query = col_inp2.slider("Hora de Análisis (UTC):", 0, 23, 12, key="hour_ionex_v2")

    if st.button("🌍 Descargar y Decodificar Mapa IONEX", key="btn_execute_ionex"):
        with st.spinner("Conectando con Berna (Suiza) y procesando mallas binarias globales..."):
            try:
                lats_raw, lons_raw, matriz_raw = procesar_archivo_ionex(fecha_query, hora_query)
                
                # Almacenamos los resultados de forma segura en la memoria de la sesión
                st.session_state.matriz_ionex = matriz_raw
                st.session_state.lats_ionex = lats_raw
                st.session_state.lons_ionex = lons_raw
                st.session_state.label_fecha_ionex = f"{fecha_query.strftime('%d/%m/%Y')} - {hora_query:02d}:00 UTC"
                st.success("✅ Datos IONEX cargados en memoria y listos para su explotación gráfica.")
            except Exception as error:
                st.error(f"❌ Error en el puente suizo: {error}. Comprueba que la fecha sea posterior a 2022/2023 (formato moderno) y que el servidor de Berna esté online.")

    # 4. Renderizado Gráfico de la Ionosfera Planetaria
        if st.session_state.matriz_ionex is not None:
            st.divider()
            
            # Dimensiones de lienzo estándar Matplotlib (14x7 óptimo para mapas mundiales)
            fig_global, ax_global = plt.subplots(figsize=(14, 7), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
            
            # SOLUCIÓN 1: set_global() evita forzar una caja delimitadora que rompa la geometría
            ax_global.set_global()
            
            # Trazado de mapas base vectoriales transparentes
            ax_global.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor='black', zorder=3)
            ax_global.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5, zorder=3)
            
            # REORDENACIÓN MATEMÁTICA: Python exige que el eje vertical crezca de Sur a Norte (-90 a 90).
            lats_ordenadas = st.session_state.lats_ionex[::-1]
            matriz_ordenada = st.session_state.matriz_ionex[::-1, :]
            
            # Pintar el mapa con contornos suavizados
            grafico_calor = ax_global.contourf(st.session_state.lons_ionex, lats_ordenadas, 
                                               matriz_ordenada, levels=60, cmap='jet', 
                                               transform=ccrs.PlateCarree(), zorder=2)
            
            # Barra lateral vertical estándar 
            cbar_global = plt.colorbar(grafico_calor, ax=ax_global, orientation='vertical', pad=0.02, aspect=35)
            cbar_global.set_label('Contenido Total de Electrones (TECU)', fontsize=12, weight='bold')
            
            plt.title(f"Mapa Ionosférico Planetario Global (IONEX) — {st.session_state.label_fecha_ionex}", fontsize=13, weight='bold', pad=15)
            
            # SOLUCIÓN 2 (Bypass a Shapely): Usamos marcas de ejes nativas en lugar de gridlines()
            ax_global.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
            ax_global.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            
            plt.tight_layout() # Fuerza el balanceo de márgenes
            st.pyplot(fig_global)
            plt.close(fig_global)

        # 5. Panel de Consulta Avanzada (Buscador de Puntos Globales)
        # (Asegúrate de dejar debajo de esto el código del buscador que ya tenías)

        # 5. Panel de Consulta Avanzada (Buscador de Puntos Globales)
        st.divider()
        st.subheader("📍 Extracción Core-TEC de Precisión (Filtro Global)")
        
        metodo_busqueda = st.radio(
            "Selecciona la entrada de coordenadas planetarias:", 
            ["Interpolar por Localidad / Ciudad", "Coordenadas directas globales (Lat/Lon)"], 
            horizontal=True, 
            key="radio_selector_ionex_v2"
        )
        
        lat_search, lon_search, label_search = None, None, ""
        
        if metodo_busqueda == "Interpolar por Localidad / Ciudad":
            ciudad_query = st.text_input("Ingresa cualquier metrópolis o región del mundo (ej. New York, Tokio, Sydney):", "Madrid", key="txt_search_ionex_v2")
            if ciudad_query:
                lat_search, lon_search, label_search = geocodificar_localidad(ciudad_query)
        else:
            c_i1, c_i2 = st.columns(2)
            lat_search = c_i1.number_input("Latitud Geográfica (°N / °S):", min_value=-90.0, max_value=90.0, value=40.41, step=0.01, key="num_lat_ionex_v2")
            lon_search = c_i2.number_input("Longitud Geográfica (°E / °W):", min_value=-180.0, max_value=180.0, value=-3.70, step=0.01, key="num_lon_ionex_v2")
            label_search = f"Punto Arbitrario"

        if lat_search is not None and lon_search is not None:
            # Reordenamos de nuevo la matriz en memoria local para alimentar de forma ascendente al RegularGridInterpolator
            lats_interpolador = st.session_state.lats_ionex[::-1]
            matriz_interpolador = st.session_state.matriz_ionex[::-1, :]
            
            # Inicialización del interpolador esférico bilinear global
            motor_interpolacion = RegularGridInterpolator(
                (lats_interpolador, st.session_state.lons_ionex), 
                matriz_interpolador, 
                method='linear', 
                bounds_error=False, 
                fill_value=None
            )
            
            # Ejecutar consulta matemática en la malla
            coordenada_punto = np.array([[lat_search, lon_search]])
            resultado_tecu = float(motor_interpolacion(coordenada_punto)[0])
            
            # Despliegue de métricas en la interfaz
            st.metric(label=f"Densidad de Electrones Calculada ({label_search})", value=f"{resultado_tecu:.3f} TECU")
            st.caption(f"Coordenadas de la auditoría: {lat_search:.4f}°N, {lon_search:.4f}°E (Datos IONEX: {st.session_state.label_fecha_ionex})")

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
    # BLOQUE 1: POR DÍAS (HORA FIJA - MOSAICO COMPARATIVO DUAL DLR/IONEX)
    # =====================================================================
    if modo_evolucion == "Por Días (Hora Fija)":
        st.subheader("📆 Análisis de Evolución Interdiaria (Hora Fija)")
        
        # 1. Variables de memoria dinámicas (Soportan mallas de distintos tamaños)
        if 'historial_vtec_3d' not in st.session_state:
            st.session_state.historial_vtec_3d = None
            st.session_state.etiquetas_fechas_reales = []
            st.session_state.matriz_maximos = None
            st.session_state.ciudades_lista = []
            st.session_state.fuente_activa_t3 = "DLR"
            st.session_state.eje_lats_t3 = None
            st.session_state.eje_lons_t3 = None

        # 2. Interfaz de Configuración
        fuente_datos_t3 = st.radio("📡 Fuente de Datos:", ["🇪🇺 DLR (Regional Europa)", "🌍 IONEX (Planetario Global)"], horizontal=True, key="radio_src_b1")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        fecha_inicial = col_c1.date_input("Fecha Inicial:", datetime.date(2026, 2, 19), key="ev_fecha_ini")
        hora_fija_sel = col_c2.slider("Hora fija de observación (UTC):", 0, 23, 15, key="ev_hour_dias")
        num_dias_sel = col_c3.slider("Número de días a evaluar:", 2, 15, 10, key="ev_num_dias")

        # 3. Motor Dual de Descarga y Extracción
        if st.button("🚀 Procesar Rango de Días", key="btn_ev_dias"):
            with st.spinner(f"Extrayendo Bloques Temporales desde {fuente_datos_t3}..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                temp_etiquetas = []
                exito_total = True
                
                # --- RUTA A: DLR (EUROPA) ---
                if "DLR" in fuente_datos_t3:
                    temp_3d = np.zeros((num_dias_sel, 43, 81))
                    for d in range(num_dias_sel):
                        fecha_actual = fecha_inicial + datetime.timedelta(days=d)
                        link_exitoso = False
                        
                        for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]: 
                            url_intento = generar_enlace_dlr_seguro(fecha_actual.year, fecha_actual.month, fecha_actual.day, hora_fija_sel, m)
                            try:
                                response = requests.get(url_intento, headers=headers, timeout=4)
                                if response.status_code == 200:
                                    data = response.json()
                                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                                    temp_3d[d, :, :] = np.array(vtec_values_list).reshape(43, 81)
                                    link_exitoso = True
                                    break
                            except: pass

                        if not link_exitoso:
                            st.error(f"❌ Sin datos en el DLR para el día {fecha_actual.strftime('%d/%m/%Y')}.")
                            exito_total = False
                            break
                        temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} ({hora_fija_sel:02d}:00)")
                    
                    if exito_total:
                        st.session_state.fuente_activa_t3 = "DLR"
                        st.session_state.eje_lats_t3 = LATS_EUROPA 
                        st.session_state.eje_lons_t3 = LONS_EUROPA

                # --- RUTA B: IONEX (GLOBAL) ---
                else:
                    lats_i = np.arange(87.5, -87.6, -2.5)
                    lons_i = np.arange(-180.0, 180.1, 5.0)
                    temp_3d = np.zeros((num_dias_sel, len(lats_i), len(lons_i)))
                    
                    for d in range(num_dias_sel):
                        fecha_actual = fecha_inicial + datetime.timedelta(days=d)
                        year, doy = fecha_actual.strftime("%Y"), fecha_actual.strftime("%j")
                        url_ionex = f"http://ftp.aiub.unibe.ch/CODE/{year}/COD0OPSFIN_{year}{doy}0000_01D_01H_GIM.INX.gz"
                        
                        tmp_gz, tmp_txt = "tmp_ev_d.INX.gz", "tmp_ev_d.inx"
                        try:
                            urllib.request.urlretrieve(url_ionex, tmp_gz)
                            with gzip.open(tmp_gz, 'rb') as f_in, open(tmp_txt, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                                
                            exponente = -1
                            mapa_ok = False
                            lat_idx = 0
                            
                            with open(tmp_txt, 'r') as f:
                                lineas = f.readlines()
                                
                            for i, linea in enumerate(lineas):
                                if "EXPONENT" in linea:
                                    nums = re.findall(r'-?\d+', linea)
                                    if nums: exponente = int(nums[0])
                                if "EPOCH OF CURRENT MAP" in linea:
                                    h = int(linea.split()[3])
                                    mapa_ok = (h == hora_fija_sel)
                                if "END OF TEC MAP" in linea and mapa_ok: break
                                if mapa_ok and "LAT/LON1/LON2/DLON/H" in linea:
                                    valores = []
                                    offset = 1
                                    while len(valores) < 73 and i + offset < len(lineas):
                                        lin_d = lineas[i+offset].rstrip('\n')
                                        for j in range(0, len(lin_d), 5):
                                            val = lin_d[j:j+5].strip()
                                            if val: valores.append(int(val))
                                        offset += 1
                                    temp_3d[d, (len(lats_i)-1) - lat_idx, :] = np.array(valores) * (10**exponente)
                                    lat_idx += 1
                                    
                            os.remove(tmp_gz)
                            os.remove(tmp_txt)
                            
                            if lat_idx == 0: raise ValueError("Hora no encontrada")
                            temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} ({hora_fija_sel:02d}:00)")
                            
                        except Exception as e:
                            st.error(f"❌ Error descargando IONEX del {fecha_actual.strftime('%d/%m/%Y')}: {e}")
                            exito_total = False
                            if os.path.exists(tmp_gz): os.remove(tmp_gz)
                            if os.path.exists(tmp_txt): os.remove(tmp_txt)
                            break

                    if exito_total:
                        st.session_state.fuente_activa_t3 = "IONEX"
                        st.session_state.eje_lats_t3 = lats_i[::-1] 
                        st.session_state.eje_lons_t3 = lons_i

                # --- GUARDADO EN MEMORIA ---
                if exito_total:
                    st.session_state.historial_vtec_3d = temp_3d
                    st.session_state.etiquetas_fechas_reales = temp_etiquetas
                    st.session_state.matriz_maximos = np.max(temp_3d, axis=0)
                    st.session_state.ciudades_lista = [] 
                    st.success(f"📊 Rango temporal procesado con éxito ({st.session_state.fuente_activa_t3}).")

        # 4. Bloque de Visualización Dinámica (Mosaico Completo)
        if st.session_state.historial_vtec_3d is not None:
            es_ionex = st.session_state.fuente_activa_t3 == "IONEX"
            fecha_final_calc = fecha_inicial + datetime.timedelta(days=num_dias_sel - 1)
            
            ajuste_local_t3_dias = st.toggle("🔍 Optimizar rango de color al Máx/Mín de este bloque de días", key="toggle_t3_dias")
            if ajuste_local_t3_dias:
                vmin_d = max(0.0, float(np.floor(np.min(st.session_state.historial_vtec_3d))))
                vmax_d = float(np.ceil(np.max(st.session_state.historial_vtec_3d)))
            else:
                vmin_d = 0.0
                vmax_d = 120.0 if es_ionex else 60.0 
            
            # --- NUEVA SECCIÓN: MOSAICO DE EVOLUCIÓN INTERDIARIA ---
            st.subheader(" Mosaico de Evolución Temporal (Malla Comparativa)")
            
            num_plots = len(st.session_state.etiquetas_fechas_reales)
            # Definimos dinámicamente las columnas para que la rejilla sea simétrica y estética
            if num_plots <= 4: ncols = num_plots
            elif num_plots <= 8: ncols = 4
            else: ncols = 5
            
            nrows = int(np.ceil(num_plots / ncols))
            
            # Dimensiones proporcionales según la fuente (IONEX requiere lienzos más anchos por ser cilíndrico)
            w_sub = 4.5 if es_ionex else 3.8
            h_sub = 2.8 if es_ionex else 2.5
            
            fig_mosaic, axes = plt.subplots(nrows, ncols, figsize=(w_sub * ncols, h_sub * nrows), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
            axes = np.array([axes]).flatten() if num_plots == 1 else np.array(axes).flatten()
            
            mesh_lon, mesh_lat = np.meshgrid(st.session_state.eje_lons_t3, st.session_state.eje_lats_t3)
            ultimo_mapeo = None
            
            for f in range(num_plots):
                ax_sub = axes[f]
                if es_ionex:
                    ax_sub.set_global()
                    ax_sub.set_xticks([-180, 0, 180], crs=ccrs.PlateCarree())
                    ax_sub.set_yticks([-90, 0, 90], crs=ccrs.PlateCarree())
                else:
                    ax_sub.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_sub.set_xticks([float(LON_MIN), (float(LON_MIN)+float(LON_MAX))/2, float(LON_MAX)], crs=ccrs.PlateCarree())
                    ax_sub.set_yticks([float(LAT_MIN), float(LAT_MAX)], crs=ccrs.PlateCarree())
                
                ax_sub.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                ax_sub.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                ax_sub.add_feature(cfeature.COASTLINE, edgecolor='#333333', linewidth=0.7, zorder=3)
                ax_sub.grid(True, color='gray', alpha=0.2, linestyle='--')
                ax_sub.tick_params(labelsize=7)
                
                ultimo_mapeo = ax_sub.pcolormesh(mesh_lon, mesh_lat, st.session_state.historial_vtec_3d[f, :, :], 
                                                 transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_d, vmax=vmax_d, zorder=2)
                
                # CORRECCIÓN DE TÍTULO: Cada casilla lleva su día y hora exacta
                ax_sub.set_title(f" {st.session_state.etiquetas_fechas_reales[f]} UTC", fontsize=9, weight='bold')
            
            # Limpiamos los cuadrantes sobrantes de la malla si el número de días no es múltiplo exacto
            for j in range(num_plots, len(axes)):
                fig_mosaic.delaxes(axes[j])
            
            # Agregamos una única barra de colores unificada debajo de la malla para todo el mosaico
            fig_mosaic.colorbar(ultimo_mapeo, ax=axes.tolist(), orientation='horizontal', shrink=0.5, pad=0.06).set_label('Escala de Densidad VTEC (TECU)', weight='bold', fontsize=10)
            
            st.pyplot(fig_mosaic)
            plt.close(fig_mosaic)

            # --- MAPA DE MÁXIMOS ABSOLUTOS ---
            # CORRECCIÓN DE TÍTULO: Rango completo y hora explícita
            st.subheader("📌 Mapa Fijo de Máximos Absolutos Registrados")
            fig_max, ax_mx = plt.subplots(figsize=(14, 7) if es_ionex else (10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            
            if es_ionex:
                ax_mx.set_global()
                ax_mx.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
                ax_mx.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            else:
                ax_mx.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                ax_mx.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40, 50], crs=ccrs.PlateCarree())
                ax_mx.set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
            
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
            ax_mx.add_feature(cfeature.COASTLINE, edgecolor='#222222')
            ax_mx.grid(True, color='gray', alpha=0.25, linestyle='--')
            
            mapa_maximos = ax_mx.pcolormesh(mesh_lon, mesh_lat, st.session_state.matriz_maximos, 
                                            transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_d, vmax=vmax_d)
            fig_max.colorbar(mapa_maximos, ax=ax_mx, orientation='vertical', pad=0.02, aspect=35).set_label('PICO MÁXIMO (TECU)', weight='bold')
            
            # Título auditado sin pérdidas de información
            plt.title(f"Picos Máximos Absolutos — Rango: {fecha_inicial.strftime('%d/%m/%Y')} al {fecha_final_calc.strftime('%d/%m/%Y')} ({hora_fija_sel:02d}:00 UTC)", fontsize=11, weight='bold', pad=12)
            
            plt.tight_layout()
            st.pyplot(fig_max)
            plt.close(fig_max)

            # --- GRÁFICAS POR CIUDAD ---
            st.subheader("📊 Gráfica Comparativa de Localidades Acumuladas")
            tipo_busqueda_t3d = st.radio("Formato de inserción de localidad:", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3d")
            lat_c, lon_c, name_c = None, None, ""
            
            if tipo_busqueda_t3d == "Por Nombre de Ciudad":
                nueva_ciudad = st.text_input("Ingresa cualquier localidad del mapa:", "Madrid", key="txt_t3d")
                if nueva_ciudad: lat_c, lon_c, name_c = geocodificar_localidad(nueva_ciudad)
            else:
                col_lc1, col_lc2 = st.columns(2)
                lat_c_man = col_lc1.number_input("Latitud punto:", min_value=-90.0, max_value=90.0, value=40.41, step=0.1, key="num_lat_t3d")
                lon_c_man = col_lc2.number_input("Longitud punto:", min_value=-180.0, max_value=180.0, value=-3.70, step=0.1, key="num_lon_t3d")
                lat_c, lon_c, name_c = lat_c_man, lon_c_man, f"Punto ({lat_c_man:.1f}, {lon_c_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico", key="btn_t3d"):
                lim_lat_min = -90 if es_ionex else LAT_MIN
                lim_lat_max = 90 if es_ionex else LAT_MAX
                lim_lon_min = -180 if es_ionex else LON_MIN
                lim_lon_max = 180 if es_ionex else LON_MAX
                
                if lat_c is not None and lon_c is not None and (lim_lat_min <= lat_c <= lim_lat_max) and (lim_lon_min <= lon_c <= lim_lon_max):
                    if name_c not in [c['name'] for c in st.session_state.ciudades_lista]: 
                        st.session_state.ciudades_lista.append({'name': name_c, 'lat': lat_c, 'lon': lon_c})
                else:
                    st.warning("Esa localidad queda fuera del mapa actualmente seleccionado.")
                    
            if st.session_state.ciudades_lista:
                fig_lineas, ax_lineas = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.ciudades_lista:
                    idx_lat = (np.abs(st.session_state.eje_lats_t3 - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(st.session_state.eje_lons_t3 - ciudad_obj['lon'])).argmin()
                    ax_lineas.plot(range(len(st.session_state.etiquetas_fechas_reales)), 
                                   st.session_state.historial_vtec_3d[:, idx_lat, idx_lon], marker='s', linewidth=2, label=ciudad_obj['name'])
                ax_lineas.grid(True, linestyle='--')
                ax_lineas.set_ylim(vmin_d, vmax_d)
                ax_lineas.set_xticks(range(len(st.session_state.etiquetas_fechas_reales)))
                ax_lineas.set_xticklabels(st.session_state.etiquetas_fechas_reales, rotation=25)
                
                # CORRECCIÓN DE TÍTULO: Identificando rango y hora fija del análisis de líneas
                ax_lineas.set_title(f"Evolución de Intensidad TECU a las {hora_fija_sel:02d}:00 UTC (Periodo: {fecha_inicial.strftime('%d/%m')} al {fecha_final_calc.strftime('%d/%m')})", fontsize=11, weight='bold', pad=10)
                ax_lineas.legend(loc="upper right")
                
                st.pyplot(fig_lineas)
                plt.close(fig_lineas)
# =====================================================================
    # BLOQUE 2: POR HORAS (24H ÚNICO DÍA - MOSAICO DUAL DLR/IONEX)
    # =====================================================================
    elif modo_evolucion == "Por Horas (24h Único Día)":
        st.subheader("⏱️ Análisis de Evolución Intradía (Hora por Hora - 24h)")
        
        # 1. Variables de memoria dinámicas
        if 'h_historial_vtec_3d' not in st.session_state:
            st.session_state.h_historial_vtec_3d = None
            st.session_state.h_etiquetas_reales = []
            st.session_state.h_matriz_maximos = None
            st.session_state.h_ciudades_lista = []
            st.session_state.h_fuente_activa = "DLR"
            st.session_state.h_eje_lats = None
            st.session_state.h_eje_lons = None

        # 2. Interfaz de Configuración
        fuente_datos_t3h = st.radio("📡 Fuente de Datos para el escaneo 24h:", ["🇪🇺 DLR (Regional Europa)", "🌍 IONEX (Planetario Global)"], horizontal=True, key="radio_fuente_24h")
        
        fecha_analisis_h = st.date_input("Selecciona el día histórico a analizar:", datetime.date(2026, 1, 24), key="ev_fecha_hor")

        # 3. Motor Dual de Extracción de Datos
        if st.button("🚀 Procesar las 24 Horas", key="btn_ev_horas"):
            with st.spinner(f"Escaneando Ciclos Diurnos (24 Frames) desde {fuente_datos_t3h}..."):
                h_temp_etiquetas = [f"{h:02d}:00" for h in range(24)]
                h_exito_total = True
                
                # --- RUTA A: DLR (EUROPA) ---
                if "DLR" in fuente_datos_t3h:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    h_temp_3d = np.zeros((24, 43, 81))
                    
                    for h in range(24):
                        link_exitoso = False
                        for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                            url_intento = generar_enlace_dlr_seguro(fecha_analisis_h.year, fecha_analisis_h.month, fecha_analisis_h.day, h, m)
                            try:
                                response = requests.get(url_intento, headers=headers, timeout=4)
                                if response.status_code == 200:
                                    data = response.json()
                                    vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                                    h_temp_3d[h, :, :] = np.array(vtec_values_list).reshape(43, 81)
                                    link_exitoso = True
                                    break
                            except: pass

                        if not link_exitoso:
                            st.error(f"❌ Error descargando del DLR la hora {h:02d}:00. Abortado.")
                            h_exito_total = False
                            break
                            
                    if h_exito_total:
                        st.session_state.h_fuente_activa = "DLR"
                        st.session_state.h_eje_lats = LATS_EUROPA 
                        st.session_state.h_eje_lons = LONS_EUROPA

                # --- RUTA B: IONEX (GLOBAL) ---
                else:
                    lats_i = np.arange(87.5, -87.6, -2.5)
                    lons_i = np.arange(-180.0, 180.1, 5.0)
                    h_temp_3d = np.zeros((24, len(lats_i), len(lons_i)))
                    
                    year = fecha_analisis_h.strftime("%Y")
                    doy = fecha_analisis_h.strftime("%j")
                    url_ionex = f"http://ftp.aiub.unibe.ch/CODE/{year}/COD0OPSFIN_{year}{doy}0000_01D_01H_GIM.INX.gz"
                    
                    tmp_gz, tmp_txt = "tmp_24h.INX.gz", "tmp_24h.inx"
                    try:
                        urllib.request.urlretrieve(url_ionex, tmp_gz)
                        with gzip.open(tmp_gz, 'rb') as f_in, open(tmp_txt, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                            
                        exponente = -1
                        hora_actual = None
                        lat_idx = 0
                        leyendo_tec = False
                        
                        with open(tmp_txt, 'r') as f:
                            lineas = f.readlines()
                            
                        for i, linea in enumerate(lineas):
                            if "EXPONENT" in linea:
                                nums = re.findall(r'-?\d+', linea)
                                if nums: exponente = int(nums[0])
                            
                            if "START OF TEC MAP" in linea: leyendo_tec = True
                            elif "END OF TEC MAP" in linea: leyendo_tec = False
                                
                            if "EPOCH OF CURRENT MAP" in linea and leyendo_tec:
                                hora_actual = int(linea.split()[3])
                                lat_idx = 0 
                                
                            if leyendo_tec and hora_actual is not None and hora_actual < 24 and "LAT/LON1/LON2/DLON/H" in linea:
                                valores = []
                                offset = 1
                                while len(valores) < 73 and i + offset < len(lineas):
                                    lin_d = lineas[i+offset].rstrip('\n')
                                    for j in range(0, len(lin_d), 5):
                                        val = lin_d[j:j+5].strip()
                                        if val: valores.append(int(val))
                                    offset += 1
                                h_temp_3d[hora_actual, (len(lats_i)-1) - lat_idx, :] = np.array(valores) * (10**exponente)
                                lat_idx += 1
                                
                        os.remove(tmp_gz)
                        os.remove(tmp_txt)
                        
                        if np.sum(h_temp_3d[0]) == 0: raise ValueError("Archivo IONEX vacío o formato corrupto.")
                            
                    except Exception as e:
                        st.error(f"❌ Error extrayendo el día completo IONEX: {e}")
                        h_exito_total = False
                        if os.path.exists(tmp_gz): os.remove(tmp_gz)
                        if os.path.exists(tmp_txt): os.remove(tmp_txt)

                    if h_exito_total:
                        st.session_state.h_fuente_activa = "IONEX"
                        st.session_state.h_eje_lats = lats_i[::-1] 
                        st.session_state.h_eje_lons = lons_i

                # --- GUARDADO EN MEMORIA ---
                if h_exito_total:
                    st.session_state.h_historial_vtec_3d = h_temp_3d
                    st.session_state.h_etiquetas_reales = h_temp_etiquetas
                    st.session_state.h_matriz_maximos = np.max(h_temp_3d, axis=0)
                    st.session_state.h_ciudades_lista = [] 
                    st.success(f"📊 ¡Éxito! 24 mapas de {st.session_state.h_fuente_activa} cargados en memoria.")

        # 4. Bloque de Visualización Dinámica (Mosaico)
        if st.session_state.h_historial_vtec_3d is not None:
            es_ionex_h = st.session_state.h_fuente_activa == "IONEX"
            
            ajuste_local_t3_horas = st.toggle("🔍 Optimizar rango de color al Máx/Mín real de estas 24 horas", key="toggle_t3_horas")
            
            # --- CORRECCIÓN DE SATURACIÓN ---
            if ajuste_local_t3_horas:
                vmin_h = max(0.0, float(np.floor(np.min(st.session_state.h_historial_vtec_3d))))
                vmax_h = float(np.ceil(np.max(st.session_state.h_historial_vtec_3d)))
            else:
                vmin_h = 0.0
                vmax_h = 120.0 if es_ionex_h else 60.0 

            # --- NUEVA SECCIÓN: MOSAICO DE EVOLUCIÓN INTRADÍA (24h) ---
            st.subheader("🧩 Mosaico de Evolución Temporal (Ciclo de 24 Horas)")
            
            ncols = 6
            nrows = 4
            
            # Dimensiones ajustadas: más anchas para mapamundi, más cuadradas para Europa
            w_sub = 3.5 if es_ionex_h else 2.8
            h_sub = 2.0 if es_ionex_h else 2.5
            
            fig_mosaic_h, axes_h = plt.subplots(nrows, ncols, figsize=(w_sub * ncols, h_sub * nrows), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
            axes_h = axes_h.flatten()
            
            mesh_lon, mesh_lat = np.meshgrid(st.session_state.h_eje_lons, st.session_state.h_eje_lats)
            ultimo_mapeo_h = None
            
            for f in range(24):
                ax_sub = axes_h[f]
                
                if es_ionex_h:
                    ax_sub.set_global()
                    # Marcas seguras y simples para evitar el error de Shapely
                    ax_sub.set_xticks([-180, 0, 180], crs=ccrs.PlateCarree())
                    ax_sub.set_yticks([-90, 0, 90], crs=ccrs.PlateCarree())
                else:
                    ax_sub.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                    ax_sub.set_xticks([float(LON_MIN), (float(LON_MIN)+float(LON_MAX))/2, float(LON_MAX)], crs=ccrs.PlateCarree())
                    ax_sub.set_yticks([float(LAT_MIN), float(LAT_MAX)], crs=ccrs.PlateCarree())
                
                ax_sub.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                ax_sub.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                ax_sub.add_feature(cfeature.COASTLINE, edgecolor='#333333', linewidth=0.7, zorder=3)
                ax_sub.grid(True, color='gray', alpha=0.2, linestyle='--')
                ax_sub.tick_params(labelsize=6)
                
                ultimo_mapeo_h = ax_sub.pcolormesh(mesh_lon, mesh_lat, st.session_state.h_historial_vtec_3d[f, :, :], 
                                                   transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_h, vmax=vmax_h, zorder=2)
                
                ax_sub.set_title(f" {f:02d}:00 UTC", fontsize=9, weight='bold')
            
            # Barra de colores unificada debajo del mosaico completo
            fig_mosaic_h.colorbar(ultimo_mapeo_h, ax=axes_h.tolist(), orientation='horizontal', shrink=0.5, pad=0.04).set_label('Escala de Densidad VTEC (TECU)', weight='bold', fontsize=10)
            
            st.pyplot(fig_mosaic_h)
            plt.close(fig_mosaic_h)

            # --- MAPA DE MÁXIMOS ABSOLUTOS ---
            st.subheader("📌 Mapa Fijo de Máximos Absolutos del Día")
            fig_max_h, ax_mxh = plt.subplots(figsize=(14, 7) if es_ionex_h else (10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            
            if es_ionex_h:
                ax_mxh.set_global()
                ax_mxh.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
                ax_mxh.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            else:
                ax_mxh.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                ax_mxh.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40, 50], crs=ccrs.PlateCarree())
                ax_mxh.set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
            
            ax_mxh.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            ax_mxh.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
            ax_mxh.add_feature(cfeature.COASTLINE, edgecolor='#222222')
            ax_mxh.grid(True, color='gray', alpha=0.3, linestyle='--')
            
            mapa_maximos_h = ax_mxh.pcolormesh(mesh_lon, mesh_lat, st.session_state.h_matriz_maximos, 
                                               transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_h, vmax=vmax_h)
            
            fig_max_h.colorbar(mapa_maximos_h, ax=ax_mxh, orientation='horizontal' if not es_ionex_h else 'vertical', 
                               pad=0.08 if not es_ionex_h else 0.02, shrink=0.7 if not es_ionex_h else 1.0, aspect=40).set_label('PICO MÁXIMO HORARIO (TECU)', weight='bold')
            
            # Título dinámico
            plt.title(f"Picos Máximos Absolutos — {fecha_analisis_h.strftime('%d/%m/%Y')} (Ciclo 24h Completo)", fontsize=11, weight='bold', pad=12)
            
            if es_ionex_h: plt.tight_layout()
            st.pyplot(fig_max_h)
            plt.close(fig_max_h)

            # --- GRÁFICAS POR CIUDAD ---
            st.subheader("📊 Gráfica Comparativa de Localidades Acumuladas (24 Horas)")
            tipo_busqueda_t3h = st.radio("Formato de inserción de localidad (24h):", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3h")
            lat_ch, lon_ch, name_ch = None, None, ""
            
            if tipo_busqueda_t3h == "Por Nombre de Ciudad":
                nueva_ciudad_h = st.text_input("Nombre de la ciudad:", "Madrid", key="txt_t3h")
                if nueva_ciudad_h: lat_ch, lon_ch, name_ch = geocodificar_localidad(nueva_ciudad_h)
            else:
                col_lch1, col_lch2 = st.columns(2)
                lat_ch_man = col_lch1.number_input("Latitud nodo:", min_value=-90.0, max_value=90.0, value=40.41, step=0.1, key="num_lat_t3h")
                lon_ch_man = col_lch2.number_input("Longitud nodo:", min_value=-180.0, max_value=180.0, value=-3.70, step=0.1, key="num_lon_t3h")
                lat_ch, lon_ch, name_ch = lat_ch_man, lon_ch_man, f"Punto ({lat_ch_man:.1f}, {lon_ch_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico Horario", key="btn_t3h"):
                lim_lat_min = -90 if es_ionex_h else LAT_MIN
                lim_lat_max = 90 if es_ionex_h else LAT_MAX
                lim_lon_min = -180 if es_ionex_h else LON_MIN
                lim_lon_max = 180 if es_ionex_h else LON_MAX
                
                if lat_ch is not None and lon_ch is not None and (lim_lat_min <= lat_ch <= lim_lat_max) and (lim_lon_min <= lon_ch <= lim_lon_max):
                    if name_ch not in [c['name'] for c in st.session_state.h_ciudades_lista]: 
                        st.session_state.h_ciudades_lista.append({'name': name_ch, 'lat': lat_ch, 'lon': lon_ch})
                else:
                    st.warning("Esa localidad queda fuera del mapa actualmente seleccionado.")
                    
            if st.session_state.h_ciudades_lista:
                fig_lineas_h, ax_lineas_h = plt.subplots(figsize=(12, 5))
                for ciudad_obj in st.session_state.h_ciudades_lista:
                    idx_lat = (np.abs(st.session_state.h_eje_lats - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(st.session_state.h_eje_lons - ciudad_obj['lon'])).argmin()
                    ax_lineas_h.plot(range(24), st.session_state.h_historial_vtec_3d[:, idx_lat, idx_lon], marker='o', linewidth=2, label=ciudad_obj['name'])
                    
                ax_lineas_h.grid(True, linestyle='--')
                ax_lineas_h.set_ylim(vmin_h, vmax_h)
                ax_lineas_h.set_xlim(-0.5, 23.5)
                ax_lineas_h.set_xticks(range(24))
                ax_lineas_h.set_xticklabels([f"{h:02d}h" for h in range(24)], rotation=45)
                
                # Título corregido
                ax_lineas_h.set_title(f"Evolución de Intensidad TECU Intradía (24h) - {fecha_analisis_h.strftime('%d/%m/%Y')}", fontsize=11, weight='bold', pad=10)
                ax_lineas_h.legend(loc="upper right")
                
                st.pyplot(fig_lineas_h)
                plt.close(fig_lineas_h)
# =====================================================================
    # BLOQUE 3: DÍAS COMPLETOS (MOSAICO CONTINUO DUAL DLR/IONEX)
    # =====================================================================
    elif modo_evolucion == "Días Completos (Rango Continuo)":
        st.subheader("📆 Análisis de Evolución Temporal Continua (24h x N Días)")
        
        # 1. Variables dinámicas de memoria
        if 'dc_historial_vtec_3d' not in st.session_state:
            st.session_state.dc_historial_vtec_3d = None
            st.session_state.dc_etiquetas_reales = []
            st.session_state.dc_matriz_maximos = None
            st.session_state.dc_ciudades_lista = []
            st.session_state.dc_fuente_activa = "DLR"
            st.session_state.dc_eje_lats = None
            st.session_state.dc_eje_lons = None

        # 2. Interfaz de Configuración
        fuente_datos_dc = st.radio("📡 Fuente de Datos Continua:", ["🇪🇺 DLR (Regional Europa)", "🌍 IONEX (Planetario Global)"], horizontal=True, key="radio_fuente_dc")
        
        col_dc1, col_dc2 = st.columns(2)
        dc_fecha_inicial = col_dc1.date_input("Fecha Inicial del rango:", datetime.date(2026, 1, 20), key="dc_fecha_ini")
        dc_num_dias = col_dc2.slider("Número de días completos a encadenar:", 2, 7, 3, key="dc_num_dias_slider")

        total_horas_rango = dc_num_dias * 24

        # 3. Motor Dual de Extracción Masiva
        if st.button("🚀 Procesar Serie Temporal Continua", key="btn_ev_dc"):
            with st.spinner(f"Descargando y encadenando {total_horas_rango} horas consecutivas desde {fuente_datos_dc}..."):
                dc_temp_etiquetas = []
                dc_exito_total = True
                contador_hora_global = 0

                # --- RUTA A: DLR (EUROPA) ---
                if "DLR" in fuente_datos_dc:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    dc_temp_3d = np.zeros((total_horas_rango, 43, 81))
                    
                    for d in range(dc_num_dias):
                        fecha_actual = dc_fecha_inicial + datetime.timedelta(days=d)
                        for h in range(24):
                            link_exitoso = False
                            for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                                url_intento = generar_enlace_dlr_seguro(fecha_actual.year, fecha_actual.month, fecha_actual.day, h, m)
                                try:
                                    response = requests.get(url_intento, headers=headers, timeout=4)
                                    if response.status_code == 200:
                                        data = response.json()
                                        link_exitoso = True
                                        break
                                except Exception: pass

                            if not link_exitoso:
                                st.error(f"❌ Corte en los datos. Sin registros en DLR el {fecha_actual.strftime('%d/%m')} a las {h:02d}:00 UTC.")
                                dc_exito_total = False
                                break

                            vtec_values_list = [f['properties']['vtec_assimilated_tecu'] for f in data['data']['grid']['features']]
                            dc_temp_3d[contador_hora_global, :, :] = np.array(vtec_values_list).reshape(43, 81)
                            dc_temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} - {h:02d}h")
                            contador_hora_global += 1
                        if not dc_exito_total: break
                        
                    if dc_exito_total:
                        st.session_state.dc_fuente_activa = "DLR"
                        st.session_state.dc_eje_lats = LATS_EUROPA 
                        st.session_state.dc_eje_lons = LONS_EUROPA

                # --- RUTA B: IONEX (GLOBAL) ---
                else:
                    lats_i = np.arange(87.5, -87.6, -2.5)
                    lons_i = np.arange(-180.0, 180.1, 5.0)
                    dc_temp_3d = np.zeros((total_horas_rango, len(lats_i), len(lons_i)))
                    
                    for d in range(dc_num_dias):
                        fecha_actual = dc_fecha_inicial + datetime.timedelta(days=d)
                        year, doy = fecha_actual.strftime("%Y"), fecha_actual.strftime("%j")
                        url_ionex = f"http://ftp.aiub.unibe.ch/CODE/{year}/COD0OPSFIN_{year}{doy}0000_01D_01H_GIM.INX.gz"
                        
                        tmp_gz, tmp_txt = "tmp_dc.INX.gz", "tmp_dc.inx"
                        try:
                            urllib.request.urlretrieve(url_ionex, tmp_gz)
                            with gzip.open(tmp_gz, 'rb') as f_in, open(tmp_txt, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                                
                            exponente = -1
                            hora_actual = None
                            leyendo_tec = False
                            mapas_procesados = 0
                            
                            with open(tmp_txt, 'r') as f:
                                lineas = f.readlines()
                                
                            for i, linea in enumerate(lineas):
                                if "EXPONENT" in linea:
                                    nums = re.findall(r'-?\d+', linea)
                                    if nums: exponente = int(nums[0])
                                    
                                if "START OF TEC MAP" in linea: leyendo_tec = True
                                elif "END OF TEC MAP" in linea: leyendo_tec = False
                                    
                                if "EPOCH OF CURRENT MAP" in linea and leyendo_tec:
                                    hora_actual = int(linea.split()[3])
                                    lat_idx = 0 
                                    
                                if leyendo_tec and hora_actual is not None and hora_actual < 24 and "LAT/LON1/LON2/DLON/H" in linea:
                                    valores = []
                                    offset = 1
                                    while len(valores) < 73 and i + offset < len(lineas):
                                        lin_d = lineas[i+offset].rstrip('\n')
                                        for j in range(0, len(lin_d), 5):
                                            val = lin_d[j:j+5].strip()
                                            if val: valores.append(int(val))
                                        offset += 1
                                    
                                    # Insertamos en el índice global continuo
                                    idx_global = (d * 24) + hora_actual
                                    dc_temp_3d[idx_global, (len(lats_i)-1) - lat_idx, :] = np.array(valores) * (10**exponente)
                                    lat_idx += 1
                                    
                                    # Si terminamos de leer un mapa completo (todas las latitudes)
                                    if lat_idx == len(lats_i):
                                        dc_temp_etiquetas.append(f"{fecha_actual.strftime('%d/%m')} - {hora_actual:02d}h")
                                        contador_hora_global += 1
                                        mapas_procesados += 1
                                        
                            os.remove(tmp_gz)
                            os.remove(tmp_txt)
                            
                        except Exception as e:
                            st.error(f"❌ Error descargando IONEX del {fecha_actual.strftime('%d/%m/%Y')}: {e}")
                            dc_exito_total = False
                            if os.path.exists(tmp_gz): os.remove(tmp_gz)
                            if os.path.exists(tmp_txt): os.remove(tmp_txt)
                            break
                            
                    if dc_exito_total:
                        st.session_state.dc_fuente_activa = "IONEX"
                        st.session_state.dc_eje_lats = lats_i[::-1] 
                        st.session_state.dc_eje_lons = lons_i

                # --- GUARDADO EN MEMORIA ---
                if dc_exito_total:
                    # Filtramos por si hubiese etiquetas duplicadas o desordenadas en IONEX
                    if len(dc_temp_etiquetas) > total_horas_rango: dc_temp_etiquetas = dc_temp_etiquetas[:total_horas_rango]
                    
                    st.session_state.dc_historial_vtec_3d = dc_temp_3d
                    st.session_state.dc_etiquetas_reales = dc_temp_etiquetas
                    st.session_state.dc_matriz_maximos = np.max(dc_temp_3d, axis=0)
                    st.session_state.dc_ciudades_lista = []
                    st.success(f"📊 Línea temporal unificada de {total_horas_rango} mapas completada ({st.session_state.dc_fuente_activa}).")

        # 4. Bloque de Visualización Dinámica (Mosaico Masivo)
        if st.session_state.dc_historial_vtec_3d is not None:
            es_ionex_dc = st.session_state.dc_fuente_activa == "IONEX"
            
            st.divider()
            col_cfg1, col_cfg2 = st.columns(2)
            ajuste_local_dc = col_cfg1.toggle("🔍 Optimizar rango vertical al Máx/Mín local de esta serie masiva", key="toggle_dc_ejes")
            salto_horas = col_cfg2.selectbox("⏱️ Resolución del Mosaico (Evita saturar la pantalla):", [1, 2, 3, 4, 6], index=2, format_func=lambda x: f"Mostrar 1 frame cada {x} horas", key="sel_salto_dc")
            
            # --- CORRECCIÓN DE SATURACIÓN ---
            if ajuste_local_dc:
                vmin_dc = max(0.0, float(np.floor(np.min(st.session_state.dc_historial_vtec_3d))))
                vmax_dc = float(np.ceil(np.max(st.session_state.dc_historial_vtec_3d)))
            else:
                vmin_dc = 0.0
                vmax_dc = 120.0 if es_ionex_dc else 60.0 

            # =====================================================================
            # 🧩 MOSAICO CONTINUO
            # =====================================================================
            st.subheader("🧩 Mosaico de Evolución Continua Multi-Día")
            
            # Seleccionamos los frames a dibujar según el salto elegido
            indices_frames = list(range(0, total_horas_rango, salto_horas))
            num_plots_dc = len(indices_frames)
            
            ncols_dc = 6
            nrows_dc = int(np.ceil(num_plots_dc / ncols_dc))
            
            w_sub_dc = 3.5 if es_ionex_dc else 2.8
            h_sub_dc = 2.0 if es_ionex_dc else 2.5
            
            with st.spinner("Renderizando mosaico masivo (puede tardar unos segundos)..."):
                fig_mosaic_dc, axes_dc = plt.subplots(nrows_dc, ncols_dc, figsize=(w_sub_dc * ncols_dc, h_sub_dc * nrows_dc), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
                axes_dc = np.array([axes_dc]).flatten() if num_plots_dc == 1 else np.array(axes_dc).flatten()
                
                mesh_lon, mesh_lat = np.meshgrid(st.session_state.dc_eje_lons, st.session_state.dc_eje_lats)
                ultimo_mapeo_dc = None
                
                for i_plot, f_real in enumerate(indices_frames):
                    ax_sub = axes_dc[i_plot]
                    
                    if es_ionex_dc:
                        ax_sub.set_global()
                        ax_sub.set_xticks([-180, 0, 180], crs=ccrs.PlateCarree())
                        ax_sub.set_yticks([-90, 0, 90], crs=ccrs.PlateCarree())
                    else:
                        ax_sub.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                        ax_sub.set_xticks([float(LON_MIN), (float(LON_MIN)+float(LON_MAX))/2, float(LON_MAX)], crs=ccrs.PlateCarree())
                        ax_sub.set_yticks([float(LAT_MIN), float(LAT_MAX)], crs=ccrs.PlateCarree())
                    
                    ax_sub.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_sub.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_sub.add_feature(cfeature.COASTLINE, edgecolor='#333333', linewidth=0.7, zorder=3)
                    ax_sub.grid(True, color='gray', alpha=0.2, linestyle='--')
                    ax_sub.tick_params(labelsize=6)
                    
                    ultimo_mapeo_dc = ax_sub.pcolormesh(mesh_lon, mesh_lat, st.session_state.dc_historial_vtec_3d[f_real, :, :], 
                                                       transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_dc, vmax=vmax_dc, zorder=2)
                    
                    # Etiqueta con el día y hora exactos
                    if f_real < len(st.session_state.dc_etiquetas_reales):
                        ax_sub.set_title(f" {st.session_state.dc_etiquetas_reales[f_real]}", fontsize=9, weight='bold')
                
                # Ocultar paneles vacíos si no es múltiplo de 6
                for j in range(num_plots_dc, len(axes_dc)):
                    fig_mosaic_dc.delaxes(axes_dc[j])
                
                # Barra global
                if ultimo_mapeo_dc:
                    fig_mosaic_dc.colorbar(ultimo_mapeo_dc, ax=axes_dc.tolist(), orientation='horizontal', shrink=0.5, pad=0.04).set_label('Escala de Densidad VTEC (TECU)', weight='bold', fontsize=10)
                
                st.pyplot(fig_mosaic_dc)
                plt.close(fig_mosaic_dc)

            # =====================================================================
            # 📌 MAPA DE MÁXIMOS ABSOLUTOS
            # =====================================================================
            st.subheader("📌 Mapa Fijo de Máximos Absolutos del Rango Completo")
            fig_max_dc, ax_mxdc = plt.subplots(figsize=(14, 7) if es_ionex_dc else (10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            
            if es_ionex_dc:
                ax_mxdc.set_global()
                ax_mxdc.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
                ax_mxdc.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
            else:
                ax_mxdc.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
                ax_mxdc.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40, 50], crs=ccrs.PlateCarree())
                ax_mxdc.set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
            
            ax_mxdc.add_feature(cfeature.LAND, facecolor='#f6f6f6')
            ax_mxdc.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
            ax_mxdc.add_feature(cfeature.COASTLINE, edgecolor='#222222')
            ax_mxdc.grid(True, color='gray', alpha=0.3, linestyle='--')
            
            mapa_maximos_dc = ax_mxdc.pcolormesh(mesh_lon, mesh_lat, st.session_state.dc_matriz_maximos, 
                                                 transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=vmin_dc, vmax=vmax_dc)
            
            fig_max_dc.colorbar(mapa_maximos_dc, ax=ax_mxdc, orientation='horizontal' if not es_ionex_dc else 'vertical', 
                                pad=0.08 if not es_ionex_dc else 0.02, shrink=0.7 if not es_ionex_dc else 1.0, aspect=40).set_label('PICO MÁXIMO DEL PERIODO (TECU)', weight='bold')
            
            fecha_final_rango = dc_fecha_inicial + datetime.timedelta(days=dc_num_dias - 1)
            plt.title(f"Valores Máximos Acumulados — Del {dc_fecha_inicial.strftime('%d/%m/%Y')} al {fecha_final_rango.strftime('%d/%m/%Y')}", fontsize=11, weight='bold', pad=12)
            
            if es_ionex_dc: plt.tight_layout()
            st.pyplot(fig_max_dc)
            plt.close(fig_max_dc)

            # =====================================================================
            # 📊 GRÁFICA DE LOCALIZACIÓN CONTINUA
            # =====================================================================
            st.subheader("📊 Gráfica Continua del Ciclo de Días Completos Encadenados")
            tipo_busqueda_t3dc = st.radio("Formato de inserción de localidad (Modo Continuo):", ["Por Nombre de Ciudad", "Por Coordenadas de Estación"], horizontal=True, key="radio_t3dc")
            lat_dcl, lon_dcl, name_dcl = None, None, ""
            
            if tipo_busqueda_t3dc == "Por Nombre de Ciudad":
                nueva_ciudad_dc = st.text_input("Nombre del municipio:", "Madrid", key="txt_t3dc")
                if nueva_ciudad_dc: lat_dcl, lon_dcl, name_dcl = geocodificar_localidad(nueva_ciudad_dc)
            else:
                col_ldc1, col_ldc2 = st.columns(2)
                lat_dc_man = col_ldc1.number_input("Latitud nodo de análisis:", min_value=-90.0, max_value=90.0, value=40.41, step=0.1, key="num_lat_t3dc")
                lon_dc_man = col_ldc2.number_input("Longitud nodo de análisis:", min_value=-180.0, max_value=180.0, value=-3.70, step=0.1, key="num_lon_t3dc")
                lat_dcl, lon_dcl, name_dcl = lat_dc_man, lon_dc_man, f"Nodo ({lat_dc_man:.1f}, {lon_dc_man:.1f})"

            if st.button("➕ Añadir Localidad al Gráfico Continuo", key="btn_t3dc"):
                lim_lat_min = -90 if es_ionex_dc else LAT_MIN
                lim_lat_max = 90 if es_ionex_dc else LAT_MAX
                lim_lon_min = -180 if es_ionex_dc else LON_MIN
                lim_lon_max = 180 if es_ionex_dc else LON_MAX
                
                if lat_dcl is not None and lon_dcl is not None and (lim_lat_min <= lat_dcl <= lim_lat_max) and (lim_lon_min <= lon_dcl <= lim_lon_max):
                    if name_dcl not in [c['name'] for c in st.session_state.dc_ciudades_lista]: 
                        st.session_state.dc_ciudades_lista.append({'name': name_dcl, 'lat': lat_dcl, 'lon': lon_dcl})
                else:
                    st.warning("Esa localidad queda fuera del mapa actualmente seleccionado.")
            
            if st.session_state.dc_ciudades_lista:
                fig_lineas_dc, ax_lineas_dc = plt.subplots(figsize=(15, 5.5))
                for ciudad_obj in st.session_state.dc_ciudades_lista:
                    idx_lat = (np.abs(st.session_state.dc_eje_lats - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(st.session_state.dc_eje_lons - ciudad_obj['lon'])).argmin()
                    ax_lineas_dc.plot(range(total_horas_rango), st.session_state.dc_historial_vtec_3d[:, idx_lat, idx_lon], linewidth=2.5, label=ciudad_obj['name'])
                
                ax_lineas_dc.grid(True, linestyle='--')
                ax_lineas_dc.set_ylim(vmin_dc, vmax_dc)
                ax_lineas_dc.set_xlim(0, total_horas_rango - 1)
                ax_lineas_dc.set_xticks(range(total_horas_rango))
                
                # Etiquetamos el eje X solo cada X horas (basado en el salto elegido) para no emborronar el texto
                ax_lineas_dc.set_xticklabels([st.session_state.dc_etiquetas_reales[k] if k % salto_horas == 0 else "" for k in range(total_horas_rango)], rotation=45, fontsize=8)
                
                ax_lineas_dc.set_ylabel("TECU", weight='bold')
                ax_lineas_dc.set_title(f"Evolución Ininterrumpida de Intensidad TECU ({total_horas_rango} horas encadenadas)", fontsize=12, weight='bold', pad=10)
                ax_lineas_dc.legend(loc="upper right")
                
                st.pyplot(fig_lineas_dc)
                plt.close(fig_lineas_dc)
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
            # Si es tiempo real en sombreado rosa 
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

    # Factor de conversión unificado de la ecuación física de refracción
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
        
        # Sistema dual alternativo de entrada de localización
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
            
            # --- BYPASS A SHAPELY GRIDLINES ---
            ax_p5_d.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40, 50], crs=ccrs.PlateCarree())
            ax_p5_d.set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
            ax_p5_d.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            ax_p5_d.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            ax_p5_d.grid(True, color='gray', alpha=0.3, linestyle='--')
            # ----------------------------------

            mapa_d = ax_p5_d.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.p5_historial_desviacion_3d[p5_hora_vista], 
                                        transform=ccrs.PlateCarree(), cmap='seismic', alpha=0.85, shading='gouraud', vmin=vmin_p5, vmax=vmax_p5, zorder=2)
            
            plt.colorbar(mapa_d, ax=ax_p5_d, orientation='vertical', pad=0.02, aspect=35).set_label(f'DESVIACIÓN DEL MODELO DE FONDO (METROS) [{str_status}]', weight='bold')
            ax_p5_d.set_title(f"MAPA ESTÁTICO DE RESIDUOS A LAS {p5_hora_vista:02d}:00 UTC\n[Blanco = Coincidencia | Rojo = Sobreestima | Azul = Subestima]", fontsize=10, weight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_p5_d)
            plt.close(fig_p5_d)

            # Reproductor dinámico
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
                    
                    # --- BYPASS A SHAPELY GRIDLINES ---
                    ax_a.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40, 50], crs=ccrs.PlateCarree())
                    ax_a.set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
                    ax_a.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
                    ax_a.yaxis.set_major_formatter(LATITUDE_FORMATTER)
                    ax_a.grid(True, color='gray', alpha=0.3, linestyle='--')
                    # ----------------------------------
                    
                    mapa_a = ax_a.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.p5_historial_desviacion_3d[f, :, :], 
                                             transform=ccrs.PlateCarree(), cmap='seismic', alpha=0.85, shading='gouraud', vmin=vmin_p5, vmax=vmax_p5, zorder=2)
                    
                    plt.colorbar(mapa_a, ax=ax_a, orientation='vertical', pad=0.02, aspect=35).set_label('DESVIACIÓN EN METROS', weight='bold')
                    ax_a.set_title(f"FRAME HORARIO DE CONTROL: {st.session_state.p5_etiquetas_fechas_reales[f]} UTC", fontsize=10, weight='bold')
                    
                    plt.tight_layout()
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
            
            # --- BYPASS A SHAPELY GRIDLINES ---
            ax_p5_r.set_xticks([-30, -20, -10, 0, 10, 20, 30, 40, 50], crs=ccrs.PlateCarree())
            ax_p5_r.set_yticks([30, 40, 50, 60, 70], crs=ccrs.PlateCarree())
            ax_p5_r.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
            ax_p5_r.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            ax_p5_r.grid(True, color='gray', alpha=0.3, linestyle='--')
            # ----------------------------------

            mapa_r = ax_p5_r.pcolormesh(GRID_LON_EUR, GRID_LAT_GRID, st.session_state.p5_historial_rms_3d[p5_hora_vista], 
                                        transform=ccrs.PlateCarree(), cmap='YlOrRd', alpha=0.85, shading='gouraud', vmin=vmin_rms, vmax=vmax_rms, zorder=2)
            
            plt.colorbar(mapa_r, ax=ax_p5_r, orientation='vertical', pad=0.02, aspect=35).set_label(f'INCERTIDUMBRE DEL DATO (METROS RMS) [{str_status_r}]', weight='bold')
            ax_p5_r.set_title(f"MAPA DE MARGEN DE TOLERANCIA RMS A LAS {p5_hora_vista:02d}:00 UTC\n[Muestra la calidad métrica intrínseca de los datos asimilados por los satélites]", fontsize=10, weight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_p5_r)
            plt.close(fig_p5_r)



# =====================================================================
# PESTAÑA 6: COMENTARIOS Y FEEDBACK 
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
            # Configuramos tus datos de recepción
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



# =====================================================================
# PESTAÑA: AVIACIÓN Y CLIMA ESPACIAL (HERRAMIENTA DUAL)
# =====================================================================
with tab_aviacion:
    
    # =====================================================================
    # SECCIÓN 1: FIABILIDAD DE SATÉLITES (TRACKING GNSS + MAPA TEC)
    # =====================================================================
    st.title("🛰️ Fiabilidad de Satélites (Tracking GNSS)")
    st.markdown("""
    Configura tu punto de observación y pega los datos orbitales (TLE). El sistema calculará la línea 
    de vista de los satélites y los proyectará sobre el mapa global de Contenido Total de Electrones (TEC).
    """)
    st.divider()

    @st.cache_resource
    def inicializar_skyfield():
        return load.timescale()

    @st.cache_data(ttl=900)
    def obtener_fondo_tecu_global():
        try:
            url_global = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL/v2.0.0/latest/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL_latest_D.json"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url_global, headers=headers, timeout=15)
            res.raise_for_status()
            matriz = np.array([f['properties']['vtec_assimilated_tecu'] for f in res.json()['data']['grid']['features']]).reshape(73, 73)
            return matriz
        except:
            return None

    try:
        ts = inicializar_skyfield()
        tiempo_actual = ts.now()

        st.subheader("📍 1. Configuración del Observador")
        tipo_posicionamiento = st.radio(
            "Selecciona el método de ubicación:", 
            ["Buscar por localidad / ciudad", "Coordenadas manuales (Lat/Lon)"], 
            horizontal=True, 
            key="radio_track_man"
        )
        
        lat_target, lon_target, label_ubicacion = None, None, ""

        if tipo_posicionamiento == "Buscar por localidad / ciudad":
            ciudad_usuario = st.text_input("Escribe el nombre de la ciudad o región:", "Madrid", key="txt_ciudad_man")
            if ciudad_usuario:
                # Usa la función global que ya definida para sacar las coordenadas
                lat_target, lon_target, label_ubicacion = geocodificar_localidad(ciudad_usuario)
        else:
            col_u1, col_u2 = st.columns(2)
            lat_target = col_u1.number_input("Latitud (°N/°S):", min_value=-90.0, max_value=90.0, value=40.41, step=0.01, key="num_lat_man")
            lon_target = col_u2.number_input("Longitud (°E/°W):", min_value=-180.0, max_value=180.0, value=-3.70, step=0.01, key="num_lon_man")
            label_ubicacion = f"Coordenadas {lat_target:.2f}°, {lon_target:.2f}°"

        angulo_mascara = st.slider("Resolución / Ángulo de máscara de seguridad (°):", 0.0, 20.0, 5.0, step=0.5, key="sld_mascara_man")

        def procesar_texto_tle_con_mapa(nombre_red, texto_tle_crudo, observador_topos, lat_obs, lon_obs):
            if not texto_tle_crudo.strip():
                st.warning("El cuadro de texto está vacío. Pega los datos TLE antes de escanear.")
                return

            tle_lines = [line.strip() for line in texto_tle_crudo.strip().split('\n') if line.strip()]
            total_red, linea_vista_teorica, linea_vista_segura = 0, 0, 0
            tabla_resultados = []
            id_satelite = 1  
            
            for i in range(0, len(tle_lines), 3):
                if i + 2 >= len(tle_lines): break
                    
                nombre_sat = tle_lines[i]
                linea1, linea2 = tle_lines[i+1], tle_lines[i+2]
                
                if not (linea1.startswith('1 ') and linea2.startswith('2 ')): continue
                    
                total_red += 1
                try:
                    satelite = EarthSatellite(linea1, linea2, nombre_sat, ts)
                    diferencia = satelite - observador_topos
                    topocentrico = diferencia.at(tiempo_actual)
                    alt, az, _ = topocentrico.altaz()
                    
                    if alt.degrees > 0:
                        linea_vista_teorica += 1
                        
                    if alt.degrees >= angulo_mascara:
                        linea_vista_segura += 1
                        subpoint = satelite.at(tiempo_actual).subpoint()
                        
                        tabla_resultados.append({
                            "ID": id_satelite,
                            "Satélite": nombre_sat,
                            "Lat (°N)": round(subpoint.latitude.degrees, 2),
                            "Lon (°E)": round(subpoint.longitude.degrees, 2),
                            "Altitud (km)": round(subpoint.elevation.km, 1),
                            "Elevación (°)": round(alt.degrees, 2),
                            "Azimut (°)": round(az.degrees, 1)
                        })
                        id_satelite += 1
                except Exception:
                    continue

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Satélites en Lista TLE", total_red)
            c2.metric("En Vista Teórica (>0°)", linea_vista_teorica)
            c3.metric(f"Enlace Seguro (>={angulo_mascara}°)", linea_vista_segura)
            
            if tabla_resultados:
                st.success(f"✅ Análisis completado. Se detectaron **{linea_vista_segura}** satélites en línea de vista para {nombre_red}.")
                
                fig, ax = plt.subplots(figsize=(14, 7), dpi=100, subplot_kw={'projection': ccrs.PlateCarree()})
                ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
                ax.add_feature(cfeature.OCEAN, facecolor='#e3f2fd')
                ax.add_feature(cfeature.COASTLINE, edgecolor='#444444', linewidth=0.8)

                matriz_tecu = obtener_fondo_tecu_global()
                if matriz_tecu is not None:
                    lons_glb, lats_glb = np.linspace(-180, 180, 73), np.linspace(-90, 90, 73)
                    grid_lon, grid_lat = np.meshgrid(lons_glb, lats_glb)
                    mapa_calor = ax.pcolormesh(grid_lon, grid_lat, matriz_tecu, transform=ccrs.PlateCarree(), cmap='jet', alpha=0.6, shading='gouraud')
                    fig.colorbar(mapa_calor, ax=ax, orientation='vertical', pad=0.02, shrink=0.7).set_label('TECU Global', weight='bold')
                
                ax.scatter(lon_obs, lat_obs, color='white', edgecolor='red', marker='*', s=350, transform=ccrs.PlateCarree(), zorder=6, label='Observador (Tú)')
                
                lats_sats = [s["Lat (°N)"] for s in tabla_resultados]
                lons_sats = [s["Lon (°E)"] for s in tabla_resultados]
                ax.scatter(lons_sats, lats_sats, color='black', edgecolor='white', s=60, transform=ccrs.PlateCarree(), zorder=5, label='Satélites en Vista')
                
                for sat in tabla_resultados:
                    ax.text(sat["Lon (°E)"] + 2, sat["Lat (°N)"] + 2, str(sat["ID"]), 
                            color='black', fontsize=9, weight='bold', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                            transform=ccrs.PlateCarree(), zorder=7)

                ax.legend(loc='lower left', framealpha=0.9)
                plt.title(f"Cobertura Espacial de {nombre_red} sobre Mapa TECU Global", weight='bold', fontsize=12)
                
                st.pyplot(fig)
                plt.close(fig)

                st.dataframe(pd.DataFrame(tabla_resultados), use_container_width=True, hide_index=True)
            elif total_red > 0:
                st.warning(f"Se leyeron {total_red} satélites, pero ninguno supera el ángulo de máscara.")
            else:
                st.error("No se pudo reconocer ningún formato TLE válido.")

        if lat_target is not None and lon_target is not None:
            st.success(f"✅ **Observatorio configurado en:** {label_ubicacion} ({lat_target:.4f}°, {lon_target:.4f}°)")
            observador = Topos(latitude_degrees=lat_target, longitude_degrees=lon_target)
            
            st.subheader("📡 2. Introducción de Datos Orbitales")
            tab_gps, tab_glo, tab_gal = st.tabs(["🇺🇸 Constelación GPS", "🇷🇺 Constelación GLONASS", "🇪🇺 Constelación Galileo"])
            
            with tab_gps:
                st.info("Obtén los datos oficiales aquí: [CelesTrak GPS TLE](https://celestrak.org/NORAD/elements/gps-ops.txt)")
                texto_gps = st.text_area("Pega aquí el bloque completo de texto TLE para GPS:", height=200, key="area_gps")
                if st.button("🚀 Procesar Datos GPS y Generar Mapa", key="btn_gps_man"):
                    procesar_texto_tle_con_mapa("GPS", texto_gps, observador, lat_target, lon_target)

            with tab_glo:
                st.info("Obtén los datos oficiales aquí: [CelesTrak GLONASS TLE](https://celestrak.org/NORAD/elements/glo-ops.txt)")
                texto_glo = st.text_area("Pega aquí el bloque completo de texto TLE para GLONASS:", height=200, key="area_glo")
                if st.button("🚀 Procesar Datos GLONASS y Generar Mapa", key="btn_glo_man"):
                    procesar_texto_tle_con_mapa("GLONASS", texto_glo, observador, lat_target, lon_target)

            with tab_gal:
                st.info("Obtén los datos oficiales aquí: [CelesTrak Galileo TLE](https://celestrak.org/NORAD/elements/galileo.txt)")
                texto_gal = st.text_area("Pega aquí el bloque completo de texto TLE para Galileo:", height=200, key="area_gal")
                if st.button("🚀 Procesar Datos Galileo y Generar Mapa", key="btn_gal_man"):
                    procesar_texto_tle_con_mapa("Galileo", texto_gal, observador, lat_target, lon_target)
        else:
            st.error("Configura una ubicación válida arriba para desbloquear los paneles.")

    except Exception as e:
        st.error(f"Error crítico en el módulo de Fiabilidad: {e}")

    # =====================================================================
    # SECCIÓN 2: CERTIFICADO IONOSFÉRICO DE VUELO
    # =====================================================================
    st.write("\n" * 3) # Espacio en blanco
    st.divider()
    st.title("📜 Certificado Ionosférico de Vuelo")
    st.markdown("""
    Simula la trayectoria ortodrómica de una aeronave y cruza sus coordenadas espaciotemporales 
    con los mapas de la ionosfera. Obtén un perfil del Contenido Total de Electrones (TEC) que 
    experimentará el vuelo a lo largo de su ruta.
    """)
    
    # Limites geográficos de la malla Europa DLR
    C_LAT_MIN, C_LAT_MAX, C_DELTA_LAT = 30, 72, 1
    C_LON_MIN, C_LON_MAX, C_DELTA_LON = -30, 50, 1

    st.subheader("🛫 Plan de Vuelo")
    col_v1, col_v2 = st.columns(2)
    origen_vuelo = col_v1.text_input("Origen (Ciudad o 'Lat,Lon'):", "Madrid", key="in_origen")
    destino_vuelo = col_v2.text_input("Destino (Ciudad o 'Lat,Lon'):", "Helsinki", key="in_destino")
    
    col_v3, col_v4, col_v5 = st.columns(3)
    fecha_vuelo = col_v3.date_input("Fecha de Salida:", datetime.date.today(), help="Usa fechas recientes (máx. 1 semana atrás).")
    hora_vuelo = col_v4.time_input("Hora de Salida (UTC):", datetime.time(8, 0))
    velocidad_nudos = col_v5.number_input("Velocidad Crucero (nudos):", min_value=100, max_value=1000, value=450, step=10)

    def obtener_coords_vuelo(punto_str):
        try:
            lat, lon = map(float, punto_str.split(','))
            return lat, lon
        except ValueError:
            geolocator = Nominatim(user_agent="vtec_flight_tracker_app")
            loc = geolocator.geocode(punto_str)
            if loc:
                return loc.latitude, loc.longitude
            return None, None

    def generar_enlace_dlr_historico(fecha_busqueda):
        str_año = fecha_busqueda.strftime("%Y")
        str_doy = fecha_busqueda.strftime("%j")
        str_hora = fecha_busqueda.strftime("%H")
        fecha_inicio = fecha_busqueda - datetime.timedelta(minutes=4, seconds=30)
        timestamp_inicio = fecha_inicio.strftime("%Y-%m-%dT%H-%M-%S")
        timestamp_fin = fecha_busqueda.strftime("%Y-%m-%dT%H-%M-%S")
        base_url = "https://impc.dlr.de/SWE/Total_Electron_Content/TEC_Near_Real-Time/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE/v2.0.0"
        return f"{base_url}/{str_año}/{str_doy}/{str_hora}/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_EUROPE_{timestamp_inicio}_{timestamp_fin}_{str_doy}_D.json"

    @st.cache_data(ttl=3600, show_spinner=False)
    def descargar_malla_vuelo(fecha_obj):
        headers = {"User-Agent": "Mozilla/5.0"}
        minutos_contiguos = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
        min_base = 5 * round(fecha_obj.minute / 5)
        if min_base == 60:
            fecha_obj = fecha_obj + datetime.timedelta(hours=1)
            min_base = 0
            
        if min_base in minutos_contiguos:
            minutos_contiguos.remove(min_base)
            minutos_contiguos.insert(0, min_base)

        for m in minutos_contiguos:
            fecha_intento = fecha_obj.replace(minute=m)
            url = generar_enlace_dlr_historico(fecha_intento)
            try:
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    vtec_list = []
                    if 'data' in data and 'grid' in data['data']:
                        for feature in data['data']['grid']['features']:
                            vtec_list.append(feature['properties']['vtec_assimilated_tecu'])
                    
                    cols = len(np.arange(C_LON_MIN, C_LON_MAX + C_DELTA_LON, C_DELTA_LON))
                    fils = len(np.arange(C_LAT_MIN, C_LAT_MAX + C_DELTA_LAT, C_DELTA_LAT))
                    
                    if len(vtec_list) == (fils * cols):
                        return np.array(vtec_list).reshape(fils, cols), f"{fecha_intento.hour:02d}:{m:02d}"
            except:
                continue
        return None, None

    if st.button("✈️ Generar Certificado de Vuelo", key="btn_certificado"):
        lat_origen, lon_origen = obtener_coords_vuelo(origen_vuelo)
        lat_destino, lon_destino = obtener_coords_vuelo(destino_vuelo)

        if None in (lat_origen, lon_origen, lat_destino, lon_destino):
            st.error("No se han podido geolocalizar el Origen o el Destino. Comprueba los nombres.")
        else:
            ruta_valida = True
            for l_lat, l_lon, nom in [(lat_origen, lon_origen, "Origen"), (lat_destino, lon_destino, "Destino")]:
                if not (C_LAT_MIN <= l_lat <= C_LAT_MAX) or not (C_LON_MIN <= l_lon <= C_LON_MAX):
                    st.error(f"¡Error! El {nom} ({l_lat:.2f}, {l_lon:.2f}) se sale de la cuadrícula de Europa (Lat: 30 a 72, Lon: -30 a 50).")
                    ruta_valida = False
            
            if ruta_valida:
                with st.spinner("Calculando ortodrómica y cruzando con datos del DLR..."):
                    dist_km = great_circle((lat_origen, lon_origen), (lat_destino, lon_destino)).kilometers
                    dist_nm = dist_km * 0.539957
                    horas_totales = dist_nm / velocidad_nudos
                    intervalo_minutos = 20
                    num_pasos = max(2, int((horas_totales * 60) / intervalo_minutos) + 2)

                    st.success(f"**Ruta válida:** {dist_km:.1f} km ({dist_nm:.1f} NM). **Tiempo estimado:** {horas_totales:.2f} horas.")

                    lat1, lon1, lat2, lon2 = map(np.radians, [lat_origen, lon_origen, lat_destino, lon_destino])
                    d_angular = np.arccos(np.clip(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2 - lon1), -1.0, 1.0))

                    posiciones_vuelo = []
                    fecha_inicial = datetime.datetime.combine(fecha_vuelo, hora_vuelo)

                    for idx, f in enumerate(np.linspace(0, 1, num_pasos)):
                        A = np.sin((1 - f) * d_angular) / np.sin(d_angular)
                        B = np.sin(f * d_angular) / np.sin(d_angular)
                        x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
                        y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
                        z = A * np.sin(lat1) + B * np.sin(lat2)
                        p_lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
                        p_lon = np.degrees(np.arctan2(y, x))
                        
                        tiempo_paso = fecha_inicial + datetime.timedelta(minutes=idx * intervalo_minutos)
                        if idx == num_pasos - 1:
                            tiempo_paso = fecha_inicial + datetime.timedelta(hours=horas_totales)
                            
                        posiciones_vuelo.append({"lat": p_lat, "lon": p_lon, "tiempo": tiempo_paso, "tecu": 0.0, "valido": False})

                    vect_lats = np.arange(C_LAT_MIN, C_LAT_MAX + C_DELTA_LAT, C_DELTA_LAT)
                    vect_lons = np.arange(C_LON_MIN, C_LON_MAX + C_DELTA_LON, C_DELTA_LON)
                    
                    barra_progreso = st.progress(0)
                    for i, pos in enumerate(posiciones_vuelo):
                        malla, _ = descargar_malla_vuelo(pos["tiempo"])
                        if malla is not None:
                            idx_lat = (np.abs(vect_lats - pos["lat"])).argmin()
                            idx_lon = (np.abs(vect_lons - pos["lon"])).argmin()
                            pos["tecu"] = malla[idx_lat, idx_lon]
                            pos["valido"] = True
                        barra_progreso.progress((i + 1) / len(posiciones_vuelo))
                    
                    pos_validas = [p for p in posiciones_vuelo if p["valido"]]
                    
                    if len(pos_validas) < 2:
                        st.error("❌ No se han encontrado datos ionosféricos en el servidor DLR para las coordenadas y fecha establecidas. Prueba con una fecha más reciente o comprueba tu conexión.")
                    else:
                        lats_ruta = [p["lat"] for p in pos_validas]
                        lons_ruta = [p["lon"] for p in pos_validas]
                        tecus_ruta = [p["tecu"] for p in pos_validas]
                        tiempos_str = [p["tiempo"].strftime("%H:%M") for p in pos_validas]

                        st.divider()
                        st.subheader("📊 Reporte de Navegación")

                        # MAPA 1
                        fig1, ax1 = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                        ax1.set_extent([C_LON_MIN, C_LON_MAX, C_LAT_MIN, C_LAT_MAX], crs=ccrs.PlateCarree())

                        ax1.add_feature(cfeature.LAND, facecolor='#eaeaea')
                        ax1.add_feature(cfeature.OCEAN, facecolor='#d9effb')
                        ax1.add_feature(cfeature.COASTLINE, edgecolor='#333333', linewidth=1)
                        ax1.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#777777')

                        ax1.plot(lons_ruta, lats_ruta, color='red', linewidth=2.5, marker='o', label='Trayectoria del Avión', transform=ccrs.PlateCarree())

                        for p in pos_validas[::2]:
                            ax1.text(p["lon"] + 0.5, p["lat"] + 0.5, p["tiempo"].strftime("%H:%M"), fontsize=8, weight='bold', transform=ccrs.PlateCarree())

                        gl = ax1.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.3)
                        gl.top_labels, gl.right_labels = False, False
                        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER

                        ax1.set_title(f"Ruta Ortodrómica: {origen_vuelo} ➔ {destino_vuelo}", fontsize=11, weight='bold')
                        ax1.legend(loc='lower left')
                        st.pyplot(fig1)
                        plt.close(fig1)

                        # GRÁFICA 2
                        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
                        ax2.plot(tiempos_str, tecus_ruta, marker='s', color='#d32f2f', linewidth=2, label='TECU experimentado')
                        ax2.fill_between(tiempos_str, tecus_ruta, color='#d32f2f', alpha=0.1)
                        ax2.grid(True, linestyle=':', alpha=0.6)
                        ax2.set_xlabel("Progreso del Tiempo de Vuelo (Hora UTC)", fontsize=10, weight='bold')
                        ax2.set_ylabel("TECU", fontsize=10, weight='bold')
                        ax2.set_title("Perfil Dinámico de Radiación Ionosférica (TEC)", fontsize=11, weight='bold')
                        ax2.set_ylim(0, max(tecus_ruta) + 5 if tecus_ruta else 30)
                        ax2.legend()
                        fig2.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)


