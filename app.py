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
tab1, tab2, tab3 = st.tabs(["🌍 Inicio y Monitoreo Real", "📊 Análisis en el pasado", "📈 Evolución TECU"])

# FUNCIÓN COMPARTIDA DE GEOCODIFICACIÓN
def geocodificar_localidad(nombre_lugar):
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
    """)
    st.divider()
    # (El resto del código de la pestaña 1 se mantiene igual de forma interna...)

# =====================================================================
# PESTAÑA 2: ANÁLISIS EN EL PASADO
# =====================================================================
with tab2:
    st.title("📊 Análisis Histórico: Mapas e Interpolar en el Pasado")
    # (El resto del código de la pestaña 2 se mantiene igual de forma interna...)

# =====================================================================
# PESTAÑA 3: EVOLUCIÓN TECU (DISEÑADA COMPLETAMENTE)
# =====================================================================
with tab3:
    st.title("📈 Estudio de Evolución Temporal del TECU")
    
    # Menú de selección de modo
    modo_evolucion = st.radio("Selecciona el tipo de análisis temporal:", ["Por Días", "Por Horas (Próximamente)"], horizontal=True)

    if modo_evolucion == "Por Días":
        st.subheader("📆 Análisis de Evolución Interdiaria (Hora Fija)")
        
        # Inicializar variables de estado persistentes en la sesión de Streamlit
        if 'historial_vtec_3d' not in st.session_state:
            st.session_state.historial_vtec_3d = None
            st.session_state.etiquetas_fechas_reales = []
            st.session_state.limites_globales = (0, 15)
            st.session_state.matriz_maximos = None
            st.session_state.ciudades_lista = []  # Almacena el historial de ciudades consultadas

        # Formulario de parámetros de rango de días
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

        if st.button("🚀 Procesar Historial de Mallas"):
            headers = {"User-Agent": "Mozilla/5.0"}
            minutos_contiguos = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
            
            temp_etiquetas = []
            temp_3d = np.zeros((num_dias_sel, 43, 81))
            progreso = st.progress(0.0)
            exito_total = True

            for d in range(num_dias_sel):
                fecha_actual = datetime.datetime(fecha_inicial.year, fecha_inicial.month, fecha_inicial.day) + datetime.timedelta(days=d)
                link_exitoso = False
                data = None
                minuto_exitoso = 0

                for m in minutos_contiguos:
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
                
                # Regla del +-2 solicitada
                max_r = np.max(temp_3d)
                min_r = np.max([0.0, np.min(temp_3d)])
                st.session_state.limites_globales = (max(0.0, float(np.floor(min_r - 2))), float(np.ceil(max_r + 2)))
                st.session_state.matriz_maximos = np.max(temp_3d, axis=0)
                st.success("📊 Rango temporal procesado y almacenado.")

        # SI LOS DATOS ESTÁN CARGADOS, HACEMOS EL DESPLIEGUE GRÁFICO
        if st.session_state.historial_vtec_3d is not None:
            st.divider()
            
            # --- SECCIÓN 1: REPRODUCTOR DE MAPAS (REEMPLAZO ESTABLE DE LA ANIMACION) ---
            st.subheader("🗺️ 1. Línea del Tiempo: Mapas de Evolución")
            frame_sel = st.slider("Desplaza el selector para ver los mapas de cada día del historial:", 
                                  0, len(st.session_state.etiquetas_fechas_reales) - 1, 0)
            
            v_min, v_max = st.session_state.limites_globales
            lons_vector = np.arange(-30, 51, 1)
            lats_vector = np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            # Graficar los dos mapas lado a lado (Día Actual vs Máximos Absolutos del periodo)
            fig_evolucion, (ax_ev, ax_mx) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)

            # Mapa Izquierdo: Frame actual
            ax_ev.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_ev.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_ev.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            ax_ev.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
            ax_ev.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
            
            mapa_dinamico = ax_ev.pcolormesh(grid_lon, grid_lat, st.session_state.historial_vtec_3d[frame_sel, :, :], 
                                             transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
            fig_evolucion.colorbar(mapa_dinamico, ax=ax_ev, orientation='horizontal', pad=0.07, shrink=0.8).set_label('VTEC (TECU)', weight='bold')
            ax_ev.set_title(f"Fotograma Actual: {st.session_state.etiquetas_fechas_reales[frame_sel]} UTC", weight='bold')

            # Mapa Derecho: Máximos Absolutos del rango
            ax_mx.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_mx.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            ax_mx.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
            ax_mx.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False

            mapa_maximos = ax_mx.pcolormesh(grid_lon, grid_lat, st.session_state.matriz_maximos, 
                                            transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
            fig_evolucion.colorbar(mapa_maximos, ax=ax_mx, orientation='horizontal', pad=0.07, shrink=0.8).set_label('PICO MÁXIMO (TECU)', weight='bold')
            ax_mx.set_title("Mapa de Máximos Registrados en el Periodo Completo", weight='bold')

            st.pyplot(fig_evolucion)

            st.divider()

            # --- SECCIÓN 2: GRÁFICA ACUMULATIVA DE LOCALIDADES ---
            st.subheader("📊 2. Gráfica Comparativa de Localidades Acumuladas")
            st.write("Escribe el nombre de una ciudad y presiona el botón. Se agregará a la gráfica sin borrar las anteriores.")

            with st.form("formulario_acumulador_ciudades"):
                nueva_ciudad = st.text_input("Nombre de la ciudad a añadir al gráfico histórico:", "Madrid")
                boton_agregar = st.form_submit_button("➕ Añadir Ciudad al Análisis")

            if boton_agregar and nueva_ciudad:
                lat_c, lon_c, _ = geocodificar_localidad(nueva_ciudad)
                if lat_c is not None and (30 <= lat_c <= 72) and (-30 <= lon_c <= 50):
                    # Guardar coordenadas y nombre en la lista de la sesión para que no se borren
                    if nueva_ciudad.capitalize() not in [c['name'] for c in st.session_state.ciudades_lista]:
                        st.session_state.ciudades_lista.append({
                            'name': nueva_ciudad.capitalize(),
                            'lat': lat_c,
                            'lon': lon_c
                        })
                        st.success(f"Añadida {nueva_ciudad.capitalize()} al histórico.")
                else:
                    st.error("Ubicación no encontrada o fuera del área de cobertura de Europa.")

            # Dibujar la gráfica acumulativa si hay ciudades registradas
            if st.session_state.ciudades_lista:
                fig_lineas, ax_lineas = plt.subplots(figsize=(12, 5))
                
                for ciudad_obj in st.session_state.ciudades_lista:
                    idx_lat = (np.abs(lats_vector - ciudad_obj['lat'])).argmin()
                    idx_lon = (np.abs(lons_vector - ciudad_obj['lon'])).argmin()
                    perfil_temporal = st.session_state.historial_vtec_3d[:, idx_lat, idx_lon]
                    
                    ax_lineas.plot(range(len(st.session_state.etiquetas_fechas_reales)), perfil_temporal, 
                                   marker='s', linestyle='-', linewidth=2, label=ciudad_obj['name'])

                ax_lineas.grid(True, linestyle='--', alpha=0.6)
                ax_lineas.set_ylim(v_min, v_max)
                ax_lineas.set_xticks(range(len(st.session_state.etiquetas_fechas_reales)))
                ax_lineas.set_xticklabels(st.session_state.etiquetas_fechas_reales, rotation=25)
                ax_lineas.set_ylabel("VTEC (TECU)", weight='bold')
                ax_lineas.set_xlabel("Muestras de la Línea Temporal (UTC)", weight='bold')
                ax_lineas.set_title(f"Evolución Comparativa (Límites Eje Y: {int(v_min)}-{int(v_max)} TECU)", weight='bold')
                ax_lineas.legend(loc="upper right")
                
                st.pyplot(fig_lineas)
                
                # Opción para limpiar la gráfica acumulada
                if st.button("🗑️ Limpiar todas las ciudades del gráfico"):
                    st.session_state.ciudades_lista = []
                    st.rerun()

    else:
        st.info("🛠️ El módulo de análisis continuo por horas se habilitará en las próximas versiones de desarrollo.")
