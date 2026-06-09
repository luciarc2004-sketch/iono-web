import datetime
import requests
import time  # Requerido para el control de tiempo del video (0.5s)
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.interpolate import RegularGridInterpolator
import streamlit as st

# (Las pestañas 1 y 2 se mantienen exactamente igual en tu archivo principal)

# =====================================================================
# PESTAÑA 3: EVOLUCIÓN TECU (REDISEÑADA POR PARTES)
# =====================================================================
with tab3:
    st.title("📈 Estudio de Evolución Temporal del TECU")
    
    modo_evolucion = st.radio("Selecciona el tipo de análisis temporal:", ["Por Días", "Por Horas (Próximamente)"], horizontal=True)

    if modo_evolucion == "Por Días":
        st.subheader("📆 Análisis de Evolución Interdiaria (Hora Fija)")
        
        # Inicialización de estados de sesión necesarios
        if 'historial_vtec_3d' not in st.session_state:
            st.session_state.historial_vtec_3d = None
            st.session_state.etiquetas_fechas_reales = []
            st.session_state.limites_globales = (0, 15)
            st.session_state.matriz_maximos = None
            st.session_state.ciudades_lista = []

        # Formulario de parámetros
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

                for m in minutes_contiguos:
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

        # COMPONENTES VISUALES UNA VEZ CARGADOS LOS DATOS
        if st.session_state.historial_vtec_3d is not None:
            st.divider()
            
            v_min, v_max = st.session_state.limites_globales
            lons_vector = np.arange(-30, 51, 1)
            lats_vector = np.arange(30, 73, 1)
            grid_lon, grid_lat = np.meshgrid(lons_vector, lats_vector)

            # -----------------------------------------------------------------
            # PARTE 1: MAPA DE MÁXIMOS (ESTRUCTURA FIJA)
            # -----------------------------------------------------------------
            st.subheader("📌 Mapa Fijo de Máximos Absolutos Registrados")
            st.write("Muestra el valor más alto alcanzado en cada punto cardinal durante todo el periodo analizado.")
            
            fig_max, ax_mx = plt.subplots(figsize=(10, 5.5), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
            ax_mx.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
            ax_mx.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
            ax_mx.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
            ax_mx.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
            ax_mx.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
            ax_mx.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False

            mapa_maximos = ax_mx.pcolormesh(grid_lon, grid_lat, st.session_state.matriz_maximos, 
                                            transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
            fig_max.colorbar(mapa_maximos, ax=ax_mx, orientation='vertical', pad=0.02, shrink=0.8).set_label('PICO MÁXIMO (TECU)', weight='bold')
            ax_mx.set_title("Distribución de Intensidades Máximas Observadas", weight='bold')
            st.pyplot(fig_max)

            st.divider()

            # -----------------------------------------------------------------
            # PARTE 2: MAPA DE FLAMES TIPO VIDEO (REPRODUCTOR AUTOMÁTICO A 0.5s)
            # -----------------------------------------------------------------
            st.subheader("🎬 Reproductor de Video: Evolución Diaria del TEC (0.5s por Frame)")
            
            # Botones de control de flujo del video
            col_b1, col_b2, col_b3 = st.columns([1, 1, 4])
            play_video = col_b1.button("▶️ Reproducir Video")
            stop_video = col_b2.button("⏹️ Detener")

            # Contenedor dinámico asíncrono para el mapa tipo video
            contenedor_video_mapa = st.empty()

            if play_video:
                # Bucle de animación iterativo
                num_frames = len(st.session_state.etiquetas_fechas_reales)
                for f in range(num_frames):
                    fig_video, ax_ev = plt.subplots(figsize=(10, 5.5), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                    ax_ev.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                    ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                    ax_ev.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                    ax_ev.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                    ax_ev.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
                    ax_ev.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
                    
                    # Cargar matriz del frame actual
                    matriz_frame = st.session_state.historial_vtec_3d[f, :, :]
                    mapa_dinamico = ax_ev.pcolormesh(grid_lon, grid_lat, matriz_frame, 
                                                     transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
                    fig_video.colorbar(mapa_dinamico, ax=ax_ev, orientation='vertical', pad=0.02, shrink=0.8).set_label('VTEC (TECU)', weight='bold')
                    
                    ax_ev.set_title(f"Video en Curso ➔ Fecha del Frame: {st.session_state.etiquetas_fechas_reales[f]} UTC", weight='bold', color='#1976d2')
                    
                    # Pintar de forma limpia sobre el mismo contenedor fijo
                    contenedor_video_mapa.pyplot(fig_video)
                    plt.close(fig_video)
                    
                    # Espera estricta de 0.5 segundos solicitada por frame
                    time.sleep(0.5)
            else:
                # Vista estática por defecto antes de dar Play (Muestra el primer día)
                fig_video, ax_ev = plt.subplots(figsize=(10, 5.5), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=100)
                ax_ev.set_extent([-30, 50, 30, 72], crs=ccrs.PlateCarree())
                ax_ev.add_feature(cfeature.LAND, facecolor='#f6f6f6', zorder=1)
                ax_ev.add_feature(cfeature.OCEAN, facecolor='#e3f2fd', zorder=1)
                ax_ev.add_feature(cfeature.COASTLINE, edgecolor='#222222', linewidth=1.1, zorder=3)
                ax_ev.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='#888888', zorder=3)
                ax_ev.gridlines(draw_labels=True, color='gray', alpha=0.2, linestyle='--').top_labels = False
                
                mapa_dinamico = ax_ev.pcolormesh(grid_lon, grid_lat, st.session_state.historial_vtec_3d[0, :, :], 
                                                 transform=ccrs.PlateCarree(), cmap='jet', alpha=0.85, shading='gouraud', vmin=v_min, vmax=v_max, zorder=2)
                fig_video.colorbar(mapa_dinamico, ax=ax_ev, orientation='vertical', pad=0.02, shrink=0.8).set_label('VTEC (TECU)', weight='bold')
                ax_ev.set_title(f"Video Detenido ➔ Presiona Play para iniciar", weight='bold')
                contenedor_video_mapa.pyplot(fig_video)
                plt.close(fig_video)

            st.divider()

            # -----------------------------------------------------------------
            # PARTE 3: GRÁFICA ACUMULATIVA DE LOCALIDADES (Sin alteraciones)
            # -----------------------------------------------------------------
            st.subheader("📊 3. Gráfica Comparativa de Localidades Acumuladas")
            st.write("Escribe el nombre de una ciudad y presiona el botón. Se agregará a la gráfica sin borrar las anteriores.")

            with st.form("formulario_acumulador_ciudades"):
                nueva_ciudad = st.text_input("Nombre de la ciudad a añadir al gráfico histórico:", "Madrid")
                boton_agregar = st.form_submit_button("➕ Añadir Ciudad al Análisis")

            if boton_agregar and nueva_ciudad:
                lat_c, lon_c, _ = geocodificar_localidad(nueva_ciudad)
                if lat_c is not None and (30 <= lat_c <= 72) and (-30 <= lon_c <= 50):
                    if nueva_ciudad.capitalize() not in [c['name'] for c in st.session_state.ciudades_lista]:
                        st.session_state.ciudades_lista.append({
                            'name': nueva_ciudad.capitalize(),
                            'lat': lat_c,
                            'lon': lon_c
                        })
                        st.success(f"Añadida {nueva_ciudad.capitalize()} al histórico.")
                else:
                    st.error("Ubicación no encontrada o fuera del área de cobertura de Europa.")

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
                plt.close(fig_lineas)
                
                if st.button("🗑️ Limpiar todas las ciudades del gráfico"):
                    st.session_state.ciudades_lista = []
                    st.rerun()
    else:
        st.info("🛠️ El módulo de análisis continuo por horas se habilitará en las próximas versiones de desarrollo.")
