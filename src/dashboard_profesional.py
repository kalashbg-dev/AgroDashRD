"""
Módulo de Dashboard Profesional para AgroDashRD.

Este módulo implementa visualizaciones y análisis avanzados
orientados a profesionales, analistas y tomadores de decisiones.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

logger = logging.getLogger(__name__)

# --- OPTIMIZACIÓN: Funciones cacheadas para agrupaciones y pivoteos ---
@st.cache_data(show_spinner=False)
def cached_groupby_mean(df, group_cols, value_col):
    return df.groupby(group_cols)[value_col].mean().reset_index()

@st.cache_data(show_spinner=False)
def cached_groupby_agg(df, group_col, value_col):
    return df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()

@st.cache_data(show_spinner=False)
def cached_pivot(df, index_col, columns_col, values_col):
    return df.pivot(index=index_col, columns=columns_col, values=values_col)

def analisis_tendencias_avanzado(df: pd.DataFrame) -> None:
    """
    Implementa análisis avanzado de tendencias de precios
    utilizando técnicas estadísticas y visualizaciones detalladas.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.header("Análisis Avanzado de Tendencias")
    
    st.markdown("""
    <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
    <p style="font-size: 16px; margin-bottom: 0;">
    Este módulo permite realizar análisis detallado de tendencias utilizando métodos
    estadísticos avanzados como descomposición de series temporales, regresiones y 
    detección de patrones estacionales.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or 'Fecha' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para realizar el análisis avanzado.")
        return
    
    # Asegurarse que la fecha es datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Obtener lista de productos
    productos = sorted(df['Producto'].unique())
    
    if len(productos) == 0:
        st.warning("No hay productos disponibles en los datos.")
        return
    
    # Interfaz de usuario para selección
    col1, col2 = st.columns([2, 2])
    
    with col1:
        producto_seleccionado = st.selectbox(
            "Seleccione un producto:",
            productos,
            key="producto_tendencias"
        )
        
        # Determinar columna de precio a usar
        precio_cols = []
        if 'Precio_Mayorista' in df.columns:
            precio_cols.append('Precio Mayorista')
        if 'Precio_Minorista' in df.columns:
            precio_cols.append('Precio Minorista')
        
        tipo_precio = st.selectbox(
            "Tipo de precio:",
            precio_cols,
            key="tipo_precio_tendencias"
        )
        
        precio_col = 'Precio_Mayorista' if tipo_precio == 'Precio Mayorista' else 'Precio_Minorista'
    
    with col2:
        if 'Mercado' in df.columns:
            mercados = ['Todos'] + sorted(df['Mercado'].unique())
            
            mercado_seleccionado = st.selectbox(
                "Seleccione un mercado (opcional):",
                mercados,
                key="mercado_tendencias"
            )
        else:
            mercado_seleccionado = None
            st.info("No hay datos de mercados disponibles para filtrar.")
        
        # Periodo de análisis
        periodo_opciones = ['Último año', 'Últimos 6 meses', 'Últimos 3 meses', 'Todo el periodo']
        
        periodo_analisis = st.selectbox(
            "Periodo de análisis:",
            periodo_opciones,
            key="periodo_tendencias"
        )
    
    # Filtrar datos por producto y mercado
    df_producto = df[df['Producto'] == producto_seleccionado].copy()
    
    if mercado_seleccionado != 'Todos' and mercado_seleccionado is not None:
        df_producto = df_producto[df_producto['Mercado'] == mercado_seleccionado]
    
    if len(df_producto) == 0:
        st.warning(f"No hay datos disponibles para {producto_seleccionado}.")
        return
    
    # Verificar que la columna de precio seleccionada existe
    if precio_col not in df_producto.columns:
        st.warning(f"No hay datos de {tipo_precio} disponibles para este producto.")
        return
    
    # Filtrar por periodo
    fecha_maxima = df_producto['Fecha'].max()
    
    if periodo_analisis == 'Último año':
        fecha_minima = fecha_maxima - pd.Timedelta(days=365)
    elif periodo_analisis == 'Últimos 6 meses':
        fecha_minima = fecha_maxima - pd.Timedelta(days=180)
    elif periodo_analisis == 'Últimos 3 meses':
        fecha_minima = fecha_maxima - pd.Timedelta(days=90)
    else:  # Todo el periodo
        fecha_minima = df_producto['Fecha'].min()
    
    df_periodo = df_producto[(df_producto['Fecha'] >= fecha_minima) & 
                            (df_producto['Fecha'] <= fecha_maxima)].copy()
    
    if len(df_periodo) < 10:
        st.warning("No hay suficientes datos para el análisis en el periodo seleccionado.")
        return
    
    # Asegurarse que la columna de precios es numérica
    df_periodo[precio_col] = pd.to_numeric(df_periodo[precio_col], errors='coerce')
    
    # Filtrar valores no numéricos
    df_periodo = df_periodo[df_periodo[precio_col].notna()]
    
    # Agrupar por fecha para obtener precio promedio diario
    df_periodo[precio_col] = pd.to_numeric(df_periodo[precio_col], errors='coerce')
    df_diario = cached_groupby_mean(df_periodo, 'Fecha', precio_col)
    
    # Crear serie temporal para análisis
    serie_temporal = df_diario.set_index('Fecha')[precio_col]
    
    # 1. Visualización de tendencia con línea de regresión
    st.subheader("Tendencia de Precios")
    
    # Crear datos para regresión
    X = np.array(range(len(serie_temporal))).reshape(-1, 1)
    y = serie_temporal.values
    
    # Ajustar modelo de regresión
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Predecir valores
    y_pred = modelo.predict(X)
    
    # Calcular métricas
    pendiente = modelo.coef_[0]
    intercepto = modelo.intercept_
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir datos reales
    fig.add_trace(go.Scatter(
        x=serie_temporal.index,
        y=serie_temporal.values,
        mode='markers+lines',
        name='Precio real',
        line=dict(color='#5a8f7b', width=1),
        marker=dict(size=5)
    ))
    
    # Añadir línea de tendencia
    fig.add_trace(go.Scatter(
        x=serie_temporal.index,
        y=y_pred,
        mode='lines',
        name='Tendencia',
        line=dict(color='#e63946', width=2, dash='dash')
    ))
    
    # Actualizar diseño
    fig.update_layout(
        title=f"Tendencia de Precios para {producto_seleccionado}",
        xaxis_title="Fecha",
        yaxis_title=f"{tipo_precio} (RD$)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretación de tendencia
    tendencia_texto = "ascendente" if pendiente > 0 else "descendente"
    cambio_mensual = pendiente * 30  # Cambio aproximado en un mes
    
    st.markdown(f"""
    ### Interpretación de la Tendencia
    
    * La tendencia general para **{producto_seleccionado}** es **{tendencia_texto}**.
    * La pendiente de la recta de regresión es **{pendiente:.4f}**, lo que indica un cambio promedio de 
      **RD$ {abs(pendiente):.2f}** por día en el precio.
    * En términos mensuales, esto representa un cambio aproximado de **RD$ {abs(cambio_mensual):.2f}**
      por mes en el precio del producto.
    * El precio base estimado (intercepto) es **RD$ {intercepto:.2f}**.
    """)
    
    # 2. Descomposición de series temporales
    try:
        # Crear serie temporal con frecuencia diaria o apropiada
        if len(serie_temporal) >= 14:  # Necesitamos suficientes datos para la descomposición
            # Intentar descomponer la serie
            descomposicion = seasonal_decompose(
                serie_temporal, 
                model='additive', 
                period=min(14, len(serie_temporal) // 2)  # Usar un periodo razonable
            )
            
            # Crear subplots para mostrar la descomposición
            st.subheader("Descomposición de Serie Temporal")
            
            # Crear figura con 4 subplots
            fig = go.Figure()
            
            # Datos observados
            fig = make_subplots(rows=4, cols=1, subplot_titles=("Observado", "Tendencia", "Estacionalidad", "Residuos"))
            
            # Observado
            fig.add_trace(go.Scatter(
                x=serie_temporal.index, 
                y=serie_temporal.values,
                mode='lines', 
                name='Observado'
            ), row=1, col=1)
            
            # Tendencia
            fig.add_trace(go.Scatter(
                x=descomposicion.trend.index, 
                y=descomposicion.trend.values,
                mode='lines', 
                name='Tendencia',
                line=dict(color='#e63946')
            ), row=2, col=1)
            
            # Estacionalidad
            fig.add_trace(go.Scatter(
                x=descomposicion.seasonal.index, 
                y=descomposicion.seasonal.values,
                mode='lines', 
                name='Estacionalidad',
                line=dict(color='#457b9d')
            ), row=3, col=1)
            
            # Residuos
            fig.add_trace(go.Scatter(
                x=descomposicion.resid.index, 
                y=descomposicion.resid.values,
                mode='lines', 
                name='Residuos',
                line=dict(color='#2a9d8f')
            ), row=4, col=1)
            
            # Actualizar diseño
            fig.update_layout(
                height=800,
                title_text=f"Descomposición de Serie Temporal para {producto_seleccionado}",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Interpretación de la Descomposición
            
            La descomposición de una serie temporal separa los datos en tres componentes:
            
            1. **Tendencia**: Muestra la dirección general a largo plazo de los precios.
            2. **Estacionalidad**: Captura patrones que se repiten en intervalos regulares.
            3. **Residuos**: La parte aleatoria o inexplicable del precio.
            
            Observe los patrones estacionales para identificar ciclos predecibles en los precios.
            """)
    except Exception as e:
        st.info(f"No se pudo realizar la descomposición de la serie temporal: {str(e)}")
        st.info("Esto puede deberse a datos insuficientes o irregulares. Se requieren al menos 14 observaciones.")
    
    # 3. Análisis de volatilidad
    st.subheader("Análisis de Volatilidad")
    
    # Calcular cambios porcentuales diarios
    serie_temporal_pct = serie_temporal.pct_change().dropna()
    
    if len(serie_temporal_pct) > 0:
        volatilidad = serie_temporal_pct.std() * 100  # Convertir a porcentaje
        
        # Crear histograma de cambios porcentuales
        fig = px.histogram(
            x=serie_temporal_pct * 100,  # Convertir a porcentaje
            nbins=20,
            labels={'x': 'Cambio Porcentual Diario (%)'},
            title=f"Distribución de Cambios Porcentuales Diarios para {producto_seleccionado}",
            color_discrete_sequence=['#5a8f7b']
        )
        
        # Añadir línea vertical en cero
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        # Actualizar diseño
        fig.update_layout(
            xaxis_title="Cambio Porcentual (%)",
            yaxis_title="Frecuencia",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación de volatilidad
        nivel_volatilidad = "alta" if volatilidad > 5 else "moderada" if volatilidad > 2 else "baja"
        
        st.markdown(f"""
        ### Interpretación de Volatilidad
        
        * La volatilidad diaria promedio para **{producto_seleccionado}** es de **{volatilidad:.2f}%**.
        * Esto indica una volatilidad **{nivel_volatilidad}** en los precios.
        * Productos con alta volatilidad presentan mayor riesgo pero también más oportunidades para
          operaciones de compra-venta estratégicas.
        """)
    else:
        st.info("No hay suficientes datos para calcular la volatilidad.")

def analisis_comparativo_mercados(df: pd.DataFrame) -> None:
    """
    Implementa análisis comparativo avanzado entre diferentes mercados,
    incluyendo correlaciones, diferenciales de precios y oportunidades de arbitraje.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.header("Análisis Comparativo de Mercados")
    
    st.markdown("""
    <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
    <p style="font-size: 16px; margin-bottom: 0;">
    Este módulo permite comparar precios entre diferentes mercados, identificar
    correlaciones y detectar oportunidades de arbitraje en la comercialización
    de productos agrícolas.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or 'Mercado' not in df.columns or 'Fecha' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para realizar el análisis comparativo.")
        return
    
    # Asegurarse que la fecha es datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Obtener lista de productos
    productos = sorted(df['Producto'].unique())
    
    if len(productos) == 0:
        st.warning("No hay productos disponibles en los datos.")
        return
    
    # Determinar columna de precio a usar
    precio_col = None
    if 'Precio_Mayorista' in df.columns:
        precio_col = 'Precio_Mayorista'
    elif 'Precio_Minorista' in df.columns:
        precio_col = 'Precio_Minorista'
    else:
        st.warning("No hay columnas de precio disponibles para el análisis.")
        return
    
    # Interfaz de usuario para selección
    producto_seleccionado = st.selectbox(
        "Seleccione un producto:",
        productos,
        key="producto_comparacion"
    )
    
    # Filtrar datos por producto
    df_producto = df[df['Producto'] == producto_seleccionado].copy()
    
    if len(df_producto) == 0:
        st.warning(f"No hay datos disponibles para {producto_seleccionado}.")
        return
    
    # Verificar que haya mercados disponibles
    mercados = df_producto['Mercado'].unique()
    
    if len(mercados) < 2:
        st.warning("Se necesitan al menos dos mercados diferentes para realizar la comparación.")
        return
    
    # 1. Visualización de precios por mercado
    st.subheader("Comparación de Precios por Mercado")
    
    # Convertir la columna de precio a numérico, forzando los errores a NaN
    df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
    
    # Eliminar filas con valores NaN en la columna de precio
    df_producto = df_producto.dropna(subset=[precio_col])
    
    # Verificar si aún hay datos después de la limpieza
    if len(df_producto) == 0:
        st.warning(f"No hay datos numéricos válidos para {producto_seleccionado}.")
        return
    
    # Agrupar datos por mercado y fecha
    df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
    df_mercado_fecha = cached_groupby_mean(df_producto, ['Mercado', 'Fecha'], precio_col)
    
    # Crear gráfico de líneas
    fig = px.line(
        df_mercado_fecha,
        x='Fecha',
        y=precio_col,
        color='Mercado',
        title=f"Precios de {producto_seleccionado} por Mercado",
        labels={precio_col: 'Precio (RD$)', 'Fecha': 'Fecha'}
    )
    
    # Actualizar diseño
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Precio (RD$)",
        hovermode="x unified",
        legend_title="Mercado"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Diferencial de precios entre mercados
    st.subheader("Diferencial de Precios entre Mercados")
    
    # Calcular precio promedio por mercado
    df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
    precio_promedio = cached_groupby_mean(df_producto, 'Mercado', precio_col)
    precio_promedio = precio_promedio.sort_values(precio_col, ascending=False)
    
    # Calcular diferencial y oportunidad de arbitraje
    if len(precio_promedio) >= 2:
        mercado_alto = precio_promedio.iloc[0]
        mercado_bajo = precio_promedio.iloc[-1]
        
        diferencial = mercado_alto[precio_col] - mercado_bajo[precio_col]
        if mercado_bajo[precio_col] != 0:
            diferencial_pct = (diferencial / mercado_bajo[precio_col]) * 100
        else:
            diferencial_pct = np.nan
        
        # Crear gráfico de barras
        fig = px.bar(
            precio_promedio,
            x='Mercado',
            y=precio_col,
            color=precio_col,
            color_continuous_scale='Viridis',
            title=f"Precio Promedio de {producto_seleccionado} por Mercado",
            labels={precio_col: 'Precio Promedio (RD$)', 'Mercado': 'Mercado'}
        )
        
        # Añadir línea para el precio promedio general
        precio_promedio_general = precio_promedio[precio_col].mean()
        fig.add_hline(
            y=precio_promedio_general,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Promedio General: RD$ {precio_promedio_general:.2f}",
            annotation_position="top right"
        )
        
        # Actualizar diseño
        fig.update_layout(
            xaxis_title="Mercado",
            yaxis_title="Precio Promedio (RD$)",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar análisis de diferencial
        st.markdown(f"""
        ### Análisis de Diferencial de Precios
        
        * El mercado con precios más altos para **{producto_seleccionado}** es **{mercado_alto['Mercado']}**
          con un precio promedio de **RD$ {mercado_alto[precio_col]:.2f}**.
          
        * El mercado con precios más bajos es **{mercado_bajo['Mercado']}**
          con un precio promedio de **RD$ {mercado_bajo[precio_col]:.2f}**.
          
        * Esto representa una diferencia de **RD$ {diferencial:.2f}** o un **{diferencial_pct:.1f}%**.
        
        * Esta diferencia podría representar una oportunidad de arbitraje, comprando en {mercado_bajo['Mercado']}
          y vendiendo en {mercado_alto['Mercado']}, siempre que los costos de transporte y logística sean menores
          que el diferencial de precios.
        """)
        
        # 3. Crear tabla con todos los diferenciales entre mercados
        st.subheader("Matriz de Diferenciales entre Mercados")
        
        mercados_lista = precio_promedio['Mercado'].tolist()
        
        # Limitar la matriz a los 20 mercados principales para evitar sobrecarga
        mercados_lista = mercados_lista[:20]
        
        # Crear DataFrame para matriz de diferencias
        matriz_diferencias = pd.DataFrame(index=mercados_lista, columns=mercados_lista)
        
        # Llenar matriz
        for i, mercado_i in enumerate(mercados_lista):
            precio_i = precio_promedio[precio_promedio['Mercado'] == mercado_i][precio_col].values[0]
            for j, mercado_j in enumerate(mercados_lista):
                precio_j = precio_promedio[precio_promedio['Mercado'] == mercado_j][precio_col].values[0]
                # Calcular diferencia
                diferencia = precio_i - precio_j
                matriz_diferencias.loc[mercado_i, mercado_j] = diferencia
        
        # Crear heatmap de diferencias
        fig = px.imshow(
            matriz_diferencias,
            labels=dict(x="Mercado Destino", y="Mercado Origen", color="Diferencial (RD$)"),
            x=mercados_lista,
            y=mercados_lista,
            color_continuous_scale='RdBu',
            title=f"Matriz de Diferenciales de Precios para {producto_seleccionado}"
        )
        
        # Añadir anotaciones
        for i, mercado_i in enumerate(mercados_lista):
            for j, mercado_j in enumerate(mercados_lista):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{matriz_diferencias.loc[mercado_i, mercado_j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(matriz_diferencias.loc[mercado_i, mercado_j]) < 10 else "white")
                )
        
        # Actualizar diseño
        fig.update_layout(
            height=500,
            coloraxis_colorbar=dict(title="Diferencial (RD$)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interpretación de la Matriz de Diferenciales
        
        * Los valores **positivos** (azul) indican que el mercado de origen tiene precios **más altos**
          que el mercado destino.
          
        * Los valores **negativos** (rojo) indican que el mercado de origen tiene precios **más bajos**
          que el mercado destino.
          
        * Las oportunidades de arbitraje se identifican al vender en mercados donde los valores son **negativos**
          (comprando en el origen y vendiendo en el destino).
        """)
    else:
        st.info("Se necesitan al menos dos mercados con datos de precios para mostrar diferenciales.")
    
    # 4. Análisis de correlación entre mercados
    if len(mercados) >= 3:  # Solo mostrar si hay al menos 3 mercados
        st.subheader("Correlación entre Mercados")
        
        # Crear DataFrame pivotado con precios por mercado y fecha
        df_pivot = cached_pivot(df_mercado_fecha, 'Fecha', 'Mercado', precio_col)
        
        # Limitar la matriz de correlación a los 20 mercados principales
        top_cols = list(df_pivot.columns)[:20]
        df_pivot = df_pivot[top_cols]
        corr_matrix = df_pivot.corr()
        
        # Crear heatmap de correlación
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Mercado", y="Mercado", color="Correlación"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='Viridis',
            title=f"Correlación de Precios entre Mercados para {producto_seleccionado}"
        )
        
        # Añadir anotaciones
        for i, mercado_i in enumerate(corr_matrix.columns):
            for j, mercado_j in enumerate(corr_matrix.columns):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{corr_matrix.loc[mercado_i, mercado_j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if corr_matrix.loc[mercado_i, mercado_j] < 0.7 else "white")
                )
        
        # Actualizar diseño
        fig.update_layout(
            height=500,
            coloraxis_colorbar=dict(title="Correlación")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interpretación de la Correlación
        
        * Una correlación cercana a **1.0** indica que los precios en ambos mercados se mueven
          juntos en la misma dirección y magnitud.
          
        * Una correlación cercana a **0** indica que no hay relación entre los movimientos de precios.
          
        * Correlaciones bajas entre mercados sugieren oportunidades de diversificación para comerciantes
          y distribuidores.
        """)

def analisis_estacional_avanzado(df: pd.DataFrame) -> None:
    """
    Implementa análisis estacional avanzado para detectar patrones
    y ciclos en los precios a lo largo del tiempo.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.header("Análisis Estacional Avanzado")
    
    st.markdown("""
    <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
    <p style="font-size: 16px; margin-bottom: 0;">
    Este módulo permite identificar patrones estacionales en los precios agrícolas,
    detectando ciclos mensuales, trimestrales y anuales para optimizar decisiones
    de producción y comercialización.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or 'Fecha' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para realizar el análisis estacional.")
        return
    
    # Asegurarse que la fecha es datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Obtener lista de productos
    productos = sorted(df['Producto'].unique())
    
    if len(productos) == 0:
        st.warning("No hay productos disponibles en los datos.")
        return
    
    # Determinar columna de precio a usar
    precio_col = None
    if 'Precio_Mayorista' in df.columns:
        precio_col = 'Precio_Mayorista'
    elif 'Precio_Minorista' in df.columns:
        precio_col = 'Precio_Minorista'
    else:
        st.warning("No hay columnas de precio disponibles para el análisis.")
        return
    
    # Interfaz de usuario para selección
    col1, col2 = st.columns(2)
    
    with col1:
        producto_seleccionado = st.selectbox(
            "Seleccione un producto:",
            productos,
            key="producto_estacional"
        )
    
    with col2:
        if 'Mercado' in df.columns:
            mercados = ['Todos'] + sorted(df['Mercado'].unique())
            
            mercado_seleccionado = st.selectbox(
                "Seleccione un mercado (opcional):",
                mercados,
                key="mercado_estacional"
            )
        else:
            mercado_seleccionado = None
            st.info("No hay datos de mercados disponibles para filtrar.")
    
    # Filtrar datos por producto y mercado
    df_producto = df[df['Producto'] == producto_seleccionado].copy()
    
    if mercado_seleccionado != 'Todos' and mercado_seleccionado is not None:
        df_producto = df_producto[df_producto['Mercado'] == mercado_seleccionado]
    
    if len(df_producto) == 0:
        st.warning(f"No hay datos disponibles para {producto_seleccionado}.")
        return
    
    # Verificar que la columna de precio seleccionada existe
    if precio_col not in df_producto.columns:
        st.warning(f"No hay datos de {precio_col} disponibles para este producto.")
        return
    
    # Extraer componentes temporales
    df_producto['Año'] = df_producto['Fecha'].dt.year
    df_producto['Mes'] = df_producto['Fecha'].dt.month
    df_producto['Trimestre'] = df_producto['Fecha'].dt.quarter
    df_producto['DiaSemana'] = df_producto['Fecha'].dt.dayofweek + 1  # 1 (lunes) a 7 (domingo)
    
    # 1. Análisis por mes
    st.subheader("Análisis de Precios por Mes")
    
    # Asegurar que la columna de precios sea numérica
    df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
    # Calcular estadísticas por mes
    precio_mensual = df_producto.groupby('Mes')[precio_col].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    # (No se cachea aquí porque depende de múltiples estadísticas y es menos crítico)
    
    # Añadir nombres de los meses
    nombres_meses = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    precio_mensual['Nombre_Mes'] = precio_mensual['Mes'].map(nombres_meses)
    
    # Ordenar por mes
    precio_mensual = precio_mensual.sort_values('Mes')
    
    # Crear gráfico
    fig = go.Figure()
    
    # Añadir línea para precio promedio
    fig.add_trace(go.Scatter(
        x=precio_mensual['Nombre_Mes'],
        y=precio_mensual['mean'],
        mode='lines+markers',
        name='Precio Promedio',
        line=dict(color='#5a8f7b', width=2),
        marker=dict(size=8)
    ))
    
    # Añadir rango de precio mínimo a máximo como área sombreada
    fig.add_trace(go.Scatter(
        x=precio_mensual['Nombre_Mes'],
        y=precio_mensual['max'],
        mode='lines',
        name='Máximo',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=precio_mensual['Nombre_Mes'],
        y=precio_mensual['min'],
        mode='lines',
        name='Rango Min-Max',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(90, 143, 123, 0.2)'
    ))
    
    # Actualizar diseño
    fig.update_layout(
        title=f"Patrón Estacional Mensual para {producto_seleccionado}",
        xaxis_title="Mes",
        yaxis_title="Precio (RD$)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identificar meses con precios más altos y más bajos
    mes_max = precio_mensual.loc[precio_mensual['mean'].idxmax()]
    mes_min = precio_mensual.loc[precio_mensual['mean'].idxmin()]
    
    # Calcular diferencia porcentual entre meses extremos
    if mes_min['mean'] != 0:
        dif_pct = ((mes_max['mean'] - mes_min['mean']) / mes_min['mean']) * 100
    else:
        dif_pct = np.nan
    
    st.markdown(f"""
    ### Interpretación del Patrón Mensual
    
    * Los precios de **{producto_seleccionado}** alcanzan su **punto máximo** en **{mes_max['Nombre_Mes']}**
      con un precio promedio de **RD$ {mes_max['mean']:.2f}**.
      
    * Los precios más **bajos** se observan en **{mes_min['Nombre_Mes']}**
      con un precio promedio de **RD$ {mes_min['mean']:.2f}**.
      
    * Existe una variación de **{dif_pct:.1f}%** entre el mes con precios más altos y más bajos.
    
    * La volatilidad (desviación estándar) es mayor en {precio_mensual.loc[precio_mensual['std'].idxmax()]['Nombre_Mes']},
      lo que indica mayor incertidumbre en los precios durante este mes.
    """)
    
    # 2. Análisis por trimestre
    st.subheader("Análisis de Precios por Trimestre")
    
    # Calcular estadísticas por trimestre
    df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
    precio_trimestral = df_producto.groupby('Trimestre')[precio_col].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
    
    # Añadir nombres de los trimestres
    nombres_trimestres = {
        1: 'T1 (Ene-Mar)',
        2: 'T2 (Abr-Jun)',
        3: 'T3 (Jul-Sep)',
        4: 'T4 (Oct-Dic)'
    }
    precio_trimestral['Nombre_Trimestre'] = precio_trimestral['Trimestre'].map(nombres_trimestres)
    
    # Ordenar por trimestre
    precio_trimestral = precio_trimestral.sort_values('Trimestre')
    
    # Crear gráfico de barras con barras de error
    fig = go.Figure()
    
    # Añadir barras para precio promedio
    fig.add_trace(go.Bar(
        x=precio_trimestral['Nombre_Trimestre'],
        y=precio_trimestral['mean'],
        name='Precio Promedio',
        marker_color='#5a8f7b',
        error_y=dict(
            type='data',
            array=precio_trimestral['std'],
            visible=True
        )
    ))
    
    # Actualizar diseño
    fig.update_layout(
        title=f"Precios Trimestrales para {producto_seleccionado}",
        xaxis_title="Trimestre",
        yaxis_title="Precio Promedio (RD$)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identificar trimestres con precios más altos y más bajos
    trimestre_max = precio_trimestral.loc[precio_trimestral['mean'].idxmax()]
    trimestre_min = precio_trimestral.loc[precio_trimestral['mean'].idxmin()]
    
    # Calcular diferencia porcentual entre trimestres extremos
    if trimestre_min['mean'] != 0:
        dif_pct_trimestre = ((trimestre_max['mean'] - trimestre_min['mean']) / trimestre_min['mean']) * 100
    else:
        dif_pct_trimestre = np.nan
    
    st.markdown(f"""
    ### Interpretación del Patrón Trimestral
    
    * Los precios de **{producto_seleccionado}** son más altos durante el **{trimestre_max['Nombre_Trimestre']}**
      con un precio promedio de **RD$ {trimestre_max['mean']:.2f}**.
      
    * Los precios más bajos se observan durante el **{trimestre_min['Nombre_Trimestre']}**
      con un precio promedio de **RD$ {trimestre_min['mean']:.2f}**.
      
    * La diferencia entre el trimestre más alto y más bajo es de **{dif_pct_trimestre:.1f}%**.
    """)
    
    # 3. Índice estacional (si hay datos de múltiples años)
    años_unicos = df_producto['Año'].nunique()
    
    if años_unicos > 1:
        st.subheader("Índice Estacional por Mes y Año")
        
        # Calcular precio promedio mensual por año
        df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
        precio_mensual_año = cached_groupby_mean(df_producto, ['Año', 'Mes'], precio_col)
        
        # Crear pivot table con años como columnas y meses como filas
        pivot_mensual = cached_pivot(precio_mensual_año, 'Mes', 'Año', precio_col)
        
        # Añadir nombres de los meses
        pivot_mensual['Nombre_Mes'] = pivot_mensual.index.map(nombres_meses)
        
        # Crear heatmap de precios por mes y año
        fig = px.imshow(
            pivot_mensual.drop('Nombre_Mes', axis=1),
            labels=dict(x="Año", y="Mes", color="Precio (RD$)"),
            x=pivot_mensual.columns.drop('Nombre_Mes'),
            y=[nombres_meses[m] for m in pivot_mensual.index],
            color_continuous_scale='Viridis',
            title=f"Precios Mensuales de {producto_seleccionado} por Año"
        )
        
        # Actualizar diseño
        fig.update_layout(
            height=500,
            coloraxis_colorbar=dict(title="Precio (RD$)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interpretación del Mapa de Calor
        
        * Este mapa de calor muestra cómo varían los precios por mes a lo largo de diferentes años.
        * Los colores más intensos (verde oscuro) indican precios más altos.
        * Observe patrones que se repiten en los mismos meses a través de diferentes años para identificar
          estacionalidad consistente.
        * También busque anomalías o años atípicos donde los patrones estacionales fueron diferentes.
        """)
    
    # 4. Análisis por día de la semana (si hay datos diarios)
    dias_unicos = df_producto['DiaSemana'].nunique()
    
    if dias_unicos > 3:  # Si hay datos de al menos 4 días diferentes de la semana
        st.subheader("Análisis por Día de la Semana")
        
        # Calcular estadísticas por día de la semana
        df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
        precio_diario = df_producto.groupby('DiaSemana')[precio_col].agg(['mean', 'median', 'std']).reset_index()
        
        # Añadir nombres de los días
        nombres_dias = {
            1: 'Lunes', 2: 'Martes', 3: 'Miércoles', 4: 'Jueves',
            5: 'Viernes', 6: 'Sábado', 7: 'Domingo'
        }
        precio_diario['Nombre_Dia'] = precio_diario['DiaSemana'].map(nombres_dias)
        
        # Ordenar por día de la semana
        precio_diario = precio_diario.sort_values('DiaSemana')
        
        # Crear gráfico
        fig = px.bar(
            precio_diario,
            x='Nombre_Dia',
            y='mean',
            error_y='std',
            title=f"Precios de {producto_seleccionado} por Día de la Semana",
            labels={'mean': 'Precio Promedio (RD$)', 'Nombre_Dia': 'Día de la Semana'},
            color='mean',
            color_continuous_scale='Viridis'
        )
        
        # Actualizar diseño
        fig.update_layout(
            xaxis_title="Día de la Semana",
            yaxis_title="Precio Promedio (RD$)",
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Identificar días con precios más altos y más bajos
        dia_max = precio_diario.loc[precio_diario['mean'].idxmax()]
        dia_min = precio_diario.loc[precio_diario['mean'].idxmin()]
        
        # Calcular diferencia porcentual entre días extremos
        if dia_min['mean'] != 0:
            dif_pct_dia = ((dia_max['mean'] - dia_min['mean']) / dia_min['mean']) * 100
        else:
            dif_pct_dia = np.nan
        
        st.markdown(f"""
        ### Interpretación del Patrón Semanal
        
        * Los precios de **{producto_seleccionado}** son más altos los **{dia_max['Nombre_Dia']}**
          con un precio promedio de **RD$ {dia_max['mean']:.2f}**.
          
        * Los precios más bajos se observan los **{dia_min['Nombre_Dia']}**
          con un precio promedio de **RD$ {dia_min['mean']:.2f}**.
          
        * La diferencia entre el día más alto y más bajo es de **{dif_pct_dia:.1f}%**.
        
        * Esta información puede ser útil para planificar los días de venta o compra
          para maximizar ganancias o minimizar costos.
        """)

def mostrar_dashboard_profesional(df: pd.DataFrame) -> None:
    """
    Función principal que muestra el dashboard para profesionales
    integrando todos los análisis avanzados.
    
    Args:
        df: DataFrame con datos filtrados
    """
    # Crear subpestañas para organizar el contenido 
    pestanas = [
        "📈 Análisis de Tendencias", 
        "🔄 Comparación de Mercados", 
        "🗓️ Análisis Estacional",
        "📊 Estadísticas Avanzadas",
        "🔮 Pronóstico de Precios",
        "🔍 Búsqueda Avanzada",
        "📊 Datos Filtrados"
    ]
    
    # Migración a st.tabs para pestañas principales
    tabs = st.tabs(pestanas)
    with tabs[0]:
        analisis_tendencias_avanzado(df)
    with tabs[1]:
        analisis_comparativo_mercados(df)
    with tabs[2]:
        analisis_estacional_avanzado(df)
    with tabs[3]:
        from src.optimized_outlier_detection import show_statistical_analysis
        show_statistical_analysis(df)
    with tabs[4]:
        from src.price_forecasting import show_forecast_dashboard
        show_forecast_dashboard(df)
    with tabs[5]:
        from src.search_tool import render_search_ui
        render_search_ui(df)
    with tabs[6]:
        from src.datos_filtrados import mostrar_datos_filtrados
        mostrar_datos_filtrados(df)

# Función auxiliar para crear subplots
def make_subplots(rows=1, cols=1, subplot_titles=None):
    """
    Crear una figura de Plotly con subplots
    
    Args:
        rows (int): Número de filas
        cols (int): Número de columnas
        subplot_titles (list): Lista de títulos para los subplots
        
    Returns:
        go.Figure: Figura de Plotly con subplots
    """
    from plotly.subplots import make_subplots as plotly_make_subplots
    return plotly_make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
