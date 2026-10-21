"""
Módulo de detección de outliers y análisis estadístico para AgroDashRD.

Este módulo implementa algoritmos para detección de valores atípicos
y análisis estadístico avanzado de precios agrícolas.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """
    Detecta outliers en una columna numérica usando diferentes métodos.
    
    Args:
        df: DataFrame con datos
        column: Nombre de la columna para analizar
        method: Método de detección ('iqr', 'zscore', 'modified_zscore')
        threshold: Umbral para considerar outlier
        
    Returns:
        Serie booleana donde True indica outlier
    """
    # Asegurar que la columna sea numérica
    df[column] = pd.to_numeric(df[column], errors='coerce')
    if method == 'iqr':
        # Método IQR (Rango Intercuartil)
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        # Método Z-Score
        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        return abs(z_scores) > threshold
    
    elif method == 'modified_zscore':
        # Método Z-Score Modificado (más robusto a outliers)
        median = df[column].median()
        mad = np.median(np.abs(df[column] - median))
        modified_z_scores = 0.6745 * (df[column] - median) / mad
        return abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Método '{method}' no soportado.")

def analyze_price_distribution(df: pd.DataFrame, precio_col: str, producto: Optional[str] = None) -> None:
    """
    Analiza la distribución de precios y genera visualizaciones.
    
    Args:
        df: DataFrame con datos
        precio_col: Nombre de la columna de precio
        producto: Producto específico para analizar (opcional)
    """
    st.subheader("Distribución de Precios")
    
    # Filtrar por producto si se especifica
    if producto is not None:
        df = df[df['Producto'] == producto].copy()
    
    if len(df) == 0:
        st.warning("No hay datos suficientes para el análisis.")
        return
    
    # Asegurar que la columna de precios sea numérica
    df[precio_col] = pd.to_numeric(df[precio_col], errors='coerce')
    # Estadísticas descriptivas
    stats = df[precio_col].describe()
    
    # Crear columnas para mostrar estadísticas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Promedio", f"RD$ {stats['mean']:.2f}")
    
    with col2:
        st.metric("Mediana", f"RD$ {stats['50%']:.2f}")
    
    with col3:
        st.metric("Mínimo", f"RD$ {stats['min']:.2f}")
    
    with col4:
        st.metric("Máximo", f"RD$ {stats['max']:.2f}")
    
    # Histograma de precios
    fig = px.histogram(
        df, 
        x=precio_col,
        nbins=30,
        title=f"Distribución de Precios {'de ' + producto if producto else ''}",
        labels={precio_col: 'Precio (RD$)'},
        color_discrete_sequence=['#5a8f7b']
    )
    
    # Añadir línea vertical para el promedio
    fig.add_vline(
        x=stats['mean'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Promedio: RD$ {stats['mean']:.2f}",
        annotation_position="top right"
    )
    
    # Añadir línea vertical para la mediana
    fig.add_vline(
        x=stats['50%'],
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mediana: RD$ {stats['50%']:.2f}",
        annotation_position="top left"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plot para visualizar outliers
    fig = px.box(
        df,
        y=precio_col,
        title=f"Box Plot de Precios {'de ' + producto if producto else ''}",
        labels={precio_col: 'Precio (RD$)'},
        color_discrete_sequence=['#5a8f7b']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detectar outliers
    outliers = detect_outliers(df, precio_col, method='iqr', threshold=1.5)
    outlier_count = outliers.sum()
    
    st.markdown(f"""
    ### Análisis de Outliers
    
    Se detectaron **{outlier_count}** valores atípicos de un total de **{len(df)}** registros,
    lo que representa un **{(outlier_count / len(df) * 100):.2f}%** de los datos.
    
    Los outliers pueden ser:
    - Errores de entrada de datos
    - Eventos excepcionales de mercado
    - Productos de calidad significativamente diferente
    """)
    
    if outlier_count > 0:
        # Mostrar estadísticas sin outliers
        df_no_outliers = df[~outliers]
        stats_no_outliers = df_no_outliers[precio_col].describe()
        
        # Comparar estadísticas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Con Outliers")
            st.write(f"Promedio: RD$ {stats['mean']:.2f}")
            st.write(f"Desviación Estándar: RD$ {stats['std']:.2f}")
            st.write(f"Mínimo: RD$ {stats['min']:.2f}")
            st.write(f"Máximo: RD$ {stats['max']:.2f}")
        
        with col2:
            st.subheader("Sin Outliers")
            st.write(f"Promedio: RD$ {stats_no_outliers['mean']:.2f}")
            st.write(f"Desviación Estándar: RD$ {stats_no_outliers['std']:.2f}")
            st.write(f"Mínimo: RD$ {stats_no_outliers['min']:.2f}")
            st.write(f"Máximo: RD$ {stats_no_outliers['max']:.2f}")
        
        # Histograma comparativo
        fig = go.Figure()
        
        # Histograma con outliers
        fig.add_trace(go.Histogram(
            x=df[precio_col],
            name='Con Outliers',
            opacity=0.5,
            marker_color='#5a8f7b'
        ))
        
        # Histograma sin outliers
        fig.add_trace(go.Histogram(
            x=df_no_outliers[precio_col],
            name='Sin Outliers',
            opacity=0.5,
            marker_color='#457b9d'
        ))
        
        # Actualizar diseño
        fig.update_layout(
            title=f"Comparación de Distribución: Con vs. Sin Outliers",
            xaxis_title="Precio (RD$)",
            yaxis_title="Frecuencia",
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def analyze_price_correlation(df: pd.DataFrame) -> None:
    """
    Analiza correlaciones entre precios de diferentes productos.
    
    Args:
        df: DataFrame con datos
    """
    st.subheader("Correlación entre Precios de Productos")
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or 'Fecha' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para el análisis de correlación.")
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
    
    # Obtener lista de productos
    productos = sorted(df['Producto'].unique())
    
    if len(productos) < 2:
        st.warning("Se necesitan al menos dos productos diferentes para calcular correlaciones.")
        return
    
    # Seleccionar productos para comparar
    productos_seleccionados = st.multiselect(
        "Seleccione productos para analizar correlación:",
        productos,
        default=productos[:min(5, len(productos))],
        key="productos_correlacion"
    )
    
    if len(productos_seleccionados) < 2:
        st.warning("Seleccione al menos dos productos para el análisis.")
        return
    
    # Crear DataFrame pivotado con precios promedio por fecha y producto
    df_pivot = df[df['Producto'].isin(productos_seleccionados)].copy()
    
    # Asegurarse que la fecha es datetime
    if not pd.api.types.is_datetime64_dtype(df_pivot['Fecha']):
        df_pivot['Fecha'] = pd.to_datetime(df_pivot['Fecha'])
    
    # Asegurar que la columna de precios sea numérica
    df_pivot[precio_col] = pd.to_numeric(df_pivot[precio_col], errors='coerce')
    # Agrupar por fecha y producto
    df_precio_diario = df_pivot.groupby(['Fecha', 'Producto'])[precio_col].mean().reset_index()
    
    # Crear pivot table
    matriz_precios = df_precio_diario.pivot(index='Fecha', columns='Producto', values=precio_col)
    
    # Calcular matriz de correlación
    corr_matrix = matriz_precios.corr()
    
    # Crear heatmap de correlación
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        labels=dict(x="Producto", y="Producto", color="Correlación"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu',
        range_color=[-1, 1],
        title="Matriz de Correlación entre Precios de Productos"
    )
    
    # Actualizar diseño
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(title="Correlación")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Interpretación de la Correlación
    
    * Una correlación cercana a **1.0** (azul oscuro) indica que los precios de ambos productos tienden a moverse juntos.
    * Una correlación cercana a **-1.0** (rojo oscuro) indica que los precios se mueven en direcciones opuestas.
    * Una correlación cercana a **0** (blanco) indica que no hay relación entre los movimientos de precios.
    
    Algunas aplicaciones prácticas:
    
    * Productos con alta correlación positiva pueden tener factores de producción similares o ser sustitutos.
    * Productos con correlación negativa pueden ser complementarios o tener ciclos de producción opuestos.
    * Esta información puede ser útil para diversificar cultivos y reducir riesgos.
    """)
    
    # Scatter plots para pares de productos con mayor correlación
    st.subheader("Gráficos de Dispersión para Pares de Productos")
    
    # Encontrar pares con mayor correlación absoluta
    corr_pares = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            prod1 = corr_matrix.columns[i]
            prod2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            corr_pares.append((prod1, prod2, corr_val, abs(corr_val)))
    
    # Ordenar por correlación absoluta
    corr_pares.sort(key=lambda x: x[3], reverse=True)
    
    # Mostrar los 3 pares con mayor correlación
    for i, (prod1, prod2, corr_val, _) in enumerate(corr_pares[:3]):
        # Crear scatter plot
        fig = px.scatter(
            x=matriz_precios[prod1],
            y=matriz_precios[prod2],
            trendline="ols",
            labels={
                "x": f"Precio de {prod1} (RD$)",
                "y": f"Precio de {prod2} (RD$)"
            },
            title=f"Correlación entre {prod1} y {prod2} (r = {corr_val:.2f})"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_statistical_analysis(df: pd.DataFrame) -> None:
    """
    Muestra análisis estadísticos avanzados para los datos.
    
    Args:
        df: DataFrame con datos
    """
    st.header("Análisis Estadístico Avanzado")
    
    st.markdown("""
    <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
    <p style="font-size: 16px; margin-bottom: 0;">
    Este módulo proporciona análisis estadísticos detallados de los datos de precios,
    incluyendo distribuciones, outliers, correlaciones y otras métricas para
    mejorar la comprensión de los patrones de mercado.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para el análisis estadístico.")
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
    
    # Obtener lista de productos y rubros
    productos = sorted(df['Producto'].unique())
    rubros = sorted(df['Rubro'].unique()) if 'Rubro' in df.columns else []
    
    if len(productos) == 0:
        st.warning("No hay productos disponibles en los datos.")
        return
    
    # Crear pestañas para diferentes análisis
    analisis_tabs = st.tabs([
        "Distribución de Precios", 
        "Correlación entre Productos", 
        "Detección de Outliers"
    ])
    
    # Tab 1: Distribución de precios
    with analisis_tabs[0]:
        # Selector de producto
        producto_seleccionado = st.selectbox(
            "Seleccione un producto:",
            ['Todos los productos'] + productos,
            key="producto_distribucion"
        )
        
        # Filtro de rubro si está disponible
        if rubros:
            rubro_seleccionado = st.selectbox(
                "Filtrar por rubro (opcional):",
                ['Todos los rubros'] + rubros,
                key="rubro_distribucion"
            )
            
            # Aplicar filtro de rubro si se seleccionó uno
            if rubro_seleccionado != 'Todos los rubros':
                df_filtrado = df[df['Rubro'] == rubro_seleccionado].copy()
                # Actualizar lista de productos para el rubro seleccionado
                productos_rubro = sorted(df_filtrado['Producto'].unique())
                if producto_seleccionado not in productos_rubro and producto_seleccionado != 'Todos los productos':
                    producto_seleccionado = 'Todos los productos'
            else:
                df_filtrado = df.copy()
        else:
            df_filtrado = df.copy()
        
        if producto_seleccionado == 'Todos los productos':
            analyze_price_distribution(df_filtrado, precio_col)
        else:
            analyze_price_distribution(df_filtrado, precio_col, producto_seleccionado)
    
    # Tab 2: Correlación entre productos
    with analisis_tabs[1]:
        analyze_price_correlation(df)
    
    # Tab 3: Detección de outliers
    with analisis_tabs[2]:
        st.subheader("Detección de Valores Atípicos (Outliers)")
        
        # Filtros en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Primero seleccionar el rubro para filtrar los productos
            if rubros:
                rubro_seleccionado = st.selectbox(
                    "Filtrar por rubro:",
                    ['Todos los rubros'] + rubros,
                    key="rubro_outlier"
                )
                
                # Filtrar productos según el rubro seleccionado
                if rubro_seleccionado != 'Todos los rubros':
                    productos_filtrados = sorted(df[df['Rubro'] == rubro_seleccionado]['Producto'].unique())
                else:
                    productos_filtrados = productos
            else:
                productos_filtrados = productos
                rubro_seleccionado = None
            
            # Selector múltiple de productos (filtrados por rubro)
            productos_seleccionados = st.multiselect(
                "Seleccione uno o más productos:",
                options=productos_filtrados,
                default=productos_filtrados[:1] if len(productos_filtrados) > 0 else [],
                key="productos_outlier"
            )
            
            # Si no se seleccionó ningún producto, usar todos los productos disponibles
            if not productos_seleccionados and productos_filtrados:
                productos_seleccionados = productos_filtrados
        
        with col2:
            # Selector de método de detección
            metodo_outlier = st.selectbox(
                "Método de detección:",
                ["Rango Intercuartil (IQR)", "Z-Score", "Z-Score Modificado"],
                key="metodo_outlier"
            )
        
        with col3:
            # Configurar parámetros según el método seleccionado
            if "IQR" in metodo_outlier:
                threshold = st.slider(
                    "Umbral IQR:",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    key="threshold_iqr"
                )
                metodo = 'iqr'
            elif "Z-Score" in metodo_outlier and "Modificado" not in metodo_outlier:
                threshold = st.slider(
                    "Umbral Z-Score:",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    key="threshold_zscore"
                )
                metodo = 'zscore'
            else:  # Z-Score Modificado
                threshold = st.slider(
                    "Umbral Z-Score Modificado:",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.5,
                    step=0.1,
                    key="threshold_modified_zscore"
                )
                metodo = 'modified_zscore'
        
        # Aplicar filtros
        df_filtrado = df.copy()
        
        # Aplicar filtro de rubro si se seleccionó uno
        if rubros and 'rubro_seleccionado' in locals() and rubro_seleccionado != 'Todos los rubros':
            df_filtrado = df_filtrado[df_filtrado['Rubro'] == rubro_seleccionado]
            # Actualizar lista de productos para el rubro seleccionado
            productos_disponibles = sorted(df_filtrado['Producto'].unique())
            # Filtrar productos seleccionados que estén en los disponibles
            productos_seleccionados = [p for p in productos_seleccionados if p in productos_disponibles]
            if not productos_seleccionados and productos_disponibles:
                productos_seleccionados = [productos_disponibles[0]]
        
        # Verificar si hay datos después de filtrar
        if df_filtrado.empty:
            st.warning("No hay datos disponibles con los filtros seleccionados.")
            return
        
        # Si no hay productos seleccionados, usar todos los productos disponibles
        if not productos_seleccionados and 'productos_filtrados' in locals() and productos_filtrados:
            productos_seleccionados = productos_filtrados
            
        # Filtrar por productos seleccionados
        df_productos = df_filtrado[df_filtrado['Producto'].isin(productos_seleccionados)]
        
        if df_productos.empty:
            st.warning("No hay datos disponibles con los filtros actuales.")
            return
        
        # Detectar outliers para cada producto
        df_productos = df_productos.copy()
        df_productos['Es Outlier'] = False
        
        # Procesar cada producto por separado
        for producto in productos_seleccionados:
            df_producto = df_productos[df_productos['Producto'] == producto].copy()
            
            if len(df_producto) < 5:  # Mínimo de datos necesarios
                continue
                
            # Detectar outliers usando el método seleccionado
            outliers = detect_outliers(df_producto, precio_col, method=metodo, threshold=threshold)
            df_productos.loc[df_producto.index, 'Es Outlier'] = outliers
        
        # Mostrar resumen
        total_registros = len(df_productos)
        total_outliers = df_productos['Es Outlier'].sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de registros", total_registros)
        with col2:
            st.metric("Outliers detectados", 
                     f"{total_outliers} ({total_outliers/max(1, total_registros)*100:.1f}%)")
            st.caption(f"Método: {metodo_outlier} - Umbral: {threshold}")
        
        # Visualización de outliers
        if 'Fecha' in df_productos.columns:
            # Asegurarse que la fecha es datetime
            if not pd.api.types.is_datetime64_dtype(df_productos['Fecha']):
                df_productos['Fecha'] = pd.to_datetime(df_productos['Fecha'])
            
            # Crear gráfico de dispersión por producto
            fig = px.scatter(
                df_productos,
                x='Fecha',
                y=precio_col,
                color='Producto',
                symbol='Es Outlier',
                symbol_map={True: 'x', False: 'circle'},
                title=f"Detección de Outliers para {', '.join(productos_seleccionados)}",
                labels={precio_col: 'Precio (RD$)', 'Fecha': 'Fecha'},
                hover_data=['Mercado'] if 'Mercado' in df_productos.columns else None
            )
            
            # Mejorar la visualización de outliers
            fig.update_traces(
                marker=dict(
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey'),
                    opacity=0.8
                ),
                selector=dict(mode='markers')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar tabla de outliers si hay alguno
            if total_outliers > 0:
                st.subheader("Detalle de Valores Atípicos")
                
                # Seleccionar columnas relevantes
                cols_mostrar = ['Producto', 'Fecha', precio_col, 'Es Outlier']
                if 'Mercado' in df_productos.columns:
                    cols_mostrar.append('Mercado')
                if 'Rubro' in df_productos.columns:
                    cols_mostrar.append('Rubro')
                
                # Mostrar tabla con los outliers
                st.dataframe(
                    df_productos[df_productos['Es Outlier']][cols_mostrar]\
                        .sort_values([precio_col], ascending=False),
                    use_container_width=True
                )
        
        # Explicación del método seleccionado
        st.subheader(f"Explicación del Método {metodo_outlier}")
        
        if metodo == 'iqr':
            st.markdown(f"""
            **El método IQR (Rango Intercuartil)** identifica valores atípicos basándose en la distancia al primer y tercer cuartil:
            
            1. Calcula el primer cuartil (Q1) y el tercer cuartil (Q3)
            2. Calcula el Rango Intercuartil: IQR = Q3 - Q1
            3. Define límites: 
               - Límite inferior = Q1 - {threshold} × IQR
               - Límite superior = Q3 + {threshold} × IQR
            4. Cualquier valor fuera de estos límites se considera atípico
            
            **Ventajas:**
            - Robusto a la presencia de valores extremos
            - No asume una distribución normal de los datos
            - Fácil de interpretar y calcular
            
            **Consideraciones:**
            - El factor {threshold} se puede ajustar según el contexto (1.5 es el valor estándar)
            - Funciona mejor con distribuciones moderadamente sesgadas
            """)
        elif metodo == 'zscore':
            st.markdown(f"""
            **El método Z-Score** identifica valores atípicos basándose en la distancia a la media en términos de desviaciones estándar:
            
            1. Calcula la media (μ) y la desviación estándar (σ) de los datos
            2. Para cada punto, calcula el Z-Score: Z = (x - μ) / σ
            3. Cualquier valor con |Z| > {threshold} se considera atípico
            
            **Ventajas:**
            - Fácil de entender e implementar
            - Funciona bien con distribuciones normales
            
            **Consideraciones:**
            - Sensible a la presencia de valores atípicos en el cálculo de la media y desviación estándar
            - Asume que los datos siguen una distribución normal
            - El umbral típico es 3.0, pero se puede ajustar según las necesidades
            """)
        else:  # modified_zscore
            st.markdown(f"""
            **El método Z-Score Modificado** es una versión más robusta que usa la mediana y la desviación absoluta mediana (MAD):
            
            1. Calcula la mediana (med) y la desviación absoluta mediana (MAD) de los datos
            2. Para cada punto, calcula el Z-Score Modificado: Z = 0.6745 × (x - med) / MAD
            3. Cualquier valor con |Z| > {threshold} se considera atípico
            
            **Ventajas:**
            - Más robusto a valores atípicos que el Z-Score estándar
            - No se ve afectado por valores extremos en los datos
            - Funciona bien con distribuciones no normales
            
            **Consideraciones:**
            - Menos eficiente computacionalmente que el Z-Score estándar
            - El factor 0.6745 asume una distribución normal subyacente
            - El umbral típico es 3.5, pero se puede ajustar según las necesidades
            """)

if __name__ == "__main__":
    # Para pruebas
    import pandas as pd
    import numpy as np
    
    # Crear datos de ejemplo
    np.random.seed(42)
    datos = {
        'Fecha': pd.date_range(start='2020-01-01', periods=100),
        'Producto': np.random.choice(['Tomate', 'Plátano', 'Cebolla'], 100),
        'Precio_Mayorista': np.random.normal(50, 10, 100),
        'Mercado': np.random.choice(['Mercado A', 'Mercado B', 'Mercado C'], 100)
    }
    
    df = pd.DataFrame(datos)
    
    # Añadir algunos outliers
    df.loc[10, 'Precio_Mayorista'] = 150
    df.loc[50, 'Precio_Mayorista'] = 5
    
    # Prueba
    show_statistical_analysis(df)