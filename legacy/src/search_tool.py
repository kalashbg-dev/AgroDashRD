"""
Módulo de búsqueda avanzada para AgroDashRD.

Este módulo implementa funcionalidades de búsqueda que permiten
a los usuarios encontrar información específica en los datos.
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def search_products(df: pd.DataFrame, query: str, fuzzy: bool = True) -> pd.DataFrame:
    """
    Busca productos que coinciden con la consulta.
    
    Args:
        df: DataFrame con datos
        query: Texto a buscar
        fuzzy: Si es True, realiza búsqueda aproximada
        
    Returns:
        DataFrame con resultados filtrados
    """
    if not query or len(query.strip()) == 0:
        return df
    
    query = query.lower().strip()
    
    if 'Producto' not in df.columns:
        return df
    
    if fuzzy:
        # Búsqueda aproximada: verifica si el query es parte del nombre del producto
        mask = df['Producto'].str.lower().str.contains(query, na=False)
    else:
        # Búsqueda exacta
        mask = df['Producto'].str.lower() == query
    
    return df[mask]

def search_markets(df: pd.DataFrame, query: str, fuzzy: bool = True) -> pd.DataFrame:
    """
    Busca mercados que coinciden con la consulta.
    
    Args:
        df: DataFrame con datos
        query: Texto a buscar
        fuzzy: Si es True, realiza búsqueda aproximada
        
    Returns:
        DataFrame con resultados filtrados
    """
    if not query or len(query.strip()) == 0:
        return df
    
    query = query.lower().strip()
    
    if 'Mercado' not in df.columns:
        return df
    
    if fuzzy:
        # Búsqueda aproximada
        mask = df['Mercado'].str.lower().str.contains(query, na=False)
    else:
        # Búsqueda exacta
        mask = df['Mercado'].str.lower() == query
    
    return df[mask]

def search_price_range(df: pd.DataFrame, min_price: Optional[float] = None, 
                     max_price: Optional[float] = None, 
                     price_col: str = 'Precio_Mayorista') -> pd.DataFrame:
    """
    Filtra productos por rango de precios.
    
    Args:
        df: DataFrame con datos
        min_price: Precio mínimo
        max_price: Precio máximo
        price_col: Columna de precio a usar
        
    Returns:
        DataFrame con resultados filtrados
    """
    # Si no se proporciona ningún rango, devolver el DataFrame original
    if min_price is None and max_price is None:
        return df
    
    # Verificar si la columna de precio existe
    if price_col not in df.columns:
        # Intentar con otra columna de precio
        if 'Precio_Mayorista' in df.columns:
            price_col = 'Precio_Mayorista'
        elif 'Precio_Minorista' in df.columns:
            price_col = 'Precio_Minorista'
        else:
            # No hay columnas de precio disponibles
            return df
    
    # Aplicar filtros
    result_df = df.copy()
    
    if min_price is not None:
        result_df = result_df[result_df[price_col] >= min_price]
    
    if max_price is not None:
        result_df = result_df[result_df[price_col] <= max_price]
    
    return result_df

def find_price_trends(df: pd.DataFrame, days: int = 30, 
                    min_change_pct: Optional[float] = None,
                    price_col: str = 'Precio_Mayorista') -> pd.DataFrame:
    """
    Encuentra productos con tendencias de precios significativas.
    
    Args:
        df: DataFrame con datos
        days: Número de días para calcular la tendencia
        min_change_pct: Cambio porcentual mínimo para incluir
        price_col: Columna de precio a usar
        
    Returns:
        DataFrame con productos y sus cambios de precio
    """
    # Verificar columnas necesarias
    if 'Fecha' not in df.columns or 'Producto' not in df.columns or price_col not in df.columns:
        return pd.DataFrame()
    
    # Asegurarse que la fecha está en formato datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Calcular fecha límite
    latest_date = df['Fecha'].max()
    start_date = latest_date - pd.Timedelta(days=days)
    
    # Filtrar datos por período
    period_df = df[(df['Fecha'] >= start_date) & (df['Fecha'] <= latest_date)]
    
    if len(period_df) == 0:
        return pd.DataFrame()
    
    # Calcular precios promedio por producto y fecha
    product_prices = period_df.groupby(['Producto', 'Fecha'])[price_col].mean().reset_index()
    
    # Para cada producto, obtener el primer y último precio
    results = []
    for producto in product_prices['Producto'].unique():
        product_data = product_prices[product_prices['Producto'] == producto].sort_values('Fecha')
        
        if len(product_data) >= 2:  # Necesitamos al menos dos precios para calcular tendencia
            first_price = product_data[price_col].iloc[0]
            last_price = product_data[price_col].iloc[-1]
            
            # Calcular cambio
            change = last_price - first_price
            change_pct = (change / first_price) * 100 if first_price > 0 else 0
            
            # Agregar a resultados si cumple con el cambio mínimo
            if min_change_pct is None or abs(change_pct) >= min_change_pct:
                results.append({
                    'Producto': producto,
                    'Primer Precio': first_price,
                    'Último Precio': last_price,
                    'Cambio': change,
                    'Cambio (%)': change_pct,
                    'Tendencia': 'Subida' if change > 0 else 'Bajada' if change < 0 else 'Estable'
                })
    
    # Crear DataFrame con resultados
    result_df = pd.DataFrame(results)
    
    # Ordenar por cambio porcentual absoluto (descendente)
    if not result_df.empty:
        result_df['Cambio Abs (%)'] = result_df['Cambio (%)'].abs()
        result_df = result_df.sort_values('Cambio Abs (%)', ascending=False).drop('Cambio Abs (%)', axis=1)
    
    return result_df

def find_market_opportunities(df: pd.DataFrame, 
                            min_diff_pct: float = 10.0,
                            price_col: str = 'Precio_Mayorista') -> pd.DataFrame:
    """
    Encuentra oportunidades de arbitraje entre mercados para cada producto.
    
    Args:
        df: DataFrame con datos
        min_diff_pct: Diferencia porcentual mínima para considerar oportunidad
        price_col: Columna de precio a usar
        
    Returns:
        DataFrame con oportunidades de arbitraje
    """
    # Verificar columnas necesarias
    if 'Mercado' not in df.columns or 'Producto' not in df.columns or price_col not in df.columns:
        return pd.DataFrame()
    
    # Filtrar solo datos recientes (últimos 30 días)
    if 'Fecha' in df.columns:
        if not pd.api.types.is_datetime64_dtype(df['Fecha']):
            df['Fecha'] = pd.to_datetime(df['Fecha'])
        
        latest_date = df['Fecha'].max()
        start_date = latest_date - pd.Timedelta(days=30)
        df_recent = df[(df['Fecha'] >= start_date)]
    else:
        df_recent = df
    
    # Calcular precio promedio por producto y mercado
    market_prices = df_recent.groupby(['Producto', 'Mercado'])[price_col].mean().reset_index()
    
    # Encontrar oportunidades
    opportunities = []
    
    for producto in market_prices['Producto'].unique():
        product_data = market_prices[market_prices['Producto'] == producto]
        
        if len(product_data) >= 2:  # Necesitamos al menos dos mercados
            min_price = product_data[price_col].min()
            max_price = product_data[price_col].max()
            
            min_market = product_data.loc[product_data[price_col].idxmin(), 'Mercado']
            max_market = product_data.loc[product_data[price_col].idxmax(), 'Mercado']
            
            # Calcular diferencia
            diff = max_price - min_price
            diff_pct = (diff / min_price) * 100 if min_price > 0 else 0
            
            # Agregar a resultados si cumple con la diferencia mínima
            if diff_pct >= min_diff_pct:
                opportunities.append({
                    'Producto': producto,
                    'Mercado Compra': min_market,
                    'Precio Compra': min_price,
                    'Mercado Venta': max_market,
                    'Precio Venta': max_price,
                    'Diferencia': diff,
                    'Diferencia (%)': diff_pct
                })
    
    # Crear DataFrame con resultados
    result_df = pd.DataFrame(opportunities)
    
    # Ordenar por diferencia porcentual (descendente)
    if not result_df.empty:
        result_df = result_df.sort_values('Diferencia (%)', ascending=False)
    
    return result_df

def find_seasonal_patterns(df: pd.DataFrame, 
                         producto: Optional[str] = None,
                         price_col: str = 'Precio_Mayorista') -> pd.DataFrame:
    """
    Identifica patrones estacionales para productos.
    
    Args:
        df: DataFrame con datos
        producto: Producto específico a analizar (opcional)
        price_col: Columna de precio a usar
        
    Returns:
        DataFrame con patrones estacionales por mes
    """
    # Verificar columnas necesarias
    if 'Fecha' not in df.columns or 'Producto' not in df.columns or price_col not in df.columns:
        return pd.DataFrame()
    
    # Filtrar por producto si se especifica
    if producto is not None:
        df = df[df['Producto'] == producto].copy()
    
    # Asegurarse que la fecha está en formato datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Extraer componentes de fecha
    df['Año'] = df['Fecha'].dt.year
    df['Mes'] = df['Fecha'].dt.month
    
    # Verificar que hay datos de múltiples años
    if df['Año'].nunique() < 2:
        return pd.DataFrame()
    
    # Calcular promedio por producto y mes
    seasonal_data = df.groupby(['Producto', 'Mes'])[price_col].mean().reset_index()
    
    # Calcular estadísticas por producto
    product_stats = []
    
    for product in seasonal_data['Producto'].unique():
        product_data = seasonal_data[seasonal_data['Producto'] == product]
        
        if len(product_data) >= 6:  # Necesitamos suficientes meses para analizar
            avg_price = product_data[price_col].mean()
            
            # Calcular variación estacional
            product_data['Índice Estacional'] = (product_data[price_col] / avg_price) * 100
            
            # Encontrar meses con precios altos y bajos
            high_months = product_data.nlargest(3, price_col)['Mes'].tolist()
            low_months = product_data.nsmallest(3, price_col)['Mes'].tolist()
            
            # Nombres de los meses
            month_names = {
                1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
            }
            
            high_months_names = [month_names[m] for m in high_months]
            low_months_names = [month_names[m] for m in low_months]
            
            # Calcular variabilidad estacional
            seasonality_strength = product_data['Índice Estacional'].std()
            
            product_stats.append({
                'Producto': product,
                'Meses Precios Altos': ', '.join(high_months_names),
                'Meses Precios Bajos': ', '.join(low_months_names),
                'Variabilidad Estacional': seasonality_strength,
                'Precio Promedio': avg_price
            })
    
    # Crear DataFrame con resultados
    result_df = pd.DataFrame(product_stats)
    
    # Ordenar por variabilidad estacional (descendente)
    if not result_df.empty:
        result_df = result_df.sort_values('Variabilidad Estacional', ascending=False)
    
    return result_df

def render_search_ui(df: pd.DataFrame) -> None:
    """
    Renderiza la interfaz de búsqueda avanzada.
    
    Args:
        df: DataFrame con datos completos
    """
    st.header("🔍 Búsqueda Avanzada")
    
    st.markdown("""
    <div class="fade-in">
    <p>Utilice las herramientas de búsqueda para encontrar información específica
    en los datos, descubrir tendencias de precios y oportunidades de mercado.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Crear pestañas para diferentes tipos de búsqueda
    search_tabs = st.tabs([
        "Búsqueda por Texto", 
        "Búsqueda por Precio", 
        "Tendencias", 
        "Oportunidades",
        "Patrones Estacionales"
    ])
    
    # Tab 1: Búsqueda por texto
    with search_tabs[0]:
        st.subheader("Búsqueda por Texto")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_type = st.radio(
                "Buscar en:",
                ["Productos", "Mercados", "Ambos"],
                key="search_text_type"
            )
        
        with col2:
            search_query = st.text_input(
                "Texto a buscar:",
                key="search_text_query"
            )
            
            fuzzy_search = st.checkbox(
                "Búsqueda aproximada",
                value=True,
                key="search_text_fuzzy"
            )
        
        if st.button("Buscar", key="btn_search_text", use_container_width=True):
            with st.spinner("Buscando..."):
                # Aplicar búsqueda según selección
                if search_type == "Productos" or search_type == "Ambos":
                    results = search_products(df, search_query, fuzzy_search)
                    
                    if search_type == "Ambos" and not results.empty:
                        # Si buscamos en ambos, filtrar también por mercados
                        results = search_markets(results, search_query, fuzzy_search)
                else:
                    results = search_markets(df, search_query, fuzzy_search)
                
                # Mostrar resultados
                if results.empty:
                    st.warning(f"No se encontraron resultados para '{search_query}'.")
                else:
                    st.success(f"Se encontraron {len(results)} registros.")
                    
                    # Mostrar tabla de resultados
                    if 'Producto' in results.columns and 'Mercado' in results.columns and 'Fecha' in results.columns:
                        # Agrupar por producto y mercado para obtener precios recientes
                        if 'Precio_Mayorista' in results.columns:
                            price_col = 'Precio_Mayorista'
                        elif 'Precio_Minorista' in results.columns:
                            price_col = 'Precio_Minorista'
                        else:
                            price_col = None
                        
                        if price_col:
                            # Obtener precio promedio más reciente
                            latest_date = results['Fecha'].max()
                            recent_results = results[results['Fecha'] == latest_date]
                            
                            summary = recent_results.groupby(['Producto', 'Mercado'])[price_col].mean().reset_index()
                            
                            # Formatear precios
                            summary['Precio'] = summary[price_col].apply(lambda x: f"RD$ {x:.2f}")
                            
                            # Mostrar tabla resumida
                            st.dataframe(
                                summary[['Producto', 'Mercado', 'Precio']],
                                use_container_width=True,
                                column_config={
                                    "Producto": st.column_config.TextColumn("Producto"),
                                    "Mercado": st.column_config.TextColumn("Mercado"),
                                    "Precio": st.column_config.TextColumn("Precio Reciente")
                                }
                            )
                        else:
                            # Mostrar resultados sin formateo especial
                            st.dataframe(results, use_container_width=True)
                    else:
                        # Mostrar resultados sin formateo especial
                        st.dataframe(results, use_container_width=True)
    
    # Tab 2: Búsqueda por rango de precio
    with search_tabs[1]:
        st.subheader("Búsqueda por Rango de Precio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Seleccionar tipo de precio
            price_type = st.radio(
                "Tipo de Precio:",
                ["Mayorista", "Minorista"],
                key="search_price_type",
                horizontal=True
            )
            
            price_col = f"Precio_{price_type}"
        
        with col2:
            # Rango de precios
            min_price = st.number_input(
                "Precio Mínimo (RD$):",
                min_value=0.0,
                value=0.0,
                step=10.0,
                key="search_min_price"
            )
        
        with col3:
            max_price = st.number_input(
                "Precio Máximo (RD$):",
                min_value=0.0,
                value=1000.0,
                step=10.0,
                key="search_max_price"
            )
        
        if st.button("Buscar por Precio", key="btn_search_price", use_container_width=True):
            with st.spinner("Buscando..."):
                # Aplicar filtro por rango de precios
                results = search_price_range(df, min_price, max_price, price_col)
                
                # Mostrar resultados
                if results.empty:
                    st.warning(f"No se encontraron productos en el rango de precios especificado.")
                else:
                    st.success(f"Se encontraron {len(results)} registros en el rango de precios.")
                    
                    # Agrupar por producto y calcular estadísticas
                    if 'Producto' in results.columns:
                        summary = results.groupby('Producto')[price_col].agg(['mean', 'min', 'max']).reset_index()
                        
                        # Formatear precios
                        summary['Precio Promedio'] = summary['mean'].apply(lambda x: f"RD$ {x:.2f}")
                        summary['Precio Mínimo'] = summary['min'].apply(lambda x: f"RD$ {x:.2f}")
                        summary['Precio Máximo'] = summary['max'].apply(lambda x: f"RD$ {x:.2f}")
                        
                        # Mostrar tabla resumida
                        st.dataframe(
                            summary[['Producto', 'Precio Promedio', 'Precio Mínimo', 'Precio Máximo']],
                            use_container_width=True
                        )
                    else:
                        # Mostrar resultados sin formateo especial
                        st.dataframe(results, use_container_width=True)
    
    # Tab 3: Tendencias de precios
    with search_tabs[2]:
        st.subheader("Análisis de Tendencias de Precios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seleccionar tipo de precio
            trend_price_type = st.radio(
                "Tipo de Precio:",
                ["Mayorista", "Minorista"],
                key="trend_price_type",
                horizontal=True
            )
            
            trend_price_col = f"Precio_{trend_price_type}"
        
        with col2:
            # Período de análisis
            trend_period = st.selectbox(
                "Período de Análisis:",
                ["Última semana", "Últimos 15 días", "Último mes", "Últimos 3 meses", "Último año"],
                index=2,
                key="trend_period"
            )
            
            # Convertir selección a días
            period_days = {
                "Última semana": 7,
                "Últimos 15 días": 15,
                "Último mes": 30,
                "Últimos 3 meses": 90,
                "Último año": 365
            }
            days = period_days[trend_period]
            
            # Cambio mínimo porcentual
            min_change = st.slider(
                "Cambio Mínimo (%):",
                min_value=0,
                max_value=50,
                value=5,
                step=1,
                key="trend_min_change"
            )
        
        if st.button("Analizar Tendencias", key="btn_analyze_trends", use_container_width=True):
            with st.spinner("Analizando tendencias..."):
                # Encontrar tendencias significativas
                trends = find_price_trends(df, days, min_change, trend_price_col)
                
                # Mostrar resultados
                if trends.empty:
                    st.warning(f"No se encontraron tendencias significativas en el período seleccionado.")
                else:
                    # Contar subidas y bajadas
                    subidas = len(trends[trends['Tendencia'] == 'Subida'])
                    bajadas = len(trends[trends['Tendencia'] == 'Bajada'])
                    
                    st.success(f"Se encontraron {len(trends)} productos con tendencias significativas: {subidas} subidas, {bajadas} bajadas.")
                    
                    # Formatear precios y cambios
                    trends['Primer Precio'] = trends['Primer Precio'].apply(lambda x: f"RD$ {x:.2f}")
                    trends['Último Precio'] = trends['Último Precio'].apply(lambda x: f"RD$ {x:.2f}")
                    trends['Cambio'] = trends['Cambio'].apply(lambda x: f"RD$ {x:.2f}")
                    trends['Cambio (%)'] = trends['Cambio (%)'].apply(lambda x: f"{x:.2f}%")
                    
                    # Mostrar tabla de tendencias
                    st.dataframe(
                        trends,
                        use_container_width=True,
                        column_config={
                            "Tendencia": st.column_config.TextColumn(
                                "Tendencia",
                                help="Dirección de la tendencia",
                                width="medium"
                            )
                        }
                    )
    
    # Tab 4: Oportunidades de mercado
    with search_tabs[3]:
        st.subheader("Oportunidades de Arbitraje entre Mercados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seleccionar tipo de precio
            opp_price_type = st.radio(
                "Tipo de Precio:",
                ["Mayorista", "Minorista"],
                key="opp_price_type",
                horizontal=True
            )
            
            opp_price_col = f"Precio_{opp_price_type}"
        
        with col2:
            # Diferencia mínima
            min_diff = st.slider(
                "Diferencia Mínima (%):",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                key="opp_min_diff"
            )
        
        if st.button("Buscar Oportunidades", key="btn_find_opportunities", use_container_width=True):
            with st.spinner("Buscando oportunidades..."):
                # Encontrar oportunidades de arbitraje
                opportunities = find_market_opportunities(df, min_diff, opp_price_col)
                
                # Mostrar resultados
                if opportunities.empty:
                    st.warning(f"No se encontraron oportunidades de arbitraje con una diferencia mínima del {min_diff}%.")
                else:
                    st.success(f"Se encontraron {len(opportunities)} oportunidades de arbitraje.")
                    
                    # Formatear precios y diferencias
                    opportunities['Precio Compra'] = opportunities['Precio Compra'].apply(lambda x: f"RD$ {x:.2f}")
                    opportunities['Precio Venta'] = opportunities['Precio Venta'].apply(lambda x: f"RD$ {x:.2f}")
                    opportunities['Diferencia'] = opportunities['Diferencia'].apply(lambda x: f"RD$ {x:.2f}")
                    opportunities['Diferencia (%)'] = opportunities['Diferencia (%)'].apply(lambda x: f"{x:.2f}%")
                    
                    # Mostrar tabla de oportunidades
                    st.dataframe(
                        opportunities,
                        use_container_width=True
                    )
    
    # Tab 5: Patrones estacionales
    with search_tabs[4]:
        st.subheader("Análisis de Patrones Estacionales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seleccionar tipo de precio
            seasonal_price_type = st.radio(
                "Tipo de Precio:",
                ["Mayorista", "Minorista"],
                key="seasonal_price_type",
                horizontal=True
            )
            
            seasonal_price_col = f"Precio_{seasonal_price_type}"
        
        with col2:
            # Seleccionar producto específico o todos
            productos = ['Todos los Productos'] + sorted(df['Producto'].unique().tolist())
            producto_seleccionado = st.selectbox(
                "Producto:",
                productos,
                key="seasonal_product"
            )
            
            producto = None if producto_seleccionado == 'Todos los Productos' else producto_seleccionado
        
        if st.button("Analizar Estacionalidad", key="btn_analyze_seasonality", use_container_width=True):
            with st.spinner("Analizando patrones estacionales..."):
                # Encontrar patrones estacionales
                seasonal_patterns = find_seasonal_patterns(df, producto, seasonal_price_col)
                
                # Mostrar resultados
                if seasonal_patterns.empty:
                    st.warning(f"No se encontraron patrones estacionales significativos. Se requieren datos de múltiples años.")
                else:
                    st.success(f"Se encontraron patrones estacionales para {len(seasonal_patterns)} productos.")
                    
                    # Formatear precio promedio
                    seasonal_patterns['Precio Promedio'] = seasonal_patterns['Precio Promedio'].apply(lambda x: f"RD$ {x:.2f}")
                    
                    # Mostrar tabla de patrones estacionales
                    st.dataframe(
                        seasonal_patterns,
                        use_container_width=True,
                        column_config={
                            "Variabilidad Estacional": st.column_config.NumberColumn(
                                "Variabilidad Estacional",
                                help="Mayor valor indica estacionalidad más fuerte",
                                format="%.2f"
                            )
                        }
                    )
                    
                    # Si se seleccionó un producto específico, mostrar análisis detallado
                    if producto is not None and not seasonal_patterns.empty:
                        st.subheader(f"Análisis Detallado para {producto}")
                        
                        # Extraer mes y calcular precio promedio mensual
                        df_product = df[df['Producto'] == producto].copy()
                        df_product['Mes'] = pd.to_datetime(df_product['Fecha']).dt.month
                        
                        monthly_avg = df_product.groupby('Mes')[seasonal_price_col].mean().reset_index()
                        
                        # Nombres de los meses
                        month_names = {
                            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
                        }
                        
                        monthly_avg['Nombre Mes'] = monthly_avg['Mes'].map(month_names)
                        
                        # Ordenar por mes
                        monthly_avg = monthly_avg.sort_values('Mes')
                        
                        # Crear gráfico de barras
                        import plotly.express as px
                        
                        fig = px.bar(
                            monthly_avg,
                            x='Nombre Mes',
                            y=seasonal_price_col,
                            title=f"Precio Promedio Mensual de {producto}",
                            labels={seasonal_price_col: 'Precio Promedio (RD$)', 'Nombre Mes': 'Mes'},
                            color=seasonal_price_col,
                            color_continuous_scale='Viridis'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar recomendaciones
                        st.markdown("### Recomendaciones Estratégicas")
                        
                        # Encontrar meses con precios altos y bajos
                        high_months = monthly_avg.nlargest(3, seasonal_price_col)
                        low_months = monthly_avg.nsmallest(3, seasonal_price_col)
                        
                        st.markdown(f"""
                        * **Mejores meses para vender**: {', '.join(high_months['Nombre Mes'].tolist())}
                        * **Mejores meses para comprar**: {', '.join(low_months['Nombre Mes'].tolist())}
                        
                        Basado en el análisis estacional, se recomienda:
                        
                        1. **Planificar las ventas** para los meses de {high_months['Nombre Mes'].iloc[0]} y {high_months['Nombre Mes'].iloc[1]}, cuando los precios tienden a ser más altos.
                        2. **Almacenar producto** comprado durante {low_months['Nombre Mes'].iloc[0]} y {low_months['Nombre Mes'].iloc[1]} para vender en los meses de precios altos.
                        3. **Planificar la producción** considerando estos ciclos estacionales para maximizar rentabilidad.
                        """)

if __name__ == "__main__":
    # Para pruebas
    import pandas as pd
    import numpy as np
    
    # Crear datos de ejemplo
    np.random.seed(42)
    
    # Fechas
    fechas = pd.date_range(start='2020-01-01', periods=500)
    
    # Productos
    productos = ['Tomate', 'Plátano', 'Cebolla', 'Papa', 'Yuca']
    
    # Mercados
    mercados = ['Mercado Central', 'MERCA', 'Mercado del Este', 'Mercado Nuevo']
    
    # Crear DataFrame
    n_rows = 2000
    
    datos = {
        'Fecha': np.random.choice(fechas, n_rows),
        'Producto': np.random.choice(productos, n_rows),
        'Mercado': np.random.choice(mercados, n_rows),
        'Precio_Mayorista': np.random.uniform(10, 100, n_rows),
        'Precio_Minorista': np.random.uniform(15, 120, n_rows)
    }
    
    df = pd.DataFrame(datos)
    
    # Ordenar por fecha
    df = df.sort_values('Fecha')
    
    # Prueba
    render_search_ui(df)