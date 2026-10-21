"""
Módulo de Dashboard para Agricultores de AgroDashRD.

Este módulo implementa visualizaciones simplificadas y herramientas
prácticas orientadas a agricultores y productores sin conocimientos técnicos.
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

# --- OPTIMIZACIÓN: Funciones cacheadas para agrupaciones y resúmenes ---
@st.cache_data(show_spinner=False)
def cached_groupby_mean(df, group_cols, value_col):
    return df.groupby(group_cols)[value_col].mean().reset_index()

@st.cache_data(show_spinner=False)
def cached_groupby_agg(df, group_col, value_col):
    return df.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()

@st.cache_data(show_spinner=False)
def cached_groupby_month(df, value_col):
    return df.groupby('Mes')[value_col].mean().reset_index()

def mostrar_precios_actuales(df: pd.DataFrame) -> None:
    """
    Muestra los precios actuales y tendencias de los productos agrícolas
    de forma simplificada para agricultores.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.header("Precios Actuales del Mercado")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("assets/placeholder_mercado.svg", width=150)
    with col2:
        st.markdown("""
        <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
        <p style="font-size: 16px; margin-bottom: 0;">
        Esta sección muestra los precios promedio de la última semana y su tendencia 
        comparada con la semana anterior. Use esta información para tomar mejores
        decisiones sobre cuándo y dónde vender su cosecha.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Verificamos si tenemos columnas necesarias
    if 'Fecha' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para mostrar precios actuales.")
        return
    
    # Asegurarse que la fecha es datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Obtener la fecha más reciente disponible
    fecha_mas_reciente = df['Fecha'].max()
    
    # Si no hay datos recientes, usar la fecha más reciente disponible
    if pd.isna(fecha_mas_reciente):
        st.warning("No se encontraron fechas en los datos.")
        return
    
    # Determinar qué columna de precio usar
    precio_col = 'Precio_Mayorista' if 'Precio_Mayorista' in df.columns else 'Precio_Minorista'
    
    # Obtener la fecha de inicio para la semana actual (últimos 7 días)
    fecha_inicio_semana_actual = fecha_mas_reciente - pd.Timedelta(days=7)
    
    # Filtrar datos para la semana actual
    df_semana_actual = df[df['Fecha'] >= fecha_inicio_semana_actual].copy()
    
    # Si no hay datos para la semana actual, usar un rango más amplio
    if len(df_semana_actual) == 0:
        st.info("No hay datos para la semana actual. Mostrando datos del último mes disponible.")
        fecha_inicio_semana_actual = fecha_mas_reciente - pd.Timedelta(days=30)
        df_semana_actual = df[df['Fecha'] >= fecha_inicio_semana_actual].copy()
    
    # Calcular la fecha de inicio para la semana anterior
    fecha_inicio_semana_anterior = fecha_inicio_semana_actual - pd.Timedelta(days=7)
    
    # Filtrar datos para la semana anterior
    df_semana_anterior = df[(df['Fecha'] >= fecha_inicio_semana_anterior) & 
                           (df['Fecha'] < fecha_inicio_semana_actual)].copy()
    
    # Si no hay datos para la semana anterior, ajustar el rango
    if len(df_semana_anterior) == 0:
        # Intentar con un rango más amplio (últimos 30-60 días)
        fecha_inicio_semana_anterior = fecha_inicio_semana_actual - pd.Timedelta(days=30)
        df_semana_anterior = df[(df['Fecha'] >= fecha_inicio_semana_anterior) & 
                               (df['Fecha'] < fecha_inicio_semana_actual)].copy()
    
    # Si aún no hay datos, mostrar solo los precios actuales
    if len(df_semana_anterior) == 0:
        st.info("No hay datos históricos suficientes para comparar tendencias. Mostrando solo precios actuales.")
        if 'Producto' in df.columns:
            # Asegurar que la columna de precios sea numérica
            df_semana_actual[precio_col] = pd.to_numeric(df_semana_actual[precio_col], errors='coerce')
            precio_actual = df_semana_actual.groupby('Producto')[precio_col].mean().reset_index()
            precio_actual = precio_actual.sort_values(precio_col, ascending=False)
            
            st.subheader("Precios Promedio")
            fig = px.bar(
                precio_actual.head(10), 
                x='Producto', 
                y=precio_col,
                color=precio_col,
                color_continuous_scale='Greens',
                title="Top 10 Productos por Precio"
            )
            fig.update_layout(xaxis_title="Producto", yaxis_title="Precio (RD$)")
            st.plotly_chart(fig, use_container_width=True)
        return
    
    # Calcular precios promedio por producto para ambas semanas
    if 'Producto' in df.columns:
        # Asegurar que las columnas de precios sean numéricas
        df_semana_actual[precio_col] = pd.to_numeric(df_semana_actual[precio_col], errors='coerce')
        df_semana_anterior[precio_col] = pd.to_numeric(df_semana_anterior[precio_col], errors='coerce')
        precio_actual = cached_groupby_mean(df_semana_actual, 'Producto', precio_col)
        precio_anterior = cached_groupby_mean(df_semana_anterior, 'Producto', precio_col)
        
        # Combinar datos
        comparacion = precio_actual.merge(
            precio_anterior, 
            on='Producto', 
            how='inner',
            suffixes=('_actual', '_anterior')
        )
        
        # Calcular cambio porcentual
        comparacion['cambio_pct'] = ((comparacion[f'{precio_col}_actual'] - 
                                     comparacion[f'{precio_col}_anterior']) / 
                                    comparacion[f'{precio_col}_anterior'] * 100)
        
        # Filtrar para productos con datos completos
        comparacion = comparacion.dropna(subset=['cambio_pct'])
        
        # Mostrar en tarjetas en una cuadrícula
        st.subheader("Tendencias de precios en los últimos 7 días")
        
        # Ordenar por cambio porcentual
        comparacion_subida = comparacion.sort_values('cambio_pct', ascending=False).head(5)
        comparacion_bajada = comparacion.sort_values('cambio_pct', ascending=True).head(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Productos con Mayores Subidas")
            for _, row in comparacion_subida.iterrows():
                cambio = row['cambio_pct']
                color = "green" if cambio > 0 else "red"
                icon = "↗️" if cambio > 0 else "↘️"
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid {color};">
                <h4 style="margin: 0; color: #333;">{row['Producto']}</h4>
                <p style="margin: 5px 0;">
                <span style="font-size: 18px; font-weight: bold;">RD$ {row[f'{precio_col}_actual']:.2f}</span> 
                <span style="color: {color}; font-weight: bold;">{icon} {cambio:.1f}%</span>
                </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Productos con Mayores Bajadas")
            for _, row in comparacion_bajada.iterrows():
                cambio = row['cambio_pct']
                color = "green" if cambio > 0 else "red"
                icon = "↗️" if cambio > 0 else "↘️"
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; border-left: 4px solid {color};">
                <h4 style="margin: 0; color: #333;">{row['Producto']}</h4>
                <p style="margin: 5px 0;">
                <span style="font-size: 18px; font-weight: bold;">RD$ {row[f'{precio_col}_actual']:.2f}</span> 
                <span style="color: {color}; font-weight: bold;">{icon} {cambio:.1f}%</span>
                </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Crear gráfico de precios actuales por producto
        st.subheader("Precios Actuales por Producto")
        n_productos = min(15, len(precio_actual))
        
        # Ordenar por precio descendente y tomar los primeros N productos
        precio_actual_sorted = precio_actual.sort_values(precio_col, ascending=False).head(n_productos)
        
        fig = px.bar(
            precio_actual_sorted, 
            x='Producto', 
            y=precio_col,
            color=precio_col,
            color_continuous_scale='Greens',
            title=f"Top {n_productos} Productos por Precio"
        )
        fig.update_layout(xaxis_title="Producto", yaxis_title="Precio (RD$)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar datos en tabla interactiva
        with st.expander("Ver tabla completa de precios"):
            # Ordenar por nombre de producto
            tabla_precios = precio_actual.sort_values('Producto')
            tabla_precios.columns = ['Producto', 'Precio Promedio (RD$)']
            # Formatear valores
            tabla_precios['Precio Promedio (RD$)'] = tabla_precios['Precio Promedio (RD$)'].round(2)
            # Limitar a 100 filas para evitar sobrecarga
            st.dataframe(tabla_precios.head(100), use_container_width=True)

def calculadora_mi_cosecha(df: pd.DataFrame) -> None:
    """
    Implementa una calculadora simple para estimar el valor
    de la cosecha basado en precios actuales.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.header("Calculadora: ¿Cuánto Vale Mi Cosecha?")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("assets/placeholder_agricultor.svg", width=150)
    with col2:
        st.markdown("""
        <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
        <p style="font-size: 16px; margin-bottom: 0;">
        Use esta calculadora para estimar el valor de su cosecha en diferentes mercados.
        Seleccione su producto y la cantidad estimada para ver proyecciones de ingresos.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para la calculadora.")
        return
    
    # Determinar qué columnas de precio están disponibles
    columnas_precio = []
    if 'Precio_Mayorista' in df.columns:
        columnas_precio.append('Precio_Mayorista')
    if 'Precio_Minorista' in df.columns:
        columnas_precio.append('Precio_Minorista')
    
    # Obtener lista de productos
    productos = sorted(df['Producto'].unique())
    
    if len(productos) == 0:
        st.warning("No hay productos disponibles en los datos.")
        return
    
    # Interfaz de usuario
    col1, col2 = st.columns(2)
    
    with col1:
        producto_seleccionado = st.selectbox(
            "Seleccione su producto:",
            productos
        )
        
        unidad_medida = st.selectbox(
            "Unidad de medida:",
            ["Kilogramos (kg)", "Quintales (qq)", "Libras (lb)", "Toneladas (t)"]
        )
        
    with col2:
        cantidad = st.number_input(
            "Cantidad estimada de cosecha:",
            min_value=0.1,
            value=100.0,
            step=10.0
        )
        
        tipo_mercado = st.selectbox(
            "Tipo de mercado:",
            ["Todos", "Mayorista", "Minorista"]
        )
    
    # Filtrar datos por producto seleccionado
    df_producto = df[df['Producto'] == producto_seleccionado].copy()
    
    if len(df_producto) == 0:
        st.warning(f"No hay datos disponibles para {producto_seleccionado}.")
        return
    
    # Calcular precios promedio por mercado
    if 'Mercado' in df_producto.columns:
        # Filtrar por tipo de mercado si es necesario
        if tipo_mercado == "Mayorista" and 'tipo_mercado' in df_producto.columns:
            df_producto = df_producto[df_producto['tipo_mercado'] == 'Mayorista']
        elif tipo_mercado == "Minorista" and 'tipo_mercado' in df_producto.columns:
            df_producto = df_producto[df_producto['tipo_mercado'] == 'Minorista']
        
        # Determinar columna de precio a usar
        precio_col = None
        if tipo_mercado == "Mayorista" and 'Precio_Mayorista' in df_producto.columns:
            precio_col = 'Precio_Mayorista'
        elif tipo_mercado == "Minorista" and 'Precio_Minorista' in df_producto.columns:
            precio_col = 'Precio_Minorista'
        else:
            # Usar la primera columna de precio disponible
            for col in columnas_precio:
                if col in df_producto.columns:
                    precio_col = col
                    break
        
        if precio_col is None:
            st.warning("No hay columnas de precio disponibles para el cálculo.")
            return
        
        # Asegurar que la columna de precios sea numérica
        df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
        # Calcular precios por mercado
        precios_mercado = df_producto.groupby('Mercado')[precio_col].mean().reset_index()
        precios_mercado = precios_mercado.sort_values(precio_col, ascending=False)
        
        # Aplicar factor de conversión según unidad de medida
        factor_conversion = 1.0  # Por defecto kg
        if unidad_medida == "Quintales (qq)":
            factor_conversion = 100.0  # 1 quintal = 100 kg
        elif unidad_medida == "Libras (lb)":
            factor_conversion = 0.453592  # 1 libra = 0.453592 kg
        elif unidad_medida == "Toneladas (t)":
            factor_conversion = 1000.0  # 1 tonelada = 1000 kg
        
        # Calcular valor total por mercado
        precios_mercado['Valor_Total'] = precios_mercado[precio_col] * cantidad * factor_conversion
        
        # Mostrar resultados
        st.subheader(f"Valor estimado de {cantidad} {unidad_medida} de {producto_seleccionado}")
        
        # Crear gráfico de barras
        fig = px.bar(
            precios_mercado,
            x='Mercado',
            y='Valor_Total',
            color='Valor_Total',
            color_continuous_scale='Greens',
            labels={'Valor_Total': 'Valor Estimado (RD$)', 'Mercado': 'Mercado'},
            title=f"Valor Estimado por Mercado"
        )
        
        fig.update_layout(xaxis_title="Mercado", yaxis_title="Valor Estimado (RD$)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabla con datos detallados
        precios_mercado = precios_mercado.rename(columns={precio_col: 'Precio por kg (RD$)'})
        precios_mercado['Precio por kg (RD$)'] = precios_mercado['Precio por kg (RD$)'].round(2)
        precios_mercado['Valor Estimado (RD$)'] = precios_mercado['Valor_Total'].round(2)
        precios_mercado = precios_mercado[['Mercado', 'Precio por kg (RD$)', 'Valor Estimado (RD$)']]
        
        # Limitar a 50 mercados para evitar sobrecarga
        st.dataframe(precios_mercado.head(50), use_container_width=True)
        
        # Mostrar explicación
        st.markdown(f"""
        ### Interpretación
        
        * El mejor mercado para vender su cosecha de **{producto_seleccionado}** es **{precios_mercado.iloc[0]['Mercado']}**, 
          donde podría obtener aproximadamente **RD$ {precios_mercado.iloc[0]['Valor Estimado (RD$)']:,.2f}** por su cosecha.
          
        * La diferencia entre el mejor y el peor mercado es de 
          **RD$ {(precios_mercado.iloc[0]['Valor Estimado (RD$)'] - precios_mercado.iloc[-1]['Valor Estimado (RD$)']):,.2f}**,
          lo que representa un {((precios_mercado.iloc[0]['Valor Estimado (RD$)'] / precios_mercado.iloc[-1]['Valor Estimado (RD$)'] - 1) * 100):.1f}% más de ingresos.
        """)
    else:
        # Si no hay columna de mercado, mostrar cálculo general
        precio_col = None
        for col in columnas_precio:
            if col in df_producto.columns:
                precio_col = col
                break
        
        if precio_col is None:
            st.warning("No hay columnas de precio disponibles para el cálculo.")
            return
        
        # Asegurar que la columna de precios sea numérica
        df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
        # Calcular precio promedio
        precio_promedio = df_producto[precio_col].mean()
        
        # Aplicar factor de conversión según unidad de medida
        factor_conversion = 1.0  # Por defecto kg
        if unidad_medida == "Quintales (qq)":
            factor_conversion = 100.0  # 1 quintal = 100 kg
        elif unidad_medida == "Libras (lb)":
            factor_conversion = 0.453592  # 1 libra = 0.453592 kg
        elif unidad_medida == "Toneladas (t)":
            factor_conversion = 1000.0  # 1 tonelada = 1000 kg
        
        # Calcular valor total
        valor_total = precio_promedio * cantidad * factor_conversion
        
        # Mostrar resultado
        st.subheader(f"Valor estimado de {cantidad} {unidad_medida} de {producto_seleccionado}")
        
        st.markdown(f"""
        <div style="background-color: #e8f3eb; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
            <h2 style="margin: 0; color: #2e7d32;">RD$ {valor_total:,.2f}</h2>
            <p style="margin: 5px 0 0 0;">Basado en un precio promedio de RD$ {precio_promedio:.2f} por kg</p>
        </div>
        """, unsafe_allow_html=True)

def mejores_mercados(df: pd.DataFrame) -> None:
    """
    Muestra ranking de mercados para cada producto.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.header("¿Dónde Vender? - Los Mejores Mercados")
    
    st.markdown("""
    <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
    <p style="font-size: 16px; margin-bottom: 0;">
    Esta sección muestra cuáles son los mejores mercados para vender cada producto,
    basado en los precios históricos recientes. Use esta información para decidir
    dónde llevar su producto para obtener el mejor precio.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or 'Mercado' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para analizar los mejores mercados.")
        return
    
    # Determinar qué columna de precio usar
    precio_col = 'Precio_Mayorista' if 'Precio_Mayorista' in df.columns else 'Precio_Minorista'
    
    # Permitir al usuario seleccionar un producto
    productos = sorted(df['Producto'].unique())
    
    if len(productos) == 0:
        st.warning("No hay productos disponibles en los datos.")
        return
    
    producto_seleccionado = st.selectbox(
        "Seleccione un producto para ver los mejores mercados:",
        productos
    )
    
    # Filtrar datos para el producto seleccionado
    df_producto = df[df['Producto'] == producto_seleccionado].copy()
    
    if len(df_producto) == 0:
        st.warning(f"No hay datos disponibles para {producto_seleccionado}.")
        return
    
    # Filtrar datos recientes (últimos 30 días)
    if 'Fecha' in df_producto.columns:
        if not pd.api.types.is_datetime64_dtype(df_producto['Fecha']):
            df_producto['Fecha'] = pd.to_datetime(df_producto['Fecha'])
            
        fecha_max = df_producto['Fecha'].max()
        fecha_limite = fecha_max - pd.Timedelta(days=30)
        df_reciente = df_producto[df_producto['Fecha'] >= fecha_limite]
        
        if len(df_reciente) > 0:
            df_producto = df_reciente
            st.info(f"Mostrando datos de los últimos 30 días (hasta {fecha_max.strftime('%d/%m/%Y')})")
    
    # Asegurar que la columna de precios sea numérica
    df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
    # Calcular precios promedio por mercado
    mercados_avg = cached_groupby_agg(df_producto, 'Mercado', precio_col)
    mercados_avg.columns = ['Mercado', 'Precio Promedio', 'Desviación', 'Conteo']
    
    # Ordenar por precio promedio descendente
    mercados_avg = mercados_avg.sort_values('Precio Promedio', ascending=False)
    
    # Calcular diferencia porcentual respecto al promedio general
    precio_promedio_general = df_producto[precio_col].mean()
    mercados_avg['Diferencia %'] = ((mercados_avg['Precio Promedio'] / precio_promedio_general) - 1) * 100
    
    # Formatear para visualización
    mercados_avg['Precio Promedio'] = mercados_avg['Precio Promedio'].round(2)
    mercados_avg['Desviación'] = mercados_avg['Desviación'].round(2)
    mercados_avg['Diferencia %'] = mercados_avg['Diferencia %'].round(1)
    
    # Mostrar gráfico de barras con error bars
    fig = px.bar(
        mercados_avg,
        x='Mercado',
        y='Precio Promedio',
        error_y='Desviación',
        color='Diferencia %',
        color_continuous_scale='RdYlGn',
        labels={
            'Precio Promedio': f'Precio Promedio (RD$)',
            'Mercado': 'Mercado',
            'Diferencia %': 'Diferencia vs. Promedio (%)'
        },
        title=f"Ranking de Mercados para {producto_seleccionado}"
    )
    
    # Añadir línea para el precio promedio general
    fig.add_hline(
        y=precio_promedio_general, 
        line_dash="dash", 
        line_color="gray",
        annotation_text=f"Promedio General: RD$ {precio_promedio_general:.2f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(xaxis_title="Mercado", yaxis_title="Precio Promedio (RD$)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar tabla de datos
    st.subheader("Datos detallados por mercado")
    
    # Renombrar columnas para mejor presentación
    tabla_mercados = mercados_avg.copy()
    tabla_mercados.columns = ['Mercado', 'Precio Promedio (RD$)', 'Variabilidad (±RD$)', 'Número de Registros', 'Diferencia vs. Promedio (%)']
    
    # Colorear diferencia porcentual
    def color_dif_pct(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}'
    
    # Mostrar con formato
    # Limitar a 50 mercados para evitar sobrecarga
        st.dataframe(tabla_mercados.head(50), use_container_width=True)
    
    # Añadir interpretación
    st.markdown(f"""
    ### Interpretación
    
    * El mercado con mejor precio para **{producto_seleccionado}** es **{tabla_mercados.iloc[0]['Mercado']}** 
      con un precio promedio de **RD$ {tabla_mercados.iloc[0]['Precio Promedio (RD$)']}** por unidad.
      
    * Este precio es un **{tabla_mercados.iloc[0]['Diferencia vs. Promedio (%)']:.1f}%** {'superior' if tabla_mercados.iloc[0]['Diferencia vs. Promedio (%)'] > 0 else 'inferior'} 
      al promedio general del mercado.
      
    * La diferencia entre el mejor y el peor mercado es de 
      **RD$ {(tabla_mercados.iloc[0]['Precio Promedio (RD$)'] - tabla_mercados.iloc[-1]['Precio Promedio (RD$)']):,.2f}** por unidad,
      lo que representa un {((tabla_mercados.iloc[0]['Precio Promedio (RD$)'] / tabla_mercados.iloc[-1]['Precio Promedio (RD$)'] - 1) * 100):.1f}% más de ingresos.
    """)

def calendario_siembra_cosecha(df: pd.DataFrame) -> None:
    """
    Muestra un calendario visual para identificar mejores
    momentos para cosechar y vender.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.header("Calendario de Mejores Momentos para Vender")
    
    # Pestañas para distintas herramientas de planificación 
    #   tab1, tab2, tab3 = st.tabs(["📊 Pronóstico", "🔍 Validación Cruzada", "📥 Exportar"])
    subcalendario = st.radio(
        "",
        ["📈 Mejores Momentos para Vender", "🌱 Planificador de Siembra"],
        horizontal=True
    )
    
    if subcalendario == "🌱 Planificador de Siembra":
        planificador_siembra(df)
        return
    
    st.markdown("""
    <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
    <p style="font-size: 16px; margin-bottom: 0;">
    Este calendario muestra los mejores meses para vender cada producto, basado
    en los precios históricos. Use esta información para planificar sus siembras
    y cosechas para maximizar ganancias.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or 'Fecha' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para generar el calendario.")
        return
    
    # Asegurarse que la fecha es datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Crear columna de mes
    df['Mes'] = df['Fecha'].dt.month
    df['Nombre_Mes'] = df['Fecha'].dt.strftime('%b')
    
    # Determinar qué columna de precio usar
    precio_col = 'Precio_Mayorista' if 'Precio_Mayorista' in df.columns else 'Precio_Minorista'
    
    # Permitir al usuario seleccionar un producto
    productos = sorted(df['Producto'].unique())
    tipo_analisis = st.radio(
        "Seleccione tipo de análisis:",
        ["Un producto específico", "Todos los productos"]
    )
    
    if tipo_analisis == "Un producto específico":
        if len(productos) == 0:
            st.warning("No hay productos disponibles en los datos.")
            return
            
        producto_seleccionado = st.selectbox(
            "Seleccione un producto:",
            productos
        )
        
        # Filtrar datos para el producto seleccionado
        df_producto = df[df['Producto'] == producto_seleccionado].copy()
        
        if len(df_producto) == 0:
            st.warning(f"No hay datos disponibles para {producto_seleccionado}.")
            return
        
        # Asegurar que la columna de precios sea numérica
        df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
        # Calcular precio promedio por mes
        precios_mes = cached_groupby_mean(df_producto, ['Mes', 'Nombre_Mes'], precio_col)
        
        # Ordenar por mes
        precios_mes = precios_mes.sort_values('Mes')
        
        # Normalizar precios (0-100%) para facilitar visualización
        precio_min = precios_mes[precio_col].min()
        precio_max = precios_mes[precio_col].max()
        
        if precio_max > precio_min:
            precios_mes['Precio_Normalizado'] = (precios_mes[precio_col] - precio_min) / (precio_max - precio_min) * 100
        else:
            precios_mes['Precio_Normalizado'] = 50  # Valor medio si no hay variación
        
        # Determinar mejores y peores meses
        precio_promedio = precios_mes[precio_col].mean()
        precios_mes['Categoria'] = 'Promedio'
        precios_mes.loc[precios_mes[precio_col] >= precio_promedio * 1.1, 'Categoria'] = 'Alto'
        precios_mes.loc[precios_mes[precio_col] <= precio_promedio * 0.9, 'Categoria'] = 'Bajo'
        
        # Crear heatmap por mes
        fig = px.bar(
            precios_mes,
            x='Nombre_Mes',
            y=precio_col,
            color='Categoria',
            color_discrete_map={'Alto': '#2e7d32', 'Promedio': '#fdd835', 'Bajo': '#c62828'},
            labels={
                precio_col: 'Precio Promedio (RD$)',
                'Nombre_Mes': 'Mes',
                'Categoria': 'Nivel de Precio'
            },
            title=f"Calendario de Precios para {producto_seleccionado}"
        )
        
        # Añadir línea para el precio promedio
        fig.add_hline(
            y=precio_promedio, 
            line_dash="dash", 
            line_color="gray",
            annotation_text=f"Promedio Anual: RD$ {precio_promedio:.2f}",
            annotation_position="bottom right"
        )
        
        fig.update_layout(xaxis_title="Mes", yaxis_title="Precio Promedio (RD$)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabla de datos mensuales
        precios_mes['Diferencia vs. Promedio (%)'] = ((precios_mes[precio_col] / precio_promedio) - 1) * 100
        
        # Formatear para visualización
        tabla_meses = precios_mes.copy()
        tabla_meses[precio_col] = tabla_meses[precio_col].round(2)
        tabla_meses['Diferencia vs. Promedio (%)'] = tabla_meses['Diferencia vs. Promedio (%)'].round(1)
        
        # Renombrar columnas
        tabla_meses = tabla_meses[['Nombre_Mes', precio_col, 'Diferencia vs. Promedio (%)', 'Categoria']]
        tabla_meses.columns = ['Mes', 'Precio Promedio (RD$)', 'Diferencia vs. Promedio (%)', 'Nivel de Precio']
        
        # Limitar a 20 meses para evitar sobrecarga
        st.dataframe(tabla_meses.head(20), use_container_width=True)
        
        # Mostrar recomendaciones
        mejores_meses = precios_mes[precios_mes['Categoria'] == 'Alto']['Nombre_Mes'].tolist()
        peores_meses = precios_mes[precios_mes['Categoria'] == 'Bajo']['Nombre_Mes'].tolist()
        
        st.subheader("Recomendaciones")
        
        # Corregimos la forma en que verificamos la condición para evitar errores de tipo
        if len(mejores_meses) > 0:
            max_diferencia = precios_mes[precios_mes['Categoria'] == 'Alto']['Diferencia vs. Promedio (%)'].max()
            max_dif_str = f"{max_diferencia:.1f}" if not pd.isna(max_diferencia) else "0.0"
            
            st.markdown(f"""
            ### Mejores meses para vender {producto_seleccionado}
            
            Los meses con mejores precios para vender son: **{', '.join(mejores_meses)}**
            
            * En estos meses, el precio puede estar hasta un **{max_dif_str}%** por encima del promedio anual.
            """)
        
        # Corregimos la forma de verificar la condición para el segundo bloque
        if len(peores_meses) > 0:
            min_diferencia = precios_mes[precios_mes['Categoria'] == 'Bajo']['Diferencia vs. Promedio (%)'].min()
            min_dif_str = f"{abs(min_diferencia):.1f}" if not pd.isna(min_diferencia) else "0.0"
            
            st.markdown(f"""
            ### Meses a evitar para vender {producto_seleccionado}
            
            Los meses con precios más bajos son: **{', '.join(peores_meses)}**
            
            * En estos meses, el precio puede estar hasta un **{min_dif_str}%** por debajo del promedio anual.
            """)
    
    else:  # Todos los productos
        # Calcular precio promedio por producto y mes
        mapa_calor = cached_groupby_mean(df, ['Producto', 'Mes'], precio_col)
        
        # Normalizar precios por producto (0-100%) para facilitar comparación
        productos_norm = []
        for producto in mapa_calor['Producto'].unique():
            df_prod = mapa_calor[mapa_calor['Producto'] == producto].copy()
            precio_min = df_prod[precio_col].min()
            precio_max = df_prod[precio_col].max()
            
            if precio_max > precio_min:
                df_prod['Precio_Normalizado'] = (df_prod[precio_col] - precio_min) / (precio_max - precio_min) * 100
            else:
                df_prod['Precio_Normalizado'] = 50  # Valor medio si no hay variación
                
            productos_norm.append(df_prod)
        
        mapa_calor = pd.concat(productos_norm)
        
        # Determinar mejores y peores meses por producto
        mapa_calor_categorizado = []
        for producto in mapa_calor['Producto'].unique():
            df_prod = mapa_calor[mapa_calor['Producto'] == producto].copy()
            precio_promedio = df_prod[precio_col].mean()
            
            df_prod['Categoria'] = 'Promedio'
            df_prod.loc[df_prod[precio_col] >= precio_promedio * 1.1, 'Categoria'] = 'Alto'
            df_prod.loc[df_prod[precio_col] <= precio_promedio * 0.9, 'Categoria'] = 'Bajo'
            
            mapa_calor_categorizado.append(df_prod)
        
        mapa_calor = pd.concat(mapa_calor_categorizado)
        
        # Obtener nombres de meses
        nombres_meses = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
            5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
            9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }
        mapa_calor['Nombre_Mes'] = mapa_calor['Mes'].apply(lambda x: nombres_meses.get(x, ''))
        
        # Seleccionar productos más relevantes (top 20 por volumen o presencia)
        # Usamos una solución alternativa para evitar problemas con sort_values
        if 'Volumen' in df.columns:
            # Agrupar y sumar
            vol_por_producto = df.groupby('Producto')['Volumen'].sum().to_dict()
            # Ordenar manualmente
            productos_ordenados = sorted(vol_por_producto.keys(), 
                                         key=lambda x: vol_por_producto[x], 
                                         reverse=True)
            # Obtener los 20 primeros
            top_productos = productos_ordenados[:20]
        else:
            # Contar ocurrencias
            conteo_productos = df.groupby('Producto').size().to_dict()
            # Ordenar manualmente
            productos_ordenados = sorted(conteo_productos.keys(), 
                                        key=lambda x: conteo_productos[x], 
                                        reverse=True)
            # Obtener los 20 primeros
            top_productos = productos_ordenados[:20]
        
        # Filtrar mapa de calor para top productos
        mapa_calor_filtered = mapa_calor[mapa_calor['Producto'].isin(top_productos)]
        
        # Crear heatmap
        fig = px.density_heatmap(
            mapa_calor_filtered,
            x='Nombre_Mes',
            y='Producto',
            z='Precio_Normalizado',
            color_continuous_scale='RdYlGn',
            labels={
                'Precio_Normalizado': 'Nivel de Precio (%)',
                'Nombre_Mes': 'Mes',
                'Producto': 'Producto'
            },
            title="Calendario de Mejores Momentos para Vender por Producto"
        )
        
        fig.update_layout(
            xaxis_title="Mes",
            yaxis_title="Producto",
            xaxis={'categoryorder': 'array', 'categoryarray': list(nombres_meses.values())}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Crear tabla con mejores meses por producto
        mejores_meses_por_producto = []
        
        for producto in top_productos:
            df_prod = mapa_calor[mapa_calor['Producto'] == producto]
            meses_altos = df_prod[df_prod['Categoria'] == 'Alto']['Nombre_Mes'].tolist()
            meses_bajos = df_prod[df_prod['Categoria'] == 'Bajo']['Nombre_Mes'].tolist()
            
            mejores_meses_por_producto.append({
                'Producto': producto,
                'Mejores Meses': ', '.join(meses_altos) if meses_altos else 'Sin variación significativa',
                'Meses a Evitar': ', '.join(meses_bajos) if meses_bajos else 'Sin variación significativa'
            })
        
        tabla_recomendaciones = pd.DataFrame(mejores_meses_por_producto)
        
        st.subheader("Recomendaciones por Producto")
        # Limitar a 50 productos para evitar sobrecarga
        st.dataframe(tabla_recomendaciones.head(50), use_container_width=True)

def planificador_siembra(df: pd.DataFrame) -> None:
    """
    Implementa una calculadora de planificación de siembra basada en
    datos históricos de precios y tiempos de crecimiento de los cultivos.
    
    Args:
        df: DataFrame con datos filtrados
    """
    st.subheader("🌱 Planificador de Siembra")
    
    st.markdown("""
    <div style="background-color: #f0f7f0; padding: 15px; border-radius: 10px; border-left: 5px solid #5a8f7b;">
    <p style="font-size: 16px; margin-bottom: 0;">
    Esta herramienta le ayuda a planificar cuándo sembrar para aprovechar los mejores precios
    del mercado. Seleccione su producto y conozca el mejor momento para comenzar su cultivo.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar columnas necesarias
    if 'Producto' not in df.columns or 'Fecha' not in df.columns or not any(col in df.columns for col in ['Precio_Mayorista', 'Precio_Minorista']):
        st.warning("No hay datos suficientes para el planificador de siembra.")
        return
    
    # Asegurarse que la fecha es datetime
    if not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Crear columna de mes
    df['Mes'] = df['Fecha'].dt.month
    
    # Obtener lista de productos
    productos = sorted(df['Producto'].unique())
    
    if len(productos) == 0:
        st.warning("No hay productos disponibles en los datos.")
        return
    
    # Interfaz de usuario
    col1, col2 = st.columns(2)
    
    with col1:
        producto_seleccionado = st.selectbox(
            "Seleccione su producto:",
            productos,
            key="producto_siembra"
        )
    
    # Tiempos de crecimiento aproximados para cultivos comunes (en meses)
    tiempos_crecimiento = {
        'Tomate': 3,
        'Plátano': 9,
        'Arroz': 4,
        'Cebolla': 5,
        'Yuca': 9,
        'Aguacate': 6,
        'Piña': 18,
        'Naranja': 8,
        'Zanahoria': 3,
        'Ají': 3,
        'Habichuela': 3,
        'Maíz': 4,
        'Papa': 4,
        'Batata': 5,
        'Lechuga': 2,
        'Repollo': 3,
        'Berenjena': 4,
        'Pepino': 2,
        'Calabaza': 4,
        'Limón': 6
    }
    
    # Tiempo de crecimiento predeterminado si no está en el diccionario
    tiempo_predeterminado = 4
    
    # Obtener tiempo de crecimiento para el producto seleccionado
    tiempo_crecimiento = tiempos_crecimiento.get(producto_seleccionado, tiempo_predeterminado)
    
    with col2:
        tiempo_crecimiento = st.number_input(
            "Tiempo de crecimiento (meses):",
            min_value=1,
            max_value=24,
            value=tiempo_crecimiento,
            help="Tiempo aproximado desde la siembra hasta la cosecha."
        )
    
    # Filtrar datos por producto
    df_producto = df[df['Producto'] == producto_seleccionado].copy()
    
    if len(df_producto) == 0:
        st.warning(f"No hay datos disponibles para {producto_seleccionado}.")
        return
    
    # Determinar columna de precio a usar
    precio_col = None
    if 'Precio_Mayorista' in df_producto.columns:
        precio_col = 'Precio_Mayorista'
    elif 'Precio_Minorista' in df_producto.columns:
        precio_col = 'Precio_Minorista'
    else:
        st.warning("No hay columnas de precio disponibles para el análisis.")
        return
    
    # Asegurar que la columna de precios sea numérica
    df_producto[precio_col] = pd.to_numeric(df_producto[precio_col], errors='coerce')
    # Calcular precio promedio por mes
    precio_mensual = cached_groupby_month(df_producto, precio_col)
    
    # Encontrar los meses con mejores precios
    precio_mensual = precio_mensual.sort_values(precio_col, ascending=False)
    
    # Nombres de los meses
    nombres_meses = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    
    # Añadir nombres de los meses
    precio_mensual['Nombre_Mes'] = precio_mensual['Mes'].map(nombres_meses)
    
    # Calcular meses para sembrar
    mejores_meses_cosecha = precio_mensual.head(3)['Mes'].tolist()
    
    # Calcular meses para sembrar (restando tiempo de crecimiento)
    meses_siembra = []
    for mes_cosecha in mejores_meses_cosecha:
        mes_siembra = ((mes_cosecha - tiempo_crecimiento) % 12)
        if mes_siembra == 0:
            mes_siembra = 12
        meses_siembra.append(mes_siembra)
    
    # Mostrar resultados
    st.subheader("Recomendaciones de Siembra")
    
    # Crear una tabla con recomendaciones
    recomendaciones = []
    for i, mes_cosecha in enumerate(mejores_meses_cosecha):
        mes_siembra = meses_siembra[i]
        recomendaciones.append({
            'Mes de Siembra': nombres_meses[mes_siembra],
            'Mes de Cosecha': nombres_meses[mes_cosecha],
            'Precio Promedio Esperado': f"RD$ {precio_mensual[precio_mensual['Mes'] == mes_cosecha][precio_col].values[0]:.2f}"
        })
    
    df_recomendaciones = pd.DataFrame(recomendaciones)
    
    # Mostrar gráfico de precios mensuales
    precio_mensual_ordenado = precio_mensual.sort_values('Mes')
    precio_mensual_ordenado['Nombre_Mes'] = precio_mensual_ordenado['Mes'].map(nombres_meses)
    
    fig = px.line(
        precio_mensual_ordenado,
        x='Nombre_Mes',
        y=precio_col,
        markers=True,
        labels={precio_col: 'Precio Promedio (RD$)', 'Nombre_Mes': 'Mes'},
        title=f"Precios Promedio de {producto_seleccionado} por Mes"
    )
    
    # Resaltar mejores meses
    for mes in mejores_meses_cosecha:
        nombre_mes = nombres_meses[mes]
        precio = precio_mensual_ordenado[precio_mensual_ordenado['Mes'] == mes][precio_col].values[0]
        fig.add_annotation(
            x=nombre_mes,
            y=precio,
            text="Mejor precio",
            showarrow=True,
            arrowhead=1,
            arrowcolor="#2e7d32",
            bgcolor="#e8f5e9",
            font=dict(color="#2e7d32")
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar tabla de recomendaciones
    # Limitar a 20 recomendaciones para evitar sobrecarga
    st.dataframe(df_recomendaciones.head(20), use_container_width=True)
    
    # Explicación detallada
    st.markdown(f"""
    ### Interpretación
    
    * Para **{producto_seleccionado}**, el mejor mes para vender es **{nombres_meses[mejores_meses_cosecha[0]]}** 
      cuando el precio promedio llega a **RD$ {precio_mensual[precio_mensual['Mes'] == mejores_meses_cosecha[0]][precio_col].values[0]:.2f}**.
      
    * Considerando que este cultivo tarda aproximadamente **{tiempo_crecimiento} meses** en crecer,
      el momento óptimo para sembrar es en **{nombres_meses[meses_siembra[0]]}**.
      
    * Si siembra en **{nombres_meses[meses_siembra[0]]}**, su cosecha estará lista para vender
      durante **{nombres_meses[mejores_meses_cosecha[0]]}**, cuando los precios están más altos.
    """)
    
    # Información adicional
    with st.expander("Ver consejos de planificación"):
        st.markdown("""
        ### Consejos para planificar su siembra
        
        1. **Considere las condiciones climáticas**: Además de los precios, tome en cuenta si las condiciones
           climáticas del mes recomendado son adecuadas para la siembra de su cultivo.
           
        2. **Diversifique sus cultivos**: Distribuya su siembra en diferentes momentos para reducir riesgos
           y aprovechar diferentes épocas de buenos precios.
           
        3. **Preste atención a la calidad del suelo**: Asegúrese de que su suelo está preparado adecuadamente
           antes de sembrar, especialmente si está siguiendo ciclos intensivos.
           
        4. **Coordine con otros agricultores**: Establecer un calendario coordinado con otros productores puede
           ayudar a estabilizar los precios al evitar sobreoferta en ciertos períodos.
        """)

def mostrar_dashboard_agricultor(df: pd.DataFrame) -> None:
    """
    Función principal que muestra el dashboard para agricultores
    integrando todas las visualizaciones.
    
    Args:
        df: DataFrame con datos filtrados
    """
    # Crear subpestañas para organizar el contenido
    pestanas = [
        "📊 Precios Actuales", 
        "💰 Mi Cosecha Vale", 
        "🏪 Dónde Vender", 
        "📅 Calendario de Siembra"
    ]
    
    tabs = st.tabs(pestanas)
    with tabs[0]:
        mostrar_precios_actuales(df)
    with tabs[1]:
        calculadora_mi_cosecha(df)
    with tabs[2]:
        mejores_mercados(df)
    with tabs[3]:
        calendario_siembra_cosecha(df)