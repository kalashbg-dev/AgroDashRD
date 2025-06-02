"""
Módulo para mostrar y exportar datos filtrados en AgroDashRD.

Este módulo implementa la funcionalidad para visualizar y exportar
datos filtrados en diferentes formatos, incluyendo tablas, gráficos
y exportación a varios formatos.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO
import base64
from typing import Tuple, List, Optional

def get_table_download_link(df: pd.DataFrame, file_format: str = 'csv') -> str:
    """
    Genera un enlace para descargar el DataFrame en el formato especificado.
    
    Args:
        df: DataFrame a exportar
        file_format: Formato de archivo ('csv', 'excel' o 'json')
        
    Returns:
        str: Enlace HTML para descargar el archivo
    """
    if file_format == 'csv':
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="datos_filtrados.csv">Descargar archivo CSV</a>'
    
    elif file_format == 'excel':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='DatosFiltrados')
        b64 = base64.b64encode(output.getvalue()).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="datos_filtrados.xlsx">Descargar archivo Excel</a>'
    
    elif file_format == 'json':
        json_str = df.to_json(orient='records', force_ascii=False)
        b64 = base64.b64encode(json_str.encode('utf-8')).decode()
        return f'<a href="data:application/json;base64,{b64}" download="datos_filtrados.json">Descargar archivo JSON</a>'

def create_pie_chart(df: pd.DataFrame, column: str, title: str) -> None:
    """
    Crea un gráfico de pastel para la columna especificada.
    
    Args:
        df: DataFrame con los datos
        column: Nombre de la columna para el gráfico
        title: Título del gráfico
    """
    if column not in df.columns:
        st.warning(f"La columna '{column}' no está disponible en los datos.")
        return
    
    # Contar valores y ordenar de mayor a menor
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    counts = counts.sort_values('count', ascending=False)
    
    # Limitar a los 10 primeros para mejor visualización
    if len(counts) > 10:
        other_count = counts[10:]['count'].sum()
        counts = counts.head(10)
        counts = pd.concat([counts, pd.DataFrame({column: ['Otros'], 'count': [other_count]})])
    
    # Crear gráfico de pastel
    fig = px.pie(
        counts, 
        values='count', 
        names=column,
        title=title,
        color_discrete_sequence=px.colors.sequential.Agsunset
    )
    
    # Mejorar el diseño
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}: %{value} (%{percent})<extra></extra>'
    )
    
    fig.update_layout(
        margin=dict(t=50, b=10, l=10, r=10),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_filtered_data(df: pd.DataFrame) -> None:
    """
    Muestra los datos filtrados con tabla, gráficos y opciones de exportación.
    
    Args:
        df: DataFrame con los datos filtrados
    """
    if df.empty:
        st.warning("No hay datos para mostrar con los filtros actuales.")
        return
    
    st.header("📊 Datos Filtrados")
    
    # Mostrar resumen de datos
    st.subheader("Resumen de Datos")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de registros", len(df))
    with col2:
        st.metric("Productos únicos", df['Producto'].nunique() if 'Producto' in df.columns else "N/A")
    with col3:
        st.metric("Mercados únicos", df['Mercado'].nunique() if 'Mercado' in df.columns else "N/A")
    with col4:
        st.metric("Rango de fechas", 
                 f"{df['Fecha'].min().strftime('%d/%m/%Y') if 'Fecha' in df.columns else 'N/A'} - {df['Fecha'].max().strftime('%d/%m/%Y') if 'Fecha' in df.columns else 'N/A'}")
    
    # Sección de exportación de datos (siempre visible)
    st.subheader("Exportar Datos")
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        st.markdown(get_table_download_link(df, 'csv'), unsafe_allow_html=True)
    with export_col2:
        st.markdown(get_table_download_link(df, 'excel'), unsafe_allow_html=True)
    with export_col3:
        st.markdown(get_table_download_link(df, 'json'), unsafe_allow_html=True)
    
    # Sección de visualización de datos
    st.subheader("Vista de Datos")
    
    # Mostrar tabla con datos
    st.dataframe(df, use_container_width=True, height=400)
    
    # Sección de gráficos
    st.subheader("Análisis de Distribución")
    
    # Seleccionar columnas para análisis
    available_columns = [col for col in df.columns if df[col].nunique() < 50 and df[col].nunique() > 1]
    
    if not available_columns:
        st.warning("No hay columnas categóricas adecuadas para el análisis de distribución.")
        return
    
    # Crear pestañas para diferentes tipos de gráficos
    tab1, tab2 = st.tabs(["📊 Distribución por Categorías", "📈 Estadísticas de Precios"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Producto' in df.columns:
                create_pie_chart(df, 'Producto', 'Distribución por Producto')
            
            if 'Rubro' in df.columns:
                create_pie_chart(df, 'Rubro', 'Distribución por Rubro')
        
        with col2:
            if 'Mercado' in df.columns:
                create_pie_chart(df, 'Mercado', 'Distribución por Mercado')
            
            if 'Tipo' in df.columns:
                create_pie_chart(df, 'Tipo', 'Distribución por Tipo')
    
    with tab2:
        st.subheader("Estadísticas de Precios")
        
        if 'Precio_Mayorista' in df.columns or 'Precio_Minorista' in df.columns:
            price_cols = [col for col in ['Precio_Mayorista', 'Precio_Minorista'] if col in df.columns]
            
            if price_cols:
                # Mostrar estadísticas en columnas
                stats = df[price_cols].describe().T[['min', 'mean', '50%', 'max', 'std']]
                stats.columns = ['Mínimo', 'Promedio', 'Mediana', 'Máximo', 'Desv. Estándar']
                
                # Formatear números
                st.dataframe(stats.style.format('{:.2f}'), use_container_width=True)
                
                # Mostrar histograma de precios
                st.subheader("Distribución de Precios")
                
                # Crear selección para cada tipo de precio
                selected_price_col = st.radio("Selecciona un tipo de precio", price_cols)
                
                fig = px.histogram(df, x=selected_price_col, nbins=30, 
                                 title=f"Distribución de {selected_price_col}",
                                 color_discrete_sequence=px.colors.sequential.Agsunset)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de precios disponibles para mostrar estadísticas.")

def mostrar_datos_filtrados(df: pd.DataFrame) -> None:
    """
    Función principal para mostrar la pestaña de datos filtrados.
    
    Args:
        df: DataFrame con los datos filtrados
    """
    # Verificar si hay datos
    if df is None or df.empty:
        st.warning("No hay datos disponibles para mostrar.")
        return
    
    # Mostrar los datos filtrados
    show_filtered_data(df)
