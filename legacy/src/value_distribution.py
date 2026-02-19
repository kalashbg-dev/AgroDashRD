"""
Value distribution analysis for AgroDashRD.

This module contains functions for analyzing and visualizing 
the distribution of value across the agricultural supply chain.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

def calculate_value_distribution(df: pd.DataFrame, 
                              product_col: str = 'Producto',
                              mayorista_col: str = 'Precio_Mayorista',
                              minorista_col: str = 'Precio_Minorista',
                              date_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
    """
    Calculate the distribution of value across the supply chain.
    
    Args:
        df: Input DataFrame with price data
        product_col: Column name for product
        mayorista_col: Column name for wholesale price
        minorista_col: Column name for retail price
        date_range: Optional date range to filter data
        
    Returns:
        DataFrame with value distribution data
    """
    try:
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Filter by date range if provided
        if date_range is not None and 'Fecha' in data.columns:
            start_date, end_date = date_range
            data = data[(data['Fecha'] >= start_date) & (data['Fecha'] <= end_date)]
        
        # Check if we have the necessary price columns
        required_cols = [product_col, mayorista_col, minorista_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns for value distribution: {missing_cols}")
            
            # Try to create missing price columns if possible
            if mayorista_col not in data.columns and 'Mercado Nuevo' in data.columns:
                data[mayorista_col] = data['Mercado Nuevo']
                missing_cols.remove(mayorista_col)
                
            # If still missing required columns, return empty DataFrame
            if missing_cols:
                return pd.DataFrame()
        
        # Remove rows with missing values in price columns
        data = data.dropna(subset=[mayorista_col, minorista_col])
        
        # Calculate margins and percentages
        data['Margen_Absoluto'] = data[minorista_col] - data[mayorista_col]
        data['Margen_Porcentual'] = (data['Margen_Absoluto'] / data[mayorista_col] * 100).round(2)
        
        data['Valor_Productor_Pct'] = ((data[mayorista_col] * 0.7) / data[minorista_col] * 100).round(2)
        data['Valor_Mayorista_Pct'] = ((data[mayorista_col] * 0.3) / data[minorista_col] * 100).round(2)
        data['Valor_Minorista_Pct'] = (data['Margen_Absoluto'] / data[minorista_col] * 100).round(2)
        
        # Calculate absolute values
        data['Valor_Productor'] = (data[mayorista_col] * 0.7).round(2)
        data['Valor_Mayorista'] = (data[mayorista_col] * 0.3).round(2)
        data['Valor_Minorista'] = data['Margen_Absoluto'].round(2)
        
        return data
    
    except Exception as e:
        logger.error(f"Error calculating value distribution: {str(e)}")
        return pd.DataFrame()

def filter_data_for_sankey(df: pd.DataFrame, 
                          product: Optional[str] = None,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          min_records: int = 5) -> pd.DataFrame:
    """
    Filter data for Sankey diagram visualization.
    
    Args:
        df: Input DataFrame
        product: Optional product to filter by
        start_date: Optional start date
        end_date: Optional end date
        min_records: Minimum records required for valid analysis
        
    Returns:
        Filtered DataFrame
    """
    try:
        # Create a copy to avoid modifying the original
        df_filtered = df.copy()
        
        # Filter by product if specified
        if product and product != 'Todos' and 'Producto' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Producto'] == product]
        
        # Filter by date range if specified
        if start_date and end_date and 'Fecha' in df_filtered.columns:
            df_filtered = df_filtered[(df_filtered['Fecha'] >= pd.Timestamp(start_date)) & 
                                     (df_filtered['Fecha'] <= pd.Timestamp(end_date))]
        
        # Check if we have enough data
        if len(df_filtered) < min_records:
            logger.warning(f"Insufficient data for Sankey diagram: {len(df_filtered)} records")
            return pd.DataFrame()
        
        return df_filtered
    
    except Exception as e:
        logger.error(f"Error filtering data for Sankey diagram: {str(e)}")
        return pd.DataFrame()

def create_sankey_diagram(df: pd.DataFrame,
                         product_col: str = 'Producto',
                         mayorista_col: str = 'Precio_Mayorista', 
                         minorista_col: str = 'Precio_Minorista') -> go.Figure:
    """
    Create a Sankey diagram showing the flow of value.
    
    Args:
        df: DataFrame with value distribution data
        product_col: Column name for product
        mayorista_col: Column name for wholesale price
        minorista_col: Column name for retail price
        
    Returns:
        Plotly figure with Sankey diagram
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            # Return an empty figure with a message
            fig = go.Figure()
            fig.update_layout(
                title="Sin datos suficientes para el diagrama",
                annotations=[dict(
                    text="No hay suficientes datos para crear el diagrama de distribución de valor.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
            return fig
        
        # Calculate value distribution if not already done
        if 'Valor_Productor' not in df.columns:
            df = calculate_value_distribution(
                df, 
                product_col=product_col,
                mayorista_col=mayorista_col,
                minorista_col=minorista_col
            )
            
            # Check if calculation was successful
            if df.empty:
                logger.warning("Failed to calculate value distribution")
                # Return an empty figure with a message
                fig = go.Figure()
                fig.update_layout(
                    title="Error en el cálculo de distribución",
                    annotations=[dict(
                        text="No se pudo calcular la distribución de valor con los datos disponibles.",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )]
                )
                return fig
        
        # Calculate average values for Sankey diagram
        avg_productor = df['Valor_Productor'].mean()
        avg_mayorista = df['Valor_Mayorista'].mean()
        avg_minorista = df['Valor_Minorista'].mean()
        
        # Calculate percentage values
        total_value = avg_productor + avg_mayorista + avg_minorista
        pct_productor = round(avg_productor / total_value * 100, 1)
        pct_mayorista = round(avg_mayorista / total_value * 100, 1)
        pct_minorista = round(avg_minorista / total_value * 100, 1)
        
        # Prepare Sankey diagram data
        nodes = dict(
            label=["Productor", "Mayorista", "Minorista", "Consumidor"],
            color=["#77DD77", "#89CFF0", "#FFB347", "#FF6961"]
        )
        
        # Define links (value flows)
        links = dict(
            source=[0, 1, 2],  # Productor -> Mayorista -> Minorista -> Consumidor
            target=[1, 2, 3],
            value=[avg_productor, avg_productor + avg_mayorista, avg_productor + avg_mayorista + avg_minorista],
            label=[f"RD${avg_productor:.2f} ({pct_productor}%)", 
                  f"RD${avg_mayorista:.2f} ({pct_mayorista}%)", 
                  f"RD${avg_minorista:.2f} ({pct_minorista}%)"],
            color=["rgba(119, 221, 119, 0.8)", "rgba(137, 207, 240, 0.8)", "rgba(255, 179, 71, 0.8)"]
        )
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=nodes,
            link=links,
            arrangement="snap"
        )])
        
        # Update layout
        fig.update_layout(
            title_text="Distribución de Valor en la Cadena de Suministro",
            font_size=12,
            height=500
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Sankey diagram: {str(e)}")
        # Return an error figure
        fig = go.Figure()
        fig.update_layout(
            title="Error en la visualización",
            annotations=[dict(
                text=f"Error al crear el diagrama: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig

def create_distribution_chart(df: pd.DataFrame, 
                            product: str = None,
                            chart_type: str = 'bar') -> go.Figure:
    """
    Create a chart showing the value distribution percentages.
    
    Args:
        df: DataFrame with value distribution data
        product: Optional product to filter by
        chart_type: Type of chart ('bar', 'pie', or 'treemap')
        
    Returns:
        Plotly figure
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            # Return an empty figure with a message
            fig = go.Figure()
            fig.update_layout(
                title="Sin datos suficientes para el gráfico",
                annotations=[dict(
                    text="No hay suficientes datos para crear el gráfico de distribución de valor.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
            return fig
        
        # Filter by product if specified
        if product and product != 'Todos' and 'Producto' in df.columns:
            df = df[df['Producto'] == product]
            
            # Check if we still have data after filtering
            if df.empty:
                logger.warning(f"No data available for product: {product}")
                fig = go.Figure()
                fig.update_layout(
                    title=f"Sin datos para {product}",
                    annotations=[dict(
                        text=f"No hay datos disponibles para {product}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5
                    )]
                )
                return fig
        
        # Calculate average percentages
        avg_productor = df['Valor_Productor_Pct'].mean()
        avg_mayorista = df['Valor_Mayorista_Pct'].mean()
        avg_minorista = df['Valor_Minorista_Pct'].mean()
        
        # Prepare data for visualization
        actors = ['Productor', 'Mayorista', 'Minorista']
        values = [avg_productor, avg_mayorista, avg_minorista]
        colors = ['#77DD77', '#89CFF0', '#FFB347']
        
        # Create visualization based on chart type
        if chart_type == 'pie':
            fig = px.pie(
                names=actors,
                values=values,
                title="Distribución Porcentual del Valor",
                color_discrete_sequence=colors,
                labels={'value': 'Porcentaje del valor final'}
            )
            fig.update_traces(textinfo='percent+label')
            
        elif chart_type == 'treemap':
            fig = px.treemap(
                names=actors,
                values=values,
                title="Distribución del Valor",
                color_discrete_sequence=colors
            )
            fig.update_traces(textinfo='label+percent')
            
        else:  # Default to bar chart
            fig = px.bar(
                x=actors,
                y=values,
                title="Distribución Porcentual del Valor",
                color=actors,
                color_discrete_sequence=colors,
                labels={'x': 'Actor en la cadena', 'y': 'Porcentaje del valor final'}
            )
            fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            
        fig.update_layout(height=400)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating distribution chart: {str(e)}")
        # Return an error figure
        fig = go.Figure()
        fig.update_layout(
            title="Error en la visualización",
            annotations=[dict(
                text=f"Error al crear el gráfico: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        return fig

def show_value_distribution(df: pd.DataFrame):
    """
    Show the value distribution analysis in the Streamlit app.
    
    Args:
        df: DataFrame with price data
    """
    st.subheader("💰 Distribución del Valor en la Cadena Agrícola")
    
    st.markdown("""
    Este análisis muestra cómo se distribuye el valor económico entre los diferentes actores
    de la cadena de suministro agrícola, desde el productor hasta el consumidor final.
    """)
    
    # Display control options
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Product selector
        products = ['Todos']
        if 'Producto' in df.columns:
            products += sorted(df['Producto'].unique().tolist())
        
        selected_product = st.selectbox(
            "Seleccionar Producto:",
            products,
            index=0
        )
    
    with col2:
        # Date range selector
        if 'Fecha' in df.columns:
            # Get min and max dates
            min_date = df['Fecha'].min()
            max_date = df['Fecha'].max()
            
            if pd.notnull(min_date) and pd.notnull(max_date):
                # Default to last 3 months if data range is sufficient
                default_start = max_date - timedelta(days=90) if max_date - min_date > timedelta(days=90) else min_date
                
                start_date = st.date_input(
                    "Fecha Inicial:",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                start_date = None
        else:
            start_date = None
    
    with col3:
        # Date range selector (end date)
        if 'Fecha' in df.columns and start_date is not None:
            max_date = df['Fecha'].max()
            
            if pd.notnull(max_date):
                end_date = st.date_input(
                    "Fecha Final:",
                    value=max_date,
                    min_value=start_date,
                    max_value=max_date
                )
            else:
                end_date = None
        else:
            end_date = None
    
    # Filter data based on selections
    df_filtered = filter_data_for_sankey(
        df, 
        product=selected_product if selected_product != 'Todos' else None,
        start_date=start_date,
        end_date=end_date
    )
    
    # Check if we have data to show
    if df_filtered.empty:
        st.warning("""
        No hay suficientes datos para el análisis de distribución de valor con los filtros seleccionados.
        Por favor, ajuste los filtros o seleccione un producto diferente.
        """)
        return
    
    # Calculate value distribution
    df_with_value = calculate_value_distribution(df_filtered)
    
    if df_with_value.empty:
        st.warning("""
        No se pudo calcular la distribución de valor con los datos disponibles.
        Verifique que los precios mayoristas y minoristas estén presentes y sean válidos.
        """)
        return
    
    # Show Sankey diagram
    st.subheader("Flujo de Valor en la Cadena")
    fig_sankey = create_sankey_diagram(df_with_value)
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # Show distribution charts
    st.subheader("Distribución Porcentual del Valor")
    
    chart_types = ['bar', 'pie', 'treemap']
    chart_labels = ['Barras', 'Pastel', 'Mapa de Árbol']
    
    selected_chart = st.radio(
        "Tipo de Visualización:",
        options=range(len(chart_types)),
        format_func=lambda x: chart_labels[x],
        horizontal=True
    )
    
    fig_dist = create_distribution_chart(
        df_with_value, 
        product=selected_product if selected_product != 'Todos' else None,
        chart_type=chart_types[selected_chart]
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Show data table with analysis results
    with st.expander("Ver Datos de Distribución de Valor"):
        # Select relevant columns for the table
        display_cols = [
            'Producto', 'Fecha', 'Mercado',
            'Precio_Mayorista', 'Precio_Minorista', 'Margen_Absoluto', 'Margen_Porcentual',
            'Valor_Productor', 'Valor_Mayorista', 'Valor_Minorista',
            'Valor_Productor_Pct', 'Valor_Mayorista_Pct', 'Valor_Minorista_Pct'
        ]
        
        # Only include columns that exist in the DataFrame
        display_cols = [col for col in display_cols if col in df_with_value.columns]
        
        if display_cols:
            st.dataframe(df_with_value[display_cols], use_container_width=True)
        else:
            st.warning("No hay columnas disponibles para mostrar")