"""
Data loading utilities for AgroDashRD.

This module contains functions for loading, caching, and preparing
data for the AgroDashRD application.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
import logging
from typing import Tuple, List, Dict, Optional, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data from a file with caching.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Loaded DataFrame or None if loading failed
    """
    try:
        # Determine file extension
        _, ext = os.path.splitext(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Check if file exists in data directory
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            alternative_path = os.path.join(data_dir, os.path.basename(file_path))
            
            if os.path.exists(alternative_path):
                file_path = alternative_path
            else:
                logger.error(f"File not found: {file_path}")
                return None
        
        # Load based on file extension
        if ext.lower() == '.csv':
            # Try different encodings and delimiters
            encodings = ['utf-8', 'cp1252', 'latin1']
            delimiters = [',', ';', '\t']
            
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                        logger.info(f"Successfully loaded {file_path} with encoding {encoding} and delimiter '{delimiter}'")
                        return df
                    except Exception as e:
                        continue
            
            # If all attempts failed
            logger.error(f"Failed to load CSV file: {file_path}")
            return None
            
        elif ext.lower() == '.parquet':
            df = pd.read_parquet(file_path)
            logger.info(f"Successfully loaded {file_path}")
            return df
            
        elif ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            logger.info(f"Successfully loaded {file_path}")
            return df
            
        else:
            logger.error(f"Unsupported file format: {ext}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess data for the AgroDashRD application.
    
    Returns:
        Tuple of (df_mayorista, df_minorista, df_combined)
    """
    # Load data
    df_mayorista = load_data('data/mayorista.csv')
    df_minorista = load_data('data/minorista.csv')
    
    # Check if data was loaded
    if df_mayorista is None or df_minorista is None:
        # Try to load from processed files
        df_mayorista = load_data('data/mayorista_preprocesado.parquet')
        df_minorista = load_data('data/minorista_preprocesado.parquet')
        df_combined = load_data('data/datos_preprocesados.parquet')
        
        if df_mayorista is not None and df_minorista is not None and df_combined is not None:
            return df_mayorista, df_minorista, df_combined
        else:
            # If still not loaded, create sample data
            logger.warning("Failed to load real data. Creating sample data.")
            return create_sample_data()
    
    # Process data
    try:
        # Procesar y mapear nombres de columnas de forma centralizada
        df_mayorista.columns = [normalize_column_name(col) for col in df_mayorista.columns]
        df_minorista.columns = [normalize_column_name(col) for col in df_minorista.columns]

        # En mayorista.csv: crear columna 'Precio_Mayorista' a partir de 'Mercado Nuevo',
        # pero conservar la columna original y asegurar columna 'Mercado' con valor único
        mercado_nuevo_col = None
        for col in df_mayorista.columns:
            if col.strip().lower().replace('_', ' ') == 'mercado nuevo':
                mercado_nuevo_col = col
                break
        if mercado_nuevo_col:
            df_mayorista['Precio_Mayorista'] = df_mayorista[mercado_nuevo_col]
            df_mayorista['Mercado'] = 'Mercado Nuevo'

        # Añadir tipo de mercado
        df_mayorista['Tipo_Mercado'] = 'Mayorista'
        df_minorista['Tipo_Mercado'] = 'Minorista'

        # Crear columna de fecha
        df_mayorista = create_date_column(df_mayorista)
        df_minorista = create_date_column(df_minorista)

        # Limpiar columnas numéricas y reestructurar datos mayorista
        if 'Precio_Mayorista' in df_mayorista.columns:
            df_mayorista['Precio_Mayorista'] = clean_numeric_column(df_mayorista['Precio_Mayorista'])
            df_mayorista['Mercado'] = 'Mercado Nuevo' if 'Mercado Nuevo' in df_mayorista.columns or 'Mercado_Nuevo' in df_mayorista.columns else df_mayorista.get('Mercado', 'Mercado Nuevo')
        else:
            # Buscar columnas de mercado y limpiar
            price_cols_mayorista = [col for col in df_mayorista.columns if is_market_name(col)]
            for col in price_cols_mayorista:
                df_mayorista[col] = clean_numeric_column(df_mayorista[col])
            # Si hay solo una columna de mercado, renombrar a Precio_Mayorista
            if len(price_cols_mayorista) == 1:
                df_mayorista = df_mayorista.rename(columns={price_cols_mayorista[0]: 'Precio_Mayorista'})
                df_mayorista['Mercado'] = price_cols_mayorista[0]

        # Limpiar y reestructurar datos minorista
        non_market_cols = ['Producto', 'Rubro', 'Categoría', 'Unidad', 'Semana', 'Mes', 'Año', 'Tipo_Mercado', 'Fecha', 'Ao', 'Anio']
        market_cols_minorista = [col for col in df_minorista.columns if col not in non_market_cols and is_market_name(col)]
        logger.info(f"Market columns found in minorista dataset: {market_cols_minorista}")
        for col in market_cols_minorista:
            df_minorista[col] = clean_numeric_column(df_minorista[col])
        if market_cols_minorista:
            id_columns = [col for col in df_minorista.columns if col not in market_cols_minorista]
            df_minorista_melted = pd.melt(
                df_minorista,
                id_vars=id_columns,
                value_vars=market_cols_minorista,
                var_name='Mercado',
                value_name='Precio_Minorista'
            )
            df_minorista_melted = df_minorista_melted[
                ~(df_minorista_melted['Precio_Minorista'].isna() | (df_minorista_melted['Precio_Minorista'] == 0))
            ]
            df_minorista = df_minorista_melted
        # Combinar datasets
        df_combined = merge_datasets(df_mayorista, df_minorista)

        # Guardar los DataFrames procesados como Parquet en la carpeta data/
        import os
        def save_parquet_if_changed(df, path):
            if os.path.exists(path):
                try:
                    df_old = pd.read_parquet(path)
                    if df.equals(df_old):
                        logger.info(f"No hay cambios en {path}, no se sobrescribe.")
                        return
                except Exception as e:
                    logger.warning(f"No se pudo comparar {path}, se sobrescribirá. Error: {str(e)}")
            df.to_parquet(path, index=False)
            logger.info(f"Archivo Parquet guardado/actualizado: {path}")
        try:
            save_parquet_if_changed(df_mayorista, 'data/mayorista_preprocesado.parquet')
            save_parquet_if_changed(df_minorista, 'data/minorista_preprocesado.parquet')
            save_parquet_if_changed(df_combined, 'data/datos_preprocesados.parquet')
        except Exception as e:
            logger.error(f"No se pudieron guardar los archivos Parquet: {str(e)}")

        return df_mayorista, df_minorista, df_combined
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Create sample data as fallback
        return create_sample_data()

def normalize_column_name(col_name: str) -> str:
    """
    Normaliza y mapea nombres de columnas a nombres estándar y explícitos.
    Centraliza todos los mapeos relevantes para evitar ambigüedades.

    Args:
        col_name: Nombre original de la columna
    Returns:
        Nombre de columna normalizado y mapeado
    """
    if not isinstance(col_name, str):
        return str(col_name)

    normalized = col_name.strip()

    # Mapeo centralizado de nombres ambiguos a nombres estándar
    mapping = {
        # Fechas y tiempo
        'Ao': 'Año',
        'Ano': 'Año',
        'Anio': 'Año',
        'Mes': 'Mes',
        'Semana': 'Semana',
        # Producto y rubro
        'Productos': 'Producto',
        'Categoria': 'Categoría',
        'Rubro': 'Rubro',
        'Unidad': 'Unidad',
        'Unidad de Venta y Empaque (RD$)': 'Unidad',
        # Mercados y precios
        'Mercado Nuevo': 'Precio_Mayorista',
        'Mercado_Nuevo': 'Precio_Mayorista',
        'Mercado Central': 'Precio_Mayorista',  # Si hay más de un mercado mayorista, ajustar lógica
        'Mercado Oriental': 'Precio_Mayorista',
        'Precio Mayorista': 'Precio_Mayorista',
        'Precio_Mayorista': 'Precio_Mayorista',
        'Precio Minorista': 'Precio_Minorista',
        'Precio_Minorista': 'Precio_Minorista',
    }
    # Aplicar mapeo si corresponde
    for key, value in mapping.items():
        if key.lower() == normalized.lower():
            return value
    return normalized

def is_market_name(col_name: str) -> bool:
    """
    Check if a column name represents a market (for price columns).
    
    Args:
        col_name: Column name to check
        
    Returns:
        True if column is likely a market name
    """
    # Common market prefixes/patterns
    market_patterns = [
        'mercado', 'market', 'mercad', 'super', 'tienda', 'feria', 
        'conaprope', 'consuelo', 'mina', 'central', 'oriental', 'nuevo'
    ]
    
    # Non-market columns that should be excluded
    non_market_cols = [
        'producto', 'rubro', 'categoría', 'categoria', 'unidad', 
        'año', 'mes', 'semana', 'fecha', 'empaque', 'origen', 'destino',
        'tipo_mercado', 'ao', 'anio'
    ]
    
    # Convert to lowercase for case-insensitive comparison
    col_lower = col_name.lower()
    
    # Check if it's a non-market column
    if any(non_col in col_lower for non_col in non_market_cols):
        return False
    
    # Check if it matches a market pattern
    return any(pattern in col_lower for pattern in market_patterns)

def clean_numeric_column(column: pd.Series) -> pd.Series:
    """
    Clean a numeric column by handling various formats and errors.
    
    Args:
        column: Series to clean
        
    Returns:
        Cleaned numeric Series
    """
    # Make a copy to avoid modifying the original
    result = column.copy()
    
    # Handle strings
    if result.dtype == 'object':
        # Replace non-numeric characters and handle thousands separators
        result = result.astype(str)
        
        # Replace N/D, ND, n/a, etc. with NaN
        result = result.replace(['N/D', 'ND', 'n/d', 'nd', 'N/A', 'n/a', '-', ''], np.nan)
        
        # Replace commas with periods for decimal separation
        result = result.str.replace(',', '.', regex=False)
        
        # Remove RD$ and other currency symbols
        result = result.str.replace('RD$', '', regex=False)
        result = result.str.replace('$', '', regex=False)
        
        # Convert to numeric
        result = pd.to_numeric(result, errors='coerce')
    
    return result

def create_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a standardized date column from year, month, week columns.
    
    Args:
        df: DataFrame with date components
        
    Returns:
        DataFrame with added 'Fecha' column
    """
    result = df.copy()
    
    # Check for required columns (with flexible naming)
    year_col = next((col for col in df.columns if col.lower() in ['año', 'ano', 'anio', 'ao', 'year']), None)
    month_col = next((col for col in df.columns if col.lower() in ['mes', 'month']), None)
    
    if year_col and month_col:
        # Create date column (15th day of month as default)
        result['Fecha'] = pd.to_datetime(
            result[year_col].astype(str) + '-' + 
            result[month_col].apply(lambda x: month_to_number(x)).astype(str) + '-15',
            errors='coerce'
        )
    else:
        # Try to find an existing date column
        date_col = next((col for col in df.columns if col.lower() in ['fecha', 'date']), None)
        
        if date_col:
            result['Fecha'] = pd.to_datetime(result[date_col], errors='coerce')
        else:
            logger.warning("Could not create date column - missing year/month information")
            # Create a dummy date column with today's date
            result['Fecha'] = datetime.now()
    
    return result

def month_to_number(month_name: Any) -> int:
    """
    Convert month name to month number (1-12).
    
    Args:
        month_name: Month name or number
        
    Returns:
        Month number (1-12)
    """
    if pd.isna(month_name):
        return 1
    
    # If already a number, return it (with boundary check)
    if isinstance(month_name, (int, float)):
        month_num = int(month_name)
        return max(1, min(12, month_num))
    
    # Convert to string
    month_str = str(month_name).lower().strip()
    
    # Month name to number mapping
    month_mapping = {
        'enero': 1, 'january': 1, 'jan': 1, 'ene': 1,
        'febrero': 2, 'february': 2, 'feb': 2,
        'marzo': 3, 'march': 3, 'mar': 3,
        'abril': 4, 'april': 4, 'apr': 4, 'abr': 4,
        'mayo': 5, 'may': 5, 'may': 5,
        'junio': 6, 'june': 6, 'jun': 6,
        'julio': 7, 'july': 7, 'jul': 7,
        'agosto': 8, 'august': 8, 'aug': 8, 'ago': 8,
        'septiembre': 9, 'september': 9, 'sep': 9, 'sept': 9,
        'octubre': 10, 'october': 10, 'oct': 10,
        'noviembre': 11, 'november': 11, 'nov': 11,
        'diciembre': 12, 'december': 12, 'dec': 12, 'dic': 12,
        'primera': 1, 'first': 1,
        'segunda': 2, 'second': 2,
        'tercera': 3, 'third': 3,
        'cuarta': 4, 'fourth': 4
    }
    
    # Try to match the month name
    for name, num in month_mapping.items():
        if name in month_str:
            return num
    
    # Try to extract a number if it's a digit string
    if month_str.isdigit():
        month_num = int(month_str)
        return max(1, min(12, month_num))
    
    # Default to January if no match found
    return 1

def merge_datasets(df_mayorista: pd.DataFrame, df_minorista: pd.DataFrame) -> pd.DataFrame:
    """
    Merge mayorista and minorista datasets into a combined dataset.
    
    Args:
        df_mayorista: Mayorista DataFrame
        df_minorista: Minorista DataFrame
        
    Returns:
        Combined DataFrame
    """
    # Make copies to avoid modifying the originals
    df_mayorista_copy = df_mayorista.copy()
    df_minorista_copy = df_minorista.copy()
    
    # Standardize column names for merging
    # Create a consolidated "Precio" column for each dataset
    if 'Precio_Mayorista' in df_mayorista_copy.columns:
        # Use existing Precio_Mayorista column
        pass
    elif 'Mercado_Nuevo' in df_mayorista_copy.columns:
        df_mayorista_copy['Precio_Mayorista'] = df_mayorista_copy['Mercado_Nuevo']
    elif 'Mercado Nuevo' in df_mayorista_copy.columns:
        df_mayorista_copy['Precio_Mayorista'] = df_mayorista_copy['Mercado Nuevo']
    
    if 'Precio_Minorista' in df_minorista_copy.columns:
        # Use existing Precio_Minorista column
        pass
    else:
        # Try to identify a price column
        market_cols = [col for col in df_minorista_copy.columns if is_market_name(col)]
        if market_cols:
            logger.warning(f"Creating Precio_Minorista from market column: {market_cols[0]}")
            df_minorista_copy['Precio_Minorista'] = df_minorista_copy[market_cols[0]]
    
    # Create a unified "Precio" column for the combined dataset
    df_mayorista_copy['Precio'] = df_mayorista_copy['Precio_Mayorista'] if 'Precio_Mayorista' in df_mayorista_copy.columns else np.nan
    if 'Precio_Minorista' in df_minorista_copy.columns:
        df_minorista_copy['Precio'] = df_minorista_copy['Precio_Minorista']
    
    # Ensure both datasets have 'Mercado' column
    if 'Mercado' not in df_mayorista_copy.columns:
        df_mayorista_copy['Mercado'] = 'Mercado Nuevo'  # Default for mayorista
    
    if 'Mercado' not in df_minorista_copy.columns:
        # Try to find market information in another column
        for col in df_minorista_copy.columns:
            if is_market_name(col):
                df_minorista_copy['Mercado'] = col
                break
        else:
            df_minorista_copy['Mercado'] = 'Desconocido'
    
    # Ensure Tipo_Mercado column exists in both datasets
    if 'Tipo_Mercado' not in df_mayorista_copy.columns:
        df_mayorista_copy['Tipo_Mercado'] = 'Mayorista'
    
    if 'Tipo_Mercado' not in df_minorista_copy.columns:
        df_minorista_copy['Tipo_Mercado'] = 'Minorista'
    
    # Use concat for combining the datasets
    df_combined = pd.concat([df_mayorista_copy, df_minorista_copy], ignore_index=True, sort=False)
    
    # Fill in any missing Tipo_Mercado values
    if df_combined['Tipo_Mercado'].isna().any():
        # Try to infer from other columns
        if 'Categoria' in df_combined.columns:
            mask = df_combined['Tipo_Mercado'].isna()
            df_combined.loc[mask, 'Tipo_Mercado'] = df_combined.loc[mask, 'Categoria']
        else:
            # Default to 'Desconocido' for missing values
            df_combined['Tipo_Mercado'].fillna('Desconocido', inplace=True)
    
    # Asegurarse que no hay valores NaN en la columna Rubro
    if 'Rubro' in df_combined.columns and df_combined['Rubro'].isna().any():
        logger.warning(f"Se detectaron {df_combined['Rubro'].isna().sum()} valores NaN en Rubro. Aplicando corrección.")
        
        # Intentar asignar un Rubro basado en el Producto
        productos_con_nan = df_combined[df_combined['Rubro'].isna()]
        
        if 'Producto' in df_combined.columns:
            for idx in productos_con_nan.index:
                producto = df_combined.loc[idx, 'Producto']
                if pd.notna(producto):
                    # Mapear productos a rubros
                    if any(fruta in str(producto).lower() for fruta in ['naranja', 'mango', 'piña', 'limón', 'chirimoya', 'uva']):
                        df_combined.loc[idx, 'Rubro'] = 'Frutas'
                    elif any(vegetal in str(producto).lower() for vegetal in ['tomate', 'cebolla', 'ají', 'zanahoria']):
                        df_combined.loc[idx, 'Rubro'] = 'Hortalizas'
                    elif any(tuberculo in str(producto).lower() for tuberculo in ['papa', 'yuca', 'batata']):
                        df_combined.loc[idx, 'Rubro'] = 'Raíces-Tubérculos'
                    elif any(legumbre in str(producto).lower() for legumbre in ['habichuela', 'guandul', 'arveja']):
                        df_combined.loc[idx, 'Rubro'] = 'Legumbres'
                    else:
                        df_combined.loc[idx, 'Rubro'] = 'Frutas'  # Default
                else:
                    df_combined.loc[idx, 'Rubro'] = 'Frutas'  # Default para NaN
        else:
            # Si no hay column Producto, asignar un valor predeterminado
            df_combined['Rubro'].fillna('Frutas', inplace=True)
    
    # Log information about the merged dataset
    logger.info(f"Merged dataset: {len(df_combined)} rows, " +
                f"Mayorista: {(df_combined['Tipo_Mercado'] == 'Mayorista').sum()}, " +
                f"Minorista: {(df_combined['Tipo_Mercado'] == 'Minorista').sum()}")
    
    return df_combined

def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create sample data for testing and demonstration purposes.
    
    Returns:
        Tuple of (df_mayorista, df_minorista, df_combined)
    """
    # Base date for sample data
    base_date = datetime.now() - pd.DateOffset(days=365)
    dates = [base_date + pd.DateOffset(days=i*30) for i in range(12)]
    
    # Sample wholesale market data
    mercados_mayoristas = ['Mercado Nuevo', 'Mercado Central', 'Mercado Oriental']
    productos = [
        'Arroz Selecto',
        'Habichuela Roja',
        'Plátano Barahonero',
        'Yuca',
        'Batata',
        'Cebolla',
        'Ajo',
        'Zanahoria',
        'Papa',
        'Tomate'
    ]
    
    # Map products to categories
    rubros = {
        'Arroz Selecto': 'Cereal',
        'Habichuela Roja': 'Leguminosas',
        'Plátano Barahonero': 'Víveres',
        'Yuca': 'Víveres',
        'Batata': 'Víveres',
        'Cebolla': 'Vegetales',
        'Ajo': 'Vegetales',
        'Zanahoria': 'Vegetales',
        'Papa': 'Vegetales',
        'Tomate': 'Vegetales'
    }
    
    # Base prices for products
    precios_base = {
        'Arroz Selecto': 2500,
        'Habichuela Roja': 4000,
        'Plátano Barahonero': 1200,
        'Yuca': 800,
        'Batata': 700,
        'Cebolla': 1800,
        'Ajo': 6000,
        'Zanahoria': 1200,
        'Papa': 1500,
        'Tomate': 1600
    }
    
    # Create sample mayorista data
    data_mayorista = []
    
    for fecha in dates:
        for producto in productos:
            precio_base = precios_base[producto]
            
            # Add seasonal pattern
            mes = fecha.month
            precio = precio_base * (1 + 0.15 * np.sin(mes / 6 * np.pi))
            
            # Add random noise
            precio = precio * (1 + np.random.uniform(-0.05, 0.05))
            
            # Create row for each market
            for mercado in mercados_mayoristas:
                # Add market-specific variation
                precio_mercado = precio * (1 + np.random.uniform(-0.08, 0.08))
                
                data_mayorista.append({
                    'Producto': producto,
                    'Rubro': rubros[producto],
                    'Tipo_Mercado': 'Mayorista',
                    'Unidad': 'Saco/100 libras' if producto == 'Arroz Selecto' else 'Quintal',
                    mercado: round(precio_mercado, 0),
                    'Semana': (fecha.day // 7) + 1,
                    'Mes': fecha.month,
                    'Año': fecha.year,
                    'Fecha': fecha
                })
    
    # Create DataFrame and pivot to wide format
    df_mayorista = pd.DataFrame(data_mayorista)
    
    # Create sample minorista data
    mercados_minoristas = ['Mercado Minorista 1', 'Mercado Minorista 2', 'Supermercado', 'Tienda Local']
    data_minorista = []
    
    for fecha in dates:
        for producto in productos:
            precio_base = precios_base[producto] / 80  # Price per pound is lower
            
            # Add seasonal pattern
            mes = fecha.month
            precio = precio_base * (1 + 0.12 * np.sin(mes / 6 * np.pi))
            
            # Create row for each market
            for mercado in mercados_minoristas:
                # Add market-specific variation
                precio_mercado = precio * (1 + np.random.uniform(-0.1, 0.2))
                
                data_minorista.append({
                    'Producto': producto,
                    'Rubro': rubros[producto],
                    'Tipo_Mercado': 'Minorista',
                    'Unidad': 'Libra',
                    mercado: round(precio_mercado, 2),
                    'Semana': (fecha.day // 7) + 1,
                    'Mes': fecha.month,
                    'Año': fecha.year,
                    'Fecha': fecha
                })
    
    # Create DataFrame for minorista
    df_minorista = pd.DataFrame(data_minorista)
    
    # Combine datasets
    df_combined = pd.concat([df_mayorista, df_minorista], ignore_index=True)
    
    return df_mayorista, df_minorista, df_combined