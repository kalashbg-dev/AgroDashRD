import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configuración de logging
from src.utils.logging_config import logger as log

# Import custom modules
from src.data_loader import load_and_preprocess_data
from src.normalize import normalize_dataframe_categories, normalize_rubro, normalize_producto
from src.animations import add_animation_css, create_animated_plotly
# Reemplazar con versión optimizada
from src.optimized_outlier_detection import show_statistical_analysis
from src.value_distribution import show_value_distribution

# Set page config - DEBE SER EL PRIMER COMANDO DE STREAMLIT
st.set_page_config(page_title="AgroDashRD",
                   page_icon="assets/favicon.ico",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Configurar logging
log.info("Iniciando aplicación AgroDashRD")

# Load all custom CSS styles
for css_file in ['assets/styles.css']:
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Agregar JavaScript para manejar la navegación en móviles
st.markdown("""
<script>
// Este script se ejecutará cuando la página esté completamente cargada
document.addEventListener('DOMContentLoaded', function() {
    // Toggle para sidebar en móviles
    const toggleSidebar = () => {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        sidebar.setAttribute('aria-expanded', 
            sidebar.getAttribute('aria-expanded') === 'true' ? 'false' : 'true');
    };
    
    // Crear botón de toggle para dispositivos móviles
    const createToggleButton = () => {
        const button = document.createElement('div');
        button.className = 'sidebar-toggle';
        button.innerHTML = '☰';
        button.addEventListener('click', toggleSidebar);
        document.body.appendChild(button);
    };
    
    // Solo mostrar en móviles
    if (window.innerWidth <= 768) {
        createToggleButton();
    }
});
</script>
""", unsafe_allow_html=True)

# Add logo to sidebar
st.sidebar.image("assets/logo.svg", width=200)

# Configuración de sesión
if 'user_type' not in st.session_state:
    st.session_state.user_type = None  # Ningún perfil seleccionado inicialmente

# Añadir CSS para animaciones
add_animation_css()

# Función para crear datos de muestra (solo usada como fallback)
def create_sample_data():
    # Crear dataset mayorista de ejemplo
    df_mayorista = pd.DataFrame({
        'Fecha': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        'Mercado': np.random.choice(['Mercado Central', 'MERCA', 'Mercado del Este'], 100),
        'Producto': np.random.choice(['Plátano', 'Tomate', 'Cebolla', 'Arroz'], 100),
        'Rubro': np.random.choice(['Vegetales', 'Frutas', 'Granos'], 100),
        'Precio_Mayorista': np.random.uniform(10, 100, 100)
    })

    # Crear dataset minorista de ejemplo
    df_minorista = pd.DataFrame({
        'Fecha': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        'Mercado': np.random.choice(['Supermercado Nacional', 'Mercado Villa Consuelo', 'Mercado Nuevo'], 100),
        'Producto': np.random.choice(['Plátano', 'Tomate', 'Cebolla', 'Arroz'], 100),
        'Rubro': np.random.choice(['Vegetales', 'Frutas', 'Granos'], 100),
        'Precio_Minorista': np.random.uniform(20, 120, 100)
    })

    # Combinar ambos datasets
    df_combined = pd.concat([df_mayorista, df_minorista]).reset_index(drop=True)

    return df_mayorista, df_minorista, df_combined

# Función para crear filtros
def create_filters(df):
    st.sidebar.header("Filtros")

    filters = {}

    # Filtro de fecha
    if 'Fecha' in df.columns:
        min_date = df['Fecha'].min() if not pd.isna(df['Fecha'].min()) else datetime(2020, 1, 1)
        max_date = df['Fecha'].max() if not pd.isna(df['Fecha'].max()) else datetime.now()

        date_range = st.sidebar.date_input(
            "Rango de Fechas",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            filters['date_range'] = date_range

    # Filtro de producto
    if 'Producto' in df.columns:
        # Convertir todos los valores a string y eliminar valores nulos
        productos = ['Todos'] + sorted(
            [str(p) for p in df['Producto'].dropna().unique()],
            key=lambda x: str(x).lower()
        )
        producto_selected = st.sidebar.selectbox("Producto", productos)

        if producto_selected != 'Todos':
            filters['Producto'] = producto_selected

    # Filtro de rubro
    if 'Rubro' in df.columns:
        # Convertir todos los valores a string y eliminar valores nulos
        rubros = ['Todos'] + sorted(
            [str(r) for r in df['Rubro'].dropna().unique()],
            key=lambda x: str(x).lower()
        )
        rubro_selected = st.sidebar.selectbox("Rubro", rubros)

        if rubro_selected != 'Todos':
            filters['Rubro'] = rubro_selected

    # Filtro de mercado
    if 'Mercado' in df.columns:
        # Convertir todos los valores a string y eliminar valores nulos
        mercados = ['Todos'] + sorted(
            [str(m) for m in df['Mercado'].dropna().unique()],
            key=lambda x: str(x).lower()
        )
        mercado_selected = st.sidebar.selectbox("Mercado", mercados)

        if mercado_selected != 'Todos':
            filters['Mercado'] = mercado_selected

    return filters

# Función para aplicar filtros
def apply_filters(df, filters):
    df_filtered = df.copy()

    # Aplicar filtro de fecha
    if 'date_range' in filters and len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        if 'Fecha' in df_filtered.columns:
            try:
                df_filtered = df_filtered[(df_filtered['Fecha'] >= pd.Timestamp(start_date)) & 
                                        (df_filtered['Fecha'] <= pd.Timestamp(end_date))]
            except Exception as e:
                st.sidebar.warning(f"Error al filtrar por fecha: {str(e)}")

    # Aplicar otros filtros
    for column, value in filters.items():
        if column != 'date_range' and column in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[column] == value]

    # Log filtered shape
    log.info(f"Filtered DataFrame shape: {df_filtered.shape}")

    return df_filtered

# Función para comparar precios
def plot_price_comparison(df, price_col, group_col='Mercado'):
    if group_col not in df.columns or price_col not in df.columns:
        return None

    # Agrupar por la columna seleccionada
    grouped = df.groupby(group_col)[price_col].mean().reset_index()
    grouped = grouped.sort_values(price_col, ascending=False)

    # Crear gráfico
    fig = px.bar(
        grouped, 
        x=group_col, 
        y=price_col,
        color=group_col,
        title=f"Comparación de {price_col} por {group_col}",
        labels={price_col: f"Promedio de {price_col}"}
    )

    return fig

# Función para visualizar precios a lo largo del tiempo
def plot_price_over_time(df, date_col, price_col, group_col=None):
    if date_col not in df.columns or price_col not in df.columns:
        return None

    if group_col and group_col in df.columns:
        # Si se proporciona una columna de agrupación, crear una línea por grupo
        grouped = df.groupby([pd.Grouper(key=date_col, freq='D'), group_col])[price_col].mean().reset_index()

        fig = px.line(
            grouped, 
            x=date_col, 
            y=price_col, 
            color=group_col,
            title=f"Evolución de {price_col} por {group_col}",
            labels={price_col: price_col, date_col: "Fecha"}
        )
    else:
        # Si no hay agrupación, crear una sola línea con el promedio diario
        grouped = df.groupby(pd.Grouper(key=date_col, freq='D'))[price_col].mean().reset_index()

        fig = px.line(
            grouped, 
            x=date_col, 
            y=price_col,
            title=f"Evolución de {price_col}",
            labels={price_col: price_col, date_col: "Fecha"}
        )

    return fig

# Función para asegurar que existe una columna de precio
def ensure_price_column_exists(df, price_col):
    """
    Ensures that the specified price column exists in the DataFrame.
    If not, tries to create it from available price data.

    Args:
        df: DataFrame to check/modify
        price_col: Name of the price column to ensure

    Returns:
        Tuple of (modified DataFrame, success flag)
    """
    if price_col in df.columns:
        return df, True

    # Si no existe, trata de crear la columna a partir de datos disponibles
    df_modified = df.copy()

    # Si estamos buscando precio mayorista pero solo tenemos minorista
    if price_col == 'Precio_Mayorista' and 'Precio_Minorista' in df.columns:
        # Crear una aproximación usando un factor de conversión
        df_modified['Precio_Mayorista'] = df['Precio_Minorista'] * 0.8
        return df_modified, True

    # Si estamos buscando precio minorista pero solo tenemos mayorista
    elif price_col == 'Precio_Minorista' and 'Precio_Mayorista' in df.columns:
        # Crear una aproximación usando un factor de conversión
        df_modified['Precio_Minorista'] = df['Precio_Mayorista'] * 1.25
        return df_modified, True

    return df, False

# Cargar datos con caché
try:
    log.info("Cargando datos...")
    from src.cache_manager import process_and_cache_data
    
    # Cargar datos usando el sistema de caché
    df_mayorista, df_minorista, df_combined = process_and_cache_data()
    
    # Asegurarse de que las columnas de fecha sean datetime
    for df in [df_mayorista, df_minorista, df_combined]:
        if 'Fecha' in df.columns and not pd.api.types.is_datetime64_dtype(df['Fecha']):
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    
    # Filtrar fechas futuras (más allá de hoy)
    today = pd.Timestamp.now().normalize()
    df_mayorista = df_mayorista[df_mayorista['Fecha'] <= today]
    df_minorista = df_minorista[df_minorista['Fecha'] <= today]
    df_combined = df_combined[df_combined['Fecha'] <= today]

    # Normalizar categorías para consistencia en filtros
    log.info("Normalizando categorías de productos y rubros...")
    df_combined = normalize_dataframe_categories(df_combined)

    # Normalizar categorías en datasets individuales para consistencia
    if 'Rubro' in df_mayorista.columns:
        df_mayorista['Rubro'] = df_mayorista['Rubro'].apply(normalize_rubro)
    if 'Producto' in df_mayorista.columns:
        df_mayorista['Producto'] = df_mayorista['Producto'].apply(normalize_producto)

    if 'Rubro' in df_minorista.columns:
        df_minorista['Rubro'] = df_minorista['Rubro'].apply(normalize_rubro)
    if 'Producto' in df_minorista.columns:
        df_minorista['Producto'] = df_minorista['Producto'].apply(normalize_producto)

except Exception as e:
    st.sidebar.warning(f"Error al cargar datos: {str(e)}")
    # Fallback to sample data
    df_mayorista, df_minorista, df_combined = create_sample_data()
    st.sidebar.info("Usando datos de muestra para demostración")

# Create and apply filters
filters = create_filters(df_combined)

# Show active filter count
filter_count = sum(1 for k, v in filters.items() if v)
st.sidebar.write(f"**{filter_count}** filtros activos")

# Apply filters
df_filtered = apply_filters(df_combined, filters)

# Display filtered data sample in sidebar
with st.sidebar.expander("Vista previa de datos filtrados"):
    st.dataframe(df_filtered.head(5))


# Verificar si ya se seleccionó un perfil
if 'user_type' not in st.session_state:
    st.session_state.user_type = None

if st.session_state.user_type is None:
    # Mostrar selección de perfil
    st.title("🌱 Bienvenido a AgroDashRD")
    st.markdown("""
    <div class="fade-in">
    <p>Seleccione su perfil para acceder a la información más relevante para sus necesidades:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contenedor con las opciones de perfil
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧑‍🌾 Agricultor
        Información práctica y sencilla para:
        - Consultar precios actuales
        - Calcular valor de cosechas
        - Encontrar mejores mercados
        - Planificar siembras
        """)
        if st.button("Seleccionar perfil de Agricultor", key="btn_agricultor", use_container_width=True):
            st.session_state.user_type = "agricultor"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 👨‍💼 Profesional
        Herramientas avanzadas para:
        - Análisis estadístico detallado
        - Comparación entre mercados
        - Detección de anomalías
        - Visualización de tendencias
        """)
        if st.button("Seleccionar perfil Profesional", key="btn_profesional", use_container_width=True):
            st.session_state.user_type = "profesional"
            st.rerun()
    
    # Salir temprano de la aplicación si no se ha seleccionado perfil
    st.stop()

# Mostrar selector para cambiar perfil
st.sidebar.markdown("---")
current_profile = "🧑‍🌾 Agricultor" if st.session_state.user_type == "agricultor" else "👨‍💼 Profesional"
st.sidebar.markdown(f"**Perfil actual:** {current_profile}")
if st.sidebar.button("Cambiar perfil", key="change_profile"):
    st.session_state.user_type = None
    st.rerun()

# Determinar sección principal según el perfil seleccionado
main_section = "🧑‍🌾 Dashboard Agricultor" if st.session_state.user_type == "agricultor" else "📊 Análisis Profesional"

if main_section == "🧑‍🌾 Dashboard Agricultor":
    # Importar el módulo de dashboard para agricultores
    from src.dashboard_agricultor import mostrar_dashboard_agricultor

    st.title("🌱 Dashboard para Agricultores")
    st.markdown("""
    <div class="fade-in">
    <p>Este dashboard está diseñado para agricultores y productores, ofreciendo información
    práctica y fácil de entender para tomar mejores decisiones de cultivo y comercialización.</p>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar el dashboard para agricultores
    mostrar_dashboard_agricultor(df_filtered)

else:  # Análisis Profesional
    # Importar el módulo de dashboard para profesionales
    from src.dashboard_profesional import mostrar_dashboard_profesional

    st.title("📈 Análisis Profesional")
    st.markdown("""
    <div class="fade-in">
    <p>Esta sección contiene herramientas avanzadas de análisis para profesionales
    del sector agrícola, analistas de mercado e investigadores. Explore análisis
    detallados de tendencias, estacionalidad y comportamiento del mercado.</p>
    </div>
    """, unsafe_allow_html=True)

    # Mostrar el dashboard para profesionales
    mostrar_dashboard_profesional(df_filtered)
