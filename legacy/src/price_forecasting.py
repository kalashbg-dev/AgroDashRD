"""
Módulo de pronóstico de precios para AgroDashRD.

Este módulo implementa algoritmos de predicción para precios agrícolas
utilizando técnicas como Prophet y XGBoost.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import datetime
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

# Importaciones para modelos de pronóstico
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def prepare_data_for_prophet(df: pd.DataFrame, product: str, market: Optional[str] = None, 
                           price_col: str = 'Precio_Mayorista') -> pd.DataFrame:
    """
    Prepara los datos para el modelo de Prophet.
    
    Args:
        df: DataFrame con datos
        product: Producto a analizar
        market: Mercado específico (opcional)
        price_col: Columna de precio a utilizar
        
    Returns:
        DataFrame con formato para Prophet (ds, y)
    """
    # Filtrar por producto
    filtered_df = df[df['Producto'] == product].copy()
    
    # Filtrar por mercado si se especifica
    if market and market != 'Todos los mercados':
        filtered_df = filtered_df[filtered_df['Mercado'] == market]
    
    # Verificar si hay suficientes datos
    if len(filtered_df) < 30:
        st.warning(f"No hay suficientes datos para pronosticar precios de {product}. Se necesitan al menos 30 registros.")
        return pd.DataFrame()
    
    # Asegurarse que la fecha está en formato datetime
    if not pd.api.types.is_datetime64_dtype(filtered_df['Fecha']):
        filtered_df['Fecha'] = pd.to_datetime(filtered_df['Fecha'])
    
    # Asegurarse que la columna de precios es numérica
    filtered_df[price_col] = pd.to_numeric(filtered_df[price_col], errors='coerce')
    
    # Filtrar valores no numéricos
    filtered_df = filtered_df[filtered_df[price_col].notna()]
    
    # Agregar por fecha y calcular precio promedio
    prophet_df = filtered_df.groupby('Fecha')[price_col].mean().reset_index()
    
    # Renombrar columnas para Prophet
    prophet_df = prophet_df.rename(columns={'Fecha': 'ds', price_col: 'y'})
    
    # Ordenar por fecha
    prophet_df = prophet_df.sort_values('ds')
    
    return prophet_df

def train_prophet_model(df: pd.DataFrame, forecast_days: int = 90, 
                       seasonality_mode: str = 'additive', 
                       yearly_seasonality: bool = True,
                       weekly_seasonality: bool = True,
                       daily_seasonality: bool = False) -> Tuple[Prophet, pd.DataFrame]:
    """
    Entrena un modelo de Prophet y genera pronósticos.
    
    Args:
        df: DataFrame con formato para Prophet (ds, y)
        forecast_days: Número de días a pronosticar
        seasonality_mode: Modo de estacionalidad ('additive' o 'multiplicative')
        yearly_seasonality: Incluir estacionalidad anual
        weekly_seasonality: Incluir estacionalidad semanal
        daily_seasonality: Incluir estacionalidad diaria
        
    Returns:
        Tupla con (modelo entrenado, dataframe con pronóstico)
    """
    if df.empty:
        return None, pd.DataFrame()
    
    # Configurar modelo
    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        interval_width=0.95  # Intervalo de confianza del 95%
    )
    
    # Entrenar modelo
    with st.spinner('Entrenando modelo de pronóstico...'):
        model.fit(df)
    
    # Crear DataFrame para predicción futura
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Generar pronósticos
    forecast = model.predict(future)
    
    return model, forecast

def create_time_series_features(df: pd.DataFrame, rolling_windows: list = [7, 14, 30]) -> pd.DataFrame:
    """
    Crea características temporales a partir de una columna de fecha y agrega rolling means.
    
    Args:
        df: DataFrame con columna 'ds' de tipo datetime
        rolling_windows: Ventanas para promedios móviles
    Returns:
        DataFrame con características temporales añadidas
    """
    df = df.copy()
    
    # Extraer componentes de fecha
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    
    # Añadir características cíclicas
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear']/365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear']/365)

    # Agregar rolling means
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df['y'].rolling(window=window, min_periods=1).mean()
    
    return df

def add_lag_features(df: pd.DataFrame, lags: list = [7, 14, 30, 60, 90]) -> pd.DataFrame:
    """
    Añade características de rezago a la serie temporal.
    
    Args:
        df: DataFrame con columna 'y' (valores objetivo)
        lags: Lista de rezagos a añadir
        
    Returns:
        DataFrame con columnas de rezago añadidas
    """
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['y'].shift(lag)
    return df


def get_seasonality_info(model: Prophet) -> Dict:
    """
    Extrae información de estacionalidad del modelo Prophet.
    
    Args:
        model: Modelo Prophet entrenado
        
    Returns:
        Diccionario con información de estacionalidad
    """
    if not model:
        return {}
    
    seasonality_info = {}
    
    # Componentes de estacionalidad
    if model.seasonalities:
        for name, props in model.seasonalities.items():
            seasonality_info[name] = {
                'period': props['period'],
                'mode': props['mode'],
                'fourier_order': props['fourier_order']
            }
    
    return seasonality_info

def evaluate_forecast_model(historical: pd.DataFrame, forecast: pd.DataFrame) -> Dict:
    """
    Evalúa el modelo de pronóstico calculando métricas de error.
    
    Args:
        historical: DataFrame con datos históricos (ds, y)
        forecast: DataFrame con pronóstico de Prophet
        
    Returns:
        Diccionario con métricas de evaluación
    """
    if historical.empty or forecast.empty:
        return {}
    
    # Obtener solo las fechas comunes entre histórico y pronóstico
    common_dates = set(historical['ds']).intersection(set(forecast['ds']))
    
    if not common_dates:
        return {}
    
    # Filtrar dataframes
    hist_common = historical[historical['ds'].isin(common_dates)]
    fore_common = forecast[forecast['ds'].isin(common_dates)]
    
    # Asegurar mismo orden
    hist_common = hist_common.sort_values('ds')
    fore_common = fore_common.sort_values('ds')
    
    # Calcular métricas
    y_true = hist_common['y'].values
    y_pred = fore_common['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calcular Error Porcentual Absoluto Medio (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calcular Error Porcentual Absoluto Medio Simétrico (sMAPE)
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'sMAPE': smape
    }

def create_forecast_plot(historical: pd.DataFrame, forecast: pd.DataFrame, 
                        product: str, market: Optional[str] = None,
                        confidence_interval: bool = True) -> go.Figure:
    """
    Crea un gráfico interactivo del pronóstico.
    
    Args:
        historical: DataFrame con datos históricos (ds, y)
        forecast: DataFrame con pronóstico de Prophet
        product: Nombre del producto
        market: Nombre del mercado (opcional)
        confidence_interval: Mostrar intervalo de confianza
        
    Returns:
        Figura de Plotly
    """
    if historical.empty or forecast.empty:
        # Crear figura vacía con mensaje
        fig = go.Figure()
        fig.add_annotation(
            text="No hay datos suficientes para generar el pronóstico",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Crear figura
    fig = go.Figure()
    
    # Agregar datos históricos
    fig.add_trace(
        go.Scatter(
            x=historical['ds'], 
            y=historical['y'],
            mode='markers+lines',
            name='Datos históricos',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=5, color='#1f77b4')
        )
    )
    
    # Agregar línea de pronóstico
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat'],
            mode='lines',
            name='Pronóstico',
            line=dict(color='#ff7f0e', width=3, dash='solid')
        )
    )
    
    # Agregar intervalo de confianza
    if confidence_interval and 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
        # Filtrar solo fechas futuras para el intervalo
        last_historical_date = historical['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_historical_date]
        
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Límite superior',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Límite inferior',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                showlegend=False
            )
        )
    
    # Personalizar gráfico
    title = f"Pronóstico de Precios: {product}"
    if market and market != 'Todos los mercados':
        title += f" - {market}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Precio (RD$)",
        legend_title="Datos",
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    
    # Destacar área de pronóstico futuro
    if not historical.empty:
        last_historical_date = historical['ds'].max()
        
        fig.add_shape(
            type="line",
            x0=last_historical_date,
            y0=0,
            x1=last_historical_date,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.add_annotation(
            x=last_historical_date,
            y=1,
            yref="paper",
            text="Inicio Pronóstico",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40,
            font=dict(color="red")
        )
    
    return fig

def render_forecast_wizard(df: pd.DataFrame) -> None:
    """
    Renderiza un asistente interactivo de pronóstico de precios.
    
    Args:
        df: DataFrame con datos completos
    """
    st.markdown("""
    <div class="fade-in">
    <h2>🧙‍♂️ Asistente de Pronóstico de Precios</h2>
    <p>Este asistente le guiará en la creación de un modelo de pronóstico 
    personalizado para predecir precios futuros de productos agrícolas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Asegurarse que la fecha está en formato datetime
    if 'Fecha' in df.columns and not pd.api.types.is_datetime64_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Paso 1: Selección de producto y mercado
    st.markdown("### Paso 1: Seleccione el producto y mercado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lista de productos disponibles
        productos = sorted(df['Producto'].unique().tolist())
        producto_seleccionado = st.selectbox(
            "Producto:",
            productos,
            key="forecast_product"
        )
    
    with col2:
        # Lista de mercados para el producto seleccionado
        df_producto = df[df['Producto'] == producto_seleccionado]
        mercados = ['Todos los mercados'] + sorted(df_producto['Mercado'].unique().tolist())
        mercado_seleccionado = st.selectbox(
            "Mercado:",
            mercados,
            key="forecast_market"
        )
    
    # Paso 2: Configuración del modelo
    st.markdown("### Paso 2: Configure el modelo de pronóstico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seleccionar tipo de precio
        precio_tipo = st.radio(
            "Tipo de Precio:",
            ["Mayorista", "Minorista"],
            key="forecast_price_type"
        )
        
        precio_col = f"Precio_{precio_tipo}"
        
        # Seleccionar horizonte de pronóstico
        horizonte = st.slider(
            "Horizonte de Pronóstico (días):",
            min_value=30,
            max_value=365,
            value=90,
            step=30,
            key="forecast_horizon"
        )
    
    with col2:
        # Seleccionar algoritmo
        algoritmo = st.selectbox(
            "Algoritmo de Pronóstico:",
            ["Prophet"],
            key="forecast_algorithm"
        )
        
        # Configuraciones adicionales según algoritmo
        if algoritmo == "Prophet":
            modo_estacionalidad = st.selectbox(
                "Modo de Estacionalidad:",
                ["aditivo", "multiplicativo"],
                index=0,
                key="forecast_seasonality_mode"
            )
            
            # Mapear a los valores esperados por Prophet
            modo_estacionalidad_map = {"aditivo": "additive", "multiplicativo": "multiplicative"}
            modo_estacionalidad_value = modo_estacionalidad_map[modo_estacionalidad]
            
            # Componentes estacionales
            estacionalidad_anual = st.checkbox("Estacionalidad Anual", value=True, key="forecast_yearly")
            estacionalidad_semanal = st.checkbox("Estacionalidad Semanal", value=True, key="forecast_weekly")
            estacionalidad_diaria = st.checkbox("Estacionalidad Diaria", value=False, key="forecast_daily")
    
    # Botón para entrenar modelo
    if st.button("Generar Pronóstico", use_container_width=True, key="btn_generate_forecast"):
        # Preparar datos
        datos_prophet = prepare_data_for_prophet(
            df, 
            producto_seleccionado,
            mercado_seleccionado,
            precio_col
        )
        
        if datos_prophet.empty:
            st.error(f"No hay suficientes datos para pronosticar precios de {producto_seleccionado}.")
        else:
            # Entrenar modelo según algoritmo seleccionado
            if algoritmo == "Prophet":
                model, forecast = train_prophet_model(
                    datos_prophet,
                    forecast_days=horizonte,
                    seasonality_mode=modo_estacionalidad_value,
                    yearly_seasonality=estacionalidad_anual,
                    weekly_seasonality=estacionalidad_semanal,
                    daily_seasonality=estacionalidad_diaria
                )
                        
            if forecast.empty:
                st.error("No se pudo generar el pronóstico. Intente con otros parámetros.")
            else:
                # Mostrar resultados
                st.success(f"¡Pronóstico generado con éxito para {horizonte} días!")
                
                # Visualización de pronóstico
                st.markdown("### Resultados del Pronóstico")
                
                # Crear gráfico
                fig = create_forecast_plot(
                    datos_prophet,
                    forecast,
                    producto_seleccionado,
                    mercado_seleccionado
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar métricas de evaluación
                metricas = evaluate_forecast_model(datos_prophet, forecast)
                
                if metricas:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("MAE", f"{metricas['MAE']:.2f}")
                    
                    with col2:
                        st.metric("RMSE", f"{metricas['RMSE']:.2f}")
                    
                    with col3:
                        st.metric("R²", f"{metricas['R2']:.2f}")
                    
                    with col4:
                        st.metric("MAPE", f"{metricas['MAPE']:.2f}%")
                    
                    with col5:
                        st.metric("sMAPE", f"{metricas['sMAPE']:.2f}%")
                
                # Tabla de pronósticos
                st.markdown("### Tabla de Pronósticos")
                
                # Separar pronósticos históricos de futuros
                last_historical_date = datos_prophet['ds'].max()
                future_forecast = forecast[forecast['ds'] > last_historical_date].copy()
                
                # Formatear tabla para mostrar
                future_forecast['Fecha'] = future_forecast['ds'].dt.strftime('%d/%m/%Y')
                future_forecast['Precio Pronosticado'] = future_forecast['yhat'].map('RD$ {:.2f}'.format)
                future_forecast['Precio Mínimo'] = future_forecast['yhat_lower'].map('RD$ {:.2f}'.format)
                future_forecast['Precio Máximo'] = future_forecast['yhat_upper'].map('RD$ {:.2f}'.format)
                
                # Mostrar tabla de pronósticos futuros
                st.dataframe(
                    future_forecast[['Fecha', 'Precio Pronosticado', 'Precio Mínimo', 'Precio Máximo']],
                    use_container_width=True
                )
                
                # Análisis de estacionalidad (solo para Prophet)
                if algoritmo == "Prophet" and model:
                    st.markdown("### Componentes del Modelo")
                    
                    # Mostrar componentes (tendencia, estacionalidad)
                    fig_components = model.plot_components(forecast)
                    st.pyplot(fig_components)
                    
                    # Información de estacionalidad
                    seasonality_info = get_seasonality_info(model)
                    
                    if seasonality_info:
                        st.markdown("### Información de Estacionalidad")
                        
                        for name, info in seasonality_info.items():
                            st.write(f"**{name.capitalize()}**: Período {info['period']} días, Modo {info['mode']}")
                
                # Recomendaciones basadas en pronóstico
                st.markdown("### Recomendaciones Estratégicas")
                
                # Calcular tendencia general
                first_future_price = future_forecast['yhat'].iloc[0]
                last_future_price = future_forecast['yhat'].iloc[-1]
                
                change_pct = ((last_future_price - first_future_price) / first_future_price) * 100
                
                if change_pct > 5:
                    st.info(f"""
                    * Se pronostica una tendencia **alcista** de {change_pct:.1f}% para {producto_seleccionado} en los próximos {horizonte} días.
                    * **Recomendación para productores**: Considere planificar su cosecha para capitalizar los precios más altos hacia el final del período.
                    * **Recomendación para compradores**: Considere asegurar contratos de suministro ahora para mitigar el aumento de precios.
                    """)
                elif change_pct < -5:
                    st.info(f"""
                    * Se pronostica una tendencia **bajista** de {abs(change_pct):.1f}% para {producto_seleccionado} en los próximos {horizonte} días.
                    * **Recomendación para productores**: Considere asegurar contratos de venta ahora para protegerse contra la caída de precios.
                    * **Recomendación para compradores**: Puede ser ventajoso esperar para comprar cuando los precios bajen más.
                    """)
                else:
                    st.info(f"""
                    * Se pronostica una tendencia **estable** para {producto_seleccionado} en los próximos {horizonte} días.
                    * **Recomendación general**: Las condiciones de mercado parecen estables, lo que facilita la planificación a corto plazo.
                    """)
                
                # Identificar mejor momento para comprar/vender
                best_time_to_buy = future_forecast.loc[future_forecast['yhat'].idxmin(), 'Fecha']
                best_time_to_sell = future_forecast.loc[future_forecast['yhat'].idxmax(), 'Fecha']
                
                st.markdown(f"""
                * **Mejor momento para comprar**: Alrededor del {best_time_to_buy}
                * **Mejor momento para vender**: Alrededor del {best_time_to_sell}
                """)

def show_forecast_dashboard(df: pd.DataFrame) -> None:
    """
    Muestra el dashboard de pronóstico de precios.
    
    Args:
        df: DataFrame con datos
    """
    st.markdown("""
    <div class="fade-in">
    <h1>🔮 Pronóstico de Precios</h1>
    <p>Esta herramienta utiliza modelos avanzados de aprendizaje automático para predecir
    cómo evolucionarán los precios de los productos agrícolas en el futuro.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Crear pestañas
    tab1, tab2 = st.tabs([
        "📊 Asistente de Pronóstico", 
        "📘 Guía de Uso"
    ])
    
    # Tab 1: Asistente de Pronóstico
    with tab1:
        render_forecast_wizard(df)
    
    # Tab 2: Guía de Uso
    with tab2:
        st.markdown("""
        ## Guía de Uso: Pronóstico de Precios
        
        ### ¿Qué es el pronóstico de precios?
        
        El pronóstico de precios utiliza datos históricos y técnicas estadísticas avanzadas para predecir 
        cómo evolucionarán los precios en el futuro. Estas predicciones pueden ayudarle a tomar decisiones 
        más informadas sobre cuándo comprar, vender o almacenar productos.
        
        ### Modelo disponible
        
        #### Prophet
        
        Desarrollado por Facebook, Prophet es un modelo especializado en series temporales que:
        - Maneja automáticamente los efectos estacionales (diarios, semanales, anuales)
        - Es robusto ante datos faltantes y valores atípicos
        - Proporciona intervalos de confianza para los pronósticos
        
        *Actualmente Prophet es el único modelo disponible en la plataforma, ya que ha demostrado ser el más robusto y confiable para los datos agrícolas de este sistema.*
        
        ### Interpretación de resultados
        
        #### Métricas de evaluación
        
        - **MAE (Error Absoluto Medio)**: Promedio de los errores absolutos. Valores más bajos son mejores.
        - **RMSE (Raíz del Error Cuadrático Medio)**: Similar a MAE pero penaliza más los errores grandes.
        - **R² (Coeficiente de Determinación)**: Mide qué tan bien el modelo explica la variabilidad. Valores cercanos a 1 son mejores.
        - **MAPE (Error Porcentual Absoluto Medio)**: Error promedio expresado como porcentaje.
        - **sMAPE (Error Porcentual Absoluto Medio Simétrico)**: Versión mejorada de MAPE que maneja mejor los valores cercanos a cero.
        
        #### Componentes del modelo (Prophet)
        
        El gráfico de componentes descompone el pronóstico en:
        - **Tendencia**: Dirección general a largo plazo
        - **Estacionalidad anual**: Patrones que se repiten cada año
        - **Estacionalidad semanal**: Patrones que se repiten cada semana
        - **Estacionalidad diaria**: Patrones que se repiten cada día (si está habilitada)
        
        ### Consejos para mejores resultados
        
        1. **Elija el horizonte adecuado**: Los pronósticos a corto plazo (30-90 días) suelen ser más precisos que los de largo plazo.
        2. **Compare algoritmos**: Pruebe tanto Prophet como XGBoost para ver cuál funciona mejor con sus datos.
        3. **Ajuste la estacionalidad**: Si conoce patrones estacionales específicos para un producto, ajuste la configuración en consecuencia.
        4. **Considere el contexto**: Los pronósticos son herramientas, no oráculos. Combine los resultados con su conocimiento del mercado.
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
    show_forecast_dashboard(df)