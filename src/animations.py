"""
Módulo de animaciones y efectos visuales para AgroDashRD.

Este módulo proporciona componentes y utilidades para mejorar
la experiencia de usuario con animaciones y transiciones suaves.
"""

import time
import streamlit as st

def add_animation_css():
    """
    Ya no es necesario añadir CSS aquí ya que las animaciones están en styles.css
    """
    pass

def animated_container(content_function, animation_type="fade-in"):
    """
    Envuelve el contenido en un div con una animación específica.
    """
    container = st.container()
    container.markdown(f'<div class="{animation_type}">', unsafe_allow_html=True)
    with container:
        content_function()
    container.markdown('</div>', unsafe_allow_html=True)

def staggered_animations(items, container_function, item_function):
    """
    Crea animaciones escalonadas para una lista de elementos.
    """
    st.markdown('<div class="staggered-animation">', unsafe_allow_html=True)
    containers = container_function(len(items))
    for i, item in enumerate(items):
        with containers[i]:
            item_function(item)
    st.markdown('</div>', unsafe_allow_html=True)

def animated_number(value, prefix="", suffix="", animation_duration=1.0, key=None):
    """
    Muestra un número con animación de conteo.
    """
    if key is None:
        key = f"animated_number_{int(time.time() * 1000)}"
    number_placeholder = st.empty()
    js_code = f"""
    <div id="{key}" class="animated-number">{prefix}0{suffix}</div>
    <script>
        (function() {{
            const el = document.getElementById("{key}");
            const final = {value};
            const duration = {animation_duration} * 1000;
            const start = performance.now();
            function animate(time) {{
                const elapsed = time - start;
                const progress = Math.min(elapsed / duration, 1);
                const current = Math.floor(progress * final);
                el.textContent = "{prefix}" + current.toLocaleString() + "{suffix}";
                if (progress < 1) {{ requestAnimationFrame(animate); }}
            }}
            requestAnimationFrame(animate);
        }})();
    </script>
    """
    number_placeholder.markdown(js_code, unsafe_allow_html=True)

def price_change_indicator(current_value, previous_value, prefix="RD$ ", suffix=""):
    """
    Muestra un indicador de cambio de precio con flecha animada.
    """
    if current_value > previous_value:
        change_pct = ((current_value / previous_value) - 1) * 100
        st.markdown(f"""
        <div class="price-up">
            {prefix}{current_value:.2f}{suffix} <span>▲ {change_pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    elif current_value < previous_value:
        change_pct = ((previous_value / current_value) - 1) * 100
        st.markdown(f"""
        <div class="price-down">
            {prefix}{current_value:.2f}{suffix} <span>▼ {change_pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="price-neutral">
            {prefix}{current_value:.2f}{suffix} <span>◆ 0.0%</span>
        </div>
        """, unsafe_allow_html=True)

def create_animated_plotly(fig, animation_frame=None):
    """
    Añade animación a un gráfico de Plotly si es posible.
    
    Args:
        fig: Figura de Plotly
        animation_frame: Columna a usar para animar (ej. 'Fecha')
        
    Returns:
        Figura de Plotly con animación configurada
    """
    if animation_frame:
        # Añadir configuración para una animación más suave
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 300, "easing": "quadratic-in-out"}
                                }
                            ]
                        }
                    ]
                }
            ]
        )
    
    # Añadir transiciones suaves para interacciones
    fig.update_layout(
        transition={
            'duration': 300,
            'easing': 'cubic-in-out'
        }
    )
    
    return fig

def loading_animation(text="Cargando datos..."):
    """
    Muestra una animación de carga personalizada.
    """
    st.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)

def animated_metric(label, value, delta=None, delta_suffix="%"):
    """
    Versión animada del componente métrica de Streamlit.
    """
    delta_class = "delta-neutral"
    delta_arrow = "◆"
    delta_abs = delta

    if delta is not None:
        if delta > 0:
            delta_class = "delta-up"
            delta_arrow = "▲"
            delta_abs = delta
        elif delta < 0:
            delta_class = "delta-down"
            delta_arrow = "▼"
            delta_abs = abs(delta)
        else:
            delta_class = "delta-neutral"
            delta_arrow = "◆"
            delta_abs = 0

    metric_html = f"""
    <div class="animated-metric">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    """

    if delta is not None:
        metric_html += f"""
        <div class="{delta_class}">
            {delta_arrow} {delta_abs}{delta_suffix}
        </div>
        """

    metric_html += "</div>"
    st.markdown(metric_html, unsafe_allow_html=True)