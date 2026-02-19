# Informe de Auditoría Técnica - AgroDashRD

## 1. Resumen Ejecutivo
El proyecto actual es un MVP (Producto Mínimo Viable) construido sobre **Python** y **Streamlit**. Funciona como una herramienta de visualización de datos interactiva web. Si bien cumple su función demostrativa, presenta limitaciones arquitectónicas severas para escalar hacia una "aplicación agrícola definitiva" multi-usuario y móvil.

**Estado Actual:** Prototipo Web Monolítico.
**Meta:** Aplicación Móvil/Web Multi-plataforma (SaaS).

## 2. Análisis de Arquitectura y Código

### 2.1 Estructura del Proyecto
*   **Framework:** Streamlit. Esto es excelente para prototipado rápido de ciencia de datos, pero **no apto** para aplicaciones de producción masiva o móviles nativas.
*   **Organización:** La carpeta `src/` intenta modularizar el código (`dashboard_agricultor.py`, `dashboard_profesional.py`), lo cual es una buena práctica. Sin embargo, la lógica de negocio está fuertemente acoplada a la interfaz de usuario (llamadas directas a `st.write`, `st.plotly_chart` dentro de funciones lógicas).
*   **Persistencia de Datos:** Inexistente. El sistema depende de archivos planos (`.csv`, `.parquet`) en la carpeta `data/` o generados al vuelo. No hay conexión a base de datos relacional, lo que impide la gestión de usuarios, roles o historial persistente real.

### 2.2 Inconsistencias Detectadas
1.  **Gestión de Dependencias (`requirements.txt` vs Código):**
    *   El archivo `src/price_forecasting.py` importa la librería `prophet` para predicciones avanzadas.
    *   Sin embargo, en `requirements.txt`, `prophet` está comentado/excluido para "aligerar el entorno".
    *   **Consecuencia:** El código de predicción fallará en ejecución si se descomentan esas líneas sin instalar la librería, o simplemente no se está usando la potencia real de los modelos prometidos.

2.  **Manejo de Errores:**
    *   El manejo de excepciones es genérico (`try...except Exception`), mostrando alertas en la UI (`st.warning`). Esto es aceptable para un MVP pero insuficiente para producción, donde se requiere un logueo robusto y recuperación de estados.

3.  **Duplicación de Lógica:**
    *   Hay funciones de transformación de datos similares dispersas entre los módulos de carga y los dashboards específicos.

4.  **Autenticación Ficticia:**
    *   La "gestión de usuarios" actual (`st.session_state.user_type`) es puramente volátil (en memoria del navegador/sesión). No hay seguridad, registro, ni persistencia de perfiles.

### 2.3 Seguridad y Escalabilidad
*   **Seguridad:** Nula. Al no haber backend separado ni base de datos, no hay manejo de sesiones seguras ni protección de endpoints.
*   **Escalabilidad:** Streamlit recalcula todo el script en cada interacción. Con múltiples usuarios concurrentes y modelos de ML pesados, el rendimiento se degradará drásticamente en un servidor gratuito o compartido.

## 3. Evaluación de "Compilación Móvil"
El requerimiento de "compilar para móvil como app" **no es viable con el stack actual**.
*   Streamlit es tecnología Web pura. No se puede empaquetar en un `.apk` o `.ipa` nativo de forma eficiente o mantenible.
*   Se podría usar un "wrapper" (webview), pero la experiencia de usuario (UX) sería pobre, lenta y dependiente de conexión a internet constante, sin acceso a funciones nativas del dispositivo (GPS, notificaciones, cámara).

## 4. Conclusión de la Auditoría
Para lograr la meta de una aplicación definitiva, robusta, multi-usuario (Agricultor, Técnico, Admin, SuperAdmin) y móvil, **es imperativa una migración de arquitectura**.

El código actual sirve como una excelente **prueba de concepto lógica y validación de visualizaciones**, pero no puede ser la base del producto final. Se debe desacoplar el "Cerebro" (Python/Datos) del "Cuerpo" (App Móvil/Web).
