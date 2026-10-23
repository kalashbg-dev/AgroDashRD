> **Las secciones 1 a 5 describen el sistema anterior** (Streamlit original, luego FastAPI y Flutter), reemplazado por el paquete `src/agrodash`. La [sección 6](#6-seguimiento-estado-en-octubre-de-2026) resume cómo quedó cada hallazgo; para usar el sistema actual ver [README.md](README.md).

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

---

## 5. Actualización 2026: Auditoría Post-Migración

### 5.1 Estado Actual
Se ha completado una reestructuración significativa del proyecto para abordar las limitaciones identificadas anteriormente.
*   **Arquitectura:** Separación exitosa entre Backend (API REST) y Frontend Móvil (Flutter).
*   **Backend (`backend/`):** Implementado con FastAPI. Soporta autenticación JWT, gestión de usuarios, productos, mercados y precios. Base de datos PostgreSQL/SQLite integrada.
*   **Mobile App (`mobile_app/`):** Se ha creado un esqueleto funcional en Flutter que incluye pantallas de Login, Registro, Home y Reporte de Precios.
*   **Legacy (`legacy/`):** El código original en Streamlit se ha movido a una carpeta dedicada para referencia histórica.

### 5.2 Anomalías Corregidas
1.  **Código Faltante en Mobile App:** Se detectó que la carpeta `mobile_app/` estaba vacía (solo contenía `pubspec.yaml`). Se ha restaurado la estructura de archivos (`lib/`, `screens/`, `services/`, `models/`) para permitir la compilación y desarrollo.
2.  **Documentación Desactualizada:** El archivo `README.md` raíz describía la arquitectura antigua. Se ha actualizado para reflejar la nueva estructura y guiar al desarrollador en el uso del Backend y la Mobile App.
3.  **Dependencias:** Se verificaron las dependencias del backend (`requirements.txt`) y se confirmó que la librería `prophet` sigue siendo opcional y está manejada correctamente en el código (`try...except`).

### 5.3 Próximos Pasos Recomendados
1.  **Implementación Lógica Móvil:** Completar la integración de los endpoints reales en `api_service.dart`.
2.  **Pruebas End-to-End:** Verificar el flujo completo desde el registro de usuario en la app móvil hasta la persistencia en base de datos.
3.  **Despliegue:** Configurar pipelines de CI/CD para automatizar el despliegue del backend y la compilación de la app móvil.

---

## 6. Seguimiento: estado en octubre de 2026

La sección 5 quedó superada: el backend FastAPI y la app Flutter se retiraron (la app no tenía código) y todo se reconstruyó como un solo paquete Python, `src/agrodash`, con una web Streamlit pensada para el teléfono. Esta tabla resume cómo quedó cada hallazgo.

| Hallazgo | Estado | Cómo se resolvió |
|---|---|---|
| Lógica de negocio acoplada a la interfaz (2.1) | Resuelto | Catálogo, precios, ingesta, pronósticos y cuentas no importan Streamlit; se prueban sin él (más de 400 pruebas, también sobre PostgreSQL en CI) |
| Sin persistencia, archivos planos (2.1) | Resuelto | PostgreSQL en producción, SQLite en desarrollo, migraciones Alembic |
| `prophet` importado pero excluido (2.2.1) | Resuelto | Se retiró. Los pronósticos se eligieron con pruebas sobre datos no vistos: el último precio con rangos calibrados superó a AutoETS, Theta y gradient boosting a 4 semanas, 3 meses y 12 meses |
| Manejo de errores genérico (2.2.2) | Resuelto | Las filas dudosas van a cuarentena con su motivo; cada carga y corrida de pronóstico guarda su estado y su error |
| Lógica duplicada (2.2.3) | Resuelto | Una sola normalización de nombres, unidades y semanas, reutilizada por la carga de archivos, Drive y los reportes de campo |
| Autenticación ficticia (2.2.4) | Resuelto | Cuentas en base de datos, contraseñas Argon2id, bloqueo tras 5 intentos, roles TECNICO, ADMIN y SUPERADMIN verificados en el servidor en cada página |
| Seguridad nula (2.3) | Resuelto | Lo anterior, más protección XSRF, base y web publicadas solo en localhost detrás de un proxy HTTPS, contenedor sin privilegios y revisiones de seguridad por fase |
| Escalabilidad de Streamlit (2.3) | Mitigado | Los pronósticos se calculan una vez por semana y se guardan; las consultas de las páginas se guardan en caché 10 minutos |
| Compilación móvil (3) | Decisión | Web adaptada al teléfono en lugar de app nativa; se revisó cada página a 390 px de ancho, en modo claro y oscuro |

### Problemas de datos del sistema original, corregidos

| Problema | Efecto en el sistema original | Ahora |
|---|---|---|
| Toda fecha se fijaba en el día 15 del mes | Vistas semanales y por día de la semana sin sentido | Fecha semanal real (semanas 1 a 5 de cada informe) |
| "1,200" se leía como 1.2 | Precios con separador de miles divididos por mil | Separadores decimales y de miles reconocidos |
| La mayoría de vistas usaba solo el precio mayorista | Los precios minoristas casi no se veían | Cada vista elige mayorista o minorista |
| Rubro desconocido convertido en "Frutas" | Productos en el rubro equivocado | Variantes de rubro normalizadas; lo dudoso se revisa |
| Precios inventados (mayorista = minorista × 0,8) | Datos que no existían | Los faltantes quedan como faltantes |
| Métricas de pronóstico sobre los mismos datos de entrenamiento | Precisión sobrestimada | Pruebas sobre periodos no vistos; la cobertura medida se muestra en la página |
| El mismo producto con nombres distintos según la época | Historias partidas | Grupos de nombres que un administrador une en Catálogo |

Lo que se conservó del sistema original, rehecho sobre estos datos: precios por mercado, historia, temporadas (meses altos y bajos, mapas de calor), comparación de mercados, valor de la cosecha, mayores cambios y descargas CSV. No se trasladaron el planificador de siembra (los informes no traen ciclos de cultivo) ni los gráficos por día de la semana.
