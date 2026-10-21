# Propuesta Técnica: Transformación Digital AgroDashRD

## 1. Arquitectura Recomendada: "AgroStack"
Para cumplir con los requerimientos de una aplicación móvil nativa, multi-usuario, con capacidades de Machine Learning y gestión de datos centralizada, se propone la siguiente arquitectura desacoplada:

### **Frontend (Cliente):** Flutter (Dart)
*   **¿Por qué?** Permite desarrollar una sola base de código para **Android, iOS y Web**.
*   **Ventaja:** Rendimiento nativo, acceso a hardware (GPS para geolocalizar fincas, Cámara para fotos de cultivos), y funcionamiento offline (sincronización posterior).
*   **Compilación:** Genera `.apk` (Android) y `.ipa` (iOS) directamente.

### **Backend (Servidor):** Python (FastAPI o Django Ninja)
*   **¿Por qué?** Python es indispensable para ejecutar los modelos de predicción (Prophet, XGBoost) y análisis de datos (Pandas, NumPy). FastAPI es ligero, moderno y perfecto para crear REST APIs rápidas.
*   **Función:**
    *   Gestionar usuarios y autenticación (JWT).
    *   Ejecutar modelos de predicción bajo demanda o programados.
    *   Servir datos JSON a la app móvil.

### **Base de Datos:** PostgreSQL
*   **¿Por qué?** Relacional, robusta, open-source y excelente para datos geoespaciales (PostGIS) si se requiere a futuro.
*   **Hosting:** Hostinger (VPS) o servicios gestionados gratuitos (Supabase, Neon, Render PostgreSQL).

---

## 2. Modelo de Datos y Usuarios
Se definen 4 roles principales con permisos diferenciados:

1.  **Agricultor (Usuario Final):**
    *   **Permisos:** Ver precios, recibir alertas, registrar su producción, consultar clima.
    *   **Datos:** Perfil básico, ubicación de finca, historial de ventas.
2.  **Técnico (Usuario de Campo):**
    *   **Permisos:** Registrar visitas, subir reportes de plagas/enfermedades, validar datos de agricultores.
    *   **Datos:** Asignación de zonas, bitácora de visitas.
3.  **Administrador (Gestión):**
    *   **Permisos:** Gestionar usuarios (altas/bajas), validar precios de mercado, configurar parámetros del sistema.
4.  **SuperAdmin (Dueño - Tú):**
    *   **Permisos:** Acceso total a base de datos, logs del sistema, gestión de facturación/pagos (futuro), configuración global.

### Entidades Principales (Tablas)
*   `Users` (id, email, password_hash, role, profile_data)
*   `Markets` (id, name, location, type)
*   `Products` (id, name, category, unit)
*   `Prices` (id, product_id, market_id, date, price_wholesale, price_retail, user_id_reporter)
*   `Predictions` (id, product_id, market_id, forecast_date, predicted_price, model_version)

---

## 3. Estrategia de Modelos de Predicción (MLOps)
Dado que el entorno de hosting puede ser limitado (Render Free Tier o VPS básico), se sugiere:

1.  **Entrenamiento Diferido (Batch Processing):**
    *   No entrenar modelos en tiempo real cuando el usuario lo pide.
    *   **Estrategia:** Un script nocturno (CRON job) en el servidor Python re-entrena los modelos con los nuevos datos del día y guarda las predicciones futuras en la tabla `Predictions`.
2.  **Inferencia Rápida:**
    *   Cuando el usuario consulta "Predicción de precio del plátano", la API simplemente lee el valor ya calculado de la base de datos. Respuesta instantánea (<100ms).
3.  **Librerías:**
    *   Mantener `Prophet` para series temporales estacionales.
    *   Evaluar `XGBoost` para predicciones basadas en características (clima, volumen, etc.).

---

## 4. Plan de Migración (Roadmap)

### Fase 1: Backend API (Python)
1.  Configurar FastAPI con SQLAlchemy (ORM).
2.  Diseñar endpoints: `/login`, `/markets`, `/products`, `/prices/upload`.
3.  Migrar lógica de limpieza de datos de `src/data_loader.py` a scripts de ETL (Extract, Transform, Load) en el backend.

### Fase 2: Base de Datos
1.  Provisionar PostgreSQL en Hostinger.
2.  Crear esquema relacional.
3.  Cargar datos históricos (CSV actuales) a la BD.

### Fase 3: App Móvil (Flutter)
1.  Pantallas de Login y Registro.
2.  Dashboard principal (consumiendo JSON del backend).
3.  Formulario de "Subir Precio" (para Técnicos/Admin).

### Fase 4: Integración ML
1.  Adaptar scripts de `src/price_forecasting.py` para leer de BD en lugar de CSV.
2.  Programar tarea automática de re-entrenamiento semanal.
