# Configuración Pendiente AgroDashRD (2026)

Este documento detalla los pasos manuales necesarios para poner en marcha la aplicación en producción y desarrollo local, asegurando que todas las conexiones sean seguras y funcionales.

## 1. Backend (Servidor)

### Variables de Entorno (`.env`)
Crear un archivo `.env` en la raíz del backend (`backend/`) con los siguientes valores obligatorios:

```env
# Base de Datos (PostgreSQL 16+)
# Formato: postgresql://usuario:contraseña@host:puerto/nombre_bd
DATABASE_URL=postgresql://agrodash_user:secure_password_2026@db:5432/agrodash_db

# Seguridad (JWT)
# Generar con: openssl rand -hex 32
SECRET_KEY=cambiar_esto_por_una_clave_segura_y_larga_en_produccion
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Configuración de ML (Opcional)
# ML_MODEL_PATH=/app/models/prophet_v1.pkl
```

### Dependencias Opcionales
En `backend/requirements.txt`, la librería `prophet` está comentada por defecto para facilitar el despliegue rápido. Si se requiere el módulo de predicciones:
1.  Descomentar `prophet>=1.1.5`.
2.  Asegurarse de tener `g++` y `python3-dev` instalados en el sistema operativo del servidor.

---

## 2. Frontend Móvil (Flutter)

### Conexión con API (`lib/services/api_service.dart`)
El archivo `api_service.dart` tiene configurada la URL base para el emulador de Android por defecto (`10.0.2.2`).

*   **Para Emulador Android:** Mantener `http://10.0.2.2:8000`.
*   **Para Simulador iOS:** Cambiar a `http://localhost:8000`.
*   **Para Dispositivo Físico (mismo WiFi):** Cambiar a la IP local de tu PC (ej: `http://192.168.1.50:8000`).
*   **Para Producción:** Cambiar a la URL del dominio real (ej: `https://api.agrodashrd.com`).

### Descomentar Lógica Real
En las pantallas (`screens/`), hay bloques de código comentados con `// TODO: Descomentar...` que conectan la UI con el backend real.
*   **LoginScreen (`login_screen.dart`):** Descomentar la llamada a `ApiService.login`.
*   **RegisterScreen (`register_screen.dart`):** Descomentar la llamada a `ApiService.register`.
*   **HomeScreen (`home_screen.dart`):** Descomentar la carga de precios en `_loadPrices`.
*   **ReportPriceScreen (`report_price_screen.dart`):** Descomentar la carga de productos/mercados y el envío del reporte.

---

## 3. Base de Datos (PostgreSQL)

### Inicialización (Seeds)
Al desplegar por primera vez, la base de datos estará vacía. Para cargar los mercados y productos de República Dominicana:
1.  Acceder al contenedor del backend o servidor:
    ```bash
    docker exec -it agrodash-backend python backend/scripts/seed_data.py
    ```
    O localmente:
    ```bash
    python backend/scripts/seed_data.py
    ```

---

## 4. Docker (Despliegue)
El archivo `docker-compose.yml` está configurado para levantar:
1.  Backend (FastAPI) en el puerto 8000.
2.  Base de Datos (PostgreSQL 16) en el puerto 5432.

**Nota:** La base de datos persiste sus datos en un volumen de Docker llamado `postgres_data`. Si deseas resetear la BD por completo:
```bash
docker-compose down -v
```
Esto eliminará todos los datos.
