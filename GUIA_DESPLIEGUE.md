# Guía de Despliegue de AgroStack

## 1. Configuración del Servidor Backend (Python + PostgreSQL)

### 1.1 Base de Datos (Hostinger / Render / Supabase)
Para la versión "definitiva" multi-usuario, recomendamos un servicio gestionado (PostgreSQL as a Service).

1.  **Crear Instancia:**
    *   En Hostinger o Render, busca "PostgreSQL Database".
    *   Copia la `DATABASE_URL` (formato `postgresql://user:pass@host:5432/dbname`).
2.  **Migrar Datos:**
    *   Usa `pgAdmin` o `DBeaver` (gratuito) para conectar a tu base de datos remota.
    *   Ejecuta el script SQL de creación de tablas (ver anexo en propuesta técnica).

### 1.2 Backend (FastAPI en Hostinger VPS o Render)

#### Opción A: Render (Gratis/Bajo Costo) - Recomendado MVP+
1.  **Repo en GitHub:** Asegúrate que tu código backend (Python) esté en un repositorio.
2.  **Crear Web Service:**
    *   Conecta tu cuenta de GitHub.
    *   Build Command: `pip install -r requirements.txt`
    *   Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3.  **Variables de Entorno:**
    *   Añade `DATABASE_URL` con la conexión de tu base de datos.
    *   Añade `SECRET_KEY` para tokens JWT.

#### Opción B: Hostinger VPS (Control Total)
1.  **SSH:** Conecta a tu VPS: `ssh root@tu-ip`.
2.  **Instalar Docker:**
    ```bash
    apt update && apt install docker.io docker-compose -y
    ```
3.  **Desplegar con Docker Compose:**
    *   Crea un archivo `docker-compose.yml` que levante tu app Python y (opcionalmente) la base de datos si no usas servicio gestionado.
    *   Ejecuta `docker-compose up -d --build`.
4.  **Dominio:** Configura Nginx como proxy inverso para apuntar `api.tudominio.com` al puerto de tu app (ej: 8000).

---

## 2. Desarrollo y Publicación de App Móvil (Flutter)

### 2.1 Entorno de Desarrollo
1.  Instala [Flutter SDK](https://flutter.dev/docs/get-started/install).
2.  Verifica instalación: `flutter doctor`.

### 2.2 Compilación para Android (Google Play)
1.  **Configurar firma:**
    *   Genera una llave de upload (`keytool`).
    *   Configura `android/key.properties`.
2.  **Build Release:**
    ```bash
    flutter build apk --release
    # O para Google Play App Bundle:
    flutter build appbundle
    ```
3.  **Distribución:**
    *   Sube el `.aab` a la Google Play Console.

### 2.3 Compilación para iOS (App Store) - Requiere Mac
1.  Abre `ios/Runner.xcworkspace` en Xcode.
2.  Configura "Signing & Capabilities" con tu cuenta de desarrollador Apple ($99/año).
3.  Product -> Archive -> Distribute App.

### 2.4 Versión Web (PWA)
1.  Habilita web: `flutter config --enable-web`.
2.  Build: `flutter build web --release`.
3.  Sube el contenido de `build/web` a cualquier hosting estático (Netlify, Vercel, o carpeta pública en Hostinger).

---

## 3. Mantenimiento y Automatización (CI/CD)
Para evitar despliegues manuales, configura GitHub Actions:
*   Cada vez que hagas `push` a `main`, el servidor Render detectará cambios y re-desplegará automáticamente.
*   Para Flutter, puedes usar Codemagic (gratis hasta cierto límite) para generar los APKs automáticamente.
