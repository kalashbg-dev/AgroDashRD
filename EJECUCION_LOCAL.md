# Guía de Ejecución Local en Linux Mint

Esta guía te permitirá correr tanto el Backend como la App Móvil en tu computadora con Linux Mint.

## 1. Prerequisitos

Abre tu terminal y asegúrate de tener instaladas las siguientes herramientas. Si te falta alguna, instálala con los comandos sugeridos:

### Git y Curl
```bash
sudo apt update
sudo apt install git curl -y
```

### Docker (Opcional pero recomendado para la Base de Datos)
Si prefieres no instalar PostgreSQL manualmente en tu sistema, usa Docker.
[Guía oficial de instalación de Docker en Ubuntu/Mint](https://docs.docker.com/engine/install/ubuntu/)

### Python 3.12+
Linux Mint suele traer Python. Verifica la versión:
```bash
python3 --version
```
Si necesitas instalar `venv` y `pip`:
```bash
sudo apt install python3-venv python3-pip -y
```

### Flutter (Para la App Móvil)
1.  Descarga el SDK de Flutter (o usa `snap`):
    ```bash
    sudo snap install flutter --classic
    ```
2.  Verifica la instalación:
    ```bash
    flutter doctor
    ```
    *Sigue las instrucciones que te dé `flutter doctor` para instalar dependencias faltantes (como Android Studio o herramientas de Linux).*

---

## 2. Ejecutar el Backend (Servidor)

El proyecto incluye un script automático para Linux.

1.  **Navega a la carpeta del proyecto:**
    ```bash
    cd /ruta/a/tu/AgroDashRD
    ```

2.  **Ejecuta el script de desarrollo:**
    ```bash
    ./start_dev.sh
    ```
    *Este script:*
    *   Creará un entorno virtual de Python (`venv`).
    *   Instalará las librerías necesarias.
    *   Levantará la base de datos (si tienes Docker) o usará SQLite por defecto.
    *   Cargará los datos de prueba (Mercados y Productos de RD).
    *   Iniciará el servidor en `http://localhost:8000`.

**Prueba:** Abre tu navegador en [http://localhost:8000/docs](http://localhost:8000/docs). Deberías ver la documentación interactiva de la API.

---

## 3. Ejecutar la App Móvil

1.  **Abre una NUEVA terminal** (deja la del backend corriendo).

2.  **Navega a la carpeta de la app:**
    ```bash
    cd mobile_app
    ```

3.  **Instala las dependencias de Flutter:**
    ```bash
    flutter pub get
    ```

4.  **Ejecuta la app:**

    *   **Opción A: Linux Desktop (Más rápido para probar lógica)**
        Asegúrate de tener los requisitos de Linux (`flutter config --enable-linux-desktop`):
        ```bash
        flutter run -d linux
        ```

    *   **Opción B: Web (Fácil y visual)**
        ```bash
        flutter run -d chrome
        ```

    *   **Opción C: Emulador Android (Si tienes Android Studio instalado)**
        Abre tu emulador primero y luego:
        ```bash
        flutter run
        ```

---

## 4. Credenciales de Prueba

Una vez en la app, puedes usar este usuario (creado automáticamente por `start_dev.sh` si usaste el script de superusuario, o regístrate uno nuevo en la app):

*   **Registro:** Ve a la opción "Regístrate" en la app y crea tu propio usuario.
*   **Rol:** Por defecto serás `AGRICULTOR`.

Si necesitas un **Administrador** para probar el reporte de precios:
1.  En la terminal del backend, presiona `Ctrl+C` para detenerlo.
2.  Ejecuta:
    ```bash
    source venv/bin/activate
    python backend/scripts/create_superuser.py admin@agrodash.com admin123
    ```
3.  Vuelve a iniciar el servidor: `./start_dev.sh`
4.  Loguéate en la app con `admin@agrodash.com` / `admin123`.
