#!/bin/bash

# Script de configuración inicial para AgrodashRD
# Este script se ejecuta durante el despliegue en Render

# Actualizar el sistema e instalar dependencias del sistema
echo "Actualizando el sistema e instalando dependencias..."
apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio para datos si no existe
mkdir -p data

# Instalar dependencias de Python
echo "Instalando dependencias de Python..."
pip install --no-cache-dir -r requirements.txt

# Configurar variables de entorno si es necesario
if [ ! -f .env ]; then
    echo "Creando archivo .env..."
    cat > .env <<EOL
# Configuración de la aplicación
PYTHONUNBUFFERED=true
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_RUN_ON_SAVE=false
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Configuración de la base de datos (si es necesaria)
# DATABASE_URL=postgresql://user:password@host:port/dbname

# Otras variables de entorno
EOL
fi

echo "Configuración completada. La aplicación está lista para ejecutarse."
