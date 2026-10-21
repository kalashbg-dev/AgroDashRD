#!/bin/bash

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Iniciando Entorno de Desarrollo AgroDashRD ===${NC}"

# Función para comprobar comandos
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    else
        return 0
    fi
}

# 1. Comprobar Python
if ! check_command python3; then
    echo -e "${RED}Error: Python 3 no está instalado.${NC}"
    echo "Instálalo con: sudo apt install python3"
    exit 1
fi

# 2. Comprobar venv (común en debian/ubuntu/mint)
# Intentamos crear un venv temporal para probar
if ! python3 -m venv test_env_check > /dev/null 2>&1; then
    echo -e "${YELLOW}Aviso: El módulo 'venv' de Python parece faltar.${NC}"
    echo -e "${GREEN}Intentando instalarlo automáticamente... (necesitará tu contraseña)${NC}"
    sudo apt update && sudo apt install python3-venv -y

    if ! python3 -m venv test_env_check > /dev/null 2>&1; then
        echo -e "${RED}Error: No se pudo instalar python3-venv. Por favor instálalo manualmente.${NC}"
        exit 1
    fi
fi
rm -rf test_env_check

# 3. Decidir método de ejecución (Docker vs Local)
if check_command docker && docker info > /dev/null 2>&1; then
    echo -e "${GREEN}Docker detectado. Usando contenedores (Recomendado).${NC}"

    if check_command docker-compose; then
        CMD="docker-compose"
    else
        CMD="docker compose"
    fi

    echo "Levantando servicios..."
    $CMD up -d

    echo "Esperando a que la base de datos esté lista..."
    sleep 10

    echo "Cargando datos iniciales (Seeds)..."
    docker exec agrodash-backend python backend/scripts/seed_data.py

    echo -e "${GREEN}¡Listo! Backend corriendo en http://localhost:8000${NC}"
    echo "Documentación API: http://localhost:8000/docs"

else
    echo -e "${YELLOW}Docker no detectado o no iniciado. Usando modo LOCAL (Python nativo).${NC}"

    # Crear entorno virtual si no existe
    if [ ! -d "venv" ]; then
        echo "Creando entorno virtual..."
        python3 -m venv venv
    fi

    echo "Activando entorno virtual..."
    source venv/bin/activate

    echo "Instalando dependencias del backend..."
    pip install -r backend/requirements.txt

    echo "Iniciando servidor en segundo plano..."
    # Usamos nohup o simplemente & pero guardando el PID
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!

    echo "Esperando arranque..."
    sleep 5

    echo "Cargando datos iniciales..."
    python backend/scripts/seed_data.py

    echo -e "${GREEN}¡Backend Local Activo! (PID: $SERVER_PID)${NC}"
    echo "Documentación API: http://localhost:8000/docs"
    echo -e "${YELLOW}Presiona Ctrl+C para detener el servidor.${NC}"

    # Esperar al proceso para que el script no termine inmediatamente
    wait $SERVER_PID
fi
