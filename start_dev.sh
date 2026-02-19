#!/bin/bash

# Verificar si Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    echo "Docker no está corriendo. Intentando levantar backend localmente..."

    # Levantar backend en segundo plano
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -r backend/requirements.txt

    echo "Iniciando servidor backend..."
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!

    # Esperar a que el servidor esté listo
    echo "Esperando a que el backend inicie..."
    sleep 5

    # Cargar datos de prueba si es necesario
    python backend/scripts/seed_data.py

    echo "Servidor corriendo en PID $SERVER_PID"
    wait $SERVER_PID
else
    echo "Levantando con Docker Compose..."
    docker-compose up -d

    echo "Esperando a que la base de datos esté lista..."
    sleep 10

    # Ejecutar seeds dentro del contenedor
    docker exec agrodash-backend python backend/scripts/seed_data.py

    echo "Despliegue local completado! Accede a http://localhost:8000/docs"
fi
