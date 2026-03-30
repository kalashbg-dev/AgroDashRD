#!/bin/bash

# Script de instalación "Todo en Uno" para Linux Mint / Ubuntu
# Instala: Git, Curl, Python, Docker, Flutter

set -e # Detener si hay error

echo "=== Instalador de Dependencias AgroDashRD ==="
echo "Este script instalará las herramientas necesarias en tu sistema."
echo "Te pedirá tu contraseña de sudo varias veces."
read -p "¿Continuar? (s/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Ss]$ ]]; then
    exit 1
fi

# 1. Actualizar sistema
echo "--> Actualizando repositorios..."
sudo apt update

# 2. Herramientas básicas
echo "--> Instalando Git, Curl, Python..."
sudo apt install -y git curl python3 python3-venv python3-pip

# 3. Docker (Opcional)
read -p "¿Quieres instalar Docker? (Recomendado para BD) (s/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo "--> Instalando Docker..."
    # Método simplificado para Ubuntu/Mint
    sudo apt install -y docker.io docker-compose

    # Añadir usuario al grupo docker (para no usar sudo)
    sudo usermod -aG docker $USER
    echo "NOTA: Deberás cerrar sesión y volver a entrar para usar Docker sin sudo."
fi

# 4. Flutter
if ! command -v flutter &> /dev/null; then
    echo "--> Instalando Flutter (vía Snap)..."
    sudo snap install flutter --classic

    echo "--> Configurando Flutter..."
    flutter config --no-analytics
    flutter doctor
else
    echo "Flutter ya está instalado."
fi

echo "=== Instalación Completada ==="
echo "Para empezar:"
echo "1. Si instalaste Docker, cierra sesión y vuelve a entrar."
echo "2. Ejecuta './start_dev.sh' para iniciar el servidor."
echo "3. En otra terminal, ve a 'mobile_app' y ejecuta 'flutter run -d linux'."
