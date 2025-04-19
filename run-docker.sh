#!/bin/bash

# Проверка наличия Docker и Docker Compose
if ! command -v docker &> /dev/null; then
    echo "ОШИБКА: Docker не установлен."
    echo "Установите Docker, следуя инструкциям по адресу: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "ОШИБКА: Docker Compose не установлен."
    echo "Установите Docker Compose, следуя инструкциям по адресу: https://docs.docker.com/compose/install/"
    exit 1
fi

# Проверка наличия .env файла
if [ ! -f .env ]; then
    echo "ПРЕДУПРЕЖДЕНИЕ: Файл .env не найден. Создаем из шаблона..."
    
    cat > .env << 'EOL'
# Telegram
TELEGRAM_TOKEN=

# Neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# OpenRouter
OPENROUTER_API_KEY=

# Модель
MODEL_NAME=x-ai/grok-3-mini-beta
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=2000
MODEL_REQUEST_TIMEOUT=120

# Настройки
MAX_CONSECUTIVE_AUTO_REPLIES=3
LOG_LEVEL=INFO
EOL
    
    echo "Файл .env создан. Пожалуйста, заполните его необходимыми данными."
    exit 1
fi

# Проверка наличия необходимых переменных в .env
if ! grep -q "TELEGRAM_TOKEN=" .env || grep -q "TELEGRAM_TOKEN=$" .env; then
    echo "ОШИБКА: В файле .env не задан TELEGRAM_TOKEN"
    exit 1
fi

if ! grep -q "OPENROUTER_API_KEY=" .env || grep -q "OPENROUTER_API_KEY=$" .env; then
    echo "ОШИБКА: В файле .env не задан OPENROUTER_API_KEY"
    exit 1
fi

# Сборка и запуск контейнеров
echo "Сборка и запуск контейнеров..."
docker-compose up -d --build

# Проверка статуса контейнеров
echo -e "\nСтатус контейнеров:"
docker-compose ps

echo -e "\nЛоги AI-репетитора (нажмите Ctrl+C для выхода):"
docker-compose logs -f ai-tutor 