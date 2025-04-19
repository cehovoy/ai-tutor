FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей для sentence-transformers и Neo4j
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода проекта
COPY . .
COPY .env .

# Добавление директории в PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Делаем entrypoint-скрипт и fix_imports.py исполняемыми
RUN chmod +x docker-entrypoint.sh && chmod +x fix_imports.py

# Запуск бота через entrypoint-скрипт
ENTRYPOINT ["/app/docker-entrypoint.sh"] 