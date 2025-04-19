#!/bin/bash
set -e

# Функция для проверки доступности Neo4j
wait_for_neo4j() {
  echo "Ожидание запуска Neo4j..."
  
  until python -c "from neo4j import GraphDatabase; \
    driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USER', '$NEO4J_PASSWORD')); \
    driver.verify_connectivity(); \
    driver.close()" 2>/dev/null; do
    echo "Neo4j еще не доступен, ожидание..."
    sleep 2
  done
  
  echo "Neo4j запущен и доступен!"
}

# Основная функция
main() {
  echo "Запуск AI-репетитора..."
  
  # Проверка наличия переменных окружения
  if [ -z "$TELEGRAM_TOKEN" ]; then
    echo "ОШИБКА: Не задан TELEGRAM_TOKEN"
    exit 1
  fi
  
  if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ОШИБКА: Не задан OPENROUTER_API_KEY"
    exit 1
  fi
  
  # Ожидание запуска Neo4j
  wait_for_neo4j
  
  # Проверяем наличие концептов в Neo4j
  python -c "
from neo4j import GraphDatabase
import os

# Получаем параметры подключения из переменных окружения
uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
user = os.getenv('NEO4J_USER', 'neo4j')
password = os.getenv('NEO4J_PASSWORD', 'password')

# Функция для подсчета концептов
def count_concepts():
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run('MATCH (c:Concept) RETURN count(c) as count')
        count = result.single()['count']
        driver.close()
        return count

# Подсчитываем концепты
concept_count = count_concepts()
print(f'В базе данных найдено {concept_count} концептов')

# Если нет концептов, можно загрузить данные
if concept_count == 0:
    print('База данных пуста, нужно загрузить данные курса')
    exit(1)
"

  # Исправляем импорты в файлах проекта
  echo "Исправление импортов в файлах проекта..."
  python fix_imports.py /app <<< "y"

  # Запуск бота
  echo "Запуск Telegram-бота..."
  exec python main.py
}

# Запуск основной функции
main 