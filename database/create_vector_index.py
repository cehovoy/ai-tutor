#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для создания и управления векторными индексами в Neo4j.
Позволяет значительно ускорить поиск по векторным эмбеддингам.
"""

import argparse
import logging
import time
from neo4j import GraphDatabase
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Импортируем настройки
try:
    from ai_tutor.config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
except ImportError:
    logger.error("Не удалось импортировать настройки. Проверьте путь и доступность модуля.")
    NEO4J_URI = "bolt://localhost:17687"  # Значения по умолчанию
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "testpassword"
    logger.info(f"Используем значения по умолчанию: URI={NEO4J_URI}")

class VectorIndexManager:
    """Класс для управления векторными индексами в Neo4j"""
    
    def __init__(self, uri, user, password):
        """
        Инициализация менеджера индексов
        
        Args:
            uri: URI для подключения к Neo4j
            user: Имя пользователя Neo4j
            password: Пароль Neo4j
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.check_neo4j_version()
    
    def close(self):
        """Закрытие соединения с Neo4j"""
        self.driver.close()
    
    def check_neo4j_version(self):
        """Проверяет версию Neo4j на совместимость с векторными индексами"""
        with self.driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
            record = result.single()
            if record:
                name, versions = record
                version = versions[0]
                logger.info(f"Обнаружена версия Neo4j: {version}")
                
                # Проверка на поддержку векторных индексов (с Neo4j 5.11)
                major, minor = map(int, version.split('.')[:2])
                if major < 5 or (major == 5 and minor < 11):
                    logger.warning(f"Внимание: Векторные индексы поддерживаются с версии Neo4j 5.11, "
                                 f"у вас установлена версия {version}. Создание индекса может не сработать.")
            else:
                logger.warning("Не удалось определить версию Neo4j")
    
    def show_indexes(self):
        """Отображает все индексы в базе данных"""
        logger.info("Получение списка индексов...")
        with self.driver.session() as session:
            try:
                result = session.run("SHOW INDEXES")
                indexes = list(result)
                
                if not indexes:
                    logger.info("В базе данных нет индексов")
                    return
                
                logger.info(f"Найдено {len(indexes)} индексов:")
                
                for i, index in enumerate(indexes, 1):
                    name = index.get("name", "Без имени")
                    index_type = index.get("type", "Неизвестный тип")
                    state = index.get("state", "Неизвестное состояние")
                    progress = index.get("populationPercent", 0)
                    entity_type = index.get("entityType", "Неизвестно")
                    labels = ", ".join(index.get("labelsOrTypes", []))
                    properties = ", ".join(index.get("properties", []))
                    
                    logger.info(f"{i}. {name} ({index_type}, {entity_type} на {labels}.{properties}) - {state} ({progress}%)")
                
            except Exception as e:
                logger.error(f"Ошибка при получении списка индексов: {str(e)}")
    
    def show_vector_indexes(self):
        """Отображает все векторные индексы в базе данных"""
        logger.info("Получение списка векторных индексов...")
        with self.driver.session() as session:
            try:
                # SHOW VECTOR INDEXES поддерживается только в Neo4j 5.11+
                result = session.run("SHOW VECTOR INDEXES")
                indexes = list(result)
                
                if not indexes:
                    logger.info("В базе данных нет векторных индексов")
                    return
                
                logger.info(f"Найдено {len(indexes)} векторных индексов:")
                
                for i, index in enumerate(indexes, 1):
                    name = index.get("name", "Без имени")
                    state = index.get("state", "Неизвестное состояние")
                    progress = index.get("populationPercent", 0)
                    labels = ", ".join(index.get("labelsOrTypes", []))
                    properties = ", ".join(index.get("properties", []))
                    
                    logger.info(f"{i}. {name} (векторный индекс на {labels}.{properties}) - {state} ({progress}%)")
                
            except Exception as e:
                logger.error(f"Ошибка при получении списка векторных индексов: {str(e)}")
                logger.info("Возможно, версия Neo4j не поддерживает команду SHOW VECTOR INDEXES.")
                self.show_indexes()
    
    def create_concept_vector_index(self, index_name="concept_vectors", vector_field="combined_embedding", dimensions=768):
        """
        Создает векторный индекс для понятий
        
        Args:
            index_name: Имя для нового индекса
            vector_field: Поле с векторным эмбеддингом
            dimensions: Размерность вектора
        """
        logger.info(f"Создание векторного индекса {index_name} для поля {vector_field}...")
        
        with self.driver.session() as session:
            try:
                # Проверяем существование индекса
                result = session.run("SHOW VECTOR INDEXES WHERE name = $name", name=index_name)
                if list(result):
                    logger.info(f"Индекс с именем {index_name} уже существует")
                    return
                
                # Проверяем, что поле существует
                result = session.run(
                    "MATCH (c:Concept) WHERE c.combined_embedding IS NOT NULL RETURN count(c) as count"
                )
                record = result.single()
                count = record["count"] if record else 0
                
                if count == 0:
                    logger.error(f"Нет узлов Concept с полем {vector_field}. Индекс не будет создан.")
                    return
                
                logger.info(f"Найдено {count} узлов Concept с полем {vector_field}")
                
                # Создаем векторный индекс
                start_time = time.time()
                
                # В Neo4j 5.11+ можно использовать CREATE VECTOR INDEX
                query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS 
                FOR (c:Concept)
                ON c.{vector_field}
                OPTIONS {{
                    indexConfig: {{
                        'vector.dimensions': {dimensions},
                        'vector.similarity_function': 'cosine'
                    }}
                }}
                """
                
                session.run(query)
                
                elapsed_time = time.time() - start_time
                logger.info(f"Команда создания индекса выполнена за {elapsed_time:.2f} секунд")
                
                # Проверяем статус индекса (заполнение может занять время)
                self._wait_for_index_online(index_name)
                
            except Exception as e:
                logger.error(f"Ошибка при создании векторного индекса: {str(e)}")
    
    def _wait_for_index_online(self, index_name, max_wait_time=300, check_interval=5):
        """
        Ждет, пока индекс не перейдет в состояние ONLINE
        
        Args:
            index_name: Имя индекса
            max_wait_time: Максимальное время ожидания в секундах
            check_interval: Интервал проверки в секундах
        
        Returns:
            bool: True если индекс онлайн, False в противном случае
        """
        logger.info(f"Ожидание перехода индекса {index_name} в состояние ONLINE...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            with self.driver.session() as session:
                result = session.run(
                    "SHOW VECTOR INDEXES WHERE name = $name", 
                    name=index_name
                )
                record = result.single()
                
                if not record:
                    logger.warning(f"Индекс {index_name} не найден")
                    return False
                
                state = record.get("state", "")
                progress = record.get("populationPercent", 0)
                
                if state == "ONLINE":
                    logger.info(f"Индекс {index_name} перешел в состояние ONLINE")
                    return True
                
                logger.info(f"Индекс {index_name} в состоянии {state}, заполнение: {progress}%")
                time.sleep(check_interval)
        
        logger.warning(f"Время ожидания истекло. Индекс {index_name} не перешел в состояние ONLINE")
        return False
    
    def drop_index(self, index_name):
        """
        Удаляет индекс
        
        Args:
            index_name: Имя индекса для удаления
        """
        logger.info(f"Удаление индекса {index_name}...")
        
        with self.driver.session() as session:
            try:
                # Проверяем существование индекса
                result = session.run("SHOW VECTOR INDEXES WHERE name = $name", name=index_name)
                if not list(result):
                    logger.info(f"Индекс с именем {index_name} не существует")
                    return
                
                # Удаляем индекс
                session.run(f"DROP INDEX {index_name}")
                logger.info(f"Индекс {index_name} успешно удален")
                
            except Exception as e:
                logger.error(f"Ошибка при удалении индекса: {str(e)}")
    
    def test_vector_search(self, query_text, index_name="concept_vectors", limit=5):
        """
        Тестирует векторный поиск с использованием индекса
        
        Args:
            query_text: Текст запроса для поиска
            index_name: Имя векторного индекса
            limit: Количество результатов
        """
        logger.info(f"Тестирование векторного поиска для запроса: '{query_text}'")
        
        try:
            # Импортируем нужные зависимости
            from sentence_transformers import SentenceTransformer
            
            # Загружаем модель
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            logger.info(f"Загрузка модели {model_name}...")
            model = SentenceTransformer(model_name)
            
            # Создаем эмбеддинг запроса
            query_embedding = model.encode(query_text).tolist()
            
            # Проверяем наличие индекса
            with self.driver.session() as session:
                result = session.run("SHOW VECTOR INDEXES WHERE name = $name", name=index_name)
                if not list(result):
                    logger.error(f"Индекс {index_name} не найден")
                    return
                
                logger.info("Выполнение векторного поиска...")
                start_time = time.time()
                
                # Выполняем поиск с использованием индекса
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
                    YIELD node, score
                    RETURN 
                        node.name AS name,
                        node.definition AS definition,
                        score
                    ORDER BY score DESC
                """, index_name=index_name, limit=limit, embedding=query_embedding)
                
                results = list(result)
                elapsed_time = time.time() - start_time
                
                if not results:
                    logger.info("Поиск не вернул результатов")
                    return
                
                logger.info(f"Поиск выполнен за {elapsed_time:.3f} секунд")
                logger.info(f"Найдено {len(results)} результатов:")
                
                for i, record in enumerate(results, 1):
                    name = record.get("name", "")
                    score = record.get("score", 0)
                    definition = record.get("definition", "")
                    
                    if len(definition) > 100:
                        definition = definition[:97] + "..."
                    
                    logger.info(f"{i}. {name} (score: {score:.4f}) - {definition}")
                
        except ImportError:
            logger.error("Не удалось импортировать SentenceTransformer. "
                       "Установите пакет: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Ошибка при тестировании векторного поиска: {str(e)}")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Управление векторными индексами Neo4j')
    
    parser.add_argument('--uri', type=str, default=NEO4J_URI,
                        help=f'URI для подключения к Neo4j (по умолчанию: {NEO4J_URI})')
    parser.add_argument('--user', type=str, default=NEO4J_USER,
                        help=f'Имя пользователя Neo4j (по умолчанию: {NEO4J_USER})')
    parser.add_argument('--password', type=str, default=NEO4J_PASSWORD,
                        help='Пароль Neo4j')
    
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Команда show для просмотра индексов
    parser.add_subparsers.add_parser('show', help='Показать все векторные индексы')
    
    # Команда create для создания индекса
    create_parser = subparsers.add_parser('create', help='Создать векторный индекс')
    create_parser.add_argument('--name', type=str, default='concept_vectors',
                             help='Имя для нового индекса (по умолчанию: concept_vectors)')
    create_parser.add_argument('--field', type=str, default='combined_embedding',
                             help='Поле с векторным эмбеддингом (по умолчанию: combined_embedding)')
    create_parser.add_argument('--dimensions', type=int, default=768,
                             help='Размерность вектора (по умолчанию: 768)')
    
    # Команда drop для удаления индекса
    drop_parser = subparsers.add_parser('drop', help='Удалить индекс')
    drop_parser.add_argument('--name', type=str, required=True,
                           help='Имя индекса для удаления')
    
    # Команда test для тестирования поиска
    test_parser = subparsers.add_parser('test', help='Тестировать векторный поиск')
    test_parser.add_argument('query', type=str, help='Текст запроса для поиска')
    test_parser.add_argument('--index', type=str, default='concept_vectors',
                           help='Имя векторного индекса (по умолчанию: concept_vectors)')
    test_parser.add_argument('--limit', type=int, default=5,
                           help='Количество результатов (по умолчанию: 5)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = VectorIndexManager(args.uri, args.user, args.password)
    
    try:
        if args.command == 'show':
            manager.show_vector_indexes()
        elif args.command == 'create':
            manager.create_concept_vector_index(args.name, args.field, args.dimensions)
        elif args.command == 'drop':
            manager.drop_index(args.name)
        elif args.command == 'test':
            manager.test_vector_search(args.query, args.index, args.limit)
    finally:
        manager.close()

if __name__ == "__main__":
    main() 