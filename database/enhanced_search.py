"""
Модуль расширенного семантического поиска для курса.
Использует векторные эмбеддинги и учитывает рейтинг достоверности источников.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
# Принудительно указываем использование CPU перед импортом трансформеров
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Отключаем использование GPU
os.environ["OMP_NUM_THREADS"] = "4"  # Ограничиваем количество потоков OpenMP
os.environ["MKL_NUM_THREADS"] = "4"  # Ограничиваем количество потоков MKL

logger = logging.getLogger(__name__)

# Пытаемся импортировать SentenceTransformer с подробной диагностикой
try:
    logger.info("Попытка импорта SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    logger.info("SentenceTransformer успешно импортирован")
except ImportError as e:
    error_message = f"Ошибка импорта SentenceTransformer: {str(e)}"
    logger.error(error_message)
    # Дополнительная диагностика
    logger.error(f"Python версия: {sys.version}")
    logger.error(f"Путь поиска модулей: {sys.path}")
    try:
        import torch
        logger.info(f"PyTorch установлен, версия: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch не установлен")
    try:
        import transformers
        logger.info(f"Transformers установлен, версия: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers не установлен")
    SentenceTransformer = None
except Exception as e:
    error_message = f"Неожиданная ошибка при импорте SentenceTransformer: {str(e)}"
    logger.error(error_message)
    logger.error(f"Тип ошибки: {type(e)}")
    import traceback
    logger.error(f"Трассировка: {traceback.format_exc()}")
    SentenceTransformer = None

import json
import traceback
import time
import concurrent.futures
from functools import partial

from ai_tutor.config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# Заглушечный класс, который будет использоваться, если не удастся загрузить SentenceTransformer
class FallbackSearch:
    """
    Заглушечный класс для поиска, который используется, когда SentenceTransformer недоступен.
    Выполняет простой текстовый поиск.
    """
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD, **kwargs):
        """
        Инициализация заглушечного поискового движка
        
        Args:
            uri: URI для подключения к Neo4j
            user: Имя пользователя Neo4j
            password: Пароль Neo4j
            **kwargs: Дополнительные аргументы (игнорируются)
        """
        self.driver = None
        logger.warning("Инициализирована заглушка для поиска без SentenceTransformer")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Соединение с Neo4j установлено для заглушечного поиска")
        except Exception as e:
            logger.error(f"Ошибка подключения к Neo4j: {str(e)}")
            logger.error(traceback.format_exc())
    
    def close(self) -> None:
        """Закрытие соединения с Neo4j"""
        if self.driver:
            self.driver.close()
            logger.info("Соединение с Neo4j закрыто")
    
    def encode_query(self, query: str) -> List[float]:
        """Заглушка для encode_query"""
        return []
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Заглушка для encode_batch"""
        return [[] for _ in texts]
    
    def semantic_search_with_ranking(self, query: str, limit: int = 10, 
                                  threshold: float = 0.5, 
                                  source_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Простой текстовый поиск вместо семантического
        
        Args:
            query: Текстовый запрос для поиска
            limit: Максимальное количество результатов
            threshold: Игнорируется
            source_types: Список типов источников для поиска ('official', 'teacher', 'student')
                        
        Returns:
            Список результатов поиска
        """
        logger.info(f"Выполняем текстовый поиск (заглушка) для запроса: '{query[:50]}...'")
        
        if not self.driver:
            logger.error("Невозможно выполнить поиск: нет соединения с Neo4j")
            return []
        
        try:
            with self.driver.session() as session:
                # Фильтр по типу источника
                source_filter = ""
                if source_types and len(source_types) > 0:
                    source_filter = "AND n.source_type IN $source_types"
                
                # Ключевые слова для поиска
                search_terms = query.split()
                where_clauses = []
                
                # Создаем условия для поиска
                for i, term in enumerate(search_terms[:5]):  # Ограничиваем до 5 ключевых слов
                    if len(term) > 3:  # Ищем только слова длиннее 3 символов
                        where_clauses.append(f"toLower(n.name) CONTAINS toLower(${i}) OR toLower(n.definition) CONTAINS toLower(${i})")
                
                where_condition = " OR ".join(where_clauses) if where_clauses else "1=1"
                
                # Формируем параметры запроса
                params = {str(i): term for i, term in enumerate(search_terms[:5]) if len(term) > 3}
                if source_types:
                    params["source_types"] = source_types
                
                # Выполняем запрос
                cypher = f"""
                MATCH (n:Concept)
                WHERE ({where_condition}) {source_filter}
                RETURN 
                    n.name AS title,
                    n.definition AS content,
                    labels(n) AS labels,
                    n.source_type AS source_type,
                    coalesce(n.credibility_score, 1.0) as credibility_score,
                    n.chapters_mentions AS chapters_mentions,
                    n.example AS example,
                    n.questions AS questions
                LIMIT {limit}
                """
                
                result = session.run(cypher, **params)
                
                # Преобразуем результаты
                results = []
                for record in result:
                    # Фиксированная оценка, так как это не семантический поиск
                    credibility_score = record.get("credibility_score", 1.0)
                    
                    results.append({
                        "title": record.get("title", ""),
                        "name": record.get("title", ""),
                        "content": record.get("content", ""),
                        "definition": record.get("content", ""),
                        "labels": record.get("labels", []),
                        "source_type": record.get("source_type", "official"),
                        "similarity": 0.7,  # Фиксированная оценка
                        "credibility_score": credibility_score,
                        "weighted_score": 0.7 * credibility_score,
                        "chapters_mentions": record.get("chapters_mentions"),
                        "example": record.get("example"),
                        "questions": record.get("questions")
                    })
                
                logger.info(f"Текстовый поиск вернул {len(results)} результатов")
                return results
                
        except Exception as e:
            logger.error(f"Ошибка при выполнении текстового поиска: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Форматирование результатов поиска для вывода
        
        Args:
            results: Список результатов поиска
            
        Returns:
            Отформатированная строка с результатами
        """
        if not results:
            return "По вашему запросу ничего не найдено."
            
        output = "Найденные результаты:\n\n"
        for i, result in enumerate(results):
            output += f"{i+1}. **{result['title']}**\n\n"
            
            # Добавляем определение
            if result.get("definition"):
                definition = result["definition"]
                output += f"   Определение: {definition}\n\n"
                
            # Добавляем пример, если есть
            if result.get("example"):
                example = result["example"]
                output += f"   Пример: {example}\n\n"
            
        return output

# Доступные модели для выбора
MODEL_VARIANTS = {
    "default": "all-MiniLM-L12-v2",     # По умолчанию, более точная
    "fast": "all-MiniLM-L6-v2",          # Быстрая, но менее точная
    "accurate": "all-mpnet-base-v2",      # Более точная, но медленнее
    "multilingual": "paraphrase-multilingual-mpnet-base-v2",  # Многоязычная
    "distilbert": "distilbert-base-nli-stsb-mean-tokens"      # Легкая модель
}

# Изменяем модель по умолчанию на более легкую
MODEL_NAME = MODEL_VARIANTS["fast"]  # Используем быструю модель по умолчанию

class EnhancedCourseSearch:
    """
    Расширенный семантический поиск по курсу с использованием векторных эмбеддингов
    и учетом уровня достоверности источников
    """
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD, 
                 model_name: str = MODEL_NAME, max_workers: int = 1):  # Уменьшаем количество потоков
        """
        Инициализация расширенного поискового движка
        
        Args:
            uri: URI для подключения к Neo4j
            user: Имя пользователя Neo4j
            password: Пароль Neo4j
            model_name: Название модели SentenceTransformer или ключ из MODEL_VARIANTS
            max_workers: Максимальное количество потоков для параллельной обработки
        """
        # Инициализируем переменные, которые будут использоваться в обработчиках исключений
        self.driver = None
        self.model = None
        self.has_vector_index = False
        self.max_workers = max_workers
        
        # Подключаемся к Neo4j
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"Соединение с Neo4j установлено: {uri}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Neo4j: {str(e)}")
            logger.error(traceback.format_exc())
            self.driver = None
            # Продолжаем выполнение, попытаемся загрузить модель даже без БД
        
        # Получаем модель по ключу, если передан ключ
        if model_name in MODEL_VARIANTS:
            model_name = MODEL_VARIANTS[model_name]
            
        logger.info(f"Загрузка модели для векторного поиска: {model_name}")
        
        # Устанавливаем максимальное количество потоков
        self.max_workers = min(max_workers, 1)  # Ограничиваем максимальное количество потоков
        logger.info(f"Установлено максимальное количество потоков: {self.max_workers}")
        
        # Пытаемся загрузить модель с обработкой ошибок
        try:
            # Принудительно указываем использование CPU для трансформеров
            import torch
            device = "cpu"  # Принудительно используем CPU
            logger.info(f"Принудительно используем устройство: {device}")
            
            # Устанавливаем лимиты памяти для torch
            torch.set_num_threads(2)  # Ограничиваем количество потоков
            logger.info(f"Количество потоков torch ограничено до 2")
            
            # Пытаемся загрузить модель
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Модель для векторного поиска успешно загружена: {self.model}")
        except Exception as primary_error:
            logger.error(f"Ошибка при загрузке основной модели: {str(primary_error)}")
            logger.error(traceback.format_exc())
            
            # Пробуем загрузить запасную, более легкую модель
            try:
                backup_model = MODEL_VARIANTS["fast"]  # Самая легкая модель
                logger.info(f"Пытаемся загрузить запасную модель: {backup_model}")
                
                self.model = SentenceTransformer(backup_model, device=device)
                logger.info(f"Запасная модель успешно загружена: {self.model}")
            except Exception as backup_error:
                logger.error(f"Ошибка при загрузке запасной модели: {str(backup_error)}")
                logger.error(traceback.format_exc())
                raise RuntimeError("Не удалось загрузить ни одну модель для векторного поиска")
        
        # Проверяем наличие векторного индекса в Neo4j только если есть подключение
        if self.driver:
            self.has_vector_index = self._check_vector_index()
            if self.has_vector_index:
                logger.info("Обнаружен векторный индекс Neo4j. Будет использоваться для быстрого поиска.")
            else:
                logger.info("Векторный индекс Neo4j не обнаружен. Будет использоваться Python-вычисление сходства.")
    
    def _check_vector_index(self) -> bool:
        """
        Проверяет наличие векторного индекса в Neo4j
        
        Returns:
            True если индекс существует, иначе False
        """
        try:
            with self.driver.session() as session:
                result = session.run("SHOW VECTOR INDEXES")
                indexes = list(result)
                return len(indexes) > 0
        except Exception as e:
            logger.warning(f"Не удалось проверить наличие векторных индексов: {str(e)}")
            return False
    
    def close(self) -> None:
        """Закрытие соединения с Neo4j"""
        self.driver.close()
        logger.info("Соединение с Neo4j закрыто")
    
    def encode_query(self, query: str) -> List[float]:
        """
        Создание вектора из запроса
        
        Args:
            query: Текстовый запрос
            
        Returns:
            Список числовых значений вектора
        """
        return self.model.encode(query).tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Создание векторов для пакета текстов (более эффективно, чем по одному)
        
        Args:
            texts: Список текстовых запросов
            
        Returns:
            Список векторов
        """
        if not texts:
            return []
            
        return self.model.encode(texts).tolist()
    
    def cosine_similarity(self, vec1, vec2):
        """
        Вычисление косинусного сходства между двумя векторами
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
        
        Returns:
            Значение косинусного сходства
        """
        # Преобразуем строковое представление в список, если нужно
        if isinstance(vec1, str):
            try:
                vec1 = json.loads(vec1)
            except:
                return 0.0
                
        if isinstance(vec2, str):
            try:
                vec2 = json.loads(vec2)
            except:
                return 0.0
        
        # Проверяем, что получили правильные типы данных
        if not isinstance(vec1, list) or not isinstance(vec2, list):
            return 0.0
            
        # Проверяем, что векторы одинаковой длины
        if len(vec1) != len(vec2):
            return 0.0
            
        # Преобразуем в numpy массивы
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Вычисляем косинусное сходство
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def semantic_search_with_ranking(self, query: str, limit: int = 10, 
                                   threshold: float = 0.5, 
                                   source_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Семантический поиск с учетом уровня достоверности источников
        
        Args:
            query: Текстовый запрос для поиска
            limit: Максимальное количество результатов
            threshold: Минимальный порог сходства (0-1)
            source_types: Список типов источников для поиска ('official', 'teacher', 'student')
                        По умолчанию используются все типы
                        
        Returns:
            Список результатов поиска
        """
        # Валидация входных данных
        if not query or not isinstance(query, str):
            logger.error(f"Некорректный запрос для поиска: {query}")
            return []
            
        if threshold < 0 or threshold > 1:
            logger.warning(f"Некорректное значение threshold: {threshold}, используем значение по умолчанию 0.5")
            threshold = 0.5
        
        # Логируем начало поиска
        logger.info(f"Начинаем векторный семантический поиск для запроса: '{query[:50]}...'")
        start_time = time.time()
        
        try:
            # Создаем векторное представление запроса
            query_embedding = self.encode_query(query)
            logger.debug(f"Создан вектор запроса длиной {len(query_embedding)}")
            
            # Если есть векторный индекс в Neo4j, используем его
            if self.has_vector_index:  # Включаем использование нативного индекса
                return self._search_with_vector_index(query_embedding, limit, threshold, source_types)
            else:
                # Иначе используем гибридный подход
                return self._search_hybrid(query, query_embedding, limit, threshold, source_types)
        
        except Exception as e:
            logger.error(f"Ошибка при выполнении семантического поиска: {str(e)}")
            logger.error(traceback.format_exc())
            return []
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"Поиск завершен за {elapsed_time:.2f} секунд")
    
    def _process_batch(self, batch: List[Dict], query_embedding: List[float], 
                     threshold: float) -> List[Dict[str, Any]]:
        """
        Обрабатывает группу документов в отдельном потоке
        
        Args:
            batch: Список документов для обработки
            query_embedding: Векторное представление запроса
            threshold: Минимальный порог сходства
            
        Returns:
            Список результатов, прошедших порог сходства
        """
        results = []
        
        # Подготавливаем тексты для векторизации пакетом
        texts = []
        batch_items = []
        
        # Ограничиваем размер батча для уменьшения потребления памяти
        max_batch_size = 5
        for i, record in enumerate(batch):
            title = record.get("title", "")
            content = record.get("content", "")
            example = record.get("example", "")
            
            # Объединяем текст для сравнения
            document_text = f"{title} {content} {example}".strip()
            
            if document_text:
                texts.append(document_text)
                batch_items.append(record)
                
                # Если достигли максимального размера батча, обрабатываем
                if len(texts) >= max_batch_size:
                    self._process_small_batch(texts, batch_items, query_embedding, threshold, results)
                    texts = []
                    batch_items = []
        
        # Обрабатываем оставшиеся элементы
        if texts:
            self._process_small_batch(texts, batch_items, query_embedding, threshold, results)
        
        return results
        
    def _process_small_batch(self, texts, batch_items, query_embedding, threshold, results):
        """
        Обрабатывает небольшой батч текстов для снижения нагрузки на память
        """
        try:
            # Создаем эмбеддинги пакетом (эффективнее)
            document_embeddings = self.encode_batch(texts)
            
            # Обрабатываем результаты
            for i, embedding in enumerate(document_embeddings):
                record = batch_items[i]
                
                # Вычисляем сходство
                similarity = self.cosine_similarity(query_embedding, embedding)
                
                if similarity >= threshold:
                    credibility_score = record.get("credibility_score", 1.0)
                    weighted_score = similarity * credibility_score
                    
                    results.append({
                        "title": record.get("title", ""),
                        "name": record.get("title", ""),  # Для совместимости с существующим форматом
                        "content": record.get("content", ""),
                        "definition": record.get("content", ""),  # Для совместимости с существующим форматом
                        "labels": record.get("labels", []),
                        "source_type": record.get("source_type", "official"),
                        "similarity": similarity,
                        "credibility_score": credibility_score,
                        "weighted_score": weighted_score,
                        "chapters_mentions": record.get("chapters_mentions"),
                        "example": record.get("example"),
                        "questions": record.get("questions")
                    })
        except Exception as e:
            logger.error(f"Ошибка при обработке пакета документов: {str(e)}")
    
    def _search_hybrid(self, query: str, query_embedding: List[float], 
                      limit: int, threshold: float, 
                      source_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """
        Гибридный подход к поиску: получение всех данных одним запросом и вычисление сходства в Python
        
        Args:
            query: Текстовый запрос
            query_embedding: Векторное представление запроса
            limit: Максимальное количество результатов
            threshold: Минимальный порог сходства
            source_types: Список типов источников для поиска
            
        Returns:
            Список результатов поиска
        """
        logger.info("Используем гибридный подход к поиску")
        
        results = []
        try:
            with self.driver.session() as session:
                # Фильтр по типу источника
                source_filter = ""
                if source_types and len(source_types) > 0:
                    source_filter = "AND n.source_type IN $source_types"
                    logger.debug(f"Установлен фильтр по типам источников: {source_types}")
                
                # Получаем все понятия за один запрос, ограничиваем количество
                max_records = 100  # Ограничиваем максимальное количество записей для уменьшения нагрузки
                records = session.run(f"""
                    MATCH (n:Concept)
                    {source_filter}
                    RETURN 
                        elementId(n) as id,
                        n.name AS title,
                        n.definition AS content,
                        labels(n) AS labels,
                        n.source_type AS source_type,
                        coalesce(n.credibility_score, 1.0) as credibility_score,
                        n.chapters_mentions AS chapters_mentions,
                        n.example AS example,
                        n.questions AS questions
                    LIMIT {max_records}
                """, source_types=source_types)
                
                # Преобразуем в список для однократного обхода
                all_records = list(records)
                logger.info(f"Получено {len(all_records)} понятий из базы данных")
                
                if not all_records:
                    logger.warning("База данных не вернула ни одного понятия")
                    return []
                
                # Обработка и вычисление сходства в Python с использованием многопоточности
                start_process_time = time.time()
                
                # Определяем оптимальное количество потоков
                batch_size = max(5, len(all_records) // self.max_workers)
                batches = [all_records[i:i+batch_size] for i in range(0, len(all_records), batch_size)]
                
                logger.info(f"Разбиваем на {len(batches)} пакета(ов) по ~{batch_size} документов, используя {self.max_workers} потоков")
                
                # Параллельная обработка батчей
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
                        # Создаем частичную функцию с фиксированными параметрами
                        process_func = partial(self._process_batch, query_embedding=query_embedding, threshold=threshold)
                        
                        # Запускаем обработку батчей в параллельных потоках
                        future_results = list(executor.map(process_func, batches))
                        
                        # Объединяем результаты из всех батчей
                        for batch_results in future_results:
                            results.extend(batch_results)
                            
                except Exception as e:
                    logger.error(f"Ошибка при параллельной обработке: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Если многопоточность не сработала, переходим к однопоточной обработке
                    logger.info("Переход к однопоточной обработке из-за ошибки...")
                    
                    # Вариант без многопоточности (как запасной вариант)
                    for record in all_records:
                        title = record.get("title", "")
                        content = record.get("content", "")
                        example = record.get("example", "")
                        
                        # Объединяем текст для сравнения
                        document_text = f"{title} {content} {example}".strip()
                        
                        if document_text:
                            try:
                                # Создаем эмбеддинг документа на лету
                                document_embedding = self.encode_query(document_text)
                                
                                # Вычисляем сходство
                                similarity = self.cosine_similarity(query_embedding, document_embedding)
                                
                                if similarity >= threshold:
                                    credibility_score = record.get("credibility_score", 1.0)
                                    weighted_score = similarity * credibility_score
                                    
                                    results.append({
                                        "title": title,
                                        "name": title,  # Для совместимости с существующим форматом
                                        "content": content,
                                        "definition": content,  # Для совместимости с существующим форматом
                                        "labels": record["labels"],
                                        "source_type": record.get("source_type", "official"),
                                        "similarity": similarity,
                                        "credibility_score": credibility_score,
                                        "weighted_score": weighted_score,
                                        "chapters_mentions": record.get("chapters_mentions"),
                                        "example": record.get("example"),
                                        "questions": record.get("questions")
                                    })
                            except Exception as e:
                                logger.error(f"Ошибка при обработке понятия '{title}': {str(e)}")
                                continue
                
                process_time = time.time() - start_process_time
                logger.info(f"Обработано {len(all_records)} понятий за {process_time:.2f} сек, из них {len(results)} превысили порог сходства {threshold}")
                
                # Сортируем по взвешенному рейтингу
                results.sort(key=lambda x: x["weighted_score"], reverse=True)
                
                # Ограничиваем количество результатов
                results = results[:limit]
                
                logger.info(f"Семантический поиск вернул {len(results)} результатов, лучшее совпадение: "
                            f"'{results[0]['title']}' с оценкой {results[0]['similarity']:.4f}" if results else "Нет результатов")
                return results
        except Exception as e:
            logger.error(f"Ошибка при выполнении гибридного поиска: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _search_with_vector_index(self, query_embedding: List[float], 
                                limit: int, threshold: float, 
                                source_types: Optional[List[str]]) -> List[Dict[str, Any]]:
        """
        Поиск с использованием нативного векторного индекса Neo4j
        
        Args:
            query_embedding: Векторное представление запроса
            limit: Максимальное количество результатов
            threshold: Минимальный порог сходства
            source_types: Список типов источников для поиска
            
        Returns:
            Список результатов поиска
        """
        logger.info("Используем нативный векторный индекс Neo4j")
        
        try:
            with self.driver.session() as session:
                # Находим имя векторного индекса
                index_name = None
                try:
                    index_result = session.run("SHOW VECTOR INDEXES")
                    indexes = list(index_result)
                    
                    if indexes:
                        # Ищем индекс для Concept с combined_embedding
                        for idx in indexes:
                            labels = idx.get("labelsOrTypes", [])
                            properties = idx.get("properties", [])
                            
                            if "Concept" in labels and "combined_embedding" in properties:
                                index_name = idx.get("name")
                                logger.info(f"Найден подходящий векторный индекс: {index_name}")
                                break
                            
                        if not index_name:
                            # Берем первый доступный индекс как запасной вариант
                            index_name = indexes[0].get("name")
                            logger.warning(f"Не найден специальный индекс для Concept.combined_embedding, " 
                                         f"используем первый доступный: {index_name}")
                except Exception as e:
                    logger.error(f"Ошибка при поиске индекса: {str(e)}")
                
                if not index_name:
                    index_name = "concept_vectors"  # Значение по умолчанию
                    logger.warning(f"Не удалось найти имя индекса, используем значение по умолчанию: {index_name}")
                
                source_filter = ""
                if source_types and len(source_types) > 0:
                    source_filter = "WHERE c.source_type IN $source_types"
                
                # Увеличиваем количество запрашиваемых результатов для более точной фильтрации
                k = min(limit * 3, 100)  # Не более 100, чтобы не перегружать базу
                
                # Используем нативный векторный поиск
                query = f"""
                    CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
                    YIELD node, score
                    WITH node as c, score
                    WHERE score >= $threshold
                    {source_filter}
                    RETURN 
                        c.name AS title,
                        c.definition AS content,
                        labels(c) AS labels,
                        c.source_type AS source_type,
                        coalesce(c.credibility_score, 1.0) as credibility_score,
                        score AS similarity,
                        c.chapters_mentions AS chapters_mentions,
                        c.example AS example,
                        c.questions AS questions
                    ORDER BY score * credibility_score DESC
                    LIMIT $limit
                """
                
                logger.debug(f"Выполняем запрос к Neo4j Vector Index: {query}")
                
                result = session.run(
                    query, 
                    index_name=index_name, 
                    k=k, 
                    query_embedding=query_embedding, 
                    threshold=threshold, 
                    source_types=source_types, 
                    limit=limit
                )
                
                # Преобразуем результаты
                results = []
                for record in result:
                    similarity = record.get("similarity", 0)
                    credibility_score = record.get("credibility_score", 1.0)
                    weighted_score = similarity * credibility_score
                    
                    results.append({
                        "title": record.get("title", ""),
                        "name": record.get("title", ""),  # Для совместимости
                        "content": record.get("content", ""),
                        "definition": record.get("content", ""),  # Для совместимости
                        "labels": record.get("labels", []),
                        "source_type": record.get("source_type", "official"),
                        "similarity": similarity,
                        "credibility_score": credibility_score,
                        "weighted_score": weighted_score,
                        "chapters_mentions": record.get("chapters_mentions"),
                        "example": record.get("example"),
                        "questions": record.get("questions")
                    })
                
                logger.info(f"Нативный векторный поиск вернул {len(results)} результатов")
                return results
        
        except Exception as e:
            logger.error(f"Ошибка при использовании нативного векторного индекса: {str(e)}")
            logger.error(traceback.format_exc())
            # Если произошла ошибка с нативным индексом, переключаемся на гибридный поиск
            logger.info("Переключение на гибридный поиск из-за ошибки")
            return self._search_hybrid(None, query_embedding, limit, threshold, source_types)
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Форматирование результатов поиска для вывода
        
        Args:
            results: Список результатов поиска
            
        Returns:
            Отформатированная строка с результатами
        """
        if not results:
            return "По вашему запросу ничего не найдено."
            
        output = "Найденные результаты:\n\n"
        for i, result in enumerate(results):
            # Рассчитываем процент сходства
            similarity_percent = round(result.get("similarity", 0) * 100, 1)
            
            # Формируем заголовок для результата
            output += f"{i+1}. **{result['title']}** "
            output += f"(Релевантность: {similarity_percent}%)\n\n"
            
            # Добавляем определение
            if result.get("definition"):
                definition = result["definition"]
                output += f"   Определение: {definition}\n\n"
                
            # Добавляем пример, если есть
            if result.get("example"):
                example = result["example"]
                output += f"   Пример: {example}\n\n"
            
        return output 