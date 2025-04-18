#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для кэширования результатов векторного поиска.
Позволяет значительно ускорить поиск для повторяющихся запросов.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional
import hashlib

from ai_tutor.database.enhanced_search import EnhancedCourseSearch

logger = logging.getLogger(__name__)

class CachedSearchResult:
    """Класс для хранения кэшированного результата с метаданными"""
    
    def __init__(self, results, timestamp, query, params):
        """
        Инициализация результата кэша
        
        Args:
            results: Результаты поиска
            timestamp: Временная метка создания
            query: Исходный запрос
            params: Параметры поиска
        """
        self.results = results
        self.timestamp = timestamp
        self.query = query
        self.params = params
    
    def is_expired(self, ttl: int) -> bool:
        """
        Проверка срока действия кэша
        
        Args:
            ttl: Время жизни в секундах
            
        Returns:
            True если кэш устарел, иначе False
        """
        return time.time() - self.timestamp > ttl

class CachedSearch:
    """
    Класс для кэширования результатов векторного поиска
    """
    
    def __init__(self, search_engine: EnhancedCourseSearch, cache_ttl: int = 3600, 
                 max_cache_size: int = 100):
        """
        Инициализация кэшированного поиска
        
        Args:
            search_engine: Экземпляр EnhancedCourseSearch
            cache_ttl: Время жизни кэша в секундах (по умолчанию 1 час)
            max_cache_size: Максимальный размер кэша
        """
        self.search_engine = search_engine
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, CachedSearchResult] = {}
        logger.info(f"Инициализирован кэширующий поиск с TTL={cache_ttl}с и размером кэша {max_cache_size}")
    
    def _generate_cache_key(self, query: str, **kwargs) -> str:
        """
        Генерирует ключ кэша для запроса и параметров
        
        Args:
            query: Текстовый запрос
            **kwargs: Дополнительные параметры поиска
            
        Returns:
            Строковый ключ кэша
        """
        # Создаем словарь с запросом и параметрами
        cache_dict = {"query": query}
        cache_dict.update(kwargs)
        
        # Сериализуем в JSON
        cache_str = json.dumps(cache_dict, sort_keys=True)
        
        # Создаем хеш для экономии памяти
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    
    def _cleanup_cache_if_needed(self):
        """
        Очищает кэш, если его размер превышает максимальный
        """
        if len(self.cache) > self.max_cache_size:
            logger.info(f"Размер кэша превысил максимум ({len(self.cache)}>{self.max_cache_size}). Очистка...")
            
            # Сортируем записи кэша по времени и оставляем только самые новые
            cache_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1].timestamp, 
                reverse=True
            )
            
            # Оставляем только половину самых новых записей
            keep_count = self.max_cache_size // 2
            self.cache = {k: v for k, v in cache_items[:keep_count]}
            
            logger.info(f"Кэш очищен. Новый размер: {len(self.cache)}")
    
    def search(self, query: str, limit: int = 10, threshold: float = 0.5, 
               source_types: Optional[List[str]] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Выполняет поиск с использованием кэша
        
        Args:
            query: Текстовый запрос для поиска
            limit: Максимальное количество результатов
            threshold: Минимальный порог сходства (0-1)
            source_types: Список типов источников для поиска
            use_cache: Использовать ли кэш (True) или принудительно выполнить новый поиск (False)
            
        Returns:
            Список результатов поиска
        """
        if not query:
            logger.warning("Пустой запрос для поиска")
            return []
        
        # Если кэширование отключено, сразу выполняем поиск
        if not use_cache:
            logger.info("Кэширование отключено для этого запроса")
            return self.search_engine.semantic_search_with_ranking(
                query, limit, threshold, source_types
            )
        
        # Генерируем ключ кэша
        params = {
            "limit": limit,
            "threshold": threshold,
            "source_types": source_types
        }
        cache_key = self._generate_cache_key(query, **params)
        
        # Проверяем наличие в кэше
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            
            # Проверяем срок действия кэша
            if not cached_result.is_expired(self.cache_ttl):
                logger.info(f"Найден актуальный кэш для запроса: '{query[:50]}...'")
                return cached_result.results
            else:
                logger.info(f"Кэш для запроса устарел: '{query[:50]}...'")
        
        # Выполняем поиск
        start_time = time.time()
        results = self.search_engine.semantic_search_with_ranking(
            query, limit, threshold, source_types
        )
        search_time = time.time() - start_time
        
        # Сохраняем результаты в кэш
        self.cache[cache_key] = CachedSearchResult(
            results, time.time(), query, params
        )
        
        # Очищаем кэш, если он слишком большой
        self._cleanup_cache_if_needed()
        
        logger.info(f"Поиск выполнен за {search_time:.2f}с и сохранен в кэш. "
                   f"Текущий размер кэша: {len(self.cache)}")
        
        return results
    
    def clear_cache(self):
        """
        Полностью очищает кэш
        """
        cache_size = len(self.cache)
        self.cache = {}
        logger.info(f"Кэш полностью очищен. Удалено {cache_size} записей.")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования кэша
        
        Returns:
            Словарь со статистикой кэша
        """
        # Подсчитываем количество записей по времени
        current_time = time.time()
        valid_count = sum(1 for item in self.cache.values() 
                         if not item.is_expired(self.cache_ttl))
        
        expired_count = len(self.cache) - valid_count
        
        # Находим самые старые и новые записи
        oldest_time = min(item.timestamp for item in self.cache.values()) if self.cache else 0
        newest_time = max(item.timestamp for item in self.cache.values()) if self.cache else 0
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_count,
            "expired_entries": expired_count,
            "cache_ttl": self.cache_ttl,
            "max_cache_size": self.max_cache_size,
            "oldest_entry_age": int(current_time - oldest_time) if oldest_time else 0,
            "newest_entry_age": int(current_time - newest_time) if newest_time else 0
        }
    
    def close(self):
        """
        Закрывает соединение с базой данных в поисковом движке
        """
        if hasattr(self.search_engine, 'close'):
            self.search_engine.close()


# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Инициализация поискового движка
    from ai_tutor.database.enhanced_search import EnhancedCourseSearch
    search_engine = EnhancedCourseSearch()
    
    # Инициализация кэширующего поиска
    cached_search = CachedSearch(search_engine, cache_ttl=1800, max_cache_size=50)
    
    try:
        # Примеры запросов
        queries = [
            "Что такое системное мышление?",
            "Как развивать критическое мышление?",
            "Что такое когнитивные искажения?",
            "Что такое системное мышление?",  # Повторный запрос для проверки кэша
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Запрос {i}: {query} ---")
            
            start_time = time.time()
            results = cached_search.search(query, limit=3, threshold=0.5)
            elapsed = time.time() - start_time
            
            print(f"Поиск выполнен за {elapsed:.3f} секунд")
            print(f"Найдено {len(results)} результатов:")
            
            for j, result in enumerate(results, 1):
                print(f"{j}. {result['title']} (Релевантность: {result['similarity']:.2f})")
        
        # Выводим статистику кэша
        print("\n--- Статистика кэша ---")
        stats = cached_search.get_cache_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    finally:
        # Закрываем соединения
        cached_search.close() 