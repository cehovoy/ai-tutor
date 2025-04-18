#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для сравнения производительности различных подходов к векторному поиску.
Сравнивает стандартный поиск, гибридный подход, нативный индекс Neo4j и кэширование.
"""

import time
import logging
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from textwrap import wrap

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импортируем нужные модули
try:
    from ai_tutor.database.enhanced_search import EnhancedCourseSearch
    from ai_tutor.database.cached_search import CachedSearch
except ImportError:
    logger.error("Не удалось импортировать модули. Убедитесь, что вы запускаете скрипт из корня проекта.")
    raise

def run_benchmark(queries: List[str], runs: int = 3, 
                  use_cache: bool = True, use_native_index: bool = False,
                  limit: int = 5, threshold: float = 0.5):
    """
    Запускает бенчмарк для различных методов поиска
    
    Args:
        queries: Список запросов для тестирования
        runs: Количество прогонов для каждого запроса
        use_cache: Использовать ли кэширование
        use_native_index: Использовать ли нативный индекс Neo4j
        limit: Максимальное количество результатов
        threshold: Минимальный порог сходства
        
    Returns:
        Словарь с результатами бенчмарка
    """
    logger.info(f"Запуск бенчмарка для {len(queries)} запросов с {runs} прогонами каждый")
    
    # Инициализируем поисковый движок
    search_engine = EnhancedCourseSearch()
    
    # Если нужно использовать нативный индекс, включаем его
    if use_native_index:
        logger.info("Включение нативного векторного индекса Neo4j")
        search_engine.has_vector_index = True
    
    # Инициализируем кэширующий поиск, если нужно
    if use_cache:
        logger.info("Инициализация кэширующего поиска")
        cached_search = CachedSearch(search_engine, cache_ttl=3600, max_cache_size=100)
    
    results = []
    
    try:
        for query_idx, query in enumerate(queries):
            logger.info(f"Тестирование запроса {query_idx+1}/{len(queries)}: '{query[:50]}...'")
            
            for run in range(runs):
                logger.info(f"Запуск {run+1}/{runs}")
                
                # Тестируем обычный поиск
                start_time = time.time()
                standard_results = search_engine.semantic_search_with_ranking(
                    query, limit, threshold
                )
                standard_time = time.time() - start_time
                
                # Тестируем кэшированный поиск, если включено
                if use_cache:
                    # Первый запуск - заполнение кэша
                    if run == 0:
                        # Очищаем кэш перед первым запуском
                        cached_search.clear_cache()
                        
                        start_time = time.time()
                        cache_results_first = cached_search.search(
                            query, limit, threshold, use_cache=True
                        )
                        cache_time_first = time.time() - start_time
                    
                    # Повторный запрос - должен быть из кэша
                    start_time = time.time()
                    cache_results = cached_search.search(
                        query, limit, threshold, use_cache=True
                    )
                    cache_time = time.time() - start_time
                
                # Записываем результаты
                run_results = {
                    "query": query,
                    "run": run + 1,
                    "standard_time": standard_time,
                    "standard_results": len(standard_results),
                }
                
                if use_cache:
                    run_results.update({
                        "cache_time_first": cache_time_first if run == 0 else None,
                        "cache_time": cache_time,
                        "cache_results": len(cache_results),
                    })
                
                results.append(run_results)
                
                logger.info(f"Стандартный поиск: {standard_time:.3f}с, {len(standard_results)} результатов")
                if use_cache:
                    if run == 0:
                        logger.info(f"Кэш (первый запрос): {cache_time_first:.3f}с, {len(cache_results_first)} результатов")
                    logger.info(f"Кэш (повторный запрос): {cache_time:.3f}с, {len(cache_results)} результатов")
    
    finally:
        # Закрываем соединения
        if use_cache:
            cached_search.close()
        else:
            search_engine.close()
    
    return results

def analyze_results(results: List[Dict[str, Any]]):
    """
    Анализирует результаты бенчмарка и выводит статистику
    
    Args:
        results: Список результатов бенчмарка
    """
    if not results:
        logger.error("Нет результатов для анализа")
        return
    
    # Преобразуем в DataFrame для удобства анализа
    df = pd.DataFrame(results)
    
    # Анализируем по запросам
    queries = df["query"].unique()
    
    print("\n=== Результаты бенчмарка ===")
    print(f"Протестировано {len(queries)} запросов")
    
    # Общая статистика
    print("\n--- Общая статистика ---")
    total_stats = {}
    
    # Стандартный поиск
    total_stats["Стандартный поиск"] = {
        "Среднее время (с)": df["standard_time"].mean(),
        "Медиана времени (с)": df["standard_time"].median(),
        "Мин. время (с)": df["standard_time"].min(),
        "Макс. время (с)": df["standard_time"].max(),
    }
    
    # Кэшированный поиск
    if "cache_time" in df.columns:
        # Первый запрос (заполнение кэша)
        df_first = df[df["run"] == 1]
        total_stats["Кэш (первый запрос)"] = {
            "Среднее время (с)": df_first["cache_time_first"].mean(),
            "Медиана времени (с)": df_first["cache_time_first"].median(),
            "Мин. время (с)": df_first["cache_time_first"].min(),
            "Макс. время (с)": df_first["cache_time_first"].max(),
        }
        
        # Повторные запросы
        total_stats["Кэш (повторный запрос)"] = {
            "Среднее время (с)": df["cache_time"].mean(),
            "Медиана времени (с)": df["cache_time"].median(),
            "Мин. время (с)": df["cache_time"].min(),
            "Макс. время (с)": df["cache_time"].max(),
        }
    
    # Выводим общую статистику
    for method, stats in total_stats.items():
        print(f"\n{method}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {stat_value:.4f}")
    
    # Статистика по запросам
    print("\n--- Статистика по запросам ---")
    
    for i, query in enumerate(queries):
        query_df = df[df["query"] == query]
        print(f"\nЗапрос {i+1}: '{query[:50]}...'")
        
        # Стандартный поиск
        std_time = query_df["standard_time"].mean()
        print(f"  Стандартный поиск: {std_time:.4f}с")
        
        # Кэшированный поиск
        if "cache_time" in df.columns:
            first_time = query_df[query_df["run"] == 1]["cache_time_first"].values[0]
            cache_time = query_df["cache_time"].mean()
            
            print(f"  Кэш (первый запрос): {first_time:.4f}с")
            print(f"  Кэш (повторный запрос): {cache_time:.4f}с")
            print(f"  Ускорение от кэша: {std_time/cache_time:.1f}x")
    
    # Создаем график
    if "cache_time" in df.columns:
        plot_results(df)

def plot_results(df):
    """
    Создает график результатов бенчмарка
    
    Args:
        df: DataFrame с результатами
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Готовим данные для графика
        queries = df["query"].unique()
        x = np.arange(len(queries))
        width = 0.25
        
        # Вычисляем средние значения по запросам
        std_times = []
        cache_first_times = []
        cache_times = []
        
        for query in queries:
            query_df = df[df["query"] == query]
            std_times.append(query_df["standard_time"].mean())
            
            if "cache_time" in df.columns:
                first_df = query_df[query_df["run"] == 1]
                cache_first_times.append(first_df["cache_time_first"].values[0])
                cache_times.append(query_df["cache_time"].mean())
        
        # Создаем набор графиков
        bar1 = plt.bar(x - width, std_times, width, label='Стандартный поиск')
        
        if cache_first_times:
            bar2 = plt.bar(x, cache_first_times, width, label='Кэш (первый запрос)')
            bar3 = plt.bar(x + width, cache_times, width, label='Кэш (повторный запрос)')
        
        # Добавляем метки и заголовок
        plt.xlabel('Запросы')
        plt.ylabel('Время выполнения (секунды)')
        plt.title('Сравнение производительности методов поиска')
        
        # Ограничиваем длину подписей запросов
        labels = ['\n'.join(wrap(q[:50] + '...', 20)) for q in queries]
        plt.xticks(x, labels)
        
        plt.legend()
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig('search_benchmark_results.png')
        logger.info("График сохранен в файл search_benchmark_results.png")
        
        # Показываем график, если возможно
        try:
            plt.show()
        except:
            pass
        
    except Exception as e:
        logger.error(f"Ошибка при создании графика: {str(e)}")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Бенчмарк для сравнения методов векторного поиска')
    
    parser.add_argument('--queries', type=str, nargs='+',
                      help='Запросы для тестирования (по умолчанию используются предустановленные)')
    parser.add_argument('--runs', type=int, default=3,
                      help='Количество прогонов для каждого запроса (по умолчанию: 3)')
    parser.add_argument('--no-cache', action='store_true',
                      help='Отключить тестирование кэширования')
    parser.add_argument('--use-native-index', action='store_true',
                      help='Использовать нативный индекс Neo4j (если доступен)')
    parser.add_argument('--limit', type=int, default=5,
                      help='Максимальное количество результатов (по умолчанию: 5)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Минимальный порог сходства (по умолчанию: 0.5)')
    
    args = parser.parse_args()
    
    # Предустановленные запросы, если не указаны свои
    default_queries = [
        "Что такое системное мышление?",
        "Как развивать критическое мышление?",
        "Что такое когнитивные искажения в мышлении?",
        "Каковы основные принципы саморазвития?",
        "Как управлять своим вниманием?"
    ]
    
    queries = args.queries if args.queries else default_queries
    
    # Запускаем бенчмарк
    results = run_benchmark(
        queries=queries,
        runs=args.runs,
        use_cache=not args.no_cache,
        use_native_index=args.use_native_index,
        limit=args.limit,
        threshold=args.threshold
    )
    
    # Анализируем результаты
    analyze_results(results)

if __name__ == "__main__":
    main() 