#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Утилита для анализа и мониторинга производительности поиска.
Помогает отслеживать время выполнения различных этапов поиска 
и находить узкие места для дальнейшей оптимизации.
"""

import logging
import time
import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импортируем нужные модули
try:
    from ai_tutor.database.enhanced_search import EnhancedCourseSearch, MODEL_VARIANTS
    from ai_tutor.database.cached_search import CachedSearch
except ImportError:
    logger.error("Не удалось импортировать модули. Убедитесь, что вы запускаете скрипт из корня проекта.")
    raise

class PerformanceMonitor:
    """Класс для мониторинга производительности поиска"""
    
    def __init__(self, output_dir: str = "performance_logs"):
        """
        Инициализация монитора производительности
        
        Args:
            output_dir: Директория для сохранения логов производительности
        """
        self.output_dir = output_dir
        self.performance_data = []
        
        # Создаем директорию для логов, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Имя файла для текущей сессии
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"search_performance_{timestamp}.json")
        
        logger.info(f"Монитор производительности инициализирован. Лог: {self.log_file}")
    
    def run_test(self, query: str, model_type: str = "default", 
                with_cache: bool = False, with_vector_index: bool = True,
                limit: int = 5, threshold: float = 0.5, 
                max_workers: int = 4) -> Dict[str, Any]:
        """
        Запускает тест производительности поиска
        
        Args:
            query: Поисковый запрос
            model_type: Тип модели из MODEL_VARIANTS
            with_cache: Использовать ли кэширование
            with_vector_index: Использовать ли векторный индекс Neo4j
            limit: Максимальное количество результатов
            threshold: Минимальный порог сходства
            max_workers: Количество потоков для параллельной обработки
            
        Returns:
            Словарь с метриками производительности
        """
        logger.info(f"Запуск теста: '{query}' с моделью {model_type}")
        
        # Инициализируем поисковый движок
        start_init = time.time()
        search_engine = EnhancedCourseSearch(model_name=model_type, max_workers=max_workers)
        
        # Переключаем использование векторного индекса
        if not with_vector_index:
            search_engine.has_vector_index = False
            logger.info("Использование векторного индекса отключено")
        
        # Кэширующий поиск, если нужно
        if with_cache:
            search = CachedSearch(search_engine, cache_ttl=3600, max_cache_size=100)
        else:
            search = search_engine
            
        init_time = time.time() - start_init
        logger.info(f"Инициализация поиска завершена за {init_time:.3f} секунд")
        
        # Запускаем поиск с замером времени
        try:
            # Первый запуск (холодный старт)
            start_first = time.time()
            
            if with_cache:
                # Очищаем кэш перед первым запуском
                search.clear_cache()
                results_first = search.search(query, limit=limit, threshold=threshold, use_cache=True)
            else:
                results_first = search.semantic_search_with_ranking(
                    query, limit=limit, threshold=threshold
                )
                
            first_time = time.time() - start_first
            logger.info(f"Первый поиск выполнен за {first_time:.3f} секунд, найдено {len(results_first)} результатов")
            
            # Второй запуск (должен быть быстрее из-за кэша или оптимизаций)
            start_second = time.time()
            
            if with_cache:
                results_second = search.search(query, limit=limit, threshold=threshold, use_cache=True)
            else:
                results_second = search.semantic_search_with_ranking(
                    query, limit=limit, threshold=threshold
                )
                
            second_time = time.time() - start_second
            logger.info(f"Второй поиск выполнен за {second_time:.3f} секунд, найдено {len(results_second)} результатов")
            
            # Собираем метрики
            metrics = {
                "query": query,
                "model_type": model_type,
                "with_cache": with_cache,
                "with_vector_index": with_vector_index,
                "limit": limit,
                "threshold": threshold,
                "max_workers": max_workers,
                "results_count": len(results_first),
                "init_time": init_time,
                "first_search_time": first_time,
                "second_search_time": second_time,
                "timestamp": time.time()
            }
            
            # Добавляем в общий список метрик
            self.performance_data.append(metrics)
            
            # Сохраняем результаты в файл после каждого теста
            self._save_performance_data()
            
            return metrics
        
        finally:
            # Закрываем соединения
            if hasattr(search, 'close'):
                search.close()
    
    def run_batch_tests(self, queries: List[str], 
                      models: Optional[List[str]] = None,
                      with_caches: Optional[List[bool]] = None,
                      with_vector_indexes: Optional[List[bool]] = None) -> List[Dict[str, Any]]:
        """
        Запускает серию тестов с различными параметрами
        
        Args:
            queries: Список поисковых запросов
            models: Список моделей для тестирования
            with_caches: Список вариантов использования кэша
            with_vector_indexes: Список вариантов использования векторного индекса
            
        Returns:
            Список с метриками производительности для всех тестов
        """
        if models is None:
            models = ["default", "fast"]
            
        if with_caches is None:
            with_caches = [False, True]
            
        if with_vector_indexes is None:
            with_vector_indexes = [False, True]
        
        logger.info(f"Запуск пакетного тестирования: {len(queries)} запросов")
        logger.info(f"Модели: {models}")
        logger.info(f"Кэширование: {with_caches}")
        logger.info(f"Векторные индексы: {with_vector_indexes}")
        
        all_metrics = []
        
        # Запускаем тесты для всех комбинаций параметров
        for query in queries:
            for model in models:
                for with_cache in with_caches:
                    for with_vector_index in with_vector_indexes:
                        # Пропускаем кэширование + векторный индекс, так как кэш перекрывает индекс
                        if with_cache and with_vector_index:
                            continue
                            
                        metrics = self.run_test(
                            query=query,
                            model_type=model,
                            with_cache=with_cache,
                            with_vector_index=with_vector_index
                        )
                        
                        all_metrics.append(metrics)
        
        logger.info(f"Пакетное тестирование завершено. Выполнено {len(all_metrics)} тестов")
        return all_metrics
    
    def _save_performance_data(self):
        """Сохраняет данные о производительности в JSON-файл"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.performance_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Данные о производительности сохранены в {self.log_file}")
    
    def load_performance_data(self, filename: Optional[str] = None):
        """
        Загружает данные о производительности из JSON-файла
        
        Args:
            filename: Имя файла для загрузки. Если None, используется текущий лог-файл
        """
        if filename is None:
            filename = self.log_file
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.performance_data = json.load(f)
                
            logger.info(f"Загружено {len(self.performance_data)} записей из {filename}")
        except FileNotFoundError:
            logger.warning(f"Файл {filename} не найден")
        except json.JSONDecodeError:
            logger.error(f"Ошибка при чтении файла {filename}")
    
    def analyze_performance(self) -> pd.DataFrame:
        """
        Анализирует собранные данные о производительности
        
        Returns:
            DataFrame с агрегированными метриками
        """
        if not self.performance_data:
            logger.warning("Нет данных для анализа")
            return pd.DataFrame()
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(self.performance_data)
        
        # Создаем агрегированные метрики
        agg_df = df.groupby(['model_type', 'with_cache', 'with_vector_index']).agg({
            'init_time': ['mean', 'min', 'max'],
            'first_search_time': ['mean', 'min', 'max'],
            'second_search_time': ['mean', 'min', 'max'],
            'query': 'count'
        }).reset_index()
        
        # Переименовываем колонки для удобства
        agg_df.columns = [
            '_'.join(col).strip() if isinstance(col, tuple) else col 
            for col in agg_df.columns.values
        ]
        
        # Переименовываем колонку количества запросов
        agg_df.rename(columns={'query_count': 'num_queries'}, inplace=True)
        
        return agg_df
    
    def generate_report(self, filename: Optional[str] = "performance_report.html"):
        """
        Генерирует HTML-отчет с графиками и таблицами
        
        Args:
            filename: Имя файла для сохранения отчета
        """
        if not self.performance_data:
            logger.warning("Нет данных для создания отчета")
            return
        
        # Получаем агрегированные данные
        agg_df = self.analyze_performance()
        
        # Создаем описательную статистику
        pd.set_option('display.precision', 4)
        df = pd.DataFrame(self.performance_data)
        
        # Формируем HTML-отчет
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Отчет о производительности поиска</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .chart-container { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <h1>Отчет о производительности поиска</h1>
            <p>Создан: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>Количество тестов: """ + str(len(self.performance_data)) + """</p>
            
            <h2>Сводная статистика</h2>
            """ + agg_df.to_html() + """
            
            <h2>Основные метрики</h2>
            <div class="chart-container">
                <h3>Среднее время выполнения поиска</h3>
                <img src="search_time_chart.png" alt="Среднее время поиска" />
            </div>
            
            <h2>Описательная статистика</h2>
            """ + df.describe().to_html() + """
            
            <h2>Детальные результаты</h2>
            """ + df.to_html() + """
        </body>
        </html>
        """
        
        # Создаем график среднего времени поиска
        self._create_search_time_chart()
        
        # Сохраняем отчет
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
        logger.info(f"Отчет о производительности сохранен в {report_path}")
    
    def _create_search_time_chart(self):
        """Создает график среднего времени поиска для разных конфигураций"""
        df = pd.DataFrame(self.performance_data)
        agg_df = self.analyze_performance()
        
        plt.figure(figsize=(12, 8))
        
        # Подготавливаем данные для графика
        config_names = []
        first_times = []
        second_times = []
        
        for _, row in agg_df.iterrows():
            config_name = f"{row['model_type']}\nCache: {row['with_cache']}\nVector: {row['with_vector_index']}"
            config_names.append(config_name)
            first_times.append(row['first_search_time_mean'])
            second_times.append(row['second_search_time_mean'])
        
        # Создаем график
        x = np.arange(len(config_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bar1 = ax.bar(x - width/2, first_times, width, label='Первый поиск')
        bar2 = ax.bar(x + width/2, second_times, width, label='Повторный поиск')
        
        # Оформляем график
        ax.set_ylabel('Время (секунды)')
        ax.set_title('Среднее время выполнения поиска для разных конфигураций')
        ax.set_xticks(x)
        ax.set_xticklabels(config_names)
        ax.legend()
        
        # Добавляем значения на столбцы
        for i, v in enumerate(first_times):
            ax.text(i - width/2, v + 0.1, f"{v:.2f}s", ha='center')
        
        for i, v in enumerate(second_times):
            ax.text(i + width/2, v + 0.1, f"{v:.2f}s", ha='center')
        
        plt.tight_layout()
        
        # Сохраняем график
        chart_path = os.path.join(self.output_dir, "search_time_chart.png")
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"График времени поиска сохранен в {chart_path}")
        
def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Анализ производительности векторного поиска')
    
    parser.add_argument('--queries', type=str, nargs='+',
                      help='Запросы для тестирования (по умолчанию используются предустановленные)')
    parser.add_argument('--models', type=str, nargs='+', default=['default', 'fast'],
                      help='Модели для тестирования (по умолчанию: default, fast)')
    parser.add_argument('--output-dir', type=str, default='performance_logs',
                      help='Директория для сохранения логов производительности')
    parser.add_argument('--report', action='store_true',
                      help='Сгенерировать HTML-отчет после тестирования')
    parser.add_argument('--verbose', action='store_true',
                      help='Включить подробное логирование')
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Предустановленные запросы
    default_queries = [
        "Что такое системное мышление?",
        "Как развивать критическое мышление?",
        "Что такое когнитивные искажения в мышлении?",
        "Каковы основные принципы саморазвития?",
        "Как управлять своим вниманием?"
    ]
    
    queries = args.queries if args.queries else default_queries
    
    # Инициализируем монитор
    monitor = PerformanceMonitor(output_dir=args.output_dir)
    
    # Запускаем пакетное тестирование
    monitor.run_batch_tests(
        queries=queries,
        models=args.models
    )
    
    # Анализируем результаты
    agg_df = monitor.analyze_performance()
    print("\n=== Агрегированные метрики ===")
    print(agg_df)
    
    # Генерируем отчет, если нужно
    if args.report:
        monitor.generate_report()

if __name__ == "__main__":
    main() 