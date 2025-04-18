# Векторный поиск в Neo4j для AI-Tutor

Этот раздел описывает модули для работы с векторными эмбеддингами и семантическим поиском в базе данных Neo4j. 

## Основные компоненты

### 1. EnhancedCourseSearch (`enhanced_search.py`)

Основной класс для выполнения семантического поиска по векторным эмбеддингам понятий курса. Поддерживает:

- Поиск по содержимому курса с учетом приоритета источников
- Вычисление косинусного сходства между векторами
- Опциональное использование нативного векторного индекса Neo4j (для версии 5.11+)
- Гибридный подход к поиску для ускорения обработки
- Параллельную обработку для ускорения вычисления сходства
- Выбор различных моделей для эмбеддингов (быстрых или точных)

```python
from ai_tutor.database.enhanced_search import EnhancedCourseSearch, MODEL_VARIANTS

# Инициализация с параметрами по умолчанию
search = EnhancedCourseSearch()

# Инициализация с параллельной обработкой и выбором модели
search = EnhancedCourseSearch(
    model_name="fast",  # Выбор быстрой модели: "fast", "default", "accurate"
    max_workers=8  # Количество потоков для параллельной обработки
)

# Поиск с базовыми параметрами
results = search.semantic_search_with_ranking("Что такое системное мышление?")

# Поиск с расширенными параметрами
results = search.semantic_search_with_ranking(
    query="Как управлять вниманием?",
    limit=10,  # Максимальное количество результатов
    threshold=0.6,  # Минимальный порог сходства (0-1)
    source_types=["official", "teacher"]  # Ограничение по типам источников
)
```

### 2. CachedSearch (`cached_search.py`)

Обертка над `EnhancedCourseSearch` для кэширования результатов поиска. Значительно ускоряет повторные запросы.

```python
from ai_tutor.database.enhanced_search import EnhancedCourseSearch
from ai_tutor.database.cached_search import CachedSearch

# Инициализация базового поиска
search_engine = EnhancedCourseSearch()

# Инициализация кэширующего поиска
cached_search = CachedSearch(
    search_engine,
    cache_ttl=3600,  # Время жизни кэша в секундах (1 час)
    max_cache_size=100  # Максимальное количество записей в кэше
)

# Поиск с использованием кэша
results = cached_search.search("Что такое системное мышление?")

# Принудительное обновление кэша
results = cached_search.search("Что такое системное мышление?", use_cache=False)

# Статистика использования кэша
stats = cached_search.get_cache_stats()
print(stats)

# Очистка кэша
cached_search.clear_cache()
```

### 3. Векторные индексы Neo4j (`create_vector_index.py`)

Утилита для создания и управления векторными индексами в Neo4j, которые значительно ускоряют поиск по векторным эмбеддингам (для Neo4j 5.11+).

```bash
# Показать все векторные индексы
python -m ai_tutor.database.create_vector_index show

# Создать векторный индекс для понятий
python -m ai_tutor.database.create_vector_index create

# Создать индекс с пользовательскими параметрами
python -m ai_tutor.database.create_vector_index create --name custom_index --field combined_embedding --dimensions 768

# Удалить индекс
python -m ai_tutor.database.create_vector_index drop --name concept_vectors

# Протестировать поиск с использованием индекса
python -m ai_tutor.database.create_vector_index test "Системное мышление"
```

### 4. Бенчмарк производительности (`benchmark_search.py`)

Инструмент для сравнения производительности различных подходов к векторному поиску.

```bash
# Запустить бенчмарк с параметрами по умолчанию
python -m ai_tutor.database.benchmark_search

# Запустить с пользовательскими запросами
python -m ai_tutor.database.benchmark_search --queries "Системное мышление" "Критическое мышление"

# Запустить с нативным индексом Neo4j (если доступен)
python -m ai_tutor.database.benchmark_search --use-native-index

# Запустить без кэширования
python -m ai_tutor.database.benchmark_search --no-cache
```

### 5. Анализ производительности (`analyze_search_performance.py`)

Утилита для мониторинга и анализа производительности поиска с генерацией отчетов.

```bash
# Запустить анализ производительности с параметрами по умолчанию
python -m ai_tutor.database.analyze_search_performance

# Запустить с генерацией HTML-отчета
python -m ai_tutor.database.analyze_search_performance --report

# Тестировать только определенные модели
python -m ai_tutor.database.analyze_search_performance --models fast accurate

# Указать собственную директорию для логов
python -m ai_tutor.database.analyze_search_performance --output-dir ./performance_results
```

Пример анализа производительности в коде:

```python
from ai_tutor.database.analyze_search_performance import PerformanceMonitor

# Инициализация монитора
monitor = PerformanceMonitor()

# Запуск теста для одного запроса
metrics = monitor.run_test(
    query="Что такое системное мышление?",
    model_type="fast",
    with_cache=True
)

# Генерация отчета
monitor.generate_report()
```

## Доступные модели для эмбеддингов

Поддерживаются различные модели для оптимизации соотношения скорости и качества:

| Ключ | Модель | Описание |
|------|--------|----------|
| `default` | paraphrase-multilingual-MiniLM-L12-v2 | Оптимальный баланс скорости и качества |
| `fast` | paraphrase-multilingual-MiniLM-L6-v2 | Быстрая, но менее точная модель |
| `accurate` | paraphrase-multilingual-mpnet-base-v2 | Более точная, но медленная модель |
| `all-MiniLM-L6` | all-MiniLM-L6-v2 | Быстрая английская модель |
| `all-mpnet` | all-mpnet-base-v2 | Точная английская модель |

## Рекомендации по оптимизации

Подробные рекомендации по оптимизации производительности векторного поиска приведены в файле [README_vector_optimization.md](README_vector_optimization.md).

Краткие рекомендации:

1. **Для наилучшей производительности** используйте Neo4j Vector Index (требуется Neo4j 5.11+)
2. **Для хорошей производительности** на любой версии Neo4j используйте `CachedSearch`
3. **Для работы с большими объемами данных** используйте гибридный подход с параллельной обработкой
4. **Для ускорения холодного старта** заранее вычислите и сохраните все эмбеддинги в базе данных
5. **Для менее требовательных задач** используйте более легкую модель `fast` вместо `default`

## Структура данных

Модули работают со следующей структурой данных в Neo4j:

### Узлы (Nodes)

1. **Concept** - понятия курса
   - `name` - название понятия
   - `definition` - определение понятия
   - `example` - пример использования
   - `combined_embedding` - векторное представление (название + определение + пример)
   - `source_type` - тип источника ('official')
   - `credibility_score` - уровень достоверности (1.0)

2. **ForumPost** - посты форума (если используются)
   - `title` - заголовок поста
   - `content` - содержимое поста
   - `embedding` - векторное представление
   - `source_type` - тип источника ('teacher' или 'student')
   - `credibility_score` - уровень достоверности (0.9 для преподавателей, 0.6 для студентов)

## Типовые сценарии использования

### 1. Поиск понятий для генерации задач

```python
from ai_tutor.database.cached_search import CachedSearch
from ai_tutor.database.enhanced_search import EnhancedCourseSearch

# Используем быструю модель для задач генерации
search_engine = EnhancedCourseSearch(model_name="fast", max_workers=4)
search = CachedSearch(search_engine)

# Поиск для генератора задач - используем только официальные материалы
task_context = search.search(
    "тема задачи", 
    limit=5, 
    source_types=["official"]
)
```

### 2. Проверка ответов студентов

```python
# Для проверки ответов - используем более точную модель
search_engine = EnhancedCourseSearch(model_name="accurate")
search = CachedSearch(search_engine)

# Используем официальные материалы и ответы преподавателей
verification_context = search.search(
    "ответ студента", 
    limit=5, 
    source_types=["official", "teacher"]
)
```

### 3. Консультации для студентов

```python
# Для консультаций - можно использовать все источники
consultation_context = search.search(
    "вопрос студента", 
    limit=10, 
    source_types=["official", "teacher", "student"]
)
```

## Зависимости

- Neo4j 4.x+ (оптимально 5.11+ для векторных индексов)
- Python 3.8+
- `sentence-transformers`
- `neo4j`
- `numpy`

Дополнительные зависимости для бенчмарка и графиков:
- `pandas`
- `matplotlib` 