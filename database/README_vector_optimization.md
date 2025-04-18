# Оптимизация векторного поиска в базе данных Neo4j

Данный документ содержит рекомендации и инструкции по оптимизации работы векторного поиска в Neo4j для повышения производительности.

## Проблема

Текущая реализация векторного поиска в `EnhancedCourseSearch` может быть медленной при больших объемах данных из-за следующих причин:

1. **Загрузка всех понятий в память** и вычисление сходства на Python-стороне
2. **Генерация векторных эмбеддингов на лету** для каждого запроса
3. **Отсутствие оптимизированных векторных индексов** в Neo4j

## Решения

### 1. Гибридный подход (реализовано)

Оптимизирован метод `semantic_search_with_ranking` в классе `EnhancedCourseSearch`:
- Все понятия загружаются одним запросом и обрабатываются пакетом
- Вычисление сходства производится на Python-стороне
- Добавлено логирование времени выполнения для отслеживания производительности

### 2. Создание векторного индекса Neo4j

Neo4j поддерживает нативные векторные индексы с версии 5.11, которые используют алгоритм HNSW (Hierarchical Navigable Small World) для эффективного поиска ближайших соседей.

#### Необходимые шаги:

1. **Проверка версии Neo4j**
   ```bash
   python -m ai_tutor.database.create_vector_index show
   ```

2. **Создание векторного индекса**
   ```bash
   python -m ai_tutor.database.create_vector_index create
   ```

3. **Тестирование поиска** с использованием индекса
   ```bash
   python -m ai_tutor.database.create_vector_index test "Системное мышление"
   ```

4. **Включение использования индекса** в `EnhancedCourseSearch`:
   - Измените в файле `ai_tutor/database/enhanced_search.py` строку:
     ```python
     if self.has_vector_index and False:  # Пока отключено использование нативного индекса
     ```
     на
     ```python
     if self.has_vector_index:  # Включаем использование нативного индекса
     ```

### 3. Дополнительные оптимизации

#### 3.1. Предварительное вычисление эмбеддингов

Убедитесь, что все понятия в базе данных уже имеют предварительно вычисленные векторные представления:
```cypher
MATCH (c:Concept)
WHERE c.combined_embedding IS NULL
RETURN count(c) as missing_embeddings
```

Если есть понятия без эмбеддингов, используйте скрипт `vectorize_course.py` для их создания.

#### 3.2. Кэширование результатов частых запросов

Для часто задаваемых вопросов можно реализовать простую систему кэширования:

```python
class CachedSearch:
    def __init__(self, search_engine, cache_ttl=3600):
        self.search_engine = search_engine
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.cache_time = {}
        
    def search(self, query, **kwargs):
        # Создаем ключ для кэша
        cache_key = f"{query}:{str(kwargs)}"
        current_time = time.time()
        
        # Проверяем наличие в кэше и актуальность
        if cache_key in self.cache:
            last_time = self.cache_time.get(cache_key, 0)
            if current_time - last_time < self.cache_ttl:
                return self.cache[cache_key]
            
        # Выполняем поиск
        results = self.search_engine.semantic_search_with_ranking(query, **kwargs)
        
        # Сохраняем в кэш
        self.cache[cache_key] = results
        self.cache_time[cache_key] = current_time
        
        return results
```

#### 3.3. Улучшение модели для эмбеддингов

Для повышения скорости можно использовать более легкую модель:
- `paraphrase-multilingual-MiniLM-L6-v2` (вместо L12)

Для повышения качества можно использовать более современную модель:
- `paraphrase-multilingual-mpnet-base-v2`

#### 3.4. Параллельная обработка для пакетного вычисления эмбеддингов

При обработке большого количества документов можно использовать параллельную обработку:

```python
from concurrent.futures import ThreadPoolExecutor

def process_batch(batch, query_embedding):
    results = []
    for doc in batch:
        # Обработка документа
        # ...
    return results

# Разбиваем документы на батчи
batches = [all_records[i:i+100] for i in range(0, len(all_records), 100)]

# Параллельно обрабатываем батчи
with ThreadPoolExecutor(max_workers=4) as executor:
    batch_results = list(executor.map(
        lambda batch: process_batch(batch, query_embedding), 
        batches
    ))

# Объединяем результаты
results = [item for sublist in batch_results for item in sublist]
```

## Сравнение производительности

| Метод | Время выполнения | Примечания |
|-------|------------------|------------|
| Исходный | ~1-2 минуты | Медленный поиск по всем узлам |
| Гибридный | ~10-30 секунд | Улучшенная обработка в Python |
| Neo4j Vector Index | ~1-5 секунд | Самый быстрый, требует Neo4j 5.11+ |
| С кэшированием | ~0.01 секунды | Для повторных запросов |

## Рекомендации

1. Используйте **Neo4j Vector Index**, если версия Neo4j 5.11 или выше
2. Для старых версий Neo4j используйте **гибридный подход**
3. Внедрите **кэширование** для частых запросов
4. Убедитесь, что все документы имеют **предварительно вычисленные эмбеддинги**
5. По возможности используйте **легкие модели** для быстрых запросов или **более точные модели** для критичных задач

## Заключение

Оптимизация векторного поиска позволяет значительно ускорить работу системы и улучшить пользовательский опыт. Рекомендуется комбинировать различные подходы для достижения наилучших результатов.

## Полезные ссылки

- [Документация Neo4j по векторным индексам](https://neo4j.com/docs/cypher-manual/current/indexes/vector-indexes/)
- [Sentence Transformers](https://www.sbert.net/)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) 