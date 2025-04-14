"""
Запросы к базе данных Neo4j для проекта ИИ-репетитор
"""
from typing import Dict, List, Any, Optional, Union
import logging

from ai_tutor.database.models import Concept, Task, Student, StudentAnswer

logger = logging.getLogger(__name__)

# Запросы для работы с понятиями (Concept)
GET_CONCEPT_BY_NAME = """
MATCH (c:Concept {name: $name})
RETURN c
"""

GET_CONCEPTS_BY_CHAPTER = """
MATCH (c:Concept)
WHERE c.chapter = $chapter
RETURN c
"""

GET_RELATED_CONCEPTS = """
MATCH (c:Concept {name: $concept_name})-[r]->(related:Concept)
RETURN related, type(r) as relation_type
"""

CREATE_CONCEPT = """
CREATE (c:Concept $properties)
RETURN c
"""

UPDATE_CONCEPT = """
MATCH (c:Concept {name: $name})
SET c += $properties
RETURN c
"""

# Запросы для работы с задачами (Task)
CREATE_TASK = """
CREATE (t:Task $properties)
RETURN t
"""

GET_TASK_BY_ID = """
MATCH (t:Task)
WHERE ID(t) = $task_id
RETURN t
"""

GET_TASKS_BY_CONCEPT = """
MATCH (t:Task)
WHERE t.concept_name = $concept_name
RETURN t
"""

# Запросы для работы со студентами (Student)
GET_STUDENT_BY_TELEGRAM_ID = """
MATCH (s:Student {telegram_id: $telegram_id})
RETURN s
"""

CREATE_STUDENT = """
CREATE (s:Student $properties)
RETURN s
"""

UPDATE_STUDENT = """
MATCH (s:Student {telegram_id: $telegram_id})
SET s += $properties
RETURN s
"""

# Запросы для работы с ответами студентов (StudentAnswer)
CREATE_STUDENT_ANSWER = """
MATCH (s:Student {telegram_id: $telegram_id})
MATCH (t:Task)
WHERE ID(t) = $task_id
CREATE (s)-[r:ANSWERED {properties: $properties}]->(t)
RETURN r
"""

GET_STUDENT_ANSWERS = """
MATCH (s:Student {telegram_id: $telegram_id})-[r:ANSWERED]->(t:Task)
RETURN r.properties as answer, t
"""

# Запросы для связей между понятиями
CREATE_CONCEPT_RELATION = """
MATCH (c1:Concept {name: $concept1_name})
MATCH (c2:Concept {name: $concept2_name})
CREATE (c1)-[r:$relation_type $properties]->(c2)
RETURN r
"""

# Функции для выполнения запросов

async def get_concept_by_name(client, concept_name: str) -> Optional[Concept]:
    """
    Получение понятия по имени
    
    Args:
        client: Neo4j клиент
        concept_name: Имя понятия
        
    Returns:
        Объект Concept или None, если понятие не найдено
    """
    try:
        result = await client.execute_query(
            GET_CONCEPT_BY_NAME,
            {"name": concept_name}
        )
        
        if result and result[0]:
            concept_data = result[0]["c"]
            return Concept.from_dict(concept_data)
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении понятия по имени: {e}")
        return None

async def get_concepts_by_chapter(client, chapter: str) -> List[Concept]:
    """
    Получение всех понятий по главе
    
    Args:
        client: Neo4j клиент
        chapter: Название главы
        
    Returns:
        Список объектов Concept
    """
    try:
        result = await client.execute_query(
            GET_CONCEPTS_BY_CHAPTER,
            {"chapter": chapter}
        )
        
        concepts = []
        if result:
            for record in result:
                concept_data = record["c"]
                concepts.append(Concept.from_dict(concept_data))
        return concepts
    except Exception as e:
        logger.error(f"Ошибка при получении понятий по главе: {e}")
        return []

async def get_related_concepts(client, concept_name: str) -> List[Dict[str, Any]]:
    """
    Получение связанных понятий
    
    Args:
        client: Neo4j клиент
        concept_name: Имя понятия
        
    Returns:
        Список связанных понятий с типом связи
    """
    try:
        result = await client.execute_query(
            GET_RELATED_CONCEPTS,
            {"concept_name": concept_name}
        )
        
        related_concepts = []
        if result:
            for record in result:
                concept = Concept.from_dict(record["related"])
                relation_type = record["relation_type"]
                related_concepts.append({
                    "concept": concept,
                    "relation_type": relation_type
                })
        return related_concepts
    except Exception as e:
        logger.error(f"Ошибка при получении связанных понятий: {e}")
        return []

async def create_concept(client, concept: Concept) -> Optional[Concept]:
    """
    Создание нового понятия
    
    Args:
        client: Neo4j клиент
        concept: Объект Concept
        
    Returns:
        Созданный объект Concept или None, если возникла ошибка
    """
    try:
        result = await client.execute_query(
            CREATE_CONCEPT,
            {"properties": concept.to_dict()}
        )
        
        if result and result[0]:
            concept_data = result[0]["c"]
            return Concept.from_dict(concept_data)
        return None
    except Exception as e:
        logger.error(f"Ошибка при создании понятия: {e}")
        return None

async def get_student_by_telegram_id(client, telegram_id: int) -> Optional[Student]:
    """
    Получение студента по Telegram ID
    
    Args:
        client: Neo4j клиент
        telegram_id: ID пользователя в Telegram
        
    Returns:
        Объект Student или None, если студент не найден
    """
    try:
        result = await client.execute_query(
            GET_STUDENT_BY_TELEGRAM_ID,
            {"telegram_id": telegram_id}
        )
        
        if result and result[0]:
            student_data = result[0]["s"]
            return Student.from_dict(student_data)
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении студента по Telegram ID: {e}")
        return None

async def create_student(client, student: Student) -> Optional[Student]:
    """
    Создание нового студента
    
    Args:
        client: Neo4j клиент
        student: Объект Student
        
    Returns:
        Созданный объект Student или None, если возникла ошибка
    """
    try:
        result = await client.execute_query(
            CREATE_STUDENT,
            {"properties": student.to_dict()}
        )
        
        if result and result[0]:
            student_data = result[0]["s"]
            return Student.from_dict(student_data)
        return None
    except Exception as e:
        logger.error(f"Ошибка при создании студента: {e}")
        return None

async def update_student(client, telegram_id: int, properties: Dict[str, Any]) -> Optional[Student]:
    """
    Обновление данных студента
    
    Args:
        client: Neo4j клиент
        telegram_id: ID пользователя в Telegram
        properties: Словарь с обновляемыми свойствами
        
    Returns:
        Обновленный объект Student или None, если возникла ошибка
    """
    try:
        result = await client.execute_query(
            UPDATE_STUDENT,
            {
                "telegram_id": telegram_id,
                "properties": properties
            }
        )
        
        if result and result[0]:
            student_data = result[0]["s"]
            return Student.from_dict(student_data)
        return None
    except Exception as e:
        logger.error(f"Ошибка при обновлении студента: {e}")
        return None

async def create_task(client, task: Task) -> Optional[Task]:
    """
    Создание новой задачи
    
    Args:
        client: Neo4j клиент
        task: Объект Task
        
    Returns:
        Созданный объект Task или None, если возникла ошибка
    """
    try:
        result = await client.execute_query(
            CREATE_TASK,
            {"properties": task.to_dict()}
        )
        
        if result and result[0]:
            task_data = result[0]["t"]
            return Task.from_dict(task_data)
        return None
    except Exception as e:
        logger.error(f"Ошибка при создании задачи: {e}")
        return None
