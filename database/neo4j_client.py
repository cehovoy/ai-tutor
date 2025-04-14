"""
Клиент для работы с Neo4j - графовой базой данных для хранения понятий и связей курса
"""
from typing import Dict, List, Any, Optional, Union
import logging

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from ai_tutor.config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Клиент для работы с Neo4j
    """
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """
        Инициализация клиента Neo4j
        
        Args:
            uri: URI для подключения к Neo4j
            user: Имя пользователя
            password: Пароль
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.connect()
    
    def connect(self) -> None:
        """
        Подключение к базе данных Neo4j
        """
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Успешное подключение к Neo4j: %s", self.uri)
        except (ServiceUnavailable, AuthError) as e:
            logger.error("Ошибка подключения к Neo4j: %s", str(e))
            raise
    
    def close(self) -> None:
        """
        Закрытие соединения с Neo4j
        """
        if self.driver:
            self.driver.close()
            logger.info("Соединение с Neo4j закрыто")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Выполнение Cypher-запроса
        
        Args:
            query: Cypher-запрос
            params: Параметры запроса
        
        Returns:
            Список результатов запроса
        """
        if not self.driver:
            self.connect()
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error("Ошибка выполнения запроса: %s", str(e))
            raise
    
    def get_course_info(self, course_name: str) -> Dict[str, Any]:
        """
        Получение информации о курсе
        
        Args:
            course_name: Название курса
        
        Returns:
            Информация о курсе
        """
        query = """
        MATCH (c:Course {name: $course_name})
        RETURN c
        """
        results = self.execute_query(query, {"course_name": course_name})
        if not results:
            return {}
        return results[0].get("c", {})
    
    def get_chapters(self, course_name: str) -> List[Dict[str, Any]]:
        """
        Получение списка глав курса
        
        Args:
            course_name: Название курса
        
        Returns:
            Список глав курса
        """
        query = """
        MATCH (c:Course {name: $course_name})-[:PART_OF]-(ch:Chapter)
        RETURN ch.title as title, ch.main_ideas as main_ideas
        ORDER BY ch.title
        """
        return self.execute_query(query, {"course_name": course_name})
    
    def get_concepts_by_chapter(self, chapter_title: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Получение списка понятий из главы
        
        Args:
            chapter_title: Название главы
            limit: Ограничение по количеству понятий
        
        Returns:
            Список понятий
        """
        # Используем связь MENTIONED_IN для получения понятий, связанных с этой главой
        query = """
        MATCH (ch:Chapter {title: $chapter_title})<-[:MENTIONED_IN]-(c:Concept)
        RETURN c.name as name, c.definition as definition, c.example as example, 
               c.questions as questions, c.chapters_mentions as chapters_mentions
        LIMIT $limit
        """
        concepts = self.execute_query(query, {"chapter_title": chapter_title, "limit": limit})
        
        # Обрабатываем результаты, извлекая контекстные определения из chapters_mentions
        processed_concepts = []
        for concept in concepts:
            # Создаем копию понятия для обработки
            processed_concept = concept.copy()
            
            # Проверяем, есть ли у понятия информация о главах
            if concept.get("chapters_mentions"):
                try:
                    import json
                    # Пытаемся распарсить JSON с упоминаниями по главам
                    chapters_mentions = json.loads(concept["chapters_mentions"]) if isinstance(concept["chapters_mentions"], str) else concept["chapters_mentions"]
                    
                    # Если есть определение для этой главы, используем его
                    if chapter_title in chapters_mentions:
                        chapter_info = chapters_mentions[chapter_title]
                        if "definition" in chapter_info:
                            processed_concept["definition"] = chapter_info["definition"]
                        if "example" in chapter_info:
                            processed_concept["example"] = chapter_info["example"]
                        if "questions" in chapter_info:
                            processed_concept["questions"] = chapter_info["questions"]
                except Exception as e:
                    # Логируем ошибку и используем общее определение
                    logging.error(f"Ошибка при обработке chapters_mentions для понятия {concept.get('name')}: {e}")
            
            processed_concepts.append(processed_concept)
        
        return processed_concepts
    
    def get_concept_with_relations(self, concept_name: str) -> Dict[str, Any]:
        """
        Получение понятия со всеми связями
        
        Args:
            concept_name: Название понятия
        
        Returns:
            Понятие со связями
        """
        query = """
        MATCH (c:Concept {name: $concept_name})
        OPTIONAL MATCH (c)-[r]->(related:Concept)
        RETURN c as concept,
               collect({type: type(r), concept: related.name, definition: related.definition}) as relations
        """
        results = self.execute_query(query, {"concept_name": concept_name})
        if not results:
            return {}
        return results[0]
    
    def get_random_concept_by_chapter_and_difficulty(
        self,
        chapter_title: str,
        difficulty: str = "basic",
        excluded_concepts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Получение случайного понятия из главы с учетом сложности
        
        Args:
            chapter_title: Название главы
            difficulty: Уровень сложности (basic/advanced)
            excluded_concepts: Список понятий для исключения
        
        Returns:
            Случайное понятие
        """
        excluded = excluded_concepts or []
        
        # Для продвинутого уровня выбираем понятия, имеющие больше связей
        if difficulty == "advanced":
            query = """
            MATCH (ch:Chapter {title: $chapter_title})<-[:MENTIONED_IN]-(c:Concept)
            WHERE NOT c.name IN $excluded_concepts
            WITH c, size((c)-[]-(:Concept)) as relation_count
            WHERE relation_count > 2
            RETURN c.name as name, c.definition as definition, c.example as example, 
                   c.questions as questions, c.chapters_mentions as chapters_mentions,
                   relation_count
            ORDER BY rand()
            LIMIT 1
            """
        else:
            query = """
            MATCH (ch:Chapter {title: $chapter_title})<-[:MENTIONED_IN]-(c:Concept)
            WHERE NOT c.name IN $excluded_concepts
            WITH c, size((c)-[]-(:Concept)) as relation_count
            WHERE relation_count <= 2
            RETURN c.name as name, c.definition as definition, c.example as example, 
                   c.questions as questions, c.chapters_mentions as chapters_mentions,
                   relation_count
            ORDER BY rand()
            LIMIT 1
            """
        
        result = self.execute_query(
            query, {"chapter_title": chapter_title, "excluded_concepts": excluded}
        )
        
        if not result:
            # Если не нашли понятий с заданным количеством связей, возвращаем любое
            query = """
            MATCH (ch:Chapter {title: $chapter_title})<-[:MENTIONED_IN]-(c:Concept)
            WHERE NOT c.name IN $excluded_concepts
            RETURN c.name as name, c.definition as definition, c.example as example, 
                   c.questions as questions, c.chapters_mentions as chapters_mentions,
                   size((c)-[]-(:Concept)) as relation_count
            ORDER BY rand()
            LIMIT 1
            """
            result = self.execute_query(
                query, {"chapter_title": chapter_title, "excluded_concepts": excluded}
            )
        
        if not result:
            return {}
        
        # Обрабатываем контекстное определение для главы
        concept = result[0]
        processed_concept = concept.copy()
        
        # Проверяем, есть ли у понятия информация о главах
        if concept.get("chapters_mentions"):
            try:
                import json
                # Пытаемся распарсить JSON с упоминаниями по главам
                chapters_mentions = json.loads(concept["chapters_mentions"]) if isinstance(concept["chapters_mentions"], str) else concept["chapters_mentions"]
                
                # Если есть определение для этой главы, используем его
                if chapter_title in chapters_mentions:
                    chapter_info = chapters_mentions[chapter_title]
                    if "definition" in chapter_info:
                        processed_concept["definition"] = chapter_info["definition"]
                    if "example" in chapter_info:
                        processed_concept["example"] = chapter_info["example"]
                    if "questions" in chapter_info:
                        processed_concept["questions"] = chapter_info["questions"]
            except Exception as e:
                # Логируем ошибку и используем общее определение
                logging.error(f"Ошибка при обработке chapters_mentions для понятия {concept.get('name')}: {e}")
        
        return processed_concept
    
    def get_related_concepts(
        self, concept_name: str, chapter_title: Optional[str] = None, 
        relation_type: Optional[str] = None, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Получение связанных понятий
        
        Args:
            concept_name: Название понятия
            chapter_title: Название главы (для контекстных определений)
            relation_type: Тип связи (IS_A, USED_IN и т.д.)
            limit: Ограничение по количеству
            
        Returns:
            Список связанных понятий
        """
        if relation_type:
            query = f"""
            MATCH (c:Concept {{name: $concept_name}})-[r:{relation_type}]->(related:Concept)
            RETURN related.name as name, related.definition as definition, 
                   type(r) as relation_type, related.chapters_mentions as chapters_mentions
            LIMIT $limit
            """
        else:
            query = """
            MATCH (c:Concept {name: $concept_name})-[r]->(related:Concept)
            RETURN related.name as name, related.definition as definition, 
                   type(r) as relation_type, related.chapters_mentions as chapters_mentions
            LIMIT $limit
            """
        
        related_concepts = self.execute_query(query, {"concept_name": concept_name, "limit": limit})
        
        # Если задана глава, обрабатываем контекстные определения
        if chapter_title:
            processed_concepts = []
            for concept in related_concepts:
                processed_concept = concept.copy()
                
                # Проверяем, есть ли у понятия информация о главах
                if concept.get("chapters_mentions"):
                    try:
                        import json
                        # Пытаемся распарсить JSON с упоминаниями по главам
                        chapters_mentions = json.loads(concept["chapters_mentions"]) if isinstance(concept["chapters_mentions"], str) else concept["chapters_mentions"]
                        
                        # Если есть определение для этой главы, используем его
                        if chapter_title in chapters_mentions:
                            chapter_info = chapters_mentions[chapter_title]
                            if "definition" in chapter_info:
                                processed_concept["definition"] = chapter_info["definition"]
                    except Exception as e:
                        # Логируем ошибку и используем общее определение
                        logging.error(f"Ошибка при обработке chapters_mentions для связанного понятия {concept.get('name')}: {e}")
                
                processed_concepts.append(processed_concept)
            
            return processed_concepts
        else:
            return related_concepts
    
    def update_student_progress(
        self, 
        student_id: str, 
        chapter_title: str, 
        concept_name: str, 
        correct: bool, 
        difficulty: str
    ) -> None:
        """
        Обновление прогресса студента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            concept_name: Название понятия
            correct: Правильность ответа
            difficulty: Уровень сложности
        """
        # Создаем узел Student, если его нет
        query = """
        MERGE (s:Student {id: $student_id})
        
        WITH s
        MATCH (ch:Chapter {title: $chapter_title})
        MATCH (c:Concept {name: $concept_name})
        
        MERGE (s)-[r:STUDIED]->(c)
        ON CREATE SET r.attempts = 1, 
                      r.correct = CASE WHEN $correct THEN 1 ELSE 0 END,
                      r.difficulty = $difficulty,
                      r.last_attempt = timestamp()
        ON MATCH SET r.attempts = r.attempts + 1, 
                     r.correct = r.correct + CASE WHEN $correct THEN 1 ELSE 0 END,
                     r.difficulty = $difficulty,
                     r.last_attempt = timestamp()
        
        MERGE (s)-[rch:STUDIED_CHAPTER]->(ch)
        ON CREATE SET rch.attempts = 1,
                      rch.last_attempt = timestamp()
        ON MATCH SET rch.attempts = rch.attempts + 1,
                     rch.last_attempt = timestamp()
        """
        
        self.execute_query(query, {
            "student_id": student_id,
            "chapter_title": chapter_title,
            "concept_name": concept_name,
            "correct": correct,
            "difficulty": difficulty
        })
    
    def get_student_progress(self, student_id: str, chapter_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение прогресса студента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы (опционально)
            
        Returns:
            Прогресс студента
        """
        if chapter_title:
            query = """
            MATCH (s:Student {id: $student_id})-[r:STUDIED]->(c:Concept)
            MATCH (c)-[:MENTIONED_IN]->(ch:Chapter {title: $chapter_title})
            RETURN c.name as concept_name, 
                   r.attempts as attempts, 
                   r.correct as correct,
                   r.difficulty as difficulty,
                   r.last_attempt as last_attempt
            ORDER BY r.last_attempt DESC
            """
            params = {"student_id": student_id, "chapter_title": chapter_title}
        else:
            query = """
            MATCH (s:Student {id: $student_id})-[r:STUDIED]->(c:Concept)
            RETURN c.name as concept_name, 
                   r.attempts as attempts, 
                   r.correct as correct,
                   r.difficulty as difficulty,
                   r.last_attempt as last_attempt
            ORDER BY r.last_attempt DESC
            """
            params = {"student_id": student_id}
        
        return self.execute_query(query, params)
    
    def get_student_chapter_stats(self, student_id: str) -> List[Dict[str, Any]]:
        """
        Получение статистики студента по главам
        
        Args:
            student_id: ID студента
            
        Returns:
            Статистика по главам
        """
        query = """
        MATCH (s:Student {id: $student_id})-[r:STUDIED_CHAPTER]->(ch:Chapter)
        RETURN ch.title as chapter_title, 
               r.attempts as attempts,
               r.last_attempt as last_attempt
        ORDER BY r.attempts DESC
        """
        return self.execute_query(query, {"student_id": student_id})
    
    def suggest_difficulty_level(self, student_id: str, chapter_title: str) -> str:
        """
        Предложение уровня сложности на основе прогресса студента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            
        Returns:
            Рекомендуемый уровень сложности (basic/advanced)
        """
        query = """
        MATCH (s:Student {id: $student_id})-[r:STUDIED]->(c:Concept)
        MATCH (c)-[:MENTIONED_IN]->(ch:Chapter {title: $chapter_title})
        WITH sum(r.correct) as total_correct, sum(r.attempts) as total_attempts
        RETURN total_correct, total_attempts
        """
        
        results = self.execute_query(query, {"student_id": student_id, "chapter_title": chapter_title})
        
        if not results or not results[0]['total_attempts']:
            return "basic"  # По умолчанию базовый уровень
        
        correct = results[0]['total_correct']
        attempts = results[0]['total_attempts']
        
        # Если студент ответил правильно на > 70% вопросов, предлагаем продвинутый уровень
        if correct / attempts > 0.7:
            return "advanced"
        else:
            return "basic"

    def get_concept_by_name(self, concept_name: str, chapter_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Получение понятия по названию
        
        Args:
            concept_name: Название понятия
            chapter_title: Название главы (для контекстных определений)
            
        Returns:
            Понятие с определением
        """
        query = """
        MATCH (c:Concept {name: $concept_name})
        RETURN c.name as name, c.definition as definition, c.example as example, 
               c.questions as questions, c.chapters_mentions as chapters_mentions
        """
        results = self.execute_query(query, {"concept_name": concept_name})
        
        if not results:
            return {}
        
        concept = results[0]
        
        # Если задана глава, обрабатываем контекстное определение
        if chapter_title and concept.get("chapters_mentions"):
            try:
                import json
                # Пытаемся распарсить JSON с упоминаниями по главам
                chapters_mentions = json.loads(concept["chapters_mentions"]) if isinstance(concept["chapters_mentions"], str) else concept["chapters_mentions"]
                
                # Если есть определение для этой главы, используем его
                if chapter_title in chapters_mentions:
                    chapter_info = chapters_mentions[chapter_title]
                    if "definition" in chapter_info:
                        concept["definition"] = chapter_info["definition"]
                    if "example" in chapter_info:
                        concept["example"] = chapter_info["example"]
                    if "questions" in chapter_info:
                        concept["questions"] = chapter_info["questions"]
            except Exception as e:
                # Логируем ошибку и используем общее определение
                logging.error(f"Ошибка при обработке chapters_mentions для понятия {concept_name}: {e}")
        
        return concept

    def save_student(self, student):
        """
        Сохранение информации о студенте в базу данных
        
        Args:
            student: Объект студента
        """
        query = """
        MERGE (s:Student {telegram_id: $telegram_id})
        ON CREATE SET s.username = $username,
                      s.first_name = $first_name,
                      s.last_name = $last_name,
                      s.created_at = timestamp(),
                      s.tasks_completed = 0,
                      s.correct_answers = 0
        ON MATCH SET s.username = $username,
                     s.first_name = $first_name,
                     s.last_name = $last_name,
                     s.updated_at = timestamp()
        RETURN s
        """
        params = {
            "telegram_id": student.telegram_id,
            "username": student.username,
            "first_name": student.first_name,
            "last_name": student.last_name
        }
        return self.execute_query(query, params)
    
    def get_student_by_telegram_id(self, telegram_id):
        """
        Получение информации о студенте по его Telegram ID
        
        Args:
            telegram_id: Telegram ID студента
        
        Returns:
            Объект студента или None, если студент не найден
        """
        query = """
        MATCH (s:Student {telegram_id: $telegram_id})
        RETURN s.telegram_id as telegram_id,
               s.username as username,
               s.first_name as first_name,
               s.last_name as last_name,
               s.tasks_completed as tasks_completed,
               s.correct_answers as correct_answers
        """
        results = self.execute_query(query, {"telegram_id": telegram_id})
        
        if not results:
            return None
            
        # Создаем объект студента из данных
        from ai_tutor.database.models import Student
        student_data = results[0]
        
        student = Student(
            telegram_id=student_data.get("telegram_id"),
            username=student_data.get("username"),
            first_name=student_data.get("first_name"),
            last_name=student_data.get("last_name")
        )
        
        # Добавляем дополнительные поля
        student.tasks_completed = student_data.get("tasks_completed", 0)
        student.correct_answers = student_data.get("correct_answers", 0)
        
        return student
    
    def update_student(self, student):
        """
        Обновление информации о студенте
        
        Args:
            student: Объект студента
        """
        query = """
        MATCH (s:Student {telegram_id: $telegram_id})
        SET s.tasks_completed = $tasks_completed,
            s.correct_answers = $correct_answers,
            s.updated_at = timestamp()
        RETURN s
        """
        params = {
            "telegram_id": student.telegram_id,
            "tasks_completed": student.tasks_completed,
            "correct_answers": student.correct_answers
        }
        return self.execute_query(query, params)
    
    def save_student_answer(self, student_id, task, answer, is_correct, feedback):
        """
        Сохранение ответа студента
        
        Args:
            student_id: ID студента
            task: Задача
            answer: Ответ студента
            is_correct: Флаг правильности ответа
            feedback: Обратная связь
        """
        query = """
        MATCH (s:Student {telegram_id: $student_id})
        CREATE (a:Answer {
            concept_name: $concept_name,
            task_type: $task_type,
            difficulty: $difficulty,
            student_answer: $answer,
            is_correct: $is_correct,
            feedback: $feedback,
            created_at: timestamp()
        })
        CREATE (s)-[:GAVE]->(a)
        RETURN a
        """
        params = {
            "student_id": student_id,
            "concept_name": task.get("concept_name", ""),
            "task_type": task.get("task_type", ""),
            "difficulty": task.get("difficulty", ""),
            "answer": answer,
            "is_correct": is_correct,
            "feedback": feedback
        }
        return self.execute_query(query, params)
    
    def save_assistant_interaction(self, student_id, question, answer, chapter_title=None):
        """
        Сохранение взаимодействия с помощником
        
        Args:
            student_id: ID студента
            question: Вопрос студента
            answer: Ответ помощника
            chapter_title: Название главы (опционально)
        """
        query = """
        MATCH (s:Student {telegram_id: $student_id})
        CREATE (i:Interaction {
            question: $question,
            answer: $answer,
            chapter_title: $chapter_title,
            created_at: timestamp()
        })
        CREATE (s)-[:ASKED]->(i)
        RETURN i
        """
        params = {
            "student_id": student_id,
            "question": question,
            "answer": answer,
            "chapter_title": chapter_title
        }
        return self.execute_query(query, params)
    
    def search_concepts_by_keywords(self, keywords, chapter_title=None, limit=10):
        """
        Поиск понятий по ключевым словам
        
        Args:
            keywords: Список ключевых слов
            chapter_title: Название главы (опционально)
            limit: Ограничение по количеству результатов
            
        Returns:
            Список понятий, соответствующих ключевым словам
        """
        # Составляем условия для поиска по каждому ключевому слову
        keyword_conditions = []
        for i, keyword in enumerate(keywords):
            # Ищем ключевое слово в названии или определении понятия
            keyword_conditions.append(f"c.name =~ '(?i).*{keyword}.*' OR c.definition =~ '(?i).*{keyword}.*'")
        
        # Объединяем условия через ИЛИ
        keyword_query = " OR ".join(keyword_conditions)
        
        # Базовый запрос
        if chapter_title:
            # Если указана глава, ищем только понятия, связанные с этой главой
            query = f"""
            MATCH (ch:Chapter {{title: $chapter_title}})<-[:MENTIONED_IN]-(c:Concept)
            WHERE {keyword_query}
            RETURN c.name as name, c.definition as definition, c.example as example, 
                  c.questions as questions, c.chapters_mentions as chapters_mentions
            LIMIT $limit
            """
        else:
            # Иначе ищем по всей базе знаний
            query = f"""
            MATCH (c:Concept)
            WHERE {keyword_query}
            RETURN c.name as name, c.definition as definition, c.example as example, 
                  c.questions as questions, c.chapters_mentions as chapters_mentions
            LIMIT $limit
            """
        
        params = {
            "chapter_title": chapter_title,
            "limit": limit
        }
        
        # Выполняем запрос
        results = self.execute_query(query, params)
        
        # Обрабатываем результаты, извлекая контекстные определения
        processed_results = []
        for concept in results:
            # Создаем копию понятия для обработки
            processed_concept = concept.copy()
            
            # Проверяем, есть ли у понятия информация о главах и указана ли глава
            if chapter_title and concept.get("chapters_mentions"):
                try:
                    import json
                    # Пытаемся распарсить JSON с упоминаниями по главам
                    chapters_mentions = json.loads(concept["chapters_mentions"]) if isinstance(concept["chapters_mentions"], str) else concept["chapters_mentions"]
                    
                    # Если есть определение для этой главы, используем его
                    if chapter_title in chapters_mentions:
                        chapter_info = chapters_mentions[chapter_title]
                        if "definition" in chapter_info:
                            processed_concept["definition"] = chapter_info["definition"]
                        if "example" in chapter_info:
                            processed_concept["example"] = chapter_info["example"]
                        if "questions" in chapter_info:
                            processed_concept["questions"] = chapter_info["questions"]
                except Exception as e:
                    # Логируем ошибку и используем общее определение
                    logging.error(f"Ошибка при обработке chapters_mentions для понятия {concept.get('name')}: {e}")
            
            processed_results.append(processed_concept)
        
        return processed_results
    
    def get_concept_connections(self, concept_name, limit=10):
        """
        Получение связей понятия с другими понятиями
        
        Args:
            concept_name: Название понятия
            limit: Ограничение по количеству результатов
            
        Returns:
            Список связей понятия
        """
        query = """
        MATCH (c:Concept {name: $concept_name})-[r]->(related:Concept)
        RETURN type(r) as type, related.name as concept
        LIMIT $limit
        """
        
        params = {
            "concept_name": concept_name,
            "limit": limit
        }
        
        return self.execute_query(query, params)
    
    def get_chapter_info(self, chapter_title):
        """
        Получение информации о главе
        
        Args:
            chapter_title: Название главы
            
        Returns:
            Информация о главе
        """
        query = """
        MATCH (ch:Chapter {title: $chapter_title})
        RETURN ch.title as title, ch.main_ideas as main_ideas
        """
        
        params = {
            "chapter_title": chapter_title
        }
        
        results = self.execute_query(query, params)
        
        if not results:
            return None
            
        return results[0]
