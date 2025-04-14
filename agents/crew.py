"""
Команда агентов для ИИ-репетитора (упрощенная версия без CrewAI)
"""
from typing import Dict, List, Any, Optional
import logging
import json
import random
import traceback
import asyncio
from neo4j.exceptions import ServiceUnavailable

from ai_tutor.api.openrouter import OpenRouterClient
from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.config.settings import COURSE_NAME

logger = logging.getLogger(__name__)


class TutorCrew:
    """
    Упрощенная команда агентов ИИ-репетитора без CrewAI
    """
    
    def __init__(self, openrouter_client: OpenRouterClient, verbose: bool = False):
        """
        Инициализация команды
        
        Args:
            openrouter_client: Клиент для работы с OpenRouter API
            verbose: Режим подробного вывода
        """
        self.verbose = verbose
        self.openrouter_client = openrouter_client
        self.neo4j_client = Neo4jClient()
        # Словарь для отслеживания последовательных правильных ответов студентов
        self.correct_answers_count = {}  # Формат: {student_id: count}
    
    async def async_full_tutor_process(self, student_id: str, chapter_title: str, 
                         task_type: str, difficulty: str) -> Dict[str, Any]:
        """
        Асинхронная версия полного процесса работы репетитора: генерация задачи с метаданными
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            task_type: Тип задачи
            difficulty: Уровень сложности
            
        Returns:
            Словарь с задачей и метаданными
        """
        try:
            logger.info(f"Начинаем генерацию задачи: глава={chapter_title}, тип={task_type}, сложность={difficulty}")
            
            # Убедимся, что подключение к Neo4j активно
            try:
                self.neo4j_client.connect()
                logger.info("Соединение с Neo4j проверено")
            except Exception as db_error:
                logger.error(f"Ошибка при подключении к Neo4j: {str(db_error)}")
                raise ValueError(f"Не удалось подключиться к базе данных: {str(db_error)}")
            
            # Пытаемся получить понятия - синхронный метод, выполняем в другом потоке
            logger.info("Получаем понятия из базы данных Neo4j")
            try:
                loop = asyncio.get_event_loop()
                concepts = await loop.run_in_executor(
                    None, 
                    lambda: self.neo4j_client.get_concepts_by_chapter(chapter_title)
                )
                logger.info(f"Понятия успешно получены: {len(concepts) if concepts else 0} понятий")
            except Exception as concept_error:
                logger.error(f"Ошибка при получении понятий: {str(concept_error)}")
                raise ValueError(f"Не удалось получить понятия для главы {chapter_title}: {str(concept_error)}")
            
            if not concepts:
                logger.warning(f"Понятия для главы {chapter_title} не найдены. Используем заглушку.")
                task = self._generate_fallback_task(chapter_title, task_type, difficulty)
                logger.info("Заглушка задачи создана успешно")
            else:
                logger.info(f"Получено {len(concepts)} понятий для главы {chapter_title}")
                
                # Выбираем случайное понятие из списка
                concept = random.choice(concepts)
                logger.info(f"Выбрано случайное понятие: {concept.get('name', 'Безымянное понятие')}")
                
                # Получаем связанные понятия - синхронный метод, выполняем в другом потоке
                logger.info(f"Получаем понятия, связанные с {concept.get('name', 'Безымянное понятие')}")
                try:
                    related_concepts = await loop.run_in_executor(
                        None,
                        lambda: self.neo4j_client.get_related_concepts(concept.get('name', ''), chapter_title)
                    )
                    logger.info(f"Получено {len(related_concepts)} связанных понятий")
                except Exception as related_error:
                    logger.error(f"Ошибка при получении связанных понятий: {str(related_error)}")
                    # В случае ошибки просто продолжаем с пустым списком связанных понятий
                    related_concepts = []
                    logger.info("Продолжаем без связанных понятий")
                
                # Пробуем сгенерировать задачу через OpenRouter API (Grok)
                try:
                    logger.info("Пытаемся вызвать OpenRouter API для генерации задачи через Grok")
                    
                    # Проверяем API ключ OpenRouter
                    if not self.openrouter_client.api_key:
                        logger.error("API ключ OpenRouter не задан")
                        raise ValueError("Отсутствует API ключ для доступа к OpenRouter")
                    
                    # Максимальное время ожидания - 45 секунд
                    timeout_seconds = 45
                    
                    # Создаем и запускаем задачу с таймаутом
                    try:
                        # Преобразуем объекты в словари для API - используем напрямую, т.к. это уже словари
                        concept_dict = concept
                        related_concepts_dicts = [rc.get("concept", {}) for rc in related_concepts]
                        
                        task = await asyncio.wait_for(
                            self.openrouter_client.generate_task(
                                concept_dict,
                                related_concepts_dicts,
                                task_type,
                                difficulty
                            ),
                            timeout=timeout_seconds
                        )
                        logger.info("Задача успешно сгенерирована через OpenRouter API")
                    except asyncio.TimeoutError:
                        logger.warning(f"Тайм-аут при обращении к OpenRouter API после {timeout_seconds} секунд")
                        # Используем заглушку
                        task = self._generate_fallback_task(chapter_title, task_type, difficulty)
                        logger.info("Заглушка задачи создана из-за тайм-аута")
                except Exception as api_error:
                    logger.exception(f"Ошибка при вызове OpenRouter API: {str(api_error)}")
                    # Используем заглушку в случае ошибки
                    task = self._generate_fallback_task(chapter_title, task_type, difficulty)
                    logger.info("Заглушка задачи создана из-за ошибки API")
            
            # Проверяем структуру задачи перед возвратом
            if not isinstance(task, dict):
                logger.error(f"Некорректный формат задачи: {type(task)}")
                task = self._generate_fallback_task(chapter_title, task_type, difficulty)
                logger.info("Заглушка задачи создана из-за некорректного формата")
            
            # Формируем полный результат
            result = {
                "task": task,
                "metadata": {
                    "student_id": student_id,
                    "chapter_title": chapter_title,
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "course_name": COURSE_NAME
                }
            }
            
            logger.info("Задача успешно создана и готова к отправке")
            return result
        except Exception as e:
            logger.error(f"Ошибка в полном процессе репетитора: {str(e)}\n{traceback.format_exc()}")
            # Возвращаем заглушку с информацией об ошибке
            return {
                "task": {
                    "error": f"Ошибка при генерации задачи: {str(e)}",
                    "concept_name": "Ошибка",
                    "task_type": task_type,
                    "difficulty": difficulty
                },
                "metadata": {
                    "student_id": student_id,
                    "chapter_title": chapter_title,
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "course_name": COURSE_NAME
                }
            }
    
    def full_tutor_process(self, student_id: str, chapter_title: str, 
                         task_type: str, difficulty: str) -> Dict[str, Any]:
        """
        Полный процесс работы репетитора: генерация задачи с метаданными
        (Синхронная обертка для async_full_tutor_process)
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            task_type: Тип задачи
            difficulty: Уровень сложности
            
        Returns:
            Словарь с задачей и метаданными
        """
        try:
            # Создаем новый цикл событий для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Вызываем асинхронный метод
                return loop.run_until_complete(
                    self.async_full_tutor_process(student_id, chapter_title, task_type, difficulty)
                )
            finally:
                try:
                    loop.close()
                except Exception as loop_error:
                    logger.warning(f"Ошибка при закрытии цикла: {str(loop_error)}")
        except Exception as e:
            logger.error(f"Ошибка в синхронной обертке full_tutor_process: {str(e)}\n{traceback.format_exc()}")
            # Возвращаем заглушку с информацией об ошибке
            return {
                "task": {
                    "error": f"Ошибка при генерации задачи: {str(e)}",
                    "concept_name": "Ошибка",
                    "task_type": task_type,
                    "difficulty": difficulty
                },
                "metadata": {
                    "student_id": student_id,
                    "chapter_title": chapter_title,
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "course_name": COURSE_NAME
                }
            }
    
    async def async_generate_task(self, chapter_title: str, task_type: str, difficulty: str, 
                    excluded_concepts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Генерация задачи
        
        Args:
            chapter_title: Название главы
            task_type: Тип задачи
            difficulty: Уровень сложности
            excluded_concepts: Список понятий для исключения (опционально)
            
        Returns:
            Сгенерированная задача
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Получаем все понятия из главы - синхронный метод, запускаем в отдельном потоке
            concepts = await loop.run_in_executor(
                None,
                lambda: self.neo4j_client.get_concepts_by_chapter(chapter_title)
            )
            
            if not concepts:
                # Если понятий не найдено, возвращаем заглушку
                logger.warning(f"Понятия для главы {chapter_title} не найдены. Используем заглушку.")
                return self._generate_fallback_task(chapter_title, task_type, difficulty)
            
            # Выбираем случайное понятие из списка
            concept = random.choice(concepts)
            
            # Получаем связанные понятия - синхронный метод, запускаем в отдельном потоке
            related_concepts = await loop.run_in_executor(
                None,
                lambda: self.neo4j_client.get_related_concepts(concept.get('name', ''), chapter_title)
            )
            
            # Генерируем задачу
            task = await self.openrouter_client.generate_task(
                concept,
                [rc.get("concept", {}) for rc in related_concepts],
                task_type,
                difficulty
            )
            
            return task
        
        except Exception as e:
            logger.error(f"Ошибка при генерации задачи: {str(e)}\n{traceback.format_exc()}")
            return self._generate_fallback_task(chapter_title, task_type, difficulty)
    
    def generate_task(
        self, 
        chapter_title: str,
        task_type: str,
        difficulty: str = "basic",
        excluded_concepts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Синхронная обертка для async_generate_task."""
        try:
            # Создаем новый цикл событий для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Вызываем асинхронный метод
                return loop.run_until_complete(
                    self.async_generate_task(chapter_title, task_type, difficulty, excluded_concepts)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Ошибка в синхронной обертке generate_task: {str(e)}\n{traceback.format_exc()}")
            return self._generate_fallback_task(chapter_title, task_type, difficulty)
    
    def _generate_fallback_task(self, chapter_title: str, task_type: str, difficulty: str) -> Dict[str, Any]:
        """
        Генерация заглушки задачи при ошибке
        
        Args:
            chapter_title: Название главы
            task_type: Тип задачи
            difficulty: Уровень сложности
            
        Returns:
            Заглушка задачи
        """
        concept_name = "Ключевое понятие главы"
        
        if task_type == "template":
            return {
                "question": f"Какое из следующих определений лучше всего описывает ключевое понятие из главы '{chapter_title}'?",
                "options": [
                    {"label": "A", "text": f"Ключевое понятие из главы '{chapter_title}' - это система взаимосвязанных элементов, работающих для достижения общей цели", "is_correct": True, 
                    "explanation": "Это обобщенное определение системного подхода, который является фундаментальным для понимания материала главы."},
                    {"label": "B", "text": "Это изолированный элемент, который не взаимодействует с другими компонентами", "is_correct": False, 
                    "explanation": "Это противоречит системному подходу, который подчеркивает взаимосвязь элементов."},
                    {"label": "C", "text": "Это только теоретическая концепция, не имеющая практического применения", "is_correct": False, 
                    "explanation": "Понятия системного менеджмента имеют важное практическое применение."},
                    {"label": "D", "text": "Это простая совокупность элементов без определенной структуры", "is_correct": False, 
                    "explanation": "Системный подход подразумевает наличие структуры и организации элементов."}
                ],
                "concept_name": concept_name,
                "task_type": task_type,
                "difficulty": difficulty
            }
        else:  # creative
            return {
                "question": f"Объясните своими словами, как ключевые понятия из главы '{chapter_title}' формируют систему знаний и как они могут быть применены в реальной жизни или профессиональной деятельности.",
                "criteria": [
                    "Точность определения понятий",
                    "Понимание взаимосвязей между понятиями",
                    "Глубина анализа практического применения"
                ],
                "example_answer": f"В главе '{chapter_title}' представлены важные системные понятия, которые взаимосвязаны следующим образом... Эти понятия могут быть применены в таких ситуациях, как...",
                "hints": [
                    "Сначала определите ключевые понятия из главы",
                    "Подумайте о том, как эти понятия связаны между собой",
                    "Приведите конкретные примеры, где эти понятия могут быть применены"
                ],
                "concept_name": concept_name,
                "task_type": task_type,
                "difficulty": difficulty
            }
    
    async def async_check_answer(
        self,
        student_id: str,
        chapter_title: str,
        task: Dict[str, Any],
        student_answer: str
    ) -> Dict[str, Any]:
        """
        Асинхронная версия проверки ответа
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            task: Задача
            student_answer: Ответ студента
            
        Returns:
            Результат проверки
        """
        try:
            logger.info(f"Начинаем проверку ответа для задачи по теме: {task.get('concept_name', '')}")
            
            # Для шаблонных задач реализуем простую проверку
            if task["task_type"] == "template":
                logger.info("Проверка ответа для шаблонной задачи")
                # Проверяем, совпадает ли ответ с правильным вариантом
                student_answer = student_answer.strip().upper()
                for option in task.get("options", []):
                    if option.get("is_correct", False) and option.get("label", "") == student_answer:
                        is_correct = True
                        explanation = option.get("explanation", "Это правильный ответ.")
                        logger.info(f"Ответ правильный: {student_answer}")
                        
                        # Увеличиваем счетчик правильных ответов
                        self.correct_answers_count[student_id] = self.correct_answers_count.get(student_id, 0) + 1
                        
                        # Проверяем, достиг ли студент 5 правильных ответов подряд
                        next_steps = []
                        if self.correct_answers_count.get(student_id, 0) >= 5:
                            # Предлагаем изменить параметры обучения
                            next_steps = [
                                {"action": "change_chapter", "text": "Перейти к следующей главе"},
                                {"action": "increase_difficulty", "text": "Повысить сложность задач"},
                                {"action": "change_task_type", "text": "Попробовать творческую задачу"}
                            ]
                            # Сбрасываем счетчик
                            self.correct_answers_count[student_id] = 0
                            
                        return {
                            "is_correct": is_correct,
                            "explanation": explanation,
                            "recommendations": ["Отлично! Продолжайте изучение."],
                            "next_steps": next_steps
                        }
                        break
                else:
                    is_correct = False
                    explanation = "Выбран неверный вариант ответа."
                    logger.info(f"Ответ неверный: {student_answer}")
                    
                    # Сбрасываем счетчик правильных ответов
                    self.correct_answers_count[student_id] = 0
                    
                    # Предлагаем опции для неправильного ответа
                    next_steps = [
                        {"action": "discuss", "text": "Обсудить задачу"},
                        {"action": "try_again", "text": "Попробовать ещё раз"},
                        {"action": "skip", "text": "Пропустить задачу"}
                    ]
                    
                    return {
                        "is_correct": is_correct,
                        "explanation": explanation,
                        "recommendations": ["Изучите материалы по данной теме."],
                        "next_steps": next_steps
                    }
            
            # Для творческих задач или других типов задач используем API
            logger.info("Проверка творческого ответа через OpenRouter API")
            
            # Получаем информацию о понятии
            concept_name = task.get("concept_name", "")
            logger.info(f"Получаем информацию о понятии: {concept_name}")
            
            try:
                # Используем синхронный метод get_concept_by_name в отдельном потоке
                loop = asyncio.get_event_loop()
                if hasattr(self.neo4j_client, 'get_concept_by_name'):
                    concept = await loop.run_in_executor(
                        None, 
                        lambda: self.neo4j_client.get_concept_by_name(concept_name, chapter_title)
                    )
                else:
                    # Fallback, если метод отсутствует
                    logger.warning(f"Метод get_concept_by_name не найден, используем заглушку")
                    concept = None
                
                if not concept:
                    logger.warning(f"Понятие {concept_name} не найдено")
                    concept_dict = {"name": concept_name, "definition": "Определение отсутствует"}
                else:
                    concept_dict = concept if isinstance(concept, dict) else concept.__dict__
                    logger.info(f"Понятие {concept_name} получено успешно")
                
                # Максимальное время ожидания - 45 секунд
                timeout_seconds = 45
                
                try:
                    # Проверяем ответ через OpenRouter API (Grok) с таймаутом
                    check_result = await asyncio.wait_for(
                        self.openrouter_client.check_answer(
                            task,
                            student_answer,
                            concept_dict
                        ),
                        timeout=timeout_seconds
                    )
                    logger.info("Ответ успешно проверен через OpenRouter API")
                    
                    # Убедимся, что результат содержит правильные ключи
                    if not check_result.get("feedback") and check_result.get("explanation"):
                        check_result["feedback"] = check_result["explanation"]
                    
                    if "recommendations" not in check_result:
                        check_result["recommendations"] = []
                    
                    # Добавляем оценку по 10-балльной шкале, если её нет
                    if "score" not in check_result:
                        # Если ответ правильный, даем оценку 7, иначе 4
                        check_result["score"] = 7 if check_result.get("is_correct", False) else 4
                    
                    # Добавляем next_steps в зависимости от правильности ответа
                    if check_result.get("is_correct", False):
                        # Увеличиваем счетчик правильных ответов для творческих заданий
                        self.correct_answers_count[student_id] = self.correct_answers_count.get(student_id, 0) + 1
                        
                        # Проверяем, достиг ли студент 5 правильных ответов подряд
                        next_steps = []
                        if self.correct_answers_count.get(student_id, 0) >= 5:
                            # Предлагаем изменить параметры обучения
                            next_steps = [
                                {"action": "change_chapter", "text": "Перейти к следующей главе"},
                                {"action": "increase_difficulty", "text": "Повысить сложность задач"},
                                {"action": "change_task_type", "text": "Попробовать шаблонную задачу"}
                            ]
                            # Сбрасываем счетчик
                            self.correct_answers_count[student_id] = 0
                        
                        check_result["next_steps"] = next_steps
                    else:
                        # Сбрасываем счетчик правильных ответов
                        self.correct_answers_count[student_id] = 0
                        
                        # Предлагаем опции для неправильного ответа
                        check_result["next_steps"] = [
                            {"action": "discuss", "text": "Обсудить задачу"},
                            {"action": "try_again", "text": "Попробовать ещё раз"},
                            {"action": "skip", "text": "Пропустить задачу"}
                        ]
                    
                    return check_result
                except asyncio.TimeoutError:
                    logger.warning(f"Тайм-аут при обращении к OpenRouter API после {timeout_seconds} секунд")
                    # Используем простой ответ из-за тайм-аута
                    return {
                        "is_correct": True,  # При тайм-ауте считаем ответ условно правильным
                        "score": 6,  # Средняя оценка при тайм-ауте
                        "explanation": "Не удалось проверить ответ из-за тайм-аута. Считаем ответ условно правильным.",
                        "feedback": "Не удалось проверить ответ из-за тайм-аута. Считаем ответ условно правильным.",
                        "recommendations": ["Продолжайте работу с материалом."],
                        "next_steps": []
                    }
            except Exception as api_error:
                logger.exception(f"Ошибка при проверке ответа через API: {str(api_error)}")
                # Простой ответ в случае ошибки
                return {
                    "is_correct": True,  # При ошибке считаем ответ условно правильным
                    "score": 5,  # Средняя оценка при ошибке
                    "explanation": f"Произошла ошибка при проверке ответа, но ваш ответ принят.",
                    "feedback": f"Произошла ошибка при проверке ответа, но ваш ответ принят.",
                    "recommendations": ["Продолжайте изучение материала."],
                    "next_steps": []
                }
                
        except Exception as e:
            logger.error(f"Ошибка при проверке ответа: {str(e)}\n{traceback.format_exc()}")
            return {
                "is_correct": False,
                "score": 0,
                "explanation": f"Ошибка при проверке ответа: {str(e)}",
                "feedback": f"Ошибка при проверке ответа: {str(e)}",
                "recommendations": ["Попробуйте еще раз с другой задачей."],
                "next_steps": [
                    {"action": "skip", "text": "Пропустить задачу"}
                ]
            }
    
    def check_answer(
        self,
        student_id: str,
        chapter_title: str,
        task: Dict[str, Any],
        student_answer: str
    ) -> Dict[str, Any]:
        """Синхронная обертка для async_check_answer."""
        try:
            logger.info(f"Начинаем проверку ответа для задачи по теме: {task.get('concept_name', '')}")
            
            # Создаем новый цикл событий для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Вызываем асинхронный метод
                return loop.run_until_complete(
                    self.async_check_answer(student_id, chapter_title, task, student_answer)
                )
            finally:
                logger.info("Закрываем цикл событий")
                loop.close()
                
        except Exception as e:
            logger.error(f"Ошибка при проверке ответа: {str(e)}\n{traceback.format_exc()}")
            return {
                "is_correct": False,
                "explanation": f"Ошибка при проверке ответа: {str(e)}",
                "feedback": f"Извините, не удалось проверить ваш ответ. Попробуйте еще раз.",
                "recommendations": ["Попробуйте еще раз с другой задачей."],
                "next_steps": [
                    {"action": "skip", "text": "Пропустить задачу"}
                ]
            }
    
    def adapt_task_difficulty(
        self,
        student_id: str,
        chapter_title: str,
        student_performance: Optional[List[Dict[str, Any]]] = None,
        current_difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """Синхронная обертка для adapt_task_difficulty."""
        try:
            # Создаем новый цикл событий для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Упрощенная версия адаптации без асинхронных вызовов
                # Всегда рекомендуем стандартный уровень для простоты, если не указан текущий
                if not current_difficulty:
                    current_difficulty = "standard"
                
                # Если текущий уровень standard и у студента хорошие показатели, рекомендуем advanced
                recommended_difficulty = "advanced" if current_difficulty == "standard" else "standard"
                
                return {
                    "student_id": student_id,
                    "chapter_title": chapter_title,
                    "recommended_difficulty": recommended_difficulty,
                    "recommended_task_type": "template" if current_difficulty == "creative" else "creative",
                    "reasoning": f"Рекомендован {'продвинутый' if recommended_difficulty == 'advanced' else 'стандартный'} уровень для вашего текущего прогресса.",
                    "problem_concepts": [],
                    "strong_concepts": []
                }
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Ошибка при адаптации сложности: {str(e)}\n{traceback.format_exc()}")
            return {
                "student_id": student_id,
                "chapter_title": chapter_title,
                "recommended_difficulty": "standard",
                "recommended_task_type": "template",
                "reasoning": "Рекомендован стандартный уровень по умолчанию из-за ошибки.",
                "problem_concepts": [],
                "strong_concepts": []
            }

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Извлекает JSON из текстового ответа.
        
        Args:
            text: Текстовый ответ.
            
        Returns:
            Словарь, извлеченный из JSON.
        """
        try:
            # Попытка найти JSON в тексте
            json_start = text.find("{")
            json_end = text.rfind("}")
            
            if json_start != -1 and json_end != -1:
                json_str = text[json_start:json_end + 1]
                return json.loads(json_str)
            
            # Если не удалось найти JSON в формате {...}, ищем в формате [...]
            json_start = text.find("[")
            json_end = text.rfind("]")
            
            if json_start != -1 and json_end != -1:
                json_str = text[json_start:json_end + 1]
                return {"data": json.loads(json_str)}
            
            # Если не удалось найти JSON, пытаемся преобразовать весь текст
            return json.loads(text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при извлечении JSON из ответа: {str(e)}\n{text}")
            # Возвращаем пустой словарь с текстовым ответом
            return {"raw_text": text}
