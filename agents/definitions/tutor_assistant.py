"""
Модуль агента-репетитора для обсуждения задач со студентами
"""
import logging
import asyncio
import traceback
from typing import Dict, List, Any, Optional

from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.api.openrouter import OpenRouterClient
from ai_tutor.agents.prompts.tutor_assistant_prompt import (
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_ANSWER_PROMPT,
    INCORRECT_ANSWER_GUIDANCE_PROMPT,
    EXPLANATION_AFTER_ATTEMPTS_PROMPT,
    GENERAL_CONSULTATION_SYSTEM_PROMPT,
    GENERAL_CONSULTATION_PROMPT
)
from ai_tutor.config.settings import COURSE_NAME, CHAPTERS

logger = logging.getLogger(__name__)


class TutorAssistant:
    """
    Агент-репетитор для обсуждения задач со студентами
    """
    
    def __init__(self, neo4j_client: Neo4jClient, openrouter_client: OpenRouterClient):
        """
        Инициализация агента-репетитора
        
        Args:
            neo4j_client: Клиент для работы с Neo4j
            openrouter_client: Клиент для работы с OpenRouter API
        """
        self.neo4j_client = neo4j_client
        self.openrouter_client = openrouter_client
    
    async def general_consultation(self, student_question: str, student_id: Optional[str] = None) -> str:
        """
        Общая консультация по темам курса, не привязанная к задачам
        
        Args:
            student_question: Вопрос студента
            student_id: ID студента (опционально)
            
        Returns:
            Ответ консультанта
        """
        try:
            # Проверяем, что вопрос не пустой
            if not student_question or student_question.strip() == "":
                logger.warning("Пустой вопрос в запросе на консультацию")
                return ("Пожалуйста, задайте ваш вопрос о системном мышлении. "
                        "Я могу рассказать о понятиях курса, объяснить взаимосвязи между ними "
                        "или помочь разобраться с конкретной темой.")
                
            logger.info(f"Запрос на общую консультацию: {student_question[:50]}...")
            
            # 1. Поиск релевантных понятий через семантический поиск
            relevant_concepts = []
            try:
                relevant_concepts = self.neo4j_client.semantic_search(
                    query=student_question,
                    limit=5,
                    min_similarity=0.5
                )
                logger.info(f"Найдено {len(relevant_concepts)} релевантных понятий")
            except Exception as e:
                logger.warning(f"Ошибка при семантическом поиске: {str(e)}")
            
            # 2. Поиск релевантных глав
            relevant_chapters = []
            chapter_info = {}
            
            # Если найдены понятия, определяем, к каким главам они относятся
            if relevant_concepts:
                for concept in relevant_concepts:
                    try:
                        chapters = self.neo4j_client.get_chapters_for_concept(concept['name'])
                        for chapter in chapters:
                            if chapter not in relevant_chapters:
                                relevant_chapters.append(chapter)
                                # Получаем информацию о главе
                                chapter_info[chapter] = self.neo4j_client.get_chapter_info(chapter)
                    except Exception as e:
                        logger.warning(f"Ошибка при поиске глав для понятия {concept['name']}: {str(e)}")
            
            # 3. Формирование контекста запроса
            query_context = "Информация по запросу:\n"
            
            # Добавляем информацию о найденных понятиях
            if relevant_concepts:
                query_context += "\nНайденные релевантные понятия:\n"
                for i, concept in enumerate(relevant_concepts):
                    query_context += f"\n{i+1}. {concept['name']}\n"
                    query_context += f"   Определение: {concept.get('definition', 'Не указано')}\n"
                    if concept.get('example'):
                        query_context += f"   Пример: {concept['example']}\n"
                    
                    # Добавляем связанные понятия, если есть
                    try:
                        related = self.neo4j_client.get_related_concepts(concept['name'])
                        if related:
                            query_context += "   Связанные понятия: "
                            query_context += ", ".join([f"{r['name']} ({r.get('relation_type', 'связано с')})" for r in related])
                            query_context += "\n"
                    except Exception as e:
                        logger.warning(f"Ошибка при получении связанных понятий: {str(e)}")
            else:
                query_context += "\nНе найдено релевантных понятий в базе данных.\n"
            
            # Добавляем информацию о главах
            if relevant_chapters:
                query_context += "\nГлавы, в которых упоминаются эти понятия:\n"
                for chapter in relevant_chapters:
                    info = chapter_info.get(chapter, {})
                    query_context += f"\n- {chapter}\n"
                    if info.get('main_ideas'):
                        query_context += f"  Основные идеи: {info['main_ideas']}\n"
            
            # 4. Формируем системный промпт
            system_prompt = GENERAL_CONSULTATION_SYSTEM_PROMPT.format(
                course_name=COURSE_NAME,
                chapters_list="\n".join([f"- {chapter}" for chapter in CHAPTERS]),
                query_context=query_context,
                student_question=student_question
            )
            
            # 5. Формируем сообщения для модели
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": GENERAL_CONSULTATION_PROMPT.format(student_question=student_question)}
            ]
            
            # 6. Отправляем запрос к модели
            response = await self.openrouter_client.generate_completion(messages, temperature=0.7, max_tokens=1500)
            answer = response["choices"][0]["message"]["content"]
            
            logger.info(f"Сгенерирован ответ на общую консультацию: {student_question[:50]}...")
            
            # 7. Логируем взаимодействие, если указан ID студента
            if student_id:
                self.log_discussion(
                    student_id=student_id,
                    concept_name="Общая консультация",
                    question=student_question,
                    answer=answer
                )
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при общей консультации: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже."
    
    async def discuss_task(self, 
                          student_question: str, 
                          concept_name: str, 
                          task_question: str, 
                          chapter_title: Optional[str] = None) -> str:
        """
        Обсуждение задачи со студентом
        
        Args:
            student_question: Вопрос студента
            concept_name: Название понятия
            task_question: Вопрос задачи
            chapter_title: Название главы (опционально)
            
        Returns:
            Ответ репетитора
        """
        try:
            # Проверяем, что concept_name не пустой
            if not concept_name or concept_name.strip() == "":
                logger.warning("Пустое имя понятия в запросе")
                
                # Попробуем получить информацию о главе
                chapter_info_text = ""
                if chapter_title:
                    try:
                        chapter_info = self.neo4j_client.get_chapter_info(chapter_title)
                        if chapter_info:
                            chapter_info_text = f"\n\nМы обсуждаем главу '{chapter_title}'. "
                            if 'main_ideas' in chapter_info:
                                chapter_info_text += f"Основные идеи главы: {chapter_info['main_ideas']}"
                    except Exception as e:
                        logger.warning(f"Ошибка при получении информации о главе: {str(e)}")
                
                return (f"Я не могу определить, о каком понятии идет речь. "
                        f"Пожалуйста, уточните название понятия, которое вы хотите обсудить.{chapter_info_text}")
            
            # Получаем информацию о понятии
            concept = self.neo4j_client.get_concept_by_name(concept_name, chapter_title)
            
            if not concept:
                logger.warning(f"Не найдено понятие: {concept_name}")
                return f"К сожалению, я не нашел информации о понятии '{concept_name}'. Пожалуйста, уточните название понятия."
            
            concept_definition = concept.get('definition', 'Определение отсутствует')
            
            # Получаем контекст главы, если указана
            chapter_context = "Информация о главе отсутствует."
            if chapter_title:
                try:
                    chapter_info = self.neo4j_client.get_chapter_info(chapter_title)
                    if chapter_info:
                        chapter_context = (
                            f"Название главы: {chapter_title}\n"
                            f"Основные идеи: {chapter_info.get('main_ideas', 'Не указаны')}\n"
                        )
                except Exception as e:
                    logger.warning(f"Ошибка при получении информации о главе {chapter_title}: {str(e)}")
            
            # Формируем системный промпт
            system_prompt = DISCUSSION_SYSTEM_PROMPT.format(
                concept_name=concept_name,
                concept_definition=concept_definition,
                task_question=task_question,
                chapter_context=chapter_context,
                student_question=student_question
            )
            
            # Формируем сообщения для модели
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": DISCUSSION_ANSWER_PROMPT.format(student_question=student_question)}
            ]
            
            # Отправляем запрос к модели
            response = await self.openrouter_client.generate_completion(messages, temperature=0.7)
            answer = response["choices"][0]["message"]["content"]
            
            logger.info(f"Сгенерирован ответ на вопрос по задаче: {student_question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при обсуждении задачи: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже."
    
    async def provide_guidance_after_incorrect_answer(self, 
                                                     student_answer: str, 
                                                     correct_answer: str, 
                                                     concept_name: str) -> str:
        """
        Предоставление подсказки после неправильного ответа
        
        Args:
            student_answer: Ответ студента
            correct_answer: Правильный ответ
            concept_name: Название понятия
            
        Returns:
            Подсказка
        """
        try:
            # Формируем сообщения для модели
            messages = [
                {"role": "system", "content": "Ты - опытный педагог, помогающий студентам разобраться в ошибках."},
                {"role": "user", "content": INCORRECT_ANSWER_GUIDANCE_PROMPT.format(
                    student_answer=student_answer,
                    correct_answer=correct_answer,
                    concept_name=concept_name
                )}
            ]
            
            # Отправляем запрос к модели
            response = await self.openrouter_client.generate_completion(messages, temperature=0.7)
            guidance = response["choices"][0]["message"]["content"]
            
            logger.info(f"Сгенерирована подсказка после неправильного ответа для понятия {concept_name}")
            return guidance
            
        except Exception as e:
            logger.error(f"Ошибка при генерации подсказки: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при анализе вашего ответа. Попробуйте ещё раз или задайте вопрос."
    
    async def explain_after_multiple_attempts(self, 
                                            concept_name: str, 
                                            task_question: str, 
                                            correct_answer: str,
                                            chapter_title: Optional[str] = None) -> str:
        """
        Подробное объяснение после нескольких неудачных попыток
        
        Args:
            concept_name: Название понятия
            task_question: Вопрос задачи
            correct_answer: Правильный ответ
            chapter_title: Название главы (опционально)
            
        Returns:
            Подробное объяснение
        """
        try:
            # Получаем информацию о понятии
            concept = self.neo4j_client.get_concept_by_name(concept_name, chapter_title)
            
            if not concept:
                logger.warning(f"Не найдено понятие: {concept_name}")
                return f"К сожалению, я не нашел информации о понятии '{concept_name}'. Пожалуйста, уточните название понятия."
            
            concept_definition = concept.get('definition', 'Определение отсутствует')
            
            # Формируем сообщения для модели
            messages = [
                {"role": "system", "content": "Ты - опытный преподаватель, объясняющий сложные понятия доступным языком."},
                {"role": "user", "content": EXPLANATION_AFTER_ATTEMPTS_PROMPT.format(
                    concept_name=concept_name,
                    concept_definition=concept_definition,
                    task_question=task_question,
                    correct_answer=correct_answer
                )}
            ]
            
            # Отправляем запрос к модели
            response = await self.openrouter_client.generate_completion(messages, temperature=0.7, max_tokens=1500)
            explanation = response["choices"][0]["message"]["content"]
            
            logger.info(f"Сгенерировано подробное объяснение для понятия {concept_name}")
            return explanation
            
        except Exception as e:
            logger.error(f"Ошибка при генерации объяснения: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при подготовке объяснения. Пожалуйста, обратитесь к преподавателю."
            
    def log_discussion(self, student_id: str, concept_name: str, question: str, answer: str, chapter_title: Optional[str] = None) -> None:
        """
        Сохранение обсуждения в базе данных
        
        Args:
            student_id: ID студента
            concept_name: Название понятия
            question: Вопрос студента
            answer: Ответ репетитора
            chapter_title: Название главы (опционально)
        """
        try:
            # Вызываем метод Neo4j клиента для логирования
            self.neo4j_client.save_assistant_interaction(
                student_id=student_id,
                question=question,
                answer=answer,
                chapter_title=chapter_title
            )
            logger.info(f"Сохранено обсуждение для студента {student_id} по понятию {concept_name}")
        except Exception as e:
            logger.error(f"Ошибка при логировании обсуждения: {str(e)}") 