"""
Универсальный агент-ассистент для ответов на вопросы и консультаций по системному мышлению

Этот модуль объединяет функциональность CourseAssistant и TutorAssistant в единый 
универсальный интерфейс UnifiedAssistant. Такое объединение обеспечивает:
1. Унифицированный доступ к функциям ответов на вопросы
2. Умное определение типа запроса и выбор оптимального метода обработки
3. Устранение дублирования кода и упрощение интеграции
"""
import logging
import asyncio
import traceback
import re
from typing import Dict, List, Any, Optional, Set, Union

from ai_tutor.database.neo4j_client import Neo4jClient
# Проверяем доступность SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logging.getLogger(__name__).warning("Не удалось импортировать sentence_transformers. Будет использоваться текстовый поиск.")

from ai_tutor.database.enhanced_search import EnhancedCourseSearch, FallbackSearch
from ai_tutor.api.openrouter import OpenRouterClient
from ai_tutor.config.settings import COURSE_NAME, CHAPTERS
from ai_tutor.agents.prompts.tutor_assistant_prompt import (
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_ANSWER_PROMPT,
    INCORRECT_ANSWER_GUIDANCE_PROMPT,
    EXPLANATION_AFTER_ATTEMPTS_PROMPT,
    GENERAL_CONSULTATION_SYSTEM_PROMPT,
    GENERAL_CONSULTATION_PROMPT
)

logger = logging.getLogger(__name__)


class UnifiedAssistant:
    """
    Универсальный агент-ассистент для ответов на вопросы по курсу и обсуждения задач
    Объединяет функциональность CourseAssistant и TutorAssistant
    
    Класс объединяет два подхода:
    1. От CourseAssistant:
       - Поиск релевантных понятий по ключевым словам
       - Формирование контекста из найденных понятий
       - Генерация ответов с использованием ключевых слов
       
    2. От TutorAssistant:
       - Обсуждение задач по конкретным понятиям
       - Общие консультации с семантическим поиском
       - Предоставление подсказок после неправильных ответов
       - Подробное объяснение задач после нескольких попыток
       
    В зависимости от типа запроса и контекста автоматически выбирается оптимальный
    способ обработки и генерации ответа.
    """
    
    def __init__(self, neo4j_client: Neo4jClient, openrouter_client: OpenRouterClient):
        """
        Инициализация универсального агента
        
        Args:
            neo4j_client: Клиент для работы с Neo4j
            openrouter_client: Клиент для работы с OpenRouter API
        """
        self.neo4j_client = neo4j_client
        self.openrouter_client = openrouter_client
        
        # Инициализация улучшенного поиска
        try:
            logger.info("Инициализация улучшенного семантического поиска с векторными embeddings...")
            
            # Проверка, доступен ли SentenceTransformer на уровне модуля
            if not SENTENCE_TRANSFORMER_AVAILABLE:
                logger.warning("SentenceTransformer недоступен на уровне модуля, использование текстового поиска")
                self.enhanced_search = FallbackSearch()
                self.use_enhanced_search = False
                logger.info("Инициализирована заглушка для текстового поиска")
                return
            
            # Если SentenceTransformer доступен, пробуем создать EnhancedCourseSearch
            self.enhanced_search = EnhancedCourseSearch()
            
            # Проверяем, что объект корректно создан
            if hasattr(self.enhanced_search, 'model') and self.enhanced_search.model:
                logger.info("Успешно инициализирован улучшенный семантический поиск с моделью: "
                           f"{getattr(self.enhanced_search, 'model', 'Неизвестная модель')}")
                self.use_enhanced_search = True
            else:
                logger.warning("Модель для векторного поиска не была корректно инициализирована")
                # Используем заглушку, если модель не инициализирована
                self.enhanced_search = FallbackSearch()
                self.use_enhanced_search = False
                logger.info("Инициализирована заглушка для текстового поиска из-за ошибки модели")
                
        except ImportError as e:
            logger.error(f"Ошибка импорта при инициализации улучшенного поиска: {str(e)}")
            logger.error("Убедитесь, что установлена библиотека sentence-transformers")
            logger.warning("Будет использован текстовый поиск (заглушка)")
            self.use_enhanced_search = False
            # Используем заглушку FallbackSearch
            self.enhanced_search = FallbackSearch()
            logger.info("Инициализирована заглушка для текстового поиска из-за ImportError")
        except ValueError as e:
            logger.error(f"Ошибка в параметрах при инициализации улучшенного поиска: {str(e)}")
            logger.warning("Будет использован текстовый поиск (заглушка)")
            self.use_enhanced_search = False
            # Используем заглушку FallbackSearch
            self.enhanced_search = FallbackSearch()
            logger.info("Инициализирована заглушка для текстового поиска из-за ValueError")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при инициализации улучшенного поиска: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("Будет использован текстовый поиск (заглушка)")
            self.use_enhanced_search = False
            # Используем заглушку FallbackSearch
            self.enhanced_search = FallbackSearch()
            logger.info("Инициализирована заглушка для текстового поиска из-за неизвестной ошибки")
    
    # --- ОСНОВНЫЕ ПУБЛИЧНЫЕ МЕТОДЫ ---
    
    async def answer_question(self, question: str, 
                             student_id: Optional[str] = None,
                             chapter_title: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> str:
        """
        Универсальный метод для ответа на вопросы, определяющий тип запроса
        
        Args:
            question: Вопрос студента
            student_id: ID студента (опционально)
            chapter_title: Название главы (опционально)
            context: Дополнительный контекст (задача, понятие и т.д.)
            
        Returns:
            Ответ на вопрос
        """
        try:
            logger.info(f"Получен вопрос: '{question[:50]}...'")
            
            # Определяем тип запроса на основе контекста
            if context and (context.get('task_question') or context.get('concept_name')):
                # Обсуждение задачи со студентом
                answer = await self._discuss_task(
                    student_question=question,
                    concept_name=context.get('concept_name', ''),
                    task_question=context.get('task_question', ''),
                    chapter_title=chapter_title
                )
            else:
                # Всегда используем семантический поиск для общих вопросов
                answer = await self._general_consultation(
                    student_question=question,
                    student_id=student_id,
                    chapter_title=chapter_title
                )
            
            # Логируем взаимодействие
            if student_id:
                # Корректная обработка отсутствующего контекста
                concept_name = 'Общая консультация'
                if context is not None and 'concept_name' in context:
                    concept_name = context.get('concept_name')
                
                self.log_interaction(
                    student_id=student_id,
                    question=question,
                    answer=answer,
                    concept_name=concept_name,
                    chapter_title=chapter_title
                )
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при ответе на вопрос: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже."

    def answer_question_sync(self, question: str, 
                           student_id: Optional[str] = None,
                           chapter_title: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        Синхронная обертка для ответа на вопрос студента
        
        Args:
            question: Вопрос студента
            student_id: ID студента (опционально)
            chapter_title: Название главы (опционально)
            context: Дополнительный контекст (задача, понятие и т.д.)
            
        Returns:
            Ответ на вопрос
        """
        try:
            # Создаем новый цикл событий для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Вызываем асинхронный метод
                return loop.run_until_complete(
                    self.answer_question(
                        question=question, 
                        student_id=student_id,
                        chapter_title=chapter_title,
                        context=context
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Ошибка в синхронной обертке answer_question_sync: {str(e)}\n{traceback.format_exc()}")
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
    
    # --- ВНУТРЕННИЕ МЕТОДЫ ОБРАБОТКИ ЗАПРОСОВ ---
    
    async def _enhanced_semantic_search(self, query: str, limit: int = 5, 
                                      threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Выполнение улучшенного семантического поиска, если он доступен
        
        Args:
            query: Текст запроса
            limit: Максимальное количество результатов
            threshold: Минимальный порог сходства
            
        Returns:
            Список релевантных понятий/документов
        """
        if self.use_enhanced_search and self.enhanced_search:
            try:
                logger.info(f"Выполняется УЛУЧШЕННЫЙ семантический поиск для запроса: '{query[:50]}...'")
                
                # Используем улучшенный поиск с ранжированием
                results = self.enhanced_search.semantic_search_with_ranking(
                    query=query,
                    limit=limit,
                    threshold=threshold
                )
                logger.info(f"Улучшенный семантический поиск вернул {len(results)} результатов")
                
                if not results:
                    logger.warning("Улучшенный поиск не нашел результатов, попробуем стандартный поиск")
                    # Если улучшенный поиск не дал результатов, используем стандартный поиск
                    return await self._fallback_standard_search(query, limit, threshold)
                
                return results
            except Exception as e:
                logger.error(f"Ошибка в улучшенном поиске: {str(e)}")
                logger.error(traceback.format_exc())
                # Если произошла ошибка, используем резервный базовый поиск
                logger.info("Переключаемся на стандартный поиск из-за ошибки")
                return await self._fallback_standard_search(query, limit, threshold)
        
        # Если улучшенный поиск недоступен, используем стандартный поиск
        logger.info(f"Используется СТАНДАРТНЫЙ семантический поиск для запроса: '{query[:50]}...'")
        return await self._fallback_standard_search(query, limit, threshold)
        
    async def _fallback_standard_search(self, query: str, limit: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Резервный стандартный поиск через Neo4j клиент
        
        Args:
            query: Текст запроса
            limit: Максимальное количество результатов
            threshold: Минимальный порог сходства
            
        Returns:
            Список релевантных понятий/документов
        """
        try:
            results = self.neo4j_client.semantic_search(
                query=query,
                limit=limit,
                min_similarity=threshold
            )
            logger.info(f"Стандартный семантический поиск вернул {len(results)} результатов")
            return results
        except Exception as e:
            logger.error(f"Ошибка в стандартном поиске: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    async def _answer_by_keywords(self, question: str, chapter_title: Optional[str] = None) -> str:
        """
        Ответ на вопрос с помощью поиска по ключевым словам (подход CourseAssistant)
        
        Args:
            question: Вопрос студента
            chapter_title: Название главы (опционально)
            
        Returns:
            Ответ на вопрос
        """
        try:
            # 1. Извлекаем ключевые слова
            keywords = self._extract_keywords(question)
            logger.info(f"Извлечены ключевые слова: {', '.join(keywords)}")
            
            # 2. Ищем релевантные понятия
            concepts = self.neo4j_client.search_concepts_by_keywords(keywords, chapter_title)
            
            if not concepts:
                logger.warning(f"Не найдено понятий по запросу: {question}")
                return "К сожалению, я не нашел информации по вашему вопросу в материалах курса. Пожалуйста, попробуйте переформулировать вопрос или уточнить, какой аспект курса вас интересует."
            
            logger.info(f"Найдено {len(concepts)} релевантных понятий")
            
            # 3. Формируем контекст
            context = self._build_concept_context(concepts, chapter_title)
            
            # 4. Генерируем ответ с помощью LLM
            messages = [
                {"role": "system", "content": f"""
                Ты - помощник по курсу '{COURSE_NAME}'. Отвечай на вопросы студентов,
                используя ТОЛЬКО информацию из предоставленного контекста понятий.
                Если в контексте нет достаточной информации, честно признай это и не выдумывай факты.
                Твоя цель - дать точную информацию из курса, помочь студенту разобраться в понятиях
                и их взаимосвязях. Язык ответа - русский.
                
                Контекст понятий:
                {context}
                """},
                {"role": "user", "content": question}
            ]
            
            # Отправляем запрос к API
            response = await self.openrouter_client.generate_completion(messages, temperature=0.3)
            answer = response["choices"][0]["message"]["content"]
            
            logger.info(f"Сгенерирован ответ на вопрос: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при ответе на вопрос: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже."
    
    async def _discuss_task(self, 
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
            
            # Определяем максимальное количество токенов для первой попытки
            context_size = len(concept_definition) + len(chapter_context) + len(task_question) + len(student_question)
            max_tokens = 1500  # Базовое значение
            
            if context_size > 4000:
                max_tokens = 1000  # Для очень большого контекста
                logger.warning(f"Очень большой контекст: {context_size} символов. Ограничиваем начальный ответ до {max_tokens} токенов.")
                
            # Добавляем инструкцию о необходимости развернутого ответа
            completion_instruction = "\n\nВАЖНО: Предоставь полный, развернутый ответ, который детально объясняет понятие и его связь с вопросом студента. Используй примеры и будь информативным."
            
            # Формируем системный промпт
            system_prompt = DISCUSSION_SYSTEM_PROMPT.format(
                concept_name=concept_name,
                concept_definition=concept_definition,
                task_question=task_question,
                chapter_context=chapter_context,
                student_question=student_question
            ) + completion_instruction
            
            # Формируем сообщения для модели
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": DISCUSSION_ANSWER_PROMPT.format(student_question=student_question)}
            ]
            
            # Отправляем запрос к модели с возможностью повторных попыток
            max_attempts = 3  # Максимальное количество попыток
            current_attempt = 0
            answer = ""
            
            while current_attempt < max_attempts:
                current_attempt += 1
                logger.info(f"Попытка генерации ответа на задачу {current_attempt}/{max_attempts} (max_tokens: {max_tokens})...")
                
                try:
                    # Отправляем запрос к модели
                    response = await self.openrouter_client.generate_completion(messages, temperature=0.7, max_tokens=max_tokens)
                    answer = response["choices"][0]["message"]["content"]
                    
                    # Логируем информацию об ответе
                    completion_tokens = response["usage"]["completion_tokens"]
                    total_tokens = response["usage"]["total_tokens"]
                    finish_reason = response["choices"][0]["finish_reason"]
                    
                    logger.info(f"Сгенерирован ответ длиной {len(answer)} символов ({completion_tokens} токенов)")
                    logger.info(f"Завершено по причине: {finish_reason}, всего токенов: {total_tokens}")
                    
                    # Проверяем качество ответа - если ответ слишком короткий, пробуем повторить запрос
                    if len(answer) < 200:
                        logger.warning(f"Ответ на задачу слишком короткий ({len(answer)} символов). Пробуем увеличить max_tokens.")
                        max_tokens += 500  # Увеличиваем лимит токенов
                        
                        # Модифицируем промпт для получения более полного ответа
                        completion_instruction = "\n\nКРИТИЧЕСКИ ВАЖНО: Предыдущий ответ был неполным. Необходимо дать МАКСИМАЛЬНО ДЕТАЛЬНЫЙ И РАЗВЕРНУТЫЙ ответ, с объяснением понятия и его применения в контексте вопроса студента."
                        
                        system_prompt = DISCUSSION_SYSTEM_PROMPT.format(
                            concept_name=concept_name,
                            concept_definition=concept_definition,
                            task_question=task_question,
                            chapter_context=chapter_context,
                            student_question=student_question
                        ) + completion_instruction
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Пожалуйста, ответь РАЗВЕРНУТО на вопрос: {student_question}. Объясни понятие {concept_name} и его связь с вопросом."}
                        ]
                        
                        continue  # Переходим к следующей попытке
                    
                    # Если ответ обрезан, но уже достаточно длинный, добавляем предупреждение
                    if finish_reason == "length" and len(answer) > 500:
                        answer += "\n\n(Ответ был ограничен по размеру. Чтобы получить дополнительную информацию, задайте уточняющий вопрос.)"
                    
                    # Если ответ удовлетворительный, выходим из цикла
                    break
                    
                except Exception as e:
                    logger.error(f"Ошибка при генерации ответа на задачу (попытка {current_attempt}): {str(e)}")
                    logger.error(traceback.format_exc())
                    # Если это последняя попытка, устанавливаем сообщение об ошибке
                    if current_attempt == max_attempts:
                        answer = "Извините, произошла ошибка при генерации ответа. Пожалуйста, попробуйте задать вопрос иначе."
            
            # Проверяем длину итогового ответа
            if len(answer) < 200 and current_attempt == max_attempts:
                # Если после всех попыток ответ всё ещё слишком короткий, формируем запасной вариант
                logger.warning(f"После {max_attempts} попыток ответ на задачу всё ещё слишком короткий. Формируем запасной ответ.")
                
                # Строим запасной ответ на основе информации о понятии
                answer = (f"Хочу помочь вам разобраться с понятием '{concept_name}'. \n\n"
                        f"Определение: {concept_definition}\n\n"
                        f"В контексте вашего вопроса '{student_question}' важно отметить, что это понятие "
                        f"является ключевым для понимания темы.\n\n"
                        f"Предлагаю начать с уточнения: какой именно аспект понятия '{concept_name}' "
                        f"вас интересует больше всего?")
            
            # Проверяем длину ответа
            if len(answer) > 4000:
                logger.warning(f"Ответ очень длинный: {len(answer)} символов. Будет разбит на части в Telegram.")
                # Добавляем предупреждение в начало ответа
                warning_msg = "Внимание: ответ получился очень объемным и будет отображаться по частям.\n\n"
                answer = warning_msg + answer
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при обсуждении задачи: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже."
    
    async def _general_consultation(self, student_question: str, student_id: Optional[str] = None,
                                 chapter_title: Optional[str] = None) -> str:
        """
        Общая консультация по темам курса с семантическим поиском (подход TutorAssistant)
        
        Args:
            student_question: Вопрос студента
            student_id: ID студента (опционально)
            chapter_title: Название главы (опционально)
            
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
            
            # 1. Поиск релевантных понятий через улучшенный семантический поиск
            relevant_concepts = await self._enhanced_semantic_search(
                query=student_question,
                limit=5,
                threshold=0.5
            )
            
            # Логирование результатов поиска
            if relevant_concepts:
                concept_names = [concept['name'] for concept in relevant_concepts]
                similarity_scores = [f"{concept['name']}:{concept.get('similarity', 0):.2f}" for concept in relevant_concepts]
                logger.info(f"Найдено {len(relevant_concepts)} релевантных понятий: {', '.join(concept_names)}")
                logger.info(f"Оценки сходства: {', '.join(similarity_scores)}")
            else:
                logger.warning(f"Не найдено релевантных понятий для запроса: '{student_question[:50]}...'")
            
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
            
            # Логируем найденные главы
            if relevant_chapters:
                logger.info(f"Найдены главы, содержащие релевантные понятия: {', '.join(relevant_chapters)}")
            
            # 3. Формирование контекста запроса
            query_context = "Информация по запросу:\n"
            
            # Добавляем информацию о найденных понятиях
            if relevant_concepts:
                query_context += "\nНайденные релевантные понятия:\n"
                for i, concept in enumerate(relevant_concepts):
                    # Вычисляем процент сходства для наглядности
                    similarity_percent = round(concept.get('similarity', 0) * 100, 1)
                    
                    query_context += f"\n{i+1}. {concept['name']} (Релевантность: {similarity_percent}%)\n"
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
                query_context += "\nНе найдено релевантных понятий в базе данных. Будет предоставлена общая информация по курсу.\n"
            
            # Добавляем информацию о главах
            if relevant_chapters:
                query_context += "\nГлавы, в которых упоминаются эти понятия:\n"
                for chapter in relevant_chapters:
                    info = chapter_info.get(chapter, {})
                    query_context += f"\n- {chapter}\n"
                    if info.get('main_ideas'):
                        query_context += f"  Основные идеи: {info['main_ideas']}\n"
            
            # Логируем контекст запроса (в отладочном режиме)
            logger.debug(f"Сформирован контекст запроса длиной {len(query_context)} символов")
            
            # Определяем максимальное количество токенов в зависимости от размера контекста
            # Чтобы избежать слишком длинных сообщений, которые могут быть обрезаны в Telegram
            context_size = len(query_context)
            
            # Устанавливаем начальное значение max_tokens достаточно высоким,
            # чтобы получить полный ответ при первой попытке
            max_tokens = 1500  # Начинаем с большого значения
            
            if context_size > 5000:
                max_tokens = 1000  # Для очень большого контекста
                logger.warning(f"Очень большой контекст: {context_size} символов. Ограничиваем ответ до {max_tokens} токенов.")
            
            # Добавляем инструкцию о длине в системный промпт
            completion_instruction = "\n\nВАЖНО: Предоставь полный, развернутый ответ с детальным объяснением понятий и их взаимосвязей. Ответ должен быть структурированным и информативным."
            
            # 4. Формируем системный промпт
            system_prompt = GENERAL_CONSULTATION_SYSTEM_PROMPT.format(
                course_name=COURSE_NAME,
                chapters_list="\n".join([f"- {chapter}" for chapter in CHAPTERS]),
                query_context=query_context,
                student_question=student_question
            ) + completion_instruction
            
            # 5. Формируем сообщения для модели
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": GENERAL_CONSULTATION_PROMPT.format(student_question=student_question)}
            ]
            
            # 6. Отправляем запрос к модели с возможностью повторных попыток
            max_attempts = 3  # Максимальное количество попыток
            current_attempt = 0
            answer = ""
            
            while current_attempt < max_attempts:
                current_attempt += 1
                logger.info(f"Попытка генерации ответа {current_attempt}/{max_attempts} (max_tokens: {max_tokens})...")
                
                try:
                    # Отправляем запрос к модели
                    response = await self.openrouter_client.generate_completion(messages, temperature=0.7, max_tokens=max_tokens)
                    answer = response["choices"][0]["message"]["content"]
                    
                    # Логируем информацию об ответе
                    completion_tokens = response["usage"]["completion_tokens"]
                    total_tokens = response["usage"]["total_tokens"]
                    finish_reason = response["choices"][0]["finish_reason"]
                    
                    logger.info(f"Сгенерирован ответ длиной {len(answer)} символов ({completion_tokens} токенов)")
                    logger.info(f"Завершено по причине: {finish_reason}, всего токенов: {total_tokens}")
                    
                    # Проверяем качество ответа - если ответ слишком короткий, пробуем повторить запрос
                    if len(answer) < 200:
                        logger.warning(f"Ответ слишком короткий ({len(answer)} символов). Пробуем увеличить max_tokens.")
                        max_tokens += 500  # Увеличиваем лимит токенов
                        
                        # Модифицируем промпт для получения более полного ответа
                        completion_instruction = "\n\nКРИТИЧЕСКИ ВАЖНО: Предыдущий ответ был неполным. Твоя задача - дать МАКСИМАЛЬНО ПОЛНЫЙ И РАЗВЕРНУТЫЙ ответ, минимум 500-1000 слов. Объясни все понятия подробно, с примерами и взаимосвязями."
                        
                        system_prompt = GENERAL_CONSULTATION_SYSTEM_PROMPT.format(
                            course_name=COURSE_NAME,
                            chapters_list="\n".join([f"- {chapter}" for chapter in CHAPTERS]),
                            query_context=query_context,
                            student_question=student_question
                        ) + completion_instruction
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Пожалуйста, ответь РАЗВЕРНУТО на вопрос: {student_question}. Не менее 500-1000 слов."}
                        ]
                        
                        continue  # Переходим к следующей попытке
                    
                    # Если ответ обрезан, но уже достаточно длинный, добавляем предупреждение
                    if finish_reason == "length" and len(answer) > 500:
                        answer += "\n\n(Ответ был ограничен по размеру. Если вам нужна дополнительная информация, уточните ваш вопрос.)"
                    
                    # Если ответ удовлетворительный, выходим из цикла
                    break
                    
                except Exception as e:
                    logger.error(f"Ошибка при генерации ответа (попытка {current_attempt}): {str(e)}")
                    logger.error(traceback.format_exc())
                    # Если это последняя попытка, устанавливаем сообщение об ошибке
                    if current_attempt == max_attempts:
                        answer = "Извините, произошла ошибка при генерации ответа. Пожалуйста, попробуйте переформулировать вопрос или задать его позже."
            
            # Проверяем длину итогового ответа
            if len(answer) < 200 and current_attempt == max_attempts:
                # Если после всех попыток ответ всё ещё слишком короткий, формируем запасной вариант
                logger.warning(f"После {max_attempts} попыток ответ всё ещё слишком короткий. Формируем запасной ответ.")
                
                # Строим запасной ответ на основе найденных понятий
                if relevant_concepts:
                    fallback_answer = f"По вашему запросу '{student_question}' я нашел следующую информацию:\n\n"
                    
                    for i, concept in enumerate(relevant_concepts[:3], 1):  # Берем только первые 3 понятия
                        fallback_answer += f"{i}. **{concept['name']}**\n"
                        fallback_answer += f"   Определение: {concept.get('definition', 'Определение не указано')}\n\n"
                        if concept.get('example'):
                            fallback_answer += f"   Пример: {concept['example']}\n\n"
                    
                    fallback_answer += "Пожалуйста, уточните, какой аспект данных понятий вас интересует, чтобы я мог предоставить более детальную информацию."
                    answer = fallback_answer
                else:
                    answer = "К сожалению, у меня не получилось сформировать полный ответ на ваш вопрос. Пожалуйста, попробуйте сформулировать запрос более конкретно или обратитесь к другим разделам курса."
            
            # Проверяем длину ответа и добавляем предупреждение, если он слишком большой
            if len(answer) > 4000:
                logger.warning(f"Ответ очень длинный: {len(answer)} символов. Будет разбит на части в Telegram.")
                
                # Добавляем предупреждение в начало ответа
                warning_msg = "Внимание: ответ получился очень объемным и будет отображаться по частям.\n\n"
                answer = warning_msg + answer
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка при общей консультации: {str(e)}\n{traceback.format_exc()}")
            return "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте еще раз позже."
    
    def _extract_keywords(self, question: str) -> List[str]:
        """
        Извлечение ключевых слов из вопроса
        
        Args:
            question: Вопрос студента
            
        Returns:
            Список ключевых слов
        """
        # Простая реализация: удаляем стоп-слова и оставляем существительные и прилагательные
        # В реальном приложении можно использовать более сложные алгоритмы (например, NLTK или spaCy)
        
        # Список стоп-слов на русском
        stop_words = {"и", "в", "на", "с", "по", "к", "у", "о", "это", "что", "как", "кто", "где", "когда", 
                      "почему", "зачем", "который", "такой", "этот", "тот", "для", "из", "от", "до", "за",
                      "при", "через", "над", "под", "около", "между", "а", "но", "или", "либо", "ни", "не",
                      "да", "же", "бы", "ли", "если", "то", "чтобы", "хотя", "потому", "так", "поэтому"}
        
        # Очищаем и токенизируем текст
        cleaned_text = re.sub(r'[^\w\s]', ' ', question.lower())
        words = cleaned_text.split()
        
        # Фильтруем стоп-слова и короткие слова
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Если ключевых слов меньше 2, возвращаем все слова длиннее 3 символов
        if len(keywords) < 2:
            keywords = [word for word in words if len(word) > 3]
        
        return keywords
    
    def _build_concept_context(self, concepts: List[Dict[str, Any]], chapter_title: Optional[str] = None) -> str:
        """
        Построение контекста из понятий
        
        Args:
            concepts: Список понятий
            chapter_title: Название главы (опционально)
            
        Returns:
            Строка с контекстом
        """
        context_parts = []
        
        # Добавляем информацию о каждом понятии
        for concept in concepts:
            name = concept.get('name', '')
            definition = concept.get('definition', '')
            example = concept.get('example', '')
            
            # Формируем текст понятия
            concept_text = f"Понятие: {name}\nОпределение: {definition}"
            
            if example:
                concept_text += f"\nПример: {example}"
            
            # Получаем связи с другими понятиями
            try:
                relations = self.neo4j_client.get_concept_connections(name)
                if relations:
                    relation_text = "Связи с другими понятиями:\n"
                    for relation in relations[:5]:  # Ограничиваем до 5 связей
                        rel_type = relation.get('type', '')
                        rel_concept = relation.get('concept', '')
                        if rel_type and rel_concept:
                            relation_text += f"- {rel_type} {rel_concept}\n"
                    
                    concept_text += f"\n{relation_text}"
            except Exception as e:
                logger.warning(f"Ошибка при получении связей для понятия {name}: {str(e)}")
            
            context_parts.append(concept_text)
        
        # Добавляем информацию о главе, если указана
        if chapter_title:
            try:
                chapter_info = self.neo4j_client.get_chapter_info(chapter_title)
                if chapter_info and 'main_ideas' in chapter_info:
                    chapter_text = f"Информация о главе '{chapter_title}':\n"
                    chapter_text += f"Основные идеи: {chapter_info['main_ideas']}\n"
                    context_parts.insert(0, chapter_text)
            except Exception as e:
                logger.warning(f"Ошибка при получении информации о главе {chapter_title}: {str(e)}")
        
        return "\n\n".join(context_parts)
    
    def log_interaction(self, student_id: str, question: str, answer: str, 
                      concept_name: Optional[str] = "Общая консультация",
                      chapter_title: Optional[str] = None) -> None:
        """
        Сохранение взаимодействия в базе данных
        
        Args:
            student_id: ID студента
            question: Вопрос студента
            answer: Ответ на вопрос
            concept_name: Название понятия (опционально)
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
            logger.info(f"Сохранено взаимодействие для студента {student_id} по понятию {concept_name}")
        except Exception as e:
            logger.error(f"Ошибка при логировании взаимодействия: {str(e)}")
    
    def close(self) -> None:
        """
        Закрытие всех соединений и ресурсов при завершении работы
        """
        try:
            if self.use_enhanced_search and self.enhanced_search:
                self.enhanced_search.close()
                logger.info("Соединение улучшенного поиска закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения улучшенного поиска: {str(e)}")

# Для поддержки существующего кода, сохраняем класс SystemicThinkingAssistant как алиас
SystemicThinkingAssistant = UnifiedAssistant 