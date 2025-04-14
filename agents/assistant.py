"""
Модуль агента-помощника для ответов на вопросы студентов на основе графа знаний курса
"""
import logging
import asyncio
import traceback
import re
from typing import Dict, List, Any, Optional, Set

from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.api.openrouter import OpenRouterClient
from ai_tutor.config.settings import COURSE_NAME

logger = logging.getLogger(__name__)


class CourseAssistant:
    """
    Агент-помощник для ответов на вопросы студентов на основе графа знаний курса
    """
    
    def __init__(self, neo4j_client: Neo4jClient, openrouter_client: OpenRouterClient):
        """
        Инициализация агента-помощника
        
        Args:
            neo4j_client: Клиент для работы с Neo4j
            openrouter_client: Клиент для работы с OpenRouter API
        """
        self.neo4j_client = neo4j_client
        self.openrouter_client = openrouter_client
    
    async def answer_question(self, question: str, chapter_title: Optional[str] = None) -> str:
        """
        Асинхронный метод для ответа на вопрос студента
        
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
    
    def answer_question_sync(self, question: str, chapter_title: Optional[str] = None) -> str:
        """
        Синхронная обертка для ответа на вопрос студента
        
        Args:
            question: Вопрос студента
            chapter_title: Название главы (опционально)
            
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
                    self.answer_question(question, chapter_title)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Ошибка в синхронной обертке answer_question_sync: {str(e)}\n{traceback.format_exc()}")
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
    
    def log_interaction(self, student_id: str, question: str, answer: str, chapter_title: Optional[str] = None) -> None:
        """
        Сохранение взаимодействия в базе данных
        
        Args:
            student_id: ID студента
            question: Вопрос студента
            answer: Ответ на вопрос
            chapter_title: Название главы (опционально)
        """
        try:
            # Вызываем метод Neo4j клиента для логирования
            self.neo4j_client.save_assistant_interaction(student_id, question, answer, chapter_title)
            logger.info(f"Сохранено взаимодействие с помощником для студента {student_id}")
        except Exception as e:
            logger.error(f"Ошибка при логировании взаимодействия: {str(e)}") 