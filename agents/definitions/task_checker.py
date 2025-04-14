"""
Агент-проверяльщик ответов студентов
"""
from typing import Dict, List, Any, Optional
import logging
import asyncio
import json

from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ai_tutor.agents.base_agent import BaseAgent
from ai_tutor.agents.prompts.task_checker_prompt import TASK_CHECKER_PROMPT
from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.api.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


class CheckMultipleChoiceAnswerInput(BaseModel):
    """
    Входные данные для проверки ответа с множественным выбором
    """
    task: Dict[str, Any] = Field(description="Задача с множественным выбором")
    student_answer: str = Field(description="Ответ студента (1, 2, 3 или 4)")


class CheckMultipleChoiceAnswerTool(BaseTool):
    """
    Инструмент для проверки ответа с множественным выбором
    """
    name: str = "check_multiple_choice_answer"
    description: str = "Проверка ответа студента на задачу с множественным выбором"
    args_schema: type[BaseModel] = CheckMultipleChoiceAnswerInput
    
    def _run(self, task: Dict[str, Any], student_answer: str) -> Dict[str, Any]:
        """
        Запуск инструмента
        
        Args:
            task: Задача с множественным выбором
            student_answer: Ответ студента (1, 2, 3 или 4)
            
        Returns:
            Результат проверки
        """
        try:
            # Проверяем формат ответа
            student_choice = student_answer.strip()
            if student_choice not in ["1", "2", "3", "4"]:
                return {
                    "is_correct": False,
                    "explanation": f"Ответ '{student_choice}' не соответствует формату задачи с множественным выбором (1, 2, 3, 4)."
                }
            
            # Находим правильный вариант
            correct_option = next(
                (opt for opt in task['options'] if opt['is_correct']), None
            )
            
            if not correct_option:
                return {
                    "is_correct": False,
                    "explanation": "Не удалось определить правильный ответ в задаче."
                }
            
            # Проверяем ответ
            is_correct = student_choice == correct_option['label']
            
            # Находим выбранный вариант
            selected_option = next(
                (opt for opt in task['options'] if opt['label'] == student_choice), None
            )
            
            if not selected_option:
                return {
                    "is_correct": False,
                    "explanation": f"Ответ '{student_choice}' не соответствует ни одному из вариантов."
                }
            
            # Формируем объяснение
            if is_correct:
                explanation = f"Верно! {selected_option['explanation']}"
            else:
                explanation = f"Неверно. {selected_option['explanation']} Правильный ответ - {correct_option['label']}: {correct_option['explanation']}"
            
            # Формируем рекомендации
            recommendations = []
            if not is_correct:
                recommendations.append(f"Изучите понятие '{task['concept_name']}' более внимательно.")
            
            return {
                "is_correct": is_correct,
                "explanation": explanation,
                "correct_answer": correct_option['label'],
                "correct_explanation": correct_option['explanation'],
                "recommendations": recommendations
            }
        except Exception as e:
            logger.error(f"Ошибка при проверке ответа с множественным выбором: {str(e)}")
            return {
                "is_correct": False,
                "explanation": f"Произошла ошибка при проверке ответа: {str(e)}",
                "recommendations": ["Попробуйте еще раз."]
            }


class CheckCreativeAnswerInput(BaseModel):
    """
    Входные данные для проверки творческого ответа
    """
    task: Dict[str, Any] = Field(description="Творческая задача")
    student_answer: str = Field(description="Ответ студента в виде текста")
    concept: Dict[str, Any] = Field(description="Понятие из графа знаний")


class CheckCreativeAnswerTool(BaseTool):
    """
    Инструмент для проверки творческого ответа
    """
    name: str = "check_creative_answer"
    description: str = "Проверка творческого ответа студента с использованием AI-модели"
    args_schema: type[BaseModel] = CheckCreativeAnswerInput
    
    def __init__(self, openrouter_client: OpenRouterClient):
        """
        Инициализация инструмента
        
        Args:
            openrouter_client: Клиент для работы с OpenRouter API
        """
        super().__init__()
        self.openrouter_client = openrouter_client
    
    def _run(self, task: Dict[str, Any], student_answer: str, concept: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск инструмента
        
        Args:
            task: Творческая задача
            student_answer: Ответ студента
            concept: Понятие из графа знаний
            
        Returns:
            Результат проверки
        """
        try:
            # Запускаем асинхронную проверку ответа
            loop = asyncio.get_event_loop()
            check_result = loop.run_until_complete(
                self.openrouter_client.check_answer(
                    task=task,
                    student_answer=student_answer,
                    concept=concept
                )
            )
            
            # Добавляем рекомендации
            if 'recommendations' not in check_result:
                check_result['recommendations'] = []
            
            # Если ответ правильный, добавляем подтверждение из курса
            if check_result.get('is_correct', False):
                # Получаем определение понятия из курса
                course_definition = concept.get('definition', '')
                if course_definition:
                    check_result['explanation'] = f"Верно! Как сказано в курсе: '{course_definition}'"
                
                # Добавляем пример из курса, если он есть
                course_example = concept.get('example', '')
                if course_example:
                    check_result['explanation'] += f"\n\nПример из курса: {course_example}"
            else:
                check_result['recommendations'].append(f"Изучите понятие '{task['concept_name']}' более внимательно.")
                check_result['recommendations'].append(f"Обратите внимание на критерии оценки: {', '.join(task['criteria'])}")
            
            return check_result
        except Exception as e:
            logger.error(f"Ошибка при проверке творческого ответа: {str(e)}")
            return {
                "is_correct": False,
                "score": 0,
                "explanation": f"Произошла ошибка при проверке ответа: {str(e)}",
                "feedback": "Попробуйте еще раз. Убедитесь, что ваш ответ соответствует заданию.",
                "recommendations": ["Попробуйте еще раз."]
            }


class GetConceptInput(BaseModel):
    """
    Входные данные для получения понятия
    """
    concept_name: str = Field(description="Название понятия")


class GetConceptTool(BaseTool):
    """
    Инструмент для получения понятия из графа знаний
    """
    name: str = "get_concept"
    description: str = "Получение понятия из графа знаний"
    args_schema: type[BaseModel] = GetConceptInput
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Инициализация инструмента
        
        Args:
            neo4j_client: Клиент Neo4j
        """
        super().__init__()
        self.neo4j_client = neo4j_client
    
    def _run(self, concept_name: str) -> Dict[str, Any]:
        """
        Запуск инструмента
        
        Args:
            concept_name: Название понятия
            
        Returns:
            Понятие со связями
        """
        try:
            result = self.neo4j_client.get_concept_with_relations(concept_name)
            if not result:
                return {"error": f"Понятие '{concept_name}' не найдено в базе данных."}
            return result
        except Exception as e:
            logger.error(f"Ошибка при получении понятия: {str(e)}")
            return {"error": f"Ошибка при получении понятия: {str(e)}"}


class UpdateStudentProgressInput(BaseModel):
    """
    Входные данные для обновления прогресса студента
    """
    student_id: str = Field(description="ID студента")
    chapter_title: str = Field(description="Название главы")
    concept_name: str = Field(description="Название понятия")
    correct: bool = Field(description="Правильность ответа")
    difficulty: str = Field(description="Уровень сложности")


class UpdateStudentProgressTool(BaseTool):
    """
    Инструмент для обновления прогресса студента
    """
    name: str = "update_student_progress"
    description: str = "Обновление прогресса студента после проверки ответа"
    args_schema: type[BaseModel] = UpdateStudentProgressInput
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Инициализация инструмента
        
        Args:
            neo4j_client: Клиент Neo4j
        """
        super().__init__()
        self.neo4j_client = neo4j_client
    
    def _run(self, student_id: str, chapter_title: str, concept_name: str, 
            correct: bool, difficulty: str) -> Dict[str, Any]:
        """
        Запуск инструмента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            concept_name: Название понятия
            correct: Правильность ответа
            difficulty: Уровень сложности
            
        Returns:
            Результат обновления
        """
        try:
            self.neo4j_client.update_student_progress(
                student_id=student_id,
                chapter_title=chapter_title,
                concept_name=concept_name,
                correct=correct,
                difficulty=difficulty
            )
            
            return {
                "success": True,
                "message": f"Прогресс студента {student_id} успешно обновлен."
            }
        except Exception as e:
            logger.error(f"Ошибка при обновлении прогресса студента: {str(e)}")
            return {
                "success": False,
                "error": f"Ошибка при обновлении прогресса студента: {str(e)}"
            }


class TaskCheckerAgent(BaseAgent):
    """
    Агент-проверяльщик ответов
    """
    
    def __init__(self, openrouter_client: OpenRouterClient, verbose: bool = False):
        """
        Инициализация агента
        
        Args:
            openrouter_client: Клиент для работы с OpenRouter API
            verbose: Режим подробного вывода
        """
        name = "Проверяльщик"
        role = "Специалист по проверке ответов и предоставлению обратной связи"
        goal = "Оценивать ответы студентов и предоставлять конструктивную обратную связь"
        backstory = """
        Я - опытный преподаватель и эксперт в педагогической оценке. 
        Моя специализация - объективная проверка ответов студентов и предоставление 
        конструктивной обратной связи, которая помогает улучшить понимание материала.
        Я внимателен к деталям и умею выявлять как сильные, так и слабые стороны в ответах.
        """
        
        # Создаем инструменты агента
        tools = [
            CheckMultipleChoiceAnswerTool(),
            CheckCreativeAnswerTool(openrouter_client),
            GetConceptTool(self.get_neo4j_client()),
            UpdateStudentProgressTool(self.get_neo4j_client())
        ]
        
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            prompt=TASK_CHECKER_PROMPT,
            backstory=backstory,
            verbose=verbose,
            allow_delegation=True,
            tools=tools
        )
    
    def create_checking_task(self, student_id: str, chapter_title: str, 
                           task: Dict[str, Any], student_answer: str) -> Task:
        """
        Создание задачи для агента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            task: Задача
            student_answer: Ответ студента
            
        Returns:
            Задача для выполнения агентом
        """
        # Преобразуем задачу в JSON-строку для передачи в описание задачи
        task_json = json.dumps(task, ensure_ascii=False, indent=2)
        
        task_description = f"""
        Проверь ответ студента с ID "{student_id}" на задачу из главы "{chapter_title}".
        
        Задача:
        ```json
        {task_json}
        ```
        
        Ответ студента: {student_answer}
        
        Для проверки:
        1. Получи детальную информацию о понятии "{task['concept_name']}" из графа знаний
        2. Проверь ответ студента в зависимости от типа задачи:
           - Для задачи с множественным выбором ("multiple_choice") используй инструмент check_multiple_choice_answer
           - Для творческой задачи ("creative") используй инструмент check_creative_answer
        3. Предоставь подробную обратную связь с объяснением
        4. Дай рекомендации по дальнейшему обучению
        5. Обнови прогресс студента в базе данных
        
        Представь результат в формате JSON в зависимости от типа задачи.
        """
        
        return Task(
            description=task_description,
            expected_output="Результат проверки ответа в формате JSON",
            agent=self.create_agent()
        )
