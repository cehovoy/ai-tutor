"""
Агент-адаптер задач
"""
from typing import Dict, List, Any, Optional
import logging
import json

from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ai_tutor.agents.base_agent import BaseAgent
from ai_tutor.agents.prompts.task_adapter_prompt import TASK_ADAPTER_PROMPT
from ai_tutor.database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GetStudentProgressInput(BaseModel):
    """
    Входные данные для получения прогресса студента
    """
    student_id: str = Field(description="ID студента")
    chapter_title: Optional[str] = Field(
        default=None, 
        description="Название главы курса (опционально)"
    )


class GetStudentProgressTool(BaseTool):
    """
    Инструмент для получения прогресса студента
    """
    name: str = "get_student_progress"
    description: str = "Получение информации о прогрессе студента по изучению понятий"
    args_schema: type[BaseModel] = GetStudentProgressInput
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Инициализация инструмента
        
        Args:
            neo4j_client: Клиент Neo4j
        """
        super().__init__()
        self.neo4j_client = neo4j_client
    
    def _run(self, student_id: str, chapter_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Запуск инструмента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы (опционально)
            
        Returns:
            Прогресс студента
        """
        try:
            progress = self.neo4j_client.get_student_progress(
                student_id=student_id,
                chapter_title=chapter_title
            )
            
            # Получаем статистику по главам
            chapter_stats = self.neo4j_client.get_student_chapter_stats(student_id)
            
            return {
                "progress": progress,
                "chapter_stats": chapter_stats
            }
        except Exception as e:
            logger.error(f"Ошибка при получении прогресса студента: {str(e)}")
            return {
                "error": f"Ошибка при получении прогресса студента: {str(e)}",
                "progress": [],
                "chapter_stats": []
            }


class SuggestDifficultyInput(BaseModel):
    """
    Входные данные для рекомендации уровня сложности
    """
    student_id: str = Field(description="ID студента")
    chapter_title: str = Field(description="Название главы курса")


class SuggestDifficultyTool(BaseTool):
    """
    Инструмент для рекомендации уровня сложности задач
    """
    name: str = "suggest_difficulty"
    description: str = "Предложение рекомендуемого уровня сложности для студента"
    args_schema: type[BaseModel] = SuggestDifficultyInput
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Инициализация инструмента
        
        Args:
            neo4j_client: Клиент Neo4j
        """
        super().__init__()
        self.neo4j_client = neo4j_client
    
    def _run(self, student_id: str, chapter_title: str) -> Dict[str, Any]:
        """
        Запуск инструмента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            
        Returns:
            Рекомендуемый уровень сложности
        """
        try:
            # Получаем рекомендованный уровень сложности
            difficulty = self.neo4j_client.suggest_difficulty_level(
                student_id=student_id,
                chapter_title=chapter_title
            )
            
            # Получаем статистику по главе
            progress = self.neo4j_client.get_student_progress(
                student_id=student_id,
                chapter_title=chapter_title
            )
            
            # Анализируем, с какими понятиями были проблемы
            problem_concepts = []
            strong_concepts = []
            
            for item in progress:
                if item.get('correct', 0) / max(item.get('attempts', 1), 1) < 0.5:
                    problem_concepts.append(item.get('concept_name'))
                elif item.get('correct', 0) / max(item.get('attempts', 1), 1) > 0.8:
                    strong_concepts.append(item.get('concept_name'))
            
            return {
                "recommended_difficulty": difficulty,
                "problem_concepts": problem_concepts,
                "strong_concepts": strong_concepts
            }
        except Exception as e:
            logger.error(f"Ошибка при определении рекомендуемого уровня сложности: {str(e)}")
            return {
                "error": f"Ошибка при определении рекомендуемого уровня сложности: {str(e)}",
                "recommended_difficulty": "basic",  # По умолчанию базовый уровень
                "problem_concepts": [],
                "strong_concepts": []
            }


class TaskAdapterAgent(BaseAgent):
    """
    Агент-адаптер задач
    """
    
    def __init__(self, verbose: bool = False):
        """
        Инициализация агента
        
        Args:
            verbose: Режим подробного вывода
        """
        name = "Адаптер Задач"
        role = "Специалист по адаптации сложности учебных задач"
        goal = "Подбирать оптимальный уровень сложности задач для каждого студента"
        backstory = """
        Я - опытный педагог-методист, специализирующийся на адаптивном обучении. 
        Моя задача - анализировать прогресс студентов и предлагать оптимальный уровень 
        сложности задач, чтобы обеспечить эффективное обучение без перегрузки или недогрузки.
        """
        
        # Создаем инструменты агента
        tools = [
            GetStudentProgressTool(self.get_neo4j_client()),
            SuggestDifficultyTool(self.get_neo4j_client())
        ]
        
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            prompt=TASK_ADAPTER_PROMPT,
            backstory=backstory,
            verbose=verbose,
            allow_delegation=True,
            tools=tools
        )
    
    def create_adaptation_task(self, student_id: str, chapter_title: str) -> Task:
        """
        Создание задачи для агента
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            
        Returns:
            Задача для выполнения агентом
        """
        task_description = f"""
        Проанализируй прогресс студента с ID "{student_id}" по главе "{chapter_title}" и предложи 
        оптимальный уровень сложности и тип задач для дальнейшего обучения.
        
        Для этого:
        1. Получи информацию о прогрессе студента 
        2. Определи рекомендуемый уровень сложности
        3. Проанализируй, с какими понятиями у студента возникли трудности
        4. Определи, какой тип задач (с множественным выбором или творческие) лучше подходит студенту
        5. Объясни свое решение
        
        Результат представь в формате JSON:
        ```json
        {{
          "student_id": "{student_id}",
          "chapter_title": "{chapter_title}",
          "recommended_difficulty": "basic/advanced",
          "recommended_task_type": "multiple_choice/creative",
          "reasoning": "Объяснение, почему данная рекомендация предложена",
          "problem_concepts": ["Понятие 1", "Понятие 2"],
          "strong_concepts": ["Понятие 3", "Понятие 4"]
        }}
        ```
        """
        
        return Task(
            description=task_description,
            expected_output="Рекомендация по адаптации сложности в формате JSON",
            agent=self.create_agent()
        )
