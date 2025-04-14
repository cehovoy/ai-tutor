"""
Агент-генератор задач
"""
from typing import Dict, List, Any, Optional
import logging
import asyncio
from functools import partial

from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ai_tutor.agents.base_agent import BaseAgent
from ai_tutor.agents.prompts.task_generator_prompt import TASK_GENERATOR_PROMPT
from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.api.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)


class GenerateTaskInput(BaseModel):
    """
    Входные данные для генерации задачи
    """
    chapter_title: str = Field(description="Название главы курса")
    task_type: str = Field(description="Тип задачи (multiple_choice или creative)")
    difficulty: str = Field(description="Уровень сложности задачи (basic или advanced)")
    excluded_concepts: Optional[List[str]] = Field(
        default=None, 
        description="Список понятий для исключения из выборки"
    )


class GenerateTaskTool(BaseTool):
    """
    Инструмент для генерации задачи на основе случайного понятия из главы
    """
    name: str = "generate_task"
    description: str = "Генерация задачи на основе случайного понятия из заданной главы"
    args_schema: type[BaseModel] = GenerateTaskInput
    
    def __init__(self, neo4j_client: Neo4jClient, openrouter_client: OpenRouterClient):
        """
        Инициализация инструмента
        
        Args:
            neo4j_client: Клиент Neo4j
            openrouter_client: Клиент для работы с OpenRouter API
        """
        super().__init__()
        self.neo4j_client = neo4j_client
        self.openrouter_client = openrouter_client
    
    def _run(self, chapter_title: str, task_type: str, difficulty: str,
            excluded_concepts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Запуск инструмента
        
        Args:
            chapter_title: Название главы
            task_type: Тип задачи (multiple_choice или creative)
            difficulty: Уровень сложности (basic или advanced)
            excluded_concepts: Список понятий для исключения
            
        Returns:
            Сгенерированная задача
        """
        excluded = excluded_concepts or []
        
        # Получаем случайное понятие из главы с учетом сложности
        concept = self.neo4j_client.get_random_concept_by_chapter_and_difficulty(
            chapter_title=chapter_title,
            difficulty=difficulty, 
            excluded_concepts=excluded
        )
        
        if not concept:
            logger.error(f"Не удалось найти понятия в главе {chapter_title} для уровня {difficulty}")
            return {
                "error": f"Не удалось найти понятия в главе {chapter_title} для уровня {difficulty}"
            }
        
        # Получаем связанные понятия
        related_concepts = self.neo4j_client.get_related_concepts(
            concept_name=concept["name"],
            limit=5
        )
        
        # Запускаем асинхронную генерацию задачи
        loop = asyncio.get_event_loop()
        task = loop.run_until_complete(
            self.openrouter_client.generate_task(
                concept=concept,
                related_concepts=related_concepts,
                task_type=task_type,
                difficulty=difficulty
            )
        )
        
        return task


class TaskGeneratorAgent(BaseAgent):
    """
    Агент-генератор задач
    """
    
    def __init__(self, openrouter_client: OpenRouterClient, verbose: bool = False):
        """
        Инициализация агента
        
        Args:
            openrouter_client: Клиент для работы с OpenRouter API
            verbose: Режим подробного вывода
        """
        name = "Генератор Задач"
        role = "Специалист по генерации учебных задач"
        goal = "Создавать учебные задачи на основе понятий из графа знаний"
        backstory = """
        Я - опытный педагог с глубоким пониманием когнитивных процессов обучения. 
        Я специализируюсь на разработке задач, которые эффективно проверяют и углубляют 
        понимание студентами сложных понятий. Я умею адаптировать сложность задач 
        под разные уровни подготовки студентов.
        """
        
        self.db_client = None
        self.openrouter_client = openrouter_client
        
        # Создаем инструменты агента
        tools = [
            GenerateTaskTool(self.get_neo4j_client(), openrouter_client)
        ]
        
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            prompt=TASK_GENERATOR_PROMPT,
            backstory=backstory,
            verbose=verbose,
            allow_delegation=True,
            tools=tools
        )
    
    def get_neo4j_client(self) -> Neo4jClient:
        """
        Получение клиента Neo4j
        
        Returns:
            Клиент Neo4j
        """
        if not self.db_client:
            self.db_client = Neo4jClient()
        return self.db_client
    
    def create_generation_task(self, chapter_title: str, task_type: str, 
                             difficulty: str, excluded_concepts: Optional[List[str]] = None) -> Task:
        """
        Создание задачи для агента
        
        Args:
            chapter_title: Название главы
            task_type: Тип задачи
            difficulty: Уровень сложности
            excluded_concepts: Список понятий для исключения
            
        Returns:
            Задача для выполнения агентом
        """
        excluded = excluded_concepts or []
        excluded_str = ", ".join(excluded) if excluded else "нет"
        
        task_description = f"""
        Создай учебную задачу для проверки знаний по главе "{chapter_title}".
        
        Тип задачи: {task_type} 
        ("multiple_choice" - с множественным выбором, "creative" - творческая)
        
        Уровень сложности: {difficulty} 
        ("basic" - базовый, доступный со средним образованием; "advanced" - продвинутый, требующий высшего образования)
        
        Исключённые понятия: {excluded_str}
        
        Задача должна проверять понимание понятий из данной главы, а не просто запоминание определений.
        Убедись, что уровень сложности соответствует указанному.
        """
        
        return Task(
            description=task_description,
            expected_output="Задача в формате JSON, соответствующая заданным требованиям.",
            agent=self.create_agent()
        )
