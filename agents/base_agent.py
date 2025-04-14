"""
Базовый класс агента для CrewAI
"""
from typing import Dict, List, Any, Optional
import logging

from crewai import Agent

from ai_tutor.database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Базовый класс для всех агентов системы
    """
    
    def __init__(self, name: str, role: str, goal: str, prompt: str, 
                 backstory: Optional[str] = None, verbose: bool = False,
                 allow_delegation: bool = True, tools: Optional[List] = None):
        """
        Инициализация базового агента
        
        Args:
            name: Имя агента
            role: Роль агента
            goal: Цель агента
            prompt: Промпт для агента
            backstory: Предыстория агента (опционально)
            verbose: Режим подробного вывода
            allow_delegation: Разрешить делегирование задач
            tools: Список инструментов агента
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.prompt = prompt
        self.backstory = backstory or ""
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.tools = tools or []
        self.db_client = Neo4jClient()
        
    def create_agent(self) -> Agent:
        """
        Создание агента CrewAI
        
        Returns:
            Экземпляр агента CrewAI
        """
        return Agent(
            name=self.name,
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            verbose=self.verbose,
            allow_delegation=self.allow_delegation,
            tools=self.tools,
            llm_config={
                "temperature": 0.7,
                "request_timeout": 120
            }
        )
    
    def get_neo4j_client(self) -> Neo4jClient:
        """
        Получение клиента Neo4j
        
        Returns:
            Клиент Neo4j
        """
        return self.db_client
