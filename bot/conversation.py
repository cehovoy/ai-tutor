"""
Модуль для управления диалогами в Telegram-боте
"""
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from ai_tutor.config.constants import MAX_HISTORY_MESSAGES
from ai_tutor.database.models import Student, Task

logger = logging.getLogger(__name__)


class Conversation:
    """
    Класс для управления диалогом между студентом и ботом
    """
    
    def __init__(self, student_id: int):
        """
        Инициализация диалога
        
        Args:
            student_id: ID студента в Telegram
        """
        self.student_id = student_id
        self.history: List[Dict[str, Any]] = []
        self.current_task: Optional[Dict[str, Any]] = None
        self.current_state: str = "idle"
        self.last_message_time: datetime = datetime.now()
    
    def add_message(self, role: str, text: str, message_id: Optional[int] = None) -> None:
        """
        Добавление сообщения в историю диалога
        
        Args:
            role: Роль отправителя ('student' или 'bot')
            text: Текст сообщения
            message_id: ID сообщения в Telegram
        """
        message = {
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "message_id": message_id
        }
        
        self.history.append(message)
        self.last_message_time = datetime.now()
        
        # Ограничиваем размер истории
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]
    
    def set_current_task(self, task: Dict[str, Any]) -> None:
        """
        Установка текущей задачи
        
        Args:
            task: Задача
        """
        self.current_task = task
        self.current_state = "waiting_for_answer"
    
    def clear_current_task(self) -> None:
        """
        Очистка текущей задачи
        """
        self.current_task = None
        self.current_state = "idle"
    
    def get_last_task(self) -> Optional[Dict[str, Any]]:
        """
        Получение последней задачи из истории
        
        Returns:
            Последняя задача или None, если истории нет
        """
        if self.current_task:
            return self.current_task
            
        # Ищем задачу в истории сообщений
        for message in reversed(self.history):
            if message.get("role") == "bot" and "task" in message:
                return message["task"]
        
        return None
    
    def format_task_for_display(self) -> str:
        """
        Форматирование задачи для отображения
        
        Returns:
            Отформатированный текст задачи
        """
        if not self.current_task:
            return "Нет активной задачи."
        
        task_type = self.current_task.get("task_type", "")
        difficulty = self.current_task.get("difficulty", "")
        concept = self.current_task.get("concept_name", "")
        question = self.current_task.get("question", "")
        
        # Преобразование кодов типа задачи и сложности в удобный для чтения формат
        task_type_display = {
            "template": "Задача с выбором ответа",
            "multiple_choice": "Задача с выбором ответа",
            "creative": "Творческая задача"
        }.get(task_type, task_type)
        
        difficulty_display = {
            "standard": "Базовый уровень", 
            "basic": "Базовый уровень",
            "advanced": "Продвинутый уровень"
        }.get(difficulty, difficulty)
        
        # Форматирование текста задачи
        formatted_text = f"📚 *Задача по теме: {concept}*\n\n"
        formatted_text += f"{question}\n\n"
        
        # Добавляем варианты ответов для задачи с выбором ответа
        if "options" in self.current_task:
            formatted_text += "\n*Варианты ответов:*\n"
            for option in self.current_task["options"]:
                # Используем display_label (цифра) вместо label (буква)
                display_label = option.get('display_label', option.get('label', ''))
                formatted_text += f"\n*{display_label}.* {option['text']}"
        
        # Добавление информации о типе и сложности
        formatted_text += f"\n\n_Тип: {task_type_display} | Сложность: {difficulty_display}_"
        
        return formatted_text
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование диалога в словарь для сохранения
        
        Returns:
            Словарь с данными диалога
        """
        return {
            "student_id": self.student_id,
            "history": self.history,
            "current_task": self.current_task,
            "current_state": self.current_state,
            "last_message_time": self.last_message_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        Создание объекта из словаря
        
        Args:
            data: Словарь с данными диалога
            
        Returns:
            Объект Conversation
        """
        conversation = cls(data.get("student_id", 0))
        conversation.history = data.get("history", [])
        conversation.current_task = data.get("current_task")
        conversation.current_state = data.get("current_state", "idle")
        
        last_message_time_str = data.get("last_message_time")
        if last_message_time_str:
            try:
                conversation.last_message_time = datetime.fromisoformat(last_message_time_str)
            except ValueError:
                pass
        
        return conversation


# Хранилище диалогов в памяти
active_conversations: Dict[int, Conversation] = {}


def get_conversation(student_id: int) -> Conversation:
    """
    Получение или создание диалога для студента
    
    Args:
        student_id: ID студента в Telegram
        
    Returns:
        Объект Conversation
    """
    if student_id not in active_conversations:
        active_conversations[student_id] = Conversation(student_id)
    
    return active_conversations[student_id]


def save_conversation(conversation: Conversation) -> None:
    """
    Сохранение диалога в хранилище
    
    Args:
        conversation: Объект Conversation
    """
    active_conversations[conversation.student_id] = conversation
