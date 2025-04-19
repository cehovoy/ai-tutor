"""
Модуль, содержащий различных ассистентов для AI-репетитора
"""

# Используем прямые импорты без префикса
from agents.crew import TutorCrew
from agents.unified_assistant import UnifiedAssistant, SystemicThinkingAssistant

__all__ = ['TutorCrew', 'UnifiedAssistant', 'SystemicThinkingAssistant']
