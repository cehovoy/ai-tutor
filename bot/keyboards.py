"""
Модуль для работы с клавиатурами Telegram-бота
"""
from typing import List, Dict, Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ai_tutor.config.settings import CHAPTERS, TASK_TYPES, DIFFICULTY_LEVELS

# Префиксы для callback-данных
PREFIX_CHAPTER = "chapter:"
PREFIX_TASK_TYPE = "task_type:"
PREFIX_DIFFICULTY = "difficulty:"


def get_chapters_keyboard() -> InlineKeyboardMarkup:
    """
    Создает клавиатуру с главами курса
    
    Returns:
        InlineKeyboardMarkup: Клавиатура с главами
    """
    keyboard = []
    row = []
    
    for i, chapter in enumerate(CHAPTERS, 1):
        short_name = f"Глава {i}"
        # Используем короткий идентификатор для callback_data
        chapter_id = f"ch{i}"
        
        button = InlineKeyboardButton(
            short_name, 
            callback_data=f"{PREFIX_CHAPTER}{chapter_id}"
        )
        
        # Размещаем по 3 кнопки в ряд
        row.append(button)
        if len(row) == 3 or i == len(CHAPTERS):
            keyboard.append(row)
            row = []
    
    return InlineKeyboardMarkup(keyboard)


def get_task_types_keyboard() -> InlineKeyboardMarkup:
    """
    Создает клавиатуру с типами задач
    
    Returns:
        InlineKeyboardMarkup: Клавиатура с типами задач
    """
    keyboard = []
    
    for task_key, task_name in TASK_TYPES.items():
        keyboard.append([
            InlineKeyboardButton(
                task_name, 
                callback_data=f"{PREFIX_TASK_TYPE}{task_key}"
            )
        ])
    
    return InlineKeyboardMarkup(keyboard)


def get_difficulty_keyboard() -> InlineKeyboardMarkup:
    """
    Создает клавиатуру с уровнями сложности
    
    Returns:
        InlineKeyboardMarkup: Клавиатура с уровнями сложности
    """
    keyboard = []
    
    for difficulty_key, difficulty_name in DIFFICULTY_LEVELS.items():
        keyboard.append([
            InlineKeyboardButton(
                difficulty_name, 
                callback_data=f"{PREFIX_DIFFICULTY}{difficulty_key}"
            )
        ])
    
    return InlineKeyboardMarkup(keyboard)


def get_feedback_keyboard() -> InlineKeyboardMarkup:
    """
    Создает клавиатуру для выбора действий после получения обратной связи
    
    Returns:
        InlineKeyboardMarkup: Клавиатура с действиями
    """
    keyboard = [
        [
            InlineKeyboardButton("Новая задача", callback_data="new_task"),
            InlineKeyboardButton("Завершить", callback_data="end_session")
        ]
    ]
    
    return InlineKeyboardMarkup(keyboard)


def get_profile_keyboard() -> InlineKeyboardMarkup:
    """
    Создает клавиатуру для профиля пользователя
    
    Returns:
        InlineKeyboardMarkup: Клавиатура для профиля
    """
    keyboard = [
        [
            InlineKeyboardButton("Статистика", callback_data="show_stats"),
            InlineKeyboardButton("Достижения", callback_data="show_achievements")
        ],
        [
            InlineKeyboardButton("Изменить главу", callback_data="change_chapter"),
            InlineKeyboardButton("Назад", callback_data="back_to_main")
        ]
    ]
    
    return InlineKeyboardMarkup(keyboard)


def get_options_keyboard(options: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    """
    Создает клавиатуру с вариантами ответов для задач с множественным выбором
    
    Args:
        options: Список вариантов ответов
        
    Returns:
        InlineKeyboardMarkup: Клавиатура с вариантами ответов
    """
    keyboard = []
    
    for option in options:
        keyboard.append([
            InlineKeyboardButton(
                f"{option['label']}. {option['text'][:30]}...", 
                callback_data=f"option:{option['label']}"
            )
        ])
    
    return InlineKeyboardMarkup(keyboard)
