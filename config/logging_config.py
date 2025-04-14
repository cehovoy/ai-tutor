"""
Конфигурация логирования для проекта ИИ-репетитор
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ai_tutor.config.settings import LOG_LEVEL, BASE_DIR


def setup_logging():
    """
    Настройка логирования проекта
    
    Returns:
        logger: Настроенный логгер
    """
    # Создание директории для логов если не существует
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Путь к файлу лога
    log_file = log_dir / "ai_tutor.log"
    
    # Уровень логирования
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    # Создание форматера для логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Настройка хендлера для файла (с ротацией)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Настройка хендлера для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Настройка логгера для проекта
    logger = logging.getLogger('ai_tutor')
    
    return logger
