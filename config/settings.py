"""
Настройки проекта ИИ-репетитора
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
load_dotenv()

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Настройки Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

# Настройки Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")

# Настройки OpenRouter API для доступа к LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "x-ai/grok-2-1212")
GROK_MODEL = "x-ai/grok-2-1212"  # Явное указание модели Grok

# Настройки LLM
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "2000"))
REQUEST_TIMEOUT = int(os.getenv("MODEL_REQUEST_TIMEOUT", "120"))

# Настройки CrewAI
MAX_CONSECUTIVE_AUTO_REPLIES = int(os.getenv("MAX_CONSECUTIVE_AUTO_REPLIES", "3"))

# Настройки логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Настройки курса
COURSE_NAME = "Системное саморазвитие"
CHAPTERS = [
    "Глава 1: Физический мир и ментальное пространство",
    "Глава 2: Обучение и время",
    "Глава 3: Собранность и внимание",
    "Глава 4: Недостаток ресурсов для выполнения работы",
    "Глава 5: Системный подход в психологии личности",
    "Глава 6: Роль, ролевое мастерство и метод",
    "Глава 7: Инженерия, менеджмент, предпринимательство",
    "Глава 8: Личность и агент: человек и ИИ",
    "Глава 9: Личная траектория развития"
]

# Типы задач
TASK_TYPES = {
    "template": "Шаблонная задача",
    "creative": "Творческая задача"
}

# Уровни сложности
DIFFICULTY_LEVELS = {
    "standard": "Стандартный уровень",
    "advanced": "Продвинутый уровень"
}
