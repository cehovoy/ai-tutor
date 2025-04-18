"""
Обработчики команд и сообщений для Telegram-бота
"""
import logging
from typing import Dict, Any, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler, CommandHandler, CallbackQueryHandler, MessageHandler, filters

from ai_tutor.bot.conversation import get_conversation, save_conversation
from ai_tutor.bot.keyboards import (
    get_chapters_keyboard, get_task_types_keyboard, 
    get_difficulty_keyboard, get_feedback_keyboard
)
from ai_tutor.config.settings import TELEGRAM_TOKEN, CHAPTERS, TASK_TYPES, DIFFICULTY_LEVELS
from ai_tutor.config.constants import MESSAGES
from ai_tutor.database.models import Student, Task
from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.api.openrouter import OpenRouterClient

# Состояния диалога
SELECTING_CHAPTER, SELECTING_TASK_TYPE, SELECTING_DIFFICULTY, WAITING_FOR_ANSWER, SHOW_FEEDBACK = range(5)

# Префиксы для callback-данных
PREFIX_CHAPTER = "chapter:"
PREFIX_TASK_TYPE = "task_type:"
PREFIX_DIFFICULTY = "difficulty:"

logger = logging.getLogger(__name__)

# Инициализация клиентов
neo4j_client = Neo4jClient()
openrouter_client = OpenRouterClient()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /start
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    user = update.effective_user
    telegram_id = user.id
    
    # Получаем или создаем диалог
    conversation = get_conversation(telegram_id)
    
    # Обновляем информацию о пользователе в базе данных
    student = Student(
        telegram_id=telegram_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name
    )
    
    try:
        # Сохраняем студента в базу данных
        neo4j_client.save_student(student)
        
        # Отправляем приветственное сообщение с новым текстом
        await update.message.reply_text(
            f"ИИ-репетитор Школы Системного Менеджмента (ШСМ). Вам будет предложено два вида задач: \n"
            f"1) С вариантами ответов\n"
            f"2) Творческие (следует применить практику мышления письмом)\n\n"
            f"Главная цель репетитора - повысить беглость в использовании понятий.\n" 
            f"Проект разработан и поддерживается волонтёрами ШСМ.",
            parse_mode="Markdown"
        )
        
        # Создаем клавиатуру с кнопками основных действий
        keyboard = [
            [
                InlineKeyboardButton("Начать", callback_data="task"),
                InlineKeyboardButton("Случайная задача", callback_data="random_task")
            ],
            [
                InlineKeyboardButton("Сменить главу", callback_data="change_chapter"),
                InlineKeyboardButton("Вопрос консультанту", callback_data="consultant")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Выберите действие:",
            reply_markup=reply_markup
        )
        
        return ConversationHandler.END
    
    except Exception as e:
        logger.error(f"Ошибка при обработке команды /start: {e}")
        await update.message.reply_text(
            MESSAGES['error']
        )
        return ConversationHandler.END


async def consultant_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /consultant
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    user = update.effective_user
    
    # Сохраняем состояние, что пользователь в режиме консультации
    if 'user_data' not in context:
        context.user_data = {}
        
    context.user_data['consultation_mode'] = True
    
    await update.message.reply_text(
        f"👨‍🏫 *Режим консультации активирован*\n\n"
        f"{user.first_name}, я готов ответить на ваши вопросы по курсу системного мышления. "
        f"Вы можете спрашивать о любых понятиях, главах курса или связях между ними.\n\n"
        f"Просто задайте ваш вопрос, и я постараюсь дать подробный ответ, используя "
        f"официальные материалы курса и другие релевантные источники.\n\n"
        f"Для выхода из режима консультации введите /cancel",
        parse_mode="Markdown"
    )
    
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /help
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    await update.message.reply_text(
        "Доступные команды:\n"
        "/task - начать решать задачи\n"
        "/help - показать эту справку\n"
        "/cancel - отменить текущую операцию\n"
        "/profile - показать профиль студента\n"
        "/stats - показать статистику обучения\n\n"
        "Курс 'Системное саморазвитие' состоит из 9 глав. "
        "Вы можете выбрать главу, тип задачи и уровень сложности."
    )
    
    return ConversationHandler.END


async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /task
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    # Создаем клавиатуру с выбором глав
    reply_markup = get_chapters_keyboard()
    
    await update.message.reply_text(
        "Выберите главу курса:",
        reply_markup=reply_markup
    )
    
    return SELECTING_CHAPTER


async def select_chapter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик выбора главы
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Извлекаем выбранную главу (короткий идентификатор)
    chapter_id = query.data.replace(PREFIX_CHAPTER, "")
    
    # Получаем номер главы из идентификатора (например, из "ch1" получаем 1)
    try:
        chapter_number = int(chapter_id.replace("ch", ""))
        chapter = CHAPTERS[chapter_number - 1]  # -1 так как нумерация начинается с 1
    except (ValueError, IndexError):
        # В случае ошибки используем сам идентификатор
        chapter = chapter_id
    
    # Сохраняем выбор в контексте
    context.user_data["chapter"] = chapter
    
    # Создаем клавиатуру с выбором типа задачи
    reply_markup = get_task_types_keyboard()
    
    await query.edit_message_text(
        f"Выбрана глава: {chapter}\n\nВыберите тип задачи:",
        reply_markup=reply_markup
    )
    
    return SELECTING_TASK_TYPE


async def select_task_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик выбора типа задачи
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Извлекаем выбранный тип задачи
    task_type = query.data.replace(PREFIX_TASK_TYPE, "")
    task_type_name = TASK_TYPES[task_type]
    
    # Сохраняем выбор в контексте
    context.user_data["task_type"] = task_type
    
    # Создаем клавиатуру с выбором сложности
    reply_markup = get_difficulty_keyboard()
    
    await query.edit_message_text(
        f"Выбрана глава: {context.user_data['chapter']}\n"
        f"Выбран тип задачи: {task_type_name}\n\n"
        "Выберите уровень сложности:",
        reply_markup=reply_markup
    )
    
    return SELECTING_DIFFICULTY


async def select_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик выбора сложности
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Извлекаем выбранную сложность
    difficulty = query.data.replace(PREFIX_DIFFICULTY, "")
    difficulty_name = DIFFICULTY_LEVELS[difficulty]
    
    # Сохраняем выбор в контексте
    context.user_data["difficulty"] = difficulty
    
    # Получаем данные о выбранных параметрах
    chapter = context.user_data["chapter"]
    task_type = context.user_data["task_type"]
    
    # Сообщаем пользователю, что генерируем задачу
    await query.edit_message_text(
        f"Генерирую задачу для вас...\n\n"
        f"Глава: {chapter}\n"
        f"Тип задачи: {TASK_TYPES[task_type]}\n"
        f"Сложность: {difficulty_name}"
    )
    
    try:
        # Получаем понятия по главе
        concepts = neo4j_client.get_concepts_by_chapter(chapter)
        
        if not concepts:
            await query.edit_message_text(
                f"К сожалению, для главы '{chapter}' пока нет понятий в базе знаний.\n"
                "Попробуйте выбрать другую главу."
            )
            return ConversationHandler.END
        
        # Выбираем случайное понятие из списка
        import random
        concept = random.choice(concepts)
        
        # Получаем связанные понятия
        related_concepts = neo4j_client.get_related_concepts(concept.get('name', ''), chapter)
        
        # Генерируем задачу
        task = await openrouter_client.generate_task(
            concept, 
            related_concepts, 
            task_type, 
            difficulty
        )
        
        # Получаем диалог пользователя
        user = update.effective_user
        conversation = get_conversation(user.id)
        
        # Устанавливаем текущую задачу
        conversation.set_current_task(task)
        
        # Если это задача с вариантами ответов, обновляем метки
        if (task_type == "multiple_choice" or task_type == "template") and "options" in task:
            options = task["options"]
            # Перемешиваем варианты ответов
            random.shuffle(options)
            
            # Сохраняем буквенные метки для API и добавляем цифровые для отображения
            for i, option in enumerate(options):
                # Сохраняем оригинальную букву
                letter_label = chr(65 + i)  # A, B, C, D...
                option['label'] = letter_label  # Оригинальная буквенная метка для API
                option['display_label'] = str(i + 1)  # Цифровая метка для отображения (1, 2, 3...)
            
        # Сохраняем обновленную задачу
        conversation.set_current_task(task)
        save_conversation(conversation)
        
        # Форматируем задачу для отображения
        task_message = conversation.format_task_for_display()
        
        # Добавляем клавиатуру с подсказками или кнопками выбора ответа
        if (task_type == "multiple_choice" or task_type == "template") and "options" in task:
            options = task["options"]
            # Создаем клавиатуру с вариантами ответов
            keyboard = []
            row = []
            
            # Добавляем кнопки для каждого варианта
            for i, option in enumerate(options):
                # Сохраняем оригинальную букву
                letter_label = chr(65 + i)  # A, B, C, D...
                option['label'] = letter_label  # Оригинальная буквенная метка для API
                option['display_label'] = str(i + 1)  # Цифровая метка для отображения (1, 2, 3...)
                display_label = option['display_label']
                row.append(InlineKeyboardButton(display_label, callback_data=f"answer:{letter_label}"))
                
                # Помещаем по 3 кнопки в ряд
                if len(row) == 3 or i == len(options) - 1:
                    keyboard.append(row.copy())
                    row = []
            
            # Добавляем кнопки управления
            keyboard.append([
                InlineKeyboardButton("Пропустить", callback_data="skip"),
                InlineKeyboardButton("Завершить", callback_data="end")
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Сначала отправляем текст задачи
            await query.message.reply_text(
                task_message,
                parse_mode="Markdown"
            )
            
            # Затем отправляем клавиатуру с вариантами ответов отдельным сообщением
            await query.message.reply_text(
                "Выберите вариант ответа:",
                reply_markup=reply_markup
            )
        else:
            # Используем клавиатуру только с кнопками управления
            keyboard = [
                [
                    InlineKeyboardButton("Пропустить", callback_data="skip"),
                    InlineKeyboardButton("Завершить", callback_data="end")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Отправляем задачу с клавиатурой
            await query.message.reply_text(
                task_message,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        
        return WAITING_FOR_ANSWER
    
    except Exception as e:
        logger.error(f"Ошибка при генерации задачи: {e}")
        await query.edit_message_text(
            MESSAGES['error']
        )
        return ConversationHandler.END


async def process_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик ответа на задачу
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    # Получаем диалог пользователя
    user = update.effective_user
    conversation = get_conversation(user.id)
    
    # Проверяем, что есть активная задача
    if not conversation.current_task:
        await update.message.reply_text(
            "У вас нет активной задачи. Используйте /task, чтобы начать новую задачу."
        )
        return ConversationHandler.END
    
    # Получаем ответ пользователя
    student_answer = update.message.text
    
    # Добавляем ответ в историю диалога
    conversation.add_message('student', student_answer, update.message.message_id)
    
    try:
        # Получаем данные для проверки
        task = conversation.current_task
        concept_name = task.get("concept_name", "")
        task_type = task.get("task_type", "template")
        
        # Получаем или создаем счетчик правильных ответов
        if not hasattr(context.user_data, "correct_answers_count"):
            context.user_data["correct_answers_count"] = 0
        
        # Получаем понятие из базы данных
        concept = neo4j_client.get_concept_by_name(concept_name, context.user_data["chapter"])
        
        if not concept:
            await update.message.reply_text(
                f"К сожалению, не удалось найти понятие '{concept_name}' в базе знаний."
            )
            return ConversationHandler.END
        
        # Проверяем ответ
        check_result = await openrouter_client.check_answer(
            task, 
            student_answer, 
            concept
        )
        
        # Формируем сообщение с обратной связью
        is_correct = check_result.get("is_correct", False)
        feedback = check_result.get("feedback", "")
        
        # Подготавливаем кнопки для ответа
        keyboard = []
        
        # Обрабатываем подсчет правильных ответов
        if is_correct:
            context.user_data["correct_answers_count"] = context.user_data.get("correct_answers_count", 0) + 1
            
            # Проверяем, достиг ли студент 3 правильных ответов подряд
            if context.user_data.get("correct_answers_count", 0) >= 3:
                # Добавляем опции для изменения параметров
                keyboard.append([
                    InlineKeyboardButton("Сменить главу", callback_data="next_step:change_chapter")
                ])
                keyboard.append([
                    InlineKeyboardButton("Повысить сложность", callback_data="next_step:increase_difficulty")
                ])
                keyboard.append([
                    InlineKeyboardButton(
                        "Попробовать творческую задачу" if task_type == "template" else "Попробовать шаблонную задачу", 
                        callback_data="next_step:change_task_type"
                    )
                ])
                # Сбрасываем счетчик
                context.user_data["correct_answers_count"] = 0
        else:
            # Сбрасываем счетчик правильных ответов при неверном ответе
            context.user_data["correct_answers_count"] = 0
            
            # Добавляем опции для неправильного ответа
            keyboard.append([
                InlineKeyboardButton("Обсудить задачу", callback_data="next_step:discuss")
            ])
            keyboard.append([
                InlineKeyboardButton("Попробовать ещё раз", callback_data="next_step:try_again")
            ])
            keyboard.append([
                InlineKeyboardButton("Пропустить задачу", callback_data="next_step:skip")
            ])
        
        # Добавляем стандартные кнопки
        keyboard.append([InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")])
        keyboard.append([InlineKeyboardButton("Завершить сессию", callback_data="feedback:end")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Добавляем тип задачи в результат проверки для правильного форматирования
        check_result['task_type'] = task_type
        
        # Если есть подсказки в задаче, добавляем их в результат проверки
        if not is_correct and "hints" in task:
            check_result['hints'] = task.get("hints", [])
        
        # Используем форматтер сообщений из TelegramBot для создания сообщения с обратной связью
        from ai_tutor.bot.telegram_bot import TelegramBot
        feedback_message = TelegramBot.format_feedback_message(None, check_result)
        
        # Добавляем ответ бота в историю диалога
        conversation.add_message('bot', feedback_message)
        
        # Обновляем диалог
        conversation.clear_current_task()
        conversation.current_state = "feedback"
        save_conversation(conversation)
        
        # Отправляем сообщение с обратной связью и клавиатурой
        await update.message.reply_text(
            feedback_message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        
        # Обновляем статистику студента
        student = neo4j_client.get_student_by_telegram_id(user.id)
        if student:
            student.tasks_completed += 1
            if is_correct:
                student.correct_answers += 1
            
            # Сохраняем ответ в базе данных
            neo4j_client.save_student_answer(
                student.telegram_id,
                task,
                student_answer,
                is_correct,
                feedback
            )
            
            # Обновляем данные студента
            neo4j_client.update_student(student)
        
        return SHOW_FEEDBACK
    
    except Exception as e:
        logger.error(f"Ошибка при обработке ответа: {e}")
        await update.message.reply_text(
            MESSAGES['error']
        )
        return ConversationHandler.END


async def skip_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик пропуска задачи
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Получаем диалог пользователя
    user = update.effective_user
    conversation = get_conversation(user.id)
    
    # Очищаем текущую задачу
    conversation.clear_current_task()
    save_conversation(conversation)
    
    # Отправляем сообщение
    await query.edit_message_text(
        "Задача пропущена. Используйте /task, чтобы начать новую задачу."
    )
    
    return ConversationHandler.END


async def new_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик запроса новой задачи
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Проверяем, есть ли выбранная глава в контексте
    if "chapter" not in context.user_data:
        # Если нет, то запрашиваем выбор главы
        await query.edit_message_text("Начинаем новую задачу...")
        
        # Создаем клавиатуру с выбором глав
        reply_markup = get_chapters_keyboard()
        
        await query.message.reply_text(
            "Выберите главу курса:",
            reply_markup=reply_markup
        )
        
        return SELECTING_CHAPTER
    
    # Если глава уже выбрана, сразу переходим к генерации задачи
    # Получаем данные о выбранных параметрах
    chapter = context.user_data["chapter"]
    task_type = context.user_data.get("task_type", "template")
    difficulty = context.user_data.get("difficulty", "standard")
    
    # Сообщаем пользователю, что генерируем задачу
    await query.edit_message_text(
        f"Генерирую новую задачу для вас...\n\n"
        f"Глава: {chapter}\n"
        f"Тип задачи: {TASK_TYPES[task_type]}\n"
        f"Сложность: {DIFFICULTY_LEVELS[difficulty]}"
    )
    
    try:
        # Получаем понятия по главе
        concepts = neo4j_client.get_concepts_by_chapter(chapter)
        
        if not concepts:
            await query.edit_message_text(
                f"К сожалению, для главы '{chapter}' пока нет понятий в базе знаний.\n"
                "Попробуйте выбрать другую главу."
            )
            
            # Создаем клавиатуру с выбором глав
            reply_markup = get_chapters_keyboard()
            
            await query.message.reply_text(
                "Выберите главу курса:",
                reply_markup=reply_markup
            )
            
            return SELECTING_CHAPTER
        
        # Выбираем случайное понятие из списка
        import random
        concept = random.choice(concepts)
        
        # Получаем связанные понятия
        related_concepts = neo4j_client.get_related_concepts(concept.get('name', ''), chapter)
        
        # Генерируем задачу
        task = await openrouter_client.generate_task(
            concept, 
            related_concepts, 
            task_type, 
            difficulty
        )
        
        # Получаем диалог пользователя
        user = update.effective_user
        conversation = get_conversation(user.id)
        
        # Устанавливаем текущую задачу
        conversation.set_current_task(task)
        
        # Если это задача с вариантами ответов, обновляем метки
        if (task_type == "multiple_choice" or task_type == "template") and "options" in task:
            options = task["options"]
            # Перемешиваем варианты ответов
            random.shuffle(options)
            
            # Сохраняем буквенные метки для API и добавляем цифровые для отображения
            for i, option in enumerate(options):
                # Сохраняем оригинальную букву
                letter_label = chr(65 + i)  # A, B, C, D...
                option['label'] = letter_label  # Оригинальная буквенная метка для API
                option['display_label'] = str(i + 1)  # Цифровая метка для отображения (1, 2, 3...)
            
        # Сохраняем обновленную задачу
        conversation.set_current_task(task)
        save_conversation(conversation)
        
        # Форматируем задачу для отображения
        task_message = conversation.format_task_for_display()
        
        # Добавляем клавиатуру с подсказками или кнопками выбора ответа
        if (task_type == "multiple_choice" or task_type == "template") and "options" in task:
            options = task["options"]
            # Создаем клавиатуру с вариантами ответов
            keyboard = []
            row = []
            
            # Добавляем кнопки для каждого варианта
            for i, option in enumerate(options):
                # Сохраняем оригинальную букву
                letter_label = chr(65 + i)  # A, B, C, D...
                option['label'] = letter_label  # Оригинальная буквенная метка для API
                option['display_label'] = str(i + 1)  # Цифровая метка для отображения (1, 2, 3...)
                display_label = option['display_label']
                row.append(InlineKeyboardButton(display_label, callback_data=f"answer:{letter_label}"))
                
                # Помещаем по 3 кнопки в ряд
                if len(row) == 3 or i == len(options) - 1:
                    keyboard.append(row.copy())
                    row = []
            
            # Добавляем кнопки управления
            keyboard.append([
                InlineKeyboardButton("Пропустить", callback_data="skip"),
                InlineKeyboardButton("Завершить", callback_data="end")
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Сначала отправляем текст задачи
            await query.message.reply_text(
                task_message,
                parse_mode="Markdown"
            )
            
            # Затем отправляем клавиатуру с вариантами ответов отдельным сообщением
            await query.message.reply_text(
                "Выберите вариант ответа:",
                reply_markup=reply_markup
            )
        else:
            # Используем клавиатуру только с кнопками управления
            keyboard = [
                [
                    InlineKeyboardButton("Пропустить", callback_data="skip"),
                    InlineKeyboardButton("Завершить", callback_data="end")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Отправляем задачу с клавиатурой
            await query.message.reply_text(
                task_message,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        
        return WAITING_FOR_ANSWER
    
    except Exception as e:
        logger.error(f"Ошибка при генерации задачи: {e}")
        await query.edit_message_text(
            MESSAGES['error']
        )
        return ConversationHandler.END


async def end_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик завершения сессии
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Отправляем сообщение
    await query.edit_message_text(
        MESSAGES['session_ended']
    )
    
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик команды /cancel
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    # Получаем диалог пользователя
    user = update.effective_user
    conversation = get_conversation(user.id)
    
    # Очищаем текущую задачу
    conversation.clear_current_task()
    save_conversation(conversation)
    
    # Отправляем сообщение
    await update.message.reply_text(
        "Отменено. Используйте /task, чтобы начать новую задачу."
    )
    
    return ConversationHandler.END


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработчик неизвестных команд
    
    Args:
        update: Объект обновления
        context: Контекст бота
    """
    await update.message.reply_text(
        "Извините, я не понимаю эту команду. Используйте /help для получения списка доступных команд."
    )


async def handle_next_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик дополнительных действий после ответа на задачу
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Извлекаем действие из callback_data
    action = query.data.split(":")[1]
    
    user = update.effective_user
    conversation = get_conversation(user.id)
    
    try:
        if action == "discuss":
            # Переходим в режим обсуждения задачи
            # Сохраняем информацию о задаче для дальнейшего обсуждения
            last_task = conversation.get_last_task()
            concept_name = last_task.get("concept_name", "") if last_task else ""
            
            # Формируем сообщение с информацией о понятии
            intro_text = "Давайте обсудим задачу. Что именно вас интересует?\n\n"
            
            # Если известно понятие, добавляем информацию о нем
            if concept_name:
                intro_text += f"Задача была о понятии: *{concept_name}*\n"
                
                # Добавляем определение понятия, если можем его получить
                try:
                    chapter = context.user_data.get("chapter", "")
                    concept = neo4j_client.get_concept_by_name(concept_name, chapter)
                    if concept and concept.get("definition"):
                        intro_text += f"\nОпределение: _{concept.get('definition')}_\n"
                except Exception as e:
                    logger.warning(f"Не удалось получить определение понятия {concept_name}: {e}")
            
            intro_text += "\nЗадайте вопрос о понятии или решении задачи."
            
            await query.edit_message_text(
                intro_text,
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                    [InlineKeyboardButton("Завершить обсуждение", callback_data="feedback:end")]
                ])
            )
            
            # Устанавливаем состояние обсуждения и сохраняем задачу
            conversation.current_state = "discussion"
            # Если задача была очищена, восстанавливаем её для контекста
            if not conversation.current_task and last_task:
                conversation.current_task = last_task
            save_conversation(conversation)
            
            return SHOW_FEEDBACK
            
        elif action == "try_again":
            # Даем возможность попробовать снова
            # Восстанавливаем задачу из истории
            last_task = conversation.get_last_task()
            if not last_task:
                await query.edit_message_text(
                    "К сожалению, не удалось восстановить предыдущую задачу. Давайте начнем новую.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                        [InlineKeyboardButton("Завершить сессию", callback_data="feedback:end")]
                    ])
                )
                return SHOW_FEEDBACK
            
            # Устанавливаем текущую задачу
            conversation.current_task = last_task
            conversation.current_state = "waiting_answer"
            save_conversation(conversation)
            
            # Форматируем задачу для отображения
            task_message = conversation.format_task_for_display()
            
            # Добавляем варианты ответов для шаблонных задач
            if last_task.get("task_type") == "template" and "options" in last_task:
                options = last_task["options"]
                options_text = "\n\n*Варианты ответов:*\n"
                for option in options:
                    options_text += f"\n{option['label']}. {option['text']}"
                task_message += options_text
            
            await query.edit_message_text(
                f"Попробуйте еще раз:\n\n{task_message}",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Отменить", callback_data="feedback:end")]
                ])
            )
            
            return WAITING_FOR_ANSWER
            
        elif action == "skip":
            # Пропускаем задачу
            await query.edit_message_text(
                "Вы пропустили задачу. Хотите получить новую?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                    [InlineKeyboardButton("Завершить сессию", callback_data="feedback:end")]
                ])
            )
            return SHOW_FEEDBACK
            
        elif action == "change_chapter":
            # Предлагаем сменить главу
            reply_markup = get_chapters_keyboard()
            await query.edit_message_text(
                "Выберите новую главу:",
                reply_markup=reply_markup
            )
            return SELECTING_CHAPTER
            
        elif action == "increase_difficulty":
            # Повышаем сложность
            context.user_data["difficulty"] = "advanced"
            await query.edit_message_text(
                "Сложность повышена до продвинутого уровня. Хотите получить новую задачу?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                    [InlineKeyboardButton("Завершить сессию", callback_data="feedback:end")]
                ])
            )
            return SHOW_FEEDBACK
            
        elif action == "change_task_type":
            # Меняем тип задачи
            current_type = context.user_data.get("task_type", "template")
            new_type = "creative" if current_type == "template" else "template"
            context.user_data["task_type"] = new_type
            
            type_name = "творческую" if new_type == "creative" else "шаблонную"
            await query.edit_message_text(
                f"Тип задач изменен на {type_name}. Хотите получить новую задачу?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                    [InlineKeyboardButton("Завершить сессию", callback_data="feedback:end")]
                ])
            )
            return SHOW_FEEDBACK
            
        else:
            # Неизвестное действие
            await query.edit_message_text(
                "Неизвестное действие. Хотите получить новую задачу?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                    [InlineKeyboardButton("Завершить сессию", callback_data="feedback:end")]
                ])
            )
            return SHOW_FEEDBACK
            
    except Exception as e:
        logger.error(f"Ошибка при обработке next_step: {e}")
        await query.edit_message_text(
            MESSAGES['error'],
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Главное меню", callback_data="feedback:end")]
            ])
        )
        return ConversationHandler.END


async def handle_answer_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Обработчик выбора ответа по кнопке
    
    Args:
        update: Объект обновления
        context: Контекст бота
        
    Returns:
        Следующее состояние диалога
    """
    query = update.callback_query
    await query.answer()
    
    # Получаем выбранный ответ (формат "answer:A")
    answer_letter = query.data.replace("answer:", "")
    
    # Получаем диалог пользователя
    user = update.effective_user
    conversation = get_conversation(user.id)
    
    # Проверяем, что есть активная задача
    if not conversation.current_task:
        await query.edit_message_text(
            "У вас нет активной задачи. Используйте /task, чтобы начать новую задачу."
        )
        return ConversationHandler.END
    
    # Находим опцию с этой буквой
    selected_option = None
    for option in conversation.current_task.get("options", []):
        if option.get("label") == answer_letter:
            selected_option = option
            break
    
    # Формируем ответ для отображения пользователю
    if selected_option:
        display_answer = f"{answer_letter}. {selected_option.get('text', '')}"
        student_answer = answer_letter
    else:
        display_answer = f"Вариант {answer_letter}"
        student_answer = answer_letter
    
    # Добавляем отображаемый ответ в историю диалога
    conversation.add_message('student', display_answer, query.message.message_id)
    
    try:
        # Получаем данные для проверки
        task = conversation.current_task
        concept_name = task.get("concept_name", "")
        task_type = task.get("task_type", "template")  # По умолчанию шаблонная задача
        
        # Получаем или создаем счетчик правильных ответов
        if not hasattr(context.user_data, "correct_answers_count"):
            context.user_data["correct_answers_count"] = 0
        
        # Получаем понятие из базы данных
        concept = neo4j_client.get_concept_by_name(concept_name, context.user_data["chapter"])
        
        if not concept:
            await query.edit_message_text(
                f"К сожалению, не удалось найти понятие '{concept_name}' в базе знаний."
            )
            return ConversationHandler.END
        
        # Проверяем ответ
        check_result = await openrouter_client.check_answer(
            task, 
            student_answer,  # Передаем буквенный вариант для проверки
            concept
        )
        
        # Формируем сообщение с обратной связью
        is_correct = check_result.get("is_correct", False)
        feedback = check_result.get("feedback", "")
        
        # Подготавливаем кнопки для ответа
        keyboard = []
        
        # Обрабатываем подсчет правильных ответов
        if is_correct:
            context.user_data["correct_answers_count"] = context.user_data.get("correct_answers_count", 0) + 1
            
            # Проверяем, достиг ли студент 3 правильных ответов подряд
            if context.user_data.get("correct_answers_count", 0) >= 3:
                # Добавляем опции для изменения параметров
                keyboard.append([
                    InlineKeyboardButton("Сменить главу", callback_data="next_step:change_chapter")
                ])
                keyboard.append([
                    InlineKeyboardButton("Повысить сложность", callback_data="next_step:increase_difficulty")
                ])
                keyboard.append([
                    InlineKeyboardButton(
                        "Попробовать творческую задачу" if task_type == "template" else "Попробовать шаблонную задачу", 
                        callback_data="next_step:change_task_type"
                    )
                ])
                # Сбрасываем счетчик
                context.user_data["correct_answers_count"] = 0
        else:
            # Сбрасываем счетчик правильных ответов при неверном ответе
            context.user_data["correct_answers_count"] = 0
            
            # Добавляем опции для неправильного ответа
            keyboard.append([
                InlineKeyboardButton("Обсудить задачу", callback_data="next_step:discuss")
            ])
            keyboard.append([
                InlineKeyboardButton("Попробовать ещё раз", callback_data="next_step:try_again")
            ])
            keyboard.append([
                InlineKeyboardButton("Пропустить задачу", callback_data="next_step:skip")
            ])
        
        # Добавляем стандартные кнопки
        keyboard.append([InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")])
        keyboard.append([InlineKeyboardButton("Завершить сессию", callback_data="feedback:end")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Добавляем тип задачи в результат проверки для правильного форматирования
        check_result['task_type'] = task_type
        
        # Если есть подсказки в задаче, добавляем их в результат проверки
        if not is_correct and "hints" in task:
            check_result['hints'] = task.get("hints", [])
            
        # Используем форматтер сообщений из TelegramBot для создания сообщения с обратной связью
        from ai_tutor.bot.telegram_bot import TelegramBot
        feedback_message = TelegramBot.format_feedback_message(None, check_result)
        
        # Добавляем ответ бота в историю диалога
        conversation.add_message('bot', feedback_message)
        
        # Обновляем диалог
        conversation.clear_current_task()
        conversation.current_state = "feedback"
        save_conversation(conversation)
        
        # Отправляем сообщение с обратной связью и клавиатурой
        await query.edit_message_text(
            feedback_message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        
        # Обновляем статистику студента
        student = neo4j_client.get_student_by_telegram_id(user.id)
        if student:
            student.tasks_completed += 1
            if is_correct:
                student.correct_answers += 1
            
            # Сохраняем ответ в базе данных
            neo4j_client.save_student_answer(
                student.telegram_id,
                task,
                display_answer,  # Сохраняем полный ответ пользователя
                is_correct,
                feedback
            )
            
            # Обновляем данные студента
            neo4j_client.update_student(student)
        
        return SHOW_FEEDBACK
    
    except Exception as e:
        logger.error(f"Ошибка при обработке ответа по кнопке: {e}")
        await query.message.reply_text(
            MESSAGES['error']
        )
        return ConversationHandler.END
