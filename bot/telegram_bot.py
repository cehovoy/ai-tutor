"""
Telegram-бот для ИИ-репетитора
"""
import logging
import asyncio
import traceback
from typing import Dict, Any, Optional, Union, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ApplicationBuilder
from telegram.ext import ContextTypes, ConversationHandler, filters

from ai_tutor.config.settings import TELEGRAM_TOKEN, CHAPTERS, TASK_TYPES, DIFFICULTY_LEVELS
from ai_tutor.agents.crew import TutorCrew
from ai_tutor.api.openrouter import OpenRouterClient
from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.agents.assistant import CourseAssistant
from ai_tutor.agents.tutor_assistant import TutorAssistant
from ai_tutor.bot.handlers import (
    start_command, help_command, task_command, cancel, unknown_command,
    select_chapter, select_task_type, select_difficulty, process_answer,
    skip_task, new_task, end_session, handle_next_step
)
from ai_tutor.bot.conversation import get_conversation, save_conversation

# Состояния диалога
SELECTING_CHAPTER, SELECTING_TASK_TYPE, SELECTING_DIFFICULTY, WAITING_FOR_ANSWER, SHOW_FEEDBACK, DISCUSSION, WAITING_FOR_ASK_CHAPTER = range(7)

# Префиксы для callback-данных
PREFIX_CHAPTER = "chapter:"
PREFIX_TASK_TYPE = "task_type:"
PREFIX_DIFFICULTY = "difficulty:"

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram-бот для взаимодействия с ИИ-репетитором
    """
    
    def __init__(self, token: str = TELEGRAM_TOKEN):
        """
        Инициализация Telegram-бота
        
        Args:
            token: Токен Telegram-бота
        """
        self.token = token
        
        # Логгер
        self.logger = logging.getLogger(__name__)
        
        # Инициализация клиентов
        self.neo4j_client = Neo4jClient()
        self.openrouter_client = OpenRouterClient()
        
        # Инициализация помощника
        self.course_assistant = CourseAssistant(self.neo4j_client, self.openrouter_client)
        
        # Инициализация репетитора для обсуждения задач
        self.tutor_assistant = TutorAssistant(self.neo4j_client, self.openrouter_client)
        
        # Инициализация бота
        self.application = ApplicationBuilder().token(token).build()
        
        # Добавление обработчиков
        self._add_handlers()
    
    def _add_handlers(self) -> None:
        """
        Добавление обработчиков команд и сообщений
        """
        # Сначала добавляем обработчик для /start, чтобы он имел наивысший приоритет
        self.application.add_handler(CommandHandler("start", self.start))
        
        # Добавляем обработчик для /help 
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Создаем обработчик диалога
        conv_handler = self.create_conversation_handler()
        
        self.application.add_handler(conv_handler)
        
        # Обработчик для неизвестных команд
        self.application.add_handler(MessageHandler(filters.COMMAND, self.unknown_command))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Обработчик команды /start
        
        Args:
            update: Объект обновления
            context: Контекст бота
            
        Returns:
            Следующее состояние диалога
        """
        user = update.effective_user
        await update.message.reply_text(
            f"👋 *Добро пожаловать в AI Tutor!*\n\n"
            f"Привет, {user.first_name}! Я интеллектуальный репетитор Школы Системного Мышления, созданный помочь вам освоить системные понятия через интерактивное обучение.\n\n"
            f"🧠 *О сервисе:*\n"
            f"AI Tutor анализирует ваши ответы, адаптирует сложность и предоставляет подробную обратную связь, делая процесс обучения максимально эффективным.\n\n"
            f"📚 *Как работает процесс обучения:*\n"
            f"1. Выберите раздел курса 'Системное саморазвитие'\n"
            f"2. Определите тип задачи (шаблонная или творческая)\n"
            f"3. Установите уровень сложности\n"
            f"4. Решайте интерактивные задачи и получайте мгновенную обратную связь\n\n"
            f"⭐ *Ваш прогресс важен:*\n"
            f"После 5 правильно решенных задач вам будет предложено перейти на следующий уровень, сменить тип задач или изучить новую главу.\n\n"
            f"Используйте команду /task, чтобы начать обучение!",
            parse_mode="Markdown"
        )
        
        # Отправляем сообщение с помощью
        await self.help_command(update, context)
        
        return ConversationHandler.END
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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
            "/cancel - отменить текущий диалог\n\n"
            "Курс 'Системное саморазвитие' состоит из 9 глав. "
            "Вы можете выбрать главу, тип задачи и уровень сложности."
        )
        
        return ConversationHandler.END
    
    def create_conversation_handler(self):
        """
        Создание обработчика диалога
        
        Returns:
            Обработчик диалога
        """
        from ai_tutor.bot.handlers import (
            task_command, select_chapter, select_difficulty, select_task_type,
            process_answer, skip_task, end_session, 
            new_task, handle_next_step, handle_answer_button, cancel
        )
        
        return ConversationHandler(
            entry_points=[
                CommandHandler("task", task_command)
            ],
            states={
                SELECTING_CHAPTER: [
                    CallbackQueryHandler(select_chapter, pattern=f"^{PREFIX_CHAPTER}")
                ],
                SELECTING_TASK_TYPE: [
                    CallbackQueryHandler(select_task_type, pattern=f"^{PREFIX_TASK_TYPE}")
                ],
                SELECTING_DIFFICULTY: [
                    CallbackQueryHandler(select_difficulty, pattern=f"^{PREFIX_DIFFICULTY}")
                ],
                WAITING_FOR_ANSWER: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, process_answer),
                    CallbackQueryHandler(handle_answer_button, pattern="^answer:"),
                    CallbackQueryHandler(skip_task, pattern="^skip"),
                    CallbackQueryHandler(end_session, pattern="^end")
                ],
                SHOW_FEEDBACK: [
                    CallbackQueryHandler(new_task, pattern="^feedback:new_task"),
                    CallbackQueryHandler(end_session, pattern="^feedback:end"),
                    CallbackQueryHandler(handle_next_step, pattern="^next_step:"),
                    # Обработчик для сообщений в режиме обсуждения
                    MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: self.discussion_handler(update, context, SHOW_FEEDBACK))
                ]
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            name="conversation_handler",
            persistent=False
        )
    
    async def discussion_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE, return_state: int) -> int:
        """
        Обработчик режима обсуждения задачи, использующий помощника для ответов на вопросы
        
        Args:
            update: Объект обновления
            context: Контекст бота
            return_state: Состояние, в которое нужно вернуться
            
        Returns:
            Следующее состояние диалога
        """
        user = update.effective_user
        conversation = get_conversation(user.id)
        
        # Проверяем, что пользователь в режиме обсуждения
        if conversation.current_state != "discussion":
            return return_state
        
        # Получаем сообщение пользователя
        question = update.message.text
        
        # Добавляем вопрос в историю диалога
        conversation.add_message('student', question, update.message.message_id)
        
        # Получаем информацию о последней задаче
        last_task = conversation.get_last_task()
        
        # Получаем текущую главу из контекста пользователя
        chapter_title = context.user_data.get("chapter", "")
        concept_name = last_task.get("concept_name", "") if last_task else ""
        task_question = last_task.get("question", "") if last_task else ""
        
        # Отправляем "печатает..." статус
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Если нет информации о задаче, сообщаем пользователю
            if not last_task or not concept_name:
                logger.warning(f"Пользователь {user.id} пытается обсудить задачу без активной задачи или понятия")
                
                # Попробуем использовать CourseAssistant для ответа на общий вопрос
                try:
                    # Используем CourseAssistant, который умеет отвечать на вопросы по содержимому курса
                    answer = await self.course_assistant.answer_question(question, chapter_title)
                    logger.info(f"Ответ на вопрос без контекста задачи сгенерирован через CourseAssistant")
                except Exception as e:
                    logger.warning(f"Не удалось получить ответ через CourseAssistant: {str(e)}")
                    
                    # Формируем общий ответ основываясь на вопросе пользователя
                    answer = (
                        "Извините, но я не могу определить, о какой задаче или понятии идет речь. "
                        "Пожалуйста, уточните ваш вопрос, указав конкретное понятие, или начните новую задачу."
                    )
                    
                    # Если есть глава, добавляем контекст
                    if chapter_title:
                        try:
                            chapter_info = self.neo4j_client.get_chapter_info(chapter_title)
                            if chapter_info and 'main_ideas' in chapter_info:
                                answer += f"\n\nМы обсуждаем главу '{chapter_title}', где рассматриваются: {chapter_info['main_ideas']}"
                        except Exception as e:
                            logger.warning(f"Ошибка при получении информации о главе: {str(e)}")
            else:
                # Используем нового туториального ассистента для ответа на вопрос
                answer = await self.tutor_assistant.discuss_task(
                    student_question=question,
                    concept_name=concept_name,
                    task_question=task_question,
                    chapter_title=chapter_title
                )
            
            # Логируем взаимодействие
            self.tutor_assistant.log_discussion(
                student_id=str(user.id),
                concept_name=concept_name,
                question=question,
                answer=answer,
                chapter_title=chapter_title
            )
            
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса к помощнику: {str(e)}")
            answer = "Извините, произошла ошибка при поиске ответа на ваш вопрос. Пожалуйста, попробуйте переформулировать или задать другой вопрос."
        
        # Добавляем ответ в историю диалога
        conversation.add_message('bot', answer)
        save_conversation(conversation)
        
        # Отправляем ответ с кнопками для продолжения
        await update.message.reply_text(
            answer,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                [InlineKeyboardButton("Завершить обсуждение", callback_data="feedback:end")]
            ]),
            parse_mode="Markdown"
        )
        
        return return_state
    
    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик неизвестных команд
        
        Args:
            update: Объект обновления
            context: Контекст бота
        """
        await update.message.reply_text(
            "Неизвестная команда. Используйте /help для просмотра доступных команд."
        )
    
    async def generate_task(self, student_id: str, chapter_title: str, 
                         task_type: str, difficulty: str) -> Dict[str, Any]:
        """
        Генерация задачи
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            task_type: Тип задачи
            difficulty: Уровень сложности
            
        Returns:
            Сгенерированная задача
        """
        # Запускаем полный процесс репетитора
        try:
            logger.info(f"Запуск генерации задачи: student_id={student_id}, глава={chapter_title}, тип={task_type}, сложность={difficulty}")
            
            # Получаем текущий цикл событий
            loop = asyncio.get_event_loop()
            
            # Логируем информацию перед вызовом
            logger.info("Вызываем TutorCrew.full_tutor_process через run_in_executor")
            
            result = await loop.run_in_executor(
                None,
                lambda: self.tutor_crew.full_tutor_process(
                    student_id=student_id,
                    chapter_title=chapter_title,
                    task_type=task_type,
                    difficulty=difficulty
                )
            )
            
            logger.info("Получен результат из TutorCrew.full_tutor_process")
            
            # Проверяем наличие ошибки в результате
            if "error" in result.get("task", {}):
                error_message = result["task"]["error"]
                logger.error(f"Ошибка в результате генерации задачи: {error_message}")
            else:
                logger.info("Задача успешно сгенерирована")
                
            return result
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Исключение при генерации задачи: {error_message}\n{stack_trace}")
            
            # Создаем подробный ответ с ошибкой
            return {
                "task": {
                    "error": f"Произошла ошибка при генерации задачи: {error_message}",
                    "concept_name": "Ошибка",
                    "task_type": task_type,
                    "difficulty": difficulty
                },
                "metadata": {
                    "student_id": student_id,
                    "chapter_title": chapter_title,
                    "task_type": task_type,
                    "difficulty": difficulty
                }
            }
    
    async def check_answer(self, student_id: str, chapter_title: str, 
                        task: Dict[str, Any], student_answer: str) -> Dict[str, Any]:
        """
        Проверка ответа
        
        Args:
            student_id: ID студента
            chapter_title: Название главы
            task: Задача
            student_answer: Ответ студента
            
        Returns:
            Результат проверки
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.tutor_crew.check_answer(
                    student_id=student_id,
                    chapter_title=chapter_title,
                    task=task,
                    student_answer=student_answer
                )
            )
        except Exception as e:
            logger.error(f"Ошибка при проверке ответа: {str(e)}")
            return {
                "is_correct": False,
                "explanation": f"Ошибка при проверке ответа: {str(e)}",
                "recommendations": ["Попробуйте еще раз."]
            }
    
    def format_task_message(self, task: Dict[str, Any]) -> str:
        """
        Форматирование сообщения с задачей
        
        Args:
            task: Задача
            
        Returns:
            Отформатированный текст задачи
        """
        message = f"📚 *Задача по теме: {task['concept_name']}*\n\n"
        
        # Добавляем вопрос, форматируя его для лучшей читабельности
        message += f"{task['question']}\n\n"
        
        # Добавляем варианты ответов для задачи с множественным выбором
        if task["task_type"] == "multiple_choice":
            message += "*Варианты ответов:*\n\n"
            # Нумеруем варианты буквами (A, B, C, D)
            for i, option in enumerate(task["options"], 1):
                letter = chr(64 + i)  # 65 - код ASCII для 'A'
                message += f"*{letter}.* {option['text']}\n"
                # Сохраняем буквенную метку в опции для последующей проверки
                option['label'] = letter
        
        # Добавляем критерии оценки для творческой задачи
        if task["task_type"] == "creative" and "criteria" in task:
            message += "\n*Критерии оценки:*\n"
            for criterion in task["criteria"]:
                message += f"• {criterion}\n"
            
            # Если есть пример ответа, тоже добавляем
            if "example_answer" in task and task["example_answer"]:
                message += "\n*Пример ответа:*\n"
                message += f"{task['example_answer']}\n"
        
        return message
    
    def create_options_keyboard(self, task: Dict[str, Any]) -> Optional[InlineKeyboardMarkup]:
        """
        Создает клавиатуру с вариантами ответов для задач с множественным выбором
        
        Args:
            task: Задача
            
        Returns:
            Клавиатура с вариантами ответов или None, если это не задача с множественным выбором
        """
        if task["task_type"] != "multiple_choice":
            return None
            
        keyboard = []
        row = []
        
        # Создаем кнопки для каждого варианта ответа
        for i, option in enumerate(task["options"], 1):
            # Преобразуем номер в букву (1 -> A, 2 -> B, и т.д.)
            letter = chr(64 + i)  # 65 - код ASCII для 'A'
            # Callback data формата answer:буква
            callback_data = f"answer:{letter}"
            row.append(InlineKeyboardButton(letter, callback_data=callback_data))
            
            # Помещаем по 3 кнопки в ряд
            if len(row) == 3 or i == len(task["options"]):
                keyboard.append(row.copy())
                row = []
                
        # Добавляем кнопки для управления задачей
        keyboard.append([
            InlineKeyboardButton("Пропустить", callback_data="skip"),
            InlineKeyboardButton("Завершить", callback_data="end")
        ])
        
        return InlineKeyboardMarkup(keyboard)
    
    def format_feedback_message(self, check_result: Dict[str, Any]) -> str:
        """
        Форматирование сообщения с обратной связью с использованием мотивационного интервьюирования
        
        Args:
            check_result: Результат проверки
            
        Returns:
            Отформатированный текст обратной связи
        """
        is_correct = check_result.get('is_correct', False)
        task_type = check_result.get('task_type', 'template')
        
        # Начинаем с поддерживающего обращения
        if is_correct:
            message = "✅ *Отлично!* Ты верно ответил на вопрос.\n\n"
        else:
            # Мотивирующее сообщение вместо просто "Неверно"
            message = "🤔 *Интересная попытка!* Давай разберемся вместе.\n\n"
        
        # Добавляем объяснение или отзыв для творческих задач
        if task_type == "creative":
            # Для творческих задач используем расширенный формат с элементами мотивационного интервьюирования
            feedback = check_result.get('feedback', '')
            message += f"{feedback}\n\n"
            
            # Добавляем сильные стороны ответа, если они не включены в основной текст
            if not "сильные стороны" in feedback.lower():
                strengths = check_result.get('strengths', [])
                if strengths:
                    message += "*Сильные стороны твоего ответа:*\n"
                    for strength in strengths:
                        message += f"• {strength}\n"
                    message += "\n"
            
            # Добавляем области для улучшения и вопросы для размышления
            if not any(marker in feedback.lower() for marker in ["для размышления", "подумай", "вопросы"]):
                # Сначала добавляем области для улучшения
                improvements = check_result.get('improvements', [])
                if improvements:
                    message += "*Для размышления:*\n"
                    for improvement in improvements:
                        message += f"• {improvement}\n"
                    message += "\n"
                
                # Затем добавляем вопросы для рефлексии
                reflection_questions = check_result.get('reflection_questions', [])
                if reflection_questions:
                    message += "*Вопросы для углубления понимания:*\n"
                    for question in reflection_questions:
                        message += f"• {question}\n"
                    message += "\n"
        else:
            # Для шаблонных задач добавляем объяснение
            explanation = check_result.get('explanation', '')
            feedback = check_result.get('feedback', '')
            
            if explanation:
                message += f"{explanation}\n\n"
            elif feedback:
                message += f"{feedback}\n\n"
            
            # Добавляем подсказку для размышления (если ответ неверный)
            if not is_correct:
                message += "*Подумай над этим:*\n"
                hints = check_result.get('hints', [])
                if hints:
                    # Используем подсказки из результата проверки
                    hint = hints[0] if isinstance(hints, list) and hints else "Внимательно прочитай определение понятия и подумай о его ключевых характеристиках."
                    message += f"• {hint}\n"
                else:
                    # Если подсказок нет, добавляем общий совет
                    message += "• Обрати внимание на ключевые слова в определении понятия.\n"
                    message += "• Попробуй взглянуть на проблему с другой стороны.\n"
        
        # Добавляем мотивационное завершение
        if not is_correct:
            message += "\nНе отчаивайся! Каждая ошибка - это шаг к лучшему пониманию. Хочешь обсудить это подробнее или попробовать ещё раз?"
        else:
            message += "\nОтлично справляешься! Продолжай в том же духе. Чувствуешь, что готов перейти к более сложным задачам?"
        
        return message
    
    async def run(self) -> None:
        """
        Асинхронный запуск бота
        """
        logger.info("Запуск Telegram-бота")
        # Используем рекомендуемый метод Application.run_polling()
        # Этот метод внутри сам корректно инициализирует HTTPXRequest
        # Обратите внимание, что это блокирующий вызов
        
        # В данном случае мы не блокируем, а просто делаем базовую инициализацию
        await self.application.initialize()
        await self.application.start()
        # Здесь мы не вызываем application.updater.start_polling()
        # Вместо этого запуск polling будет выполнен в методе main.py
        
    def run_polling(self) -> None:
        """
        Синхронный метод для запуска бота в режиме poll (блокирующий)
        Должен вызываться из основного потока
        """
        logger.info("Запуск Telegram-бота в режиме polling (блокирующий)")
        # Этот метод блокирует выполнение до остановки бота
        self.application.run_polling(drop_pending_updates=True)
    
    async def stop(self) -> None:
        """
        Асинхронная остановка бота
        """
        logger.info("Остановка Telegram-бота")
        await self.application.stop()
        await self.application.shutdown()
