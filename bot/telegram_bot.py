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
from ai_tutor.config.constants import MESSAGES
from ai_tutor.agents.crew import TutorCrew
from ai_tutor.api.openrouter import OpenRouterClient
from ai_tutor.database.neo4j_client import Neo4jClient
from ai_tutor.agents.unified_assistant import UnifiedAssistant
from ai_tutor.bot.handlers import (
    start_command, help_command, task_command, cancel, unknown_command,
    select_chapter, select_task_type, select_difficulty, process_answer,
    skip_task, new_task, end_session, handle_next_step
)
from ai_tutor.bot.conversation import get_conversation, save_conversation
from ai_tutor.bot.keyboards import get_chapters_keyboard

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
        
        # Инициализация объединенного ассистента
        self.assistant = UnifiedAssistant(self.neo4j_client, self.openrouter_client)
        
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
        
        # Добавляем обработчик для /consultant для прямого доступа к консультанту
        self.application.add_handler(CommandHandler("consultant", self.consultant_command))
        
        # Добавляем обработчик для callback-запросов от кнопок меню и выбора главы
        self.application.add_handler(CallbackQueryHandler(self.menu_button_handler, pattern="^(task|random_task|change_chapter|consultant|" + PREFIX_CHAPTER + ".+|" + PREFIX_TASK_TYPE + ".+|" + PREFIX_DIFFICULTY + ".+|answer:.+|skip|end|feedback:new_task|feedback:end|next_step:.+)$"))
        
        # Создаем обработчик диалога
        conv_handler = self.create_conversation_handler()
        
        self.application.add_handler(conv_handler)
        
        # Обработчик для команды отмены вне ConversationHandler
        self.application.add_handler(CommandHandler("cancel", cancel))
        
        # Обработчик для неизвестных команд
        self.application.add_handler(MessageHandler(filters.COMMAND, self.unknown_command))
        
        # Обработчик для обычных сообщений вне диалога
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.direct_message))
    
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
        
        # Новый текст для приветственного сообщения
        welcome_text = (
            f"ИИ-репетитор Школы Системного Менеджмента (ШСМ). Вам будет предложено два вида задач: \n"
            f"1) С вариантами ответов\n"
            f"2) Творческие (следует применить практику мышления письмом)\n\n"
            f"Главная цель репетитора - повысить беглость в использовании понятий.\n" 
            f"Проект разработан и поддерживается волонтёрами ШСМ."
        )
        await self.safe_send_message(update, welcome_text)
        
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
        
        await self.safe_send_message(update, "Выберите действие:", reply_markup=reply_markup)
        
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
                    # Обработка выбора главы теперь происходит в menu_button_handler
                ],
                SELECTING_TASK_TYPE: [
                    # Обработка выбора типа задачи теперь происходит в menu_button_handler
                ],
                SELECTING_DIFFICULTY: [
                    # Обработка выбора сложности теперь происходит в menu_button_handler
                ],
                WAITING_FOR_ANSWER: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, process_answer),
                    # Обработка кнопок ответов, пропуска, завершения теперь происходит в menu_button_handler
                ],
                SHOW_FEEDBACK: [
                    # Обработка обратной связи и перехода к новой задаче теперь происходит в menu_button_handler
                    # Обработчик для сообщений в режиме обсуждения
                    MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: self.discussion_handler(update, context, SHOW_FEEDBACK))
                ]
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            name="conversation_handler",
            persistent=False,
            per_message=False
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
            # Подготавливаем контекст для запроса
            task_context = None
            if last_task and concept_name:
                task_context = {
                    'concept_name': concept_name,
                    'task_question': task_question
                }
            
            # Используем объединенный ассистент для ответа
            answer = await self.assistant.answer_question(
                question=question,
                student_id=str(user.id),
                chapter_title=chapter_title,
                context=task_context
            )
            
            logger.info(f"Ответ на вопрос сгенерирован через UnifiedAssistant")
                    
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса к помощнику: {str(e)}")
            answer = "Извините, произошла ошибка при поиске ответа на ваш вопрос. Пожалуйста, попробуйте переформулировать или задать другой вопрос."
        
        # Добавляем ответ в историю диалога
        conversation.add_message('bot', answer)
        save_conversation(conversation)
        
        # Создаем клавиатуру для продолжения
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
            [InlineKeyboardButton("Завершить обсуждение", callback_data="feedback:end")]
        ])
        
        # Отправляем ответ с кнопками для продолжения, используя безопасный метод
        await self.safe_send_message(update, answer, reply_markup=reply_markup)
        
        return return_state
    
    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик неизвестных команд
        
        Args:
            update: Объект обновления
            context: Контекст бота
        """
        await update.message.reply_text(
            "Извините, я не знаю такой команды. Используйте /help для получения списка команд."
        )
    
    async def consultant_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /consultant для прямого доступа к консультанту по курсу
        
        Args:
            update: Объект обновления
            context: Контекст бота
        """
        user = update.effective_user
        
        # Сохраняем состояние, что пользователь в режиме консультации
        if 'user_data' not in context:
            context.user_data = {}
            
        context.user_data['consultation_mode'] = True
        
        # Создаем сообщение и отправляем его безопасным методом
        message = (
            f"👨‍🏫 *Режим консультации активирован*\n\n"
            f"{user.first_name}, я готов ответить на ваши вопросы по курсу системного мышления. "
            f"Вы можете спрашивать о любых понятиях, главах курса или связях между ними.\n\n"
            f"Просто задайте ваш вопрос, и я постараюсь дать подробный ответ, используя "
            f"официальные материалы курса и другие релевантные источники.\n\n"
            f"Для выхода из режима консультации введите /cancel"
        )
        
        await self.safe_send_message(update, message)

    async def direct_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик обычных сообщений вне диалога
        
        Args:
            update: Объект обновления
            context: Контекст бота
        """
        # Проверяем, находится ли пользователь в режиме консультации
        if context.user_data and context.user_data.get('consultation_mode', False):
            await self.process_consultation(update, context)
        else:
            # Если пользователь не в режиме консультации, предлагаем доступные команды
            await update.message.reply_text(
                "Вы можете использовать следующие команды:\n"
                "/task - начать решать задачи\n"
                "/consultant - поговорить с консультантом по курсу\n"
                "/help - показать справку"
            )

    def sanitize_text_for_telegram(self, text: str) -> str:
        """
        Подготавливает текст для отправки в Telegram, удаляя HTML-теги и
        удаляя или экранируя специальные символы Markdown.
        
        Args:
            text: Исходный текст для подготовки
            
        Returns:
            str: Текст без символов Markdown-форматирования
        """
        if not text:
            return ""
        
        try:
            # Удаляем HTML-теги
            import re
            text = re.sub(r'<[^>]+>', '', text)
            
            # Заменяем последовательности '_' на обычные пробелы
            text = re.sub(r'_{2,}', ' ', text)
            
            # Удаляем символы форматирования Markdown
            text = re.sub(r'[*_`]', '', text)
            
            # Удаляем нестандартное форматирование, которое может быть в ответах LLM
            text = text.replace('**', '').replace('__', '').replace('##', '')
            
            # Заменяем квадратные скобки на круглые (для ссылок)
            text = text.replace('[', '(').replace(']', ')')
            
            # Заменяем блоки кода (```code```) на обычный текст
            text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace('```', ''), text)
            
            # Обрабатываем обратные слеши и проблемные символы
            text = text.replace('\\', '\\\\').replace('\t', '    ')
            
            # Удаляем множественные переносы строк (более 2)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Для очень длинных текстов (более 2000 символов) - удаляем все специальные символы
            # чтобы минимизировать риски проблем с разбивкой
            if len(text) > 2000:
                # Удаляем все символы, которые могут интерпретироваться как форматирование
                text = re.sub(r'[#*_`~>]', '', text)
                # Заменяем все скобки на круглые
                text = text.replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')
            
            return text
        except Exception as e:
            logger.error(f"Ошибка при санитизации текста: {str(e)}")
            # В случае ошибки, возвращаем только алфавитно-цифровые символы и пробелы
            return re.sub(r'[^\w\s,.!?;:()\-]', '', text)
    
    async def safe_send_message(self, update, text, reply_markup=None):
        """
        Безопасно отправляет сообщение в Telegram, разбивая длинные тексты на части.
        Гарантирует доставку полного текста, даже если он очень большой.
        
        Args:
            update: Объект обновления телеграм
            text: Текст для отправки
            reply_markup: Опциональная клавиатура для сообщения
        
        Returns:
            Message: Объект последнего отправленного сообщения или None в случае ошибки

            
        """
        sanitized_text = self.sanitize_text_for_telegram(text)
        
        if not sanitized_text:
            logging.warning("Попытка отправить пустое сообщение")
            return None
            
        # Максимальная длина сообщения в Telegram (с запасом)
        max_length = 3800  # Оставляем больше запаса для заголовков частей
        last_message = None
        
        try:
            # Измеряем длину исходного текста
            total_length = len(sanitized_text)
            
            # Если текст короче максимальной длины, отправляем как есть
            if total_length <= max_length:
                return await update.message.reply_text(
                    sanitized_text,
                    reply_markup=reply_markup
                )
            
            # Для длинных сообщений: предупреждаем о большом объеме текста
            notice_message = await update.message.reply_text(
                f"Ответ получился большим (примерно {len(sanitized_text) // 1000} Кб), разбиваю на части..."
            )
            
            logger.info(f"Разбиваем длинное сообщение на части (длина: {total_length} символов)")
            
            # Определяем примерное количество частей
            estimated_parts = (total_length // max_length) + 1
            
            # Если ответ очень большой, то используем более агрессивное разбиение
            # по смысловым границам (абзацы, предложения)
            if estimated_parts > 2:
                # Используем смысловое разбиение для очень длинных ответов
                parts = self._smart_text_split(sanitized_text, max_length, estimated_parts)
            else:
                # Для не очень длинных ответов делим на две части
                half_point = len(sanitized_text) // 2
                
                # Ищем ближайший конец предложения для разделения
                # Сначала ищем две новые строки (конец абзаца)
                split_point = sanitized_text.find("\n\n", half_point - 200, half_point + 200)
                
                # Если не нашли абзац, ищем конец предложения
                if split_point == -1:
                    # Ищем конец предложения (точка, восклицательный или вопросительный знак и пробел)
                    for pattern in [". ", "! ", "? "]:
                        pos = sanitized_text.find(pattern, half_point - 300, half_point + 300)
                        if pos != -1:
                            split_point = pos + 2  # +2 чтобы включить пунктуацию и пробел
                            break
                            
                # Если всё ещё не нашли подходящее место, просто делим посередине
                if split_point == -1:
                    split_point = half_point
                
                # Делим текст на две части
                parts = [
                    sanitized_text[:split_point],
                    sanitized_text[split_point:]
                ]
            
            # Отправляем части по очереди
            for i, part in enumerate(parts):
                try:
                    # Клавиатуру прикрепляем только к последнему сообщению
                    markup = reply_markup if i == len(parts) - 1 else None
                    
                    # Добавляем нумерацию частей, если их больше одной
                    if len(parts) > 1:
                        part_info = f"📄 Часть {i+1} из {len(parts)} 📄\n\n"
                        if i > 0:
                            part_info += "(Продолжение ответа)\n\n"
                        part = part_info + part
                    
                    last_message = await update.message.reply_text(
                        part,
                        reply_markup=markup
                    )
                    
                    # Добавляем небольшую задержку между сообщениями, чтобы избежать ошибок флуда
                    if i < len(parts) - 1:
                        await asyncio.sleep(1.0)  # Увеличиваем задержку для большей надежности
                        
                except Exception as e:
                    logging.error(f"Ошибка при отправке части сообщения {i+1}: {str(e)}")
                    # Пытаемся отправить сообщение об ошибке
                    try:
                        await update.message.reply_text(
                            f"Ошибка при отправке ответа (часть {i+1}). Пожалуйста, попробуйте еще раз."
                        )
                    except:
                        logging.error("Не удалось отправить сообщение об ошибке")
            
            # Удаляем уведомление о разбиении, если части успешно отправлены
            try:
                await notice_message.delete()
            except Exception as e:
                logger.warning(f"Не удалось удалить уведомление о разбиении: {str(e)}")
            
            return last_message
        except Exception as e:
            logging.error(f"Ошибка в safe_send_message: {str(e)}")
            logging.error(traceback.format_exc())
            try:
                await update.message.reply_text(
                    "Произошла ошибка при отправке сообщения. Пожалуйста, попробуйте еще раз."
                )
            except:
                pass
            return None
            
    def _smart_text_split(self, text: str, max_length: int, estimated_parts: int) -> List[str]:
        """
        Умное разбиение текста на части с учетом смысловых границ.
        
        Args:
            text: Исходный текст
            max_length: Максимальная длина одной части
            estimated_parts: Примерное количество ожидаемых частей
            
        Returns:
            Список частей текста
        """
        # Если текст короткий, возвращаем его как единственную часть
        if len(text) <= max_length:
            return [text]
            
        parts = []
        current_part = ""
        
        # Для текстов оптимальное деление - по абзацам
        paragraphs = text.split("\n\n")
        
        # Если абзацы получились очень большие, делим их на предложения
        if any(len(p) > max_length for p in paragraphs):
            # Сначала объединяем абзацы в более крупные блоки, но не превышающие max_length
            blocks = []
            current_block = ""
            
            for paragraph in paragraphs:
                # Если параграф сам по себе больше максимальной длины
                if len(paragraph) > max_length:
                    # Если есть накопленный блок, добавляем его
                    if current_block:
                        blocks.append(current_block)
                        current_block = ""
                        
                    # Делим длинный параграф на предложения
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    
                    # Группируем предложения в блоки
                    sent_block = ""
                    for sentence in sentences:
                        if len(sent_block) + len(sentence) + 2 <= max_length:
                            if sent_block:
                                sent_block += " " + sentence
                            else:
                                sent_block = sentence
                        else:
                            blocks.append(sent_block)
                            sent_block = sentence
                            
                    # Добавляем последний блок предложений
                    if sent_block:
                        if len(current_block) + len(sent_block) + 2 <= max_length:
                            if current_block:
                                current_block += "\n\n" + sent_block
                            else:
                                current_block = sent_block
                        else:
                            blocks.append(current_block)
                            current_block = sent_block
                else:
                    # Если параграф помещается в текущий блок
                    if len(current_block) + len(paragraph) + 2 <= max_length:
                        if current_block:
                            current_block += "\n\n" + paragraph
                        else:
                            current_block = paragraph
                    else:
                        blocks.append(current_block)
                        current_block = paragraph
                        
            # Добавляем последний блок, если он есть
            if current_block:
                blocks.append(current_block)
                
            return blocks
        else:
            # Если абзацы не превышают максимальную длину, группируем их
            for paragraph in paragraphs:
                # Если абзац можно добавить к текущей части без превышения лимита
                if len(current_part) + len(paragraph) + 2 <= max_length:
                    if current_part:
                        current_part += "\n\n" + paragraph
                    else:
                        current_part = paragraph
                else:
                    # Если добавление приведет к превышению, сохраняем текущую часть и начинаем новую
                    parts.append(current_part)
                    current_part = paragraph
                    
            # Добавляем последнюю часть, если она не пуста
            if current_part:
                parts.append(current_part)
                
            return parts
    
    async def safe_edit_message_text(self, query, text: str, reply_markup=None) -> None:
        """
        Безопасно редактирует сообщение в Telegram.
        Если сообщение слишком длинное, разбивает его на части и отправляет как отдельные сообщения.
        
        Args:
            query: Объект запроса от Telegram
            text: Новый текст сообщения
            reply_markup: Опциональная клавиатура для сообщения
        """
        sanitized_text = self.sanitize_text_for_telegram(text)
        if not sanitized_text:
            logging.warning("Попытка отредактировать сообщение с пустым текстом")
            return
        
        # Максимальная длина сообщения в Telegram (с запасом)
        max_length = 3800
        
        try:
            # Если текст короче максимальной длины, редактируем как обычно
            if len(sanitized_text) <= max_length:
                await query.edit_message_text(
                    sanitized_text, 
                    reply_markup=reply_markup
                )
                return
                
            # Для длинных сообщений: обновляем исходное сообщение с уведомлением
            # и отправляем новые сообщения с частями
            try:
                notice_message = await query.edit_message_text(
                    f"Ответ получился большим (примерно {len(sanitized_text) // 1000} Кб), разбиваю на части...",
                    reply_markup=None
                )
            except Exception as e:
                logging.error(f"Ошибка при обновлении исходного сообщения: {str(e)}")
            
            logger.info(f"Разбиваем длинное сообщение на части (длина: {len(sanitized_text)} символов)")
            
            # Определяем примерное количество частей
            estimated_parts = (len(sanitized_text) // max_length) + 1
            
            # Используем новый метод умного разбиения текста
            parts = self._smart_text_split(sanitized_text, max_length, estimated_parts)
            
            # Отправляем части как новые сообщения
            for i, part in enumerate(parts):
                try:
                    # Клавиатуру прикрепляем только к последнему сообщению
                    markup = reply_markup if i == len(parts) - 1 else None
                    
                    # Добавляем нумерацию частей, если их больше одной
                    if len(parts) > 1:
                        part_info = f"📄 Часть {i+1} из {len(parts)} 📄\n\n"
                        if i > 0:
                            part_info += "(Продолжение ответа)\n\n"
                        part = part_info + part
                    
                    # Отправляем часть как новое сообщение
                    await query.message.reply_text(
                        part,
                        reply_markup=markup
                    )
                    
                    # Добавляем небольшую задержку между сообщениями, чтобы избежать ошибок флуда
                    if i < len(parts) - 1:
                        await asyncio.sleep(1.0)
                        
                except Exception as e:
                    logging.error(f"Ошибка при отправке части сообщения {i+1}: {str(e)}")
            
        except Exception as e:
            logging.error(f"Ошибка при редактировании сообщения: {str(e)}")
            logging.error(traceback.format_exc())
            # Если не получилось отредактировать, пробуем отправить новое сообщение
            try:
                await query.message.reply_text(
                    "Произошла ошибка при отправке полного ответа. Попробуйте задать вопрос иначе.",
                    reply_markup=reply_markup
                )
            except Exception as e2:
                logging.error(f"Не удалось отправить новое сообщение после ошибки редактирования: {str(e2)}")
    
    async def process_consultation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработка запроса на консультацию по курсу"""
        try:
            # Получаем ID студента
            user_id = update.effective_user.id
            
            # Получаем вопрос из текста сообщения
            question = update.message.text
            
            # Отправляем уведомление о наборе сообщения (исправленная версия)
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action="typing"  # Используем строковое значение вместо ChatAction.TYPING
            )
            
            # Получаем ответ от UnifiedAssistant
            answer = await self.assistant.answer_question(
                question=question,
                student_id=str(user_id)
            )
            
            # Используем безопасный метод отправки сообщения
            await self.safe_send_message(update, answer)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке вопроса в режиме консультации: {str(e)}")
            logger.error(traceback.format_exc())
            await update.message.reply_text(
                "Произошла ошибка при обработке вашего вопроса. Пожалуйста, попробуйте позже."
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
        
        # Для обычных задач добавляем варианты ответов
        if task["task_type"] in ["multiple_choice", "template"] and "options" in task:
            message += "*Варианты ответов:*\n\n"
            # Нумеруем варианты цифрами (1, 2, 3, 4)
            for i, option in enumerate(task["options"], 1):
                # Сохраняем буквенную метку для API и цифровую для отображения
                letter_label = chr(64 + i)  # 65 - код ASCII для 'A'
                option['label'] = letter_label
                option['display_label'] = str(i)
                
                # Отображаем вариант с цифрой и ограничиваем длину текста опции
                option_text = option['text']
                if len(option_text) > 200:  # Ограничиваем длину текста опции
                    option_text = option_text[:197] + "..."
                message += f"*{i}.* {option_text}\n\n"
        
        # Добавляем критерии оценки для творческой задачи
        if task["task_type"] == "creative" and "criteria" in task:
            message += "\n*Критерии оценки:*\n"
            for criterion in task["criteria"]:
                message += f"• {criterion}\n"
            
            # Если есть пример ответа и это базовый уровень, добавляем его
            if "example_answer" in task and task["example_answer"] and task.get("difficulty", "standard") == "basic":
                example = task['example_answer']
                if len(example) > 300:  # Ограничиваем длину примера
                    example = example[:297] + "..."
                message += "\n*Пример ответа:*\n"
                message += f"{example}\n"
        
        # Добавляем тип и сложность в конце
        difficulty_name = {
            "basic": "Базовый уровень",
            "standard": "Стандартный уровень",
            "advanced": "Продвинутый уровень"
        }.get(task.get("difficulty", "standard"), "Стандартный уровень")
        
        task_type_name = {
            "multiple_choice": "Задача с выбором ответа",
            "template": "Задача с выбором ответа", 
            "creative": "Творческая задача"
        }.get(task.get("task_type", "template"), "Задача с выбором ответа")
        
        message += f"\n\n_Тип: {task_type_name} | Сложность: {difficulty_name}_"
        
        # Безопасно обрабатываем сообщение для Telegram и проверяем длину
        message = self.sanitize_text_for_telegram(message)
        
        return message
    
    def create_options_keyboard(self, task: Dict[str, Any]) -> Optional[InlineKeyboardMarkup]:
        """
        Создает клавиатуру с вариантами ответов для задач с множественным выбором
        
        Args:
            task: Задача
            
        Returns:
            Клавиатура с вариантами ответов или None, если это не задача с множественным выбором
        """
        if task["task_type"] != "multiple_choice" and task["task_type"] != "template":
            return None
        
        if "options" not in task:
            return None
            
        keyboard = []
        row = []
        
        # Создаем кнопки для каждого варианта ответа
        for i, option in enumerate(task["options"], 1):
            # Используем цифры для отображения и callback_data
            display_label = str(i)
            callback_data = f"answer:{display_label}"
            
            # Сохраняем метки в опции
            option['display_label'] = display_label
            
            # Создаем кнопку
            row.append(InlineKeyboardButton(display_label, callback_data=callback_data))
            
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
        
        # Безопасно обрабатываем сообщение для Telegram
        return self.sanitize_text_for_telegram(message)
    
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

    async def menu_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Обработчик нажатий на кнопки главного меню
        
        Args:
            update: Объект обновления
            context: Контекст бота
            
        Returns:
            Следующее состояние диалога
        """
        query = update.callback_query
        await query.answer()
        
        action = query.data
        
        # Вспомогательная функция для безопасной отправки сообщений при колбэках
        async def safe_callback_reply(text, reply_markup=None):
            """
            Безопасно отправляет сообщение, даже если query.message == None
            Разбивает длинные сообщения на части, если они превышают лимит Telegram.
            """
            # Обрабатываем текст для Telegram
            safe_text = self.sanitize_text_for_telegram(text)
            
            # Максимальная длина сообщения в Telegram (с запасом)
            MAX_MESSAGE_LENGTH = 3900
            
            # Функция для отправки сообщения через доступный канал
            async def send_message(text_part, is_last_part=False):
                markup = reply_markup if is_last_part else None
                
                if query.message:
                    try:
                        # Если есть message, используем его
                        return await query.message.reply_text(
                            text=text_part, 
                            reply_markup=markup,
                            parse_mode=ParseMode.MARKDOWN
                        )
                    except Exception as e:
                        # Если Markdown вызвал ошибку, пробуем без него
                        try:
                            return await query.message.reply_text(
                                text=text_part, 
                                reply_markup=markup,
                                parse_mode=None
                            )
                        except Exception as e:
                            logger.error(f"Ошибка при использовании query.message.reply_text без форматирования: {e}")
                
                # Если message недоступен или произошла ошибка, используем эффективный чат
                chat_id = update.effective_chat.id if update.effective_chat else query.from_user.id
                try:
                    return await context.bot.send_message(
                        chat_id=chat_id, 
                        text=text_part, 
                        reply_markup=markup,
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as e:
                    # Если и с Markdown ошибка, пробуем без него
                    try:
                        return await context.bot.send_message(
                            chat_id=chat_id, 
                            text=text_part, 
                            reply_markup=markup,
                            parse_mode=None
                        )
                    except Exception as e:
                        logger.error(f"Критическая ошибка при отправке: {e}")
                        return None
            
            # Разбиваем сообщение на части, если оно слишком длинное
            if len(safe_text) > MAX_MESSAGE_LENGTH:
                # Логируем, что текст будет разбит на части
                logger.info(f"Сообщение в callback длиной {len(safe_text)} символов будет разбито на части")
                
                # Разбиваем текст на логические части (параграфы)
                parts = []
                current_part = ""
                
                # Сначала пытаемся разбить по параграфам (два переноса строки)
                paragraphs = safe_text.split("\n\n")
                
                for paragraph in paragraphs:
                    # Если параграф сам по себе слишком длинный
                    if len(paragraph) > MAX_MESSAGE_LENGTH:
                        # Если текущая часть не пуста, добавляем ее в список частей
                        if current_part:
                            parts.append(current_part)
                            current_part = ""
                        
                        # Разбиваем длинный параграф на части по MAX_MESSAGE_LENGTH
                        for i in range(0, len(paragraph), MAX_MESSAGE_LENGTH):
                            chunk = paragraph[i:i + MAX_MESSAGE_LENGTH]
                            parts.append(chunk)
                    elif len(current_part + "\n\n" + paragraph) <= MAX_MESSAGE_LENGTH:
                        # Если параграф помещается в текущую часть
                        if current_part:
                            current_part += "\n\n"
                        current_part += paragraph
                    else:
                        # Если параграф не помещается, начинаем новую часть
                        parts.append(current_part)
                        current_part = paragraph
                
                # Добавляем последнюю часть, если она не пуста
                if current_part:
                    parts.append(current_part)
                
                # Отправляем части по очереди
                for i, part in enumerate(parts):
                    is_last = (i == len(parts) - 1)
                    await send_message(part, is_last)
                
                return None  # Возвращаем None, так как отправлено несколько сообщений
            else:
                # Если сообщение короткое, отправляем его как обычно
                return await send_message(safe_text, True)

        # Обработка выбора главы
        if action.startswith(PREFIX_CHAPTER):
            # Извлекаем выбранную главу (короткий идентификатор)
            chapter_id = action.replace(PREFIX_CHAPTER, "")
            
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
            from ai_tutor.bot.keyboards import get_task_types_keyboard
            reply_markup = get_task_types_keyboard()
            
            await query.edit_message_text(
                f"Выбрана глава: {chapter}\n\nВыберите тип задачи:",
                reply_markup=reply_markup
            )
            
            return SELECTING_TASK_TYPE
            
        # Обработка выбора типа задачи
        elif action.startswith(PREFIX_TASK_TYPE):
            # Извлекаем выбранный тип задачи
            task_type = action.replace(PREFIX_TASK_TYPE, "")
            task_type_name = TASK_TYPES[task_type]
            
            # Сохраняем выбор в контексте
            context.user_data["task_type"] = task_type
            
            # Создаем клавиатуру с выбором сложности
            from ai_tutor.bot.keyboards import get_difficulty_keyboard
            reply_markup = get_difficulty_keyboard()
            
            await query.edit_message_text(
                f"Выбрана глава: {context.user_data['chapter']}\n"
                f"Выбран тип задачи: {task_type_name}\n\n"
                "Выберите уровень сложности:",
                reply_markup=reply_markup
            )
            
            return SELECTING_DIFFICULTY
            
        # Обработка выбора сложности
        elif action.startswith(PREFIX_DIFFICULTY):
            # Извлекаем выбранную сложность
            difficulty = action.replace(PREFIX_DIFFICULTY, "")
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
                concepts = self.neo4j_client.get_concepts_by_chapter(chapter)
                
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
                related_concepts = self.neo4j_client.get_related_concepts(concept.get('name', ''), chapter)
                
                # Генерируем задачу
                task = await self.openrouter_client.generate_task(
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
                        letter_label = option['label']
                        display_label = option['display_label']
                        row.append(InlineKeyboardButton(display_label, callback_data=f"answer:{display_label}"))
                        
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
                    await self.safe_edit_message_text(query, task_message)
                    
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
                    await self.safe_edit_message_text(query, task_message, reply_markup=reply_markup)
                
                return WAITING_FOR_ANSWER
                
            except Exception as e:
                logger.error(f"Ошибка при генерации задачи: {e}")
                await query.edit_message_text(
                    MESSAGES['error']
                )
                return ConversationHandler.END
                
        # Обработка ответа на задачу
        elif action.startswith("answer:"):
            # Извлекаем выбранный вариант ответа (теперь это цифра)
            selected_display_option = action.replace("answer:", "")
            
            # Получаем диалог пользователя
            user = update.effective_user
            conversation = get_conversation(user.id)
            
            # Проверяем, что есть активная задача
            if not conversation.current_task:
                await safe_callback_reply(
                    "У вас нет активной задачи. Используйте /task, чтобы начать новую задачу."
                )
                return ConversationHandler.END
            
            # Получаем задачу
            task = conversation.current_task
            
            # Проверяем тип задачи
            if task["task_type"] != "multiple_choice" and task["task_type"] != "template":
                await safe_callback_reply(
                    "Это не задача с вариантами ответов."
                )
                return WAITING_FOR_ANSWER
            
            # Находим выбранный вариант ответа по display_label (цифре)
            selected_text = None
            is_correct = False
            selected_option = None  # Буква для API
            for option in task.get("options", []):
                if option.get("display_label") == selected_display_option:
                    selected_text = option.get("text", "")
                    is_correct = option.get("is_correct", False)
                    selected_option = option.get("label")  # Буква (A, B, C...)
                    break
            
            if not selected_text:
                # Собираем доступные варианты ответов с их цифровыми метками
                available_options = []
                for option in task.get("options", []):
                    if "display_label" in option:
                        available_options.append(option["display_label"])
                    else:
                        # Если нет display_label, показываем номер по порядку
                        idx = task.get("options", []).index(option) + 1
                        available_options.append(str(idx))
                
                # Формируем сообщение с доступными вариантами
                options_str = ", ".join(available_options)
                await safe_callback_reply(
                    f"Произошла ошибка при обработке вашего ответа. Доступные варианты: {options_str}. Пожалуйста, попробуйте еще раз."
                )
                return WAITING_FOR_ANSWER
            
            # Добавляем ответ в историю диалога
            conversation.add_message('student', selected_text)
            
            try:
                # Получаем данные для проверки
                concept_name = task.get("concept_name", "")
                
                # Получаем понятие из базы данных
                concept = self.neo4j_client.get_concept_by_name(concept_name, context.user_data["chapter"])
                
                if not concept:
                    await safe_callback_reply(
                        f"К сожалению, не удалось найти понятие '{concept_name}' в базе знаний."
                    )
                    return ConversationHandler.END
                
                # Проверяем ответ
                check_result = await self.openrouter_client.check_answer(
                    task, 
                    selected_text, 
                    concept
                )
                
                # Формируем сообщение с обратной связью
                is_correct = check_result.get("is_correct", False)
                feedback = check_result.get("feedback", "")
                
                # Создаем клавиатуру для дальнейших действий
                from ai_tutor.bot.keyboards import get_feedback_keyboard
                reply_markup = get_feedback_keyboard()
                
                # Формируем сообщение с обратной связью с использованием метода format_feedback_message
                feedback_message = self.format_feedback_message(check_result)
                
                # Отправляем сообщение с обратной связью, используя безопасный метод
                chat_id = update.effective_chat.id
                if chat_id:
                    await self.safe_send_message(
                        update=update,
                        text=feedback_message,
                        reply_markup=reply_markup
                    )
                else:
                    # Запасной вариант - используем safe_callback_reply
                    await safe_callback_reply(
                        feedback_message,
                        reply_markup=reply_markup
                    )
                
                return SHOW_FEEDBACK
                
            except Exception as e:
                logger.error(f"Ошибка при обработке ответа: {e}")
                await safe_callback_reply(
                    "Произошла ошибка при обработке вашего ответа. Пожалуйста, попробуйте еще раз."
                )
                return WAITING_FOR_ANSWER
                
        # Обработка пропуска задачи
        elif action == "skip":
            await safe_callback_reply(
                "Вы пропустили текущую задачу. Хотите получить новую?"
            )
            
            # Создаем клавиатуру для дальнейших действий
            from ai_tutor.bot.keyboards import get_feedback_keyboard
            reply_markup = get_feedback_keyboard()
            
            await safe_callback_reply(
                "Выберите действие:",
                reply_markup=reply_markup
            )
            
            return SHOW_FEEDBACK
            
        # Обработка завершения
        elif action == "end":
            await safe_callback_reply(
                "Спасибо за занятие! Вы можете продолжить обучение в любое время, используя команду /task."
            )
            
            return ConversationHandler.END
            
        # Обработка кнопки "Новая задача" после обратной связи
        elif action == "feedback:new_task":
            # Перенаправляем пользователя к выбору главы
            reply_markup = get_chapters_keyboard()
            await safe_callback_reply(
                "Выберите главу курса:",
                reply_markup=reply_markup
            )
            
            return SELECTING_CHAPTER
            
        # Обработка кнопки "Завершить" после обратной связи
        elif action == "feedback:end":
            await safe_callback_reply(
                "Спасибо за занятие! Вы можете продолжить обучение в любое время, используя команду /task."
            )
            
            return ConversationHandler.END
            
        # Обработка кнопок следующего шага
        elif action.startswith("next_step:"):
            next_action = action.replace("next_step:", "")
            
            if next_action == "change_chapter":
                # Перенаправляем пользователя к выбору главы
                reply_markup = get_chapters_keyboard()
                await safe_callback_reply(
                    "Выберите новую главу курса:",
                    reply_markup=reply_markup
                )
                
                return SELECTING_CHAPTER
                
            elif next_action == "increase_difficulty":
                # Увеличиваем сложность задач
                context.user_data["difficulty"] = "advanced"
                
                await safe_callback_reply(
                    "Уровень сложности повышен до продвинутого. Выберите главу для новой задачи:",
                    reply_markup=get_chapters_keyboard()
                )
                
                return SELECTING_CHAPTER
                
            elif next_action == "change_task_type":
                # Меняем тип задачи
                current_type = context.user_data.get("task_type", "template")
                new_type = "creative" if current_type == "template" else "template"
                context.user_data["task_type"] = new_type
                
                await safe_callback_reply(
                    f"Тип задачи изменен на {TASK_TYPES[new_type]}. Выберите главу для новой задачи:",
                    reply_markup=get_chapters_keyboard()
                )
                
                return SELECTING_CHAPTER
                
            elif next_action == "discuss":
                # Включаем режим обсуждения
                user = update.effective_user
                conversation = get_conversation(user.id)
                conversation.current_state = "discussion"
                save_conversation(conversation)
                
                await safe_callback_reply(
                    "Режим обсуждения активирован. Задавайте вопросы по текущей задаче, и я постараюсь на них ответить.\n\n"
                    "Для выхода из режима обсуждения, нажмите кнопку 'Новая задача' или 'Завершить обсуждение'.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Новая задача", callback_data="feedback:new_task")],
                        [InlineKeyboardButton("Завершить обсуждение", callback_data="feedback:end")]
                    ])
                )
                
                return SHOW_FEEDBACK
                
        elif action == "task":
            # Перенаправляем пользователя к выбору главы
            reply_markup = get_chapters_keyboard()
            await query.edit_message_text(
                "Выберите главу курса:",
                reply_markup=reply_markup
            )
            return SELECTING_CHAPTER
            
        elif action == "random_task":
            # Генерируем случайную задачу
            from random import choice
            
            # Выбираем случайную главу, тип задачи и сложность
            chapter = choice(CHAPTERS)
            task_type = choice(list(TASK_TYPES.keys()))
            difficulty = "standard"  # Стандартная сложность для случайных задач
            
            # Сохраняем выбор в контексте
            context.user_data["chapter"] = chapter
            context.user_data["task_type"] = task_type
            context.user_data["difficulty"] = difficulty
            
            # Получаем пользователя и диалог
            user = update.effective_user
            conversation = get_conversation(user.id)
            
            # Создаем задачу
            try:
                # Получаем понятия из выбранной главы
                concepts = self.neo4j_client.get_concepts_by_chapter(chapter)
                
                if not concepts:
                    await query.edit_message_text(
                        f"К сожалению, не удалось найти понятия для главы '{chapter}'.\n"
                        f"Попробуйте выбрать другую главу или обратитесь к администратору."
                    )
                    return ConversationHandler.END
                
                # Выбираем случайное понятие
                import random
                concept = random.choice(concepts)
                
                # Получаем связанные понятия
                related_concepts = self.neo4j_client.get_related_concepts(concept.get('name', ''), chapter)
                
                # Генерируем задачу
                task = await self.openrouter_client.generate_task(
                    concept, 
                    related_concepts, 
                    task_type, 
                    difficulty
                )
                
                # Устанавливаем текущую задачу
                conversation.set_current_task(task)
                save_conversation(conversation)
                
                # Форматируем и отправляем задачу
                task_message = conversation.format_task_for_display()
                
                # Добавляем клавиатуру с вариантами ответов или кнопками управления
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
                    
                    # Обновляем задачу
                    conversation.set_current_task(task)
                    save_conversation(conversation)
                    
                    # Создаем клавиатуру с вариантами ответов
                    keyboard = []
                    row = []
                    
                    # Добавляем кнопки для каждого варианта
                    for i, option in enumerate(options):
                        letter_label = option['label']
                        display_label = option['display_label']
                        row.append(InlineKeyboardButton(display_label, callback_data=f"answer:{display_label}"))
                        
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
                    await self.safe_edit_message_text(query, task_message)
                    
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
                    await self.safe_edit_message_text(query, task_message, reply_markup=reply_markup)
                
                return WAITING_FOR_ANSWER
            
            except Exception as e:
                logger.error(f"Ошибка при генерации случайной задачи: {e}")
                await query.edit_message_text(
                    "К сожалению, произошла ошибка при генерации задачи. Пожалуйста, попробуйте еще раз."
                )
                return ConversationHandler.END
        
        elif action == "change_chapter":
            # Перенаправляем пользователя к выбору главы
            reply_markup = get_chapters_keyboard()
            await query.edit_message_text(
                "Выберите новую главу курса:",
                reply_markup=reply_markup
            )
            return SELECTING_CHAPTER
        
        elif action == "consultant":
            # Активируем режим консультации
            context.user_data['consultation_mode'] = True
            
            await query.edit_message_text(
                "Режим консультации активирован. Задайте свой вопрос, и я постараюсь на него ответить.\n\n"
                "Для выхода из режима консультации введите /cancel"
            )
            
            return ConversationHandler.END
        
        return ConversationHandler.END
