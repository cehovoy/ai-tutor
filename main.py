"""
Точка входа в приложение ИИ-репетитор
"""
import asyncio
import logging
import os
import signal
from dotenv import load_dotenv

# Заменяем импорты из ai_tutor на относительные импорты
from config.settings import TELEGRAM_TOKEN
from bot.telegram_bot import TelegramBot
from database.neo4j_client import Neo4jClient

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальная переменная для хранения бота
bot = None

# Обработчик сигналов для корректного завершения бота
async def shutdown(signal, loop):
    """
    Корректное завершение бота при получении сигнала завершения
    """
    logger.info(f"Получен сигнал {signal.name}, завершение работы...")
    if bot:
        await bot.stop()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()
    logger.info("Бот остановлен")


async def init_bot():
    """
    Асинхронная инициализация бота
    """
    global bot
    
    logger.info("Инициализация ИИ-репетитора")
    
    if not TELEGRAM_TOKEN:
        logger.error("Не задан токен Telegram бота. Убедитесь, что файл .env содержит TELEGRAM_TOKEN")
        return None
    
    # Создание и инициализация Telegram-бота
    try:
        # Проверяем соединение с Neo4j
        neo4j_client = Neo4jClient()
        
        # Инициализируем бота без запуска
        bot = TelegramBot(token=TELEGRAM_TOKEN)
        # Не запускаем бота здесь, это будет сделано через run_polling() позже
        
        return bot
    except Exception as e:
        logger.error(f"Ошибка при инициализации бота: {e}")
        return None


async def main():
    """
    Основная функция для запуска бота
    """
    logger.info("Запуск ИИ-репетитора")
    
    # Инициализация бота
    initialized_bot = await init_bot()
    
    if initialized_bot:
        # Вывод сообщения о том, что бот успешно запущен
        logger.info("Бот успешно запущен и ожидает сообщений. Нажмите Ctrl+C для завершения.")
        
        # Ждем завершения работы бота
        try:
            while True:
                await asyncio.sleep(3600)  # Проверка каждый час
                logger.info("Бот работает")
        except asyncio.CancelledError:
            logger.info("Задача ожидания отменена")
    else:
        logger.error("Не удалось запустить бота")


if __name__ == "__main__":
    # Запускаем в синхронном режиме
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Настраиваем обработчики сигналов
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop))
        )
    
    try:
        # Инициализируем бота асинхронно
        bot = loop.run_until_complete(init_bot())
        
        if bot:
            # Запускаем polling синхронно - это блокирующий вызов
            # до нажатия Ctrl+C
            bot.run_polling()
        else:
            logger.error("Не удалось инициализировать бота")
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, завершение работы...")
    finally:
        logger.info("Закрытие цикла событий")
        loop.close()
