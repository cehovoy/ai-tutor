"""
Клиент для работы с моделью Grok через OpenRouter
"""
from typing import Dict, List, Any
import logging

from openai import OpenAI

from ai_tutor.config.settings import OPENROUTER_API_KEY, GROK_MODEL

logger = logging.getLogger(__name__)


class GrokClient:
    """
    Клиент для работы с Grok через OpenRouter
    """
    
    def __init__(self, api_key: str = OPENROUTER_API_KEY, model: str = GROK_MODEL):
        """
        Инициализация клиента Grok
        
        Args:
            api_key: Ключ API OpenRouter
            model: Модель Grok для использования
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.extra_headers = {
            "HTTP-Referer": "https://ai-tutor.ru",
            "X-Title": "AI Tutor System"
        }
    
    async def generate_completion(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000,
        system_prompt: str = None
    ) -> str:
        """
        Генерация текста с помощью Grok
        
        Args:
            prompt: Запрос/инструкция
            temperature: Температура генерации (разнообразие)
            max_tokens: Максимальное количество токенов в ответе
            system_prompt: Системный промпт (если нужен)
            
        Returns:
            Сгенерированный текст
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка при генерации текста Grok: {str(e)}")
            raise
    
    async def ask_question(self, question: str, context: str = "") -> str:
        """
        Задать вопрос модели Grok
        
        Args:
            question: Вопрос
            context: Контекст (если необходим)
            
        Returns:
            Ответ на вопрос
        """
        prompt = question
        if context:
            prompt = f"Контекст: {context}\n\nВопрос: {question}"
        
        system_prompt = (
            "Ты - ИИ-репетитор, помогающий студентам изучать курс 'Системное саморазвитие'. "
            "Отвечай на вопросы ясно, точно и по существу."
        )
        
        return await self.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5
        )
