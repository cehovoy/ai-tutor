"""
Модуль для работы с OpenRouter API для доступа к модели Grok
"""
from typing import Dict, List, Any, Optional
import json
import logging
import re

from openai import OpenAI
import httpx

from ai_tutor.config.settings import OPENROUTER_API_KEY, OPENROUTER_API_URL, GROK_MODEL

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Клиент для работы с OpenRouter API для доступа к модели Grok
    """
    
    def __init__(self, api_key: str = OPENROUTER_API_KEY, api_url: str = OPENROUTER_API_URL,
                 model: str = GROK_MODEL):
        """
        Инициализация клиента OpenRouter
        
        Args:
            api_key: Ключ API OpenRouter
            api_url: URL API OpenRouter
            model: Модель для использования (например, "x-ai/grok-2-1212")
        """
        self.api_key = api_key
        self.model = model
        
        # Проверка наличия API ключа
        if not api_key or api_key.startswith("sk-") is False:
            logger.warning(f"API ключ OpenRouter отсутствует или имеет неверный формат. "
                         f"Текущее значение: '{api_key if api_key else 'Не указан'}'")
        else:
            # Скрываем часть ключа для безопасности
            visible_part = api_key[:5] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
            logger.info(f"Инициализация OpenRouter клиента с ключом {visible_part}, модель: {model}")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.extra_headers = {
            "HTTP-Referer": "https://ai-tutor.ru",  # Укажите ваш домен
            "X-Title": "AI Tutor System"
        }
    
    async def generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Генерация завершений с помощью модели через OpenRouter API
        
        Args:
            messages: Список сообщений для контекста
            temperature: Температура генерации (разнообразие)
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Ответ от API
        """
        try:
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Преобразуем объект в словарь для совместимости с существующим кодом
            response = {
                "choices": [
                    {
                        "message": {
                            "content": completion.choices[0].message.content,
                            "role": completion.choices[0].message.role
                        },
                        "index": completion.choices[0].index,
                        "finish_reason": completion.choices[0].finish_reason
                    }
                ],
                "id": completion.id,
                "model": completion.model,
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }
            }
            
            return response
        except Exception as e:
            logger.error(f"Ошибка при генерации завершения: {str(e)}")
            raise
    
    async def generate_task(
        self, 
        concept: Dict[str, Any], 
        related_concepts: List[Dict[str, Any]], 
        task_type: str, 
        difficulty: str
    ) -> Dict[str, Any]:
        """
        Генерация задачи на основе понятия и связанных понятий
        
        Args:
            concept: Понятие из графа знаний
            related_concepts: Связанные понятия
            task_type: Тип задачи ("template" или "creative")
            difficulty: Уровень сложности ("standard" или "advanced")
            
        Returns:
            Сгенерированная задача
        """
        try:
            # Проверка API ключа
            if not self.api_key:
                logger.error("Не задан API ключ для OpenRouter. Невозможно сгенерировать задачу.")
                raise ValueError("API ключ отсутствует")
                
            # Проверка типа и содержания аргумента concept
            if not isinstance(concept, dict):
                logger.error(f"Аргумент concept должен быть словарем, получено: {type(concept)}")
                raise TypeError(f"Аргумент concept должен быть словарем, получено: {type(concept)}")
                
            # Проверяем наличие необходимых полей в concept
            if not concept.get('name'):
                logger.error("Понятие не содержит поля 'name'")
                raise ValueError("Некорректное понятие - отсутствует имя")
                
            if not concept.get('definition'):
                logger.warning(f"Понятие {concept.get('name')} не содержит определения")
                # Устанавливаем пустое определение, чтобы избежать ошибок
                concept['definition'] = f"Определение для понятия {concept.get('name')} отсутствует"
                
            logger.info(f"Генерируем задачу для понятия '{concept.get('name')}', "
                        f"тип: {task_type}, сложность: {difficulty}")
            
            # Проверка правильности типа задачи
            if task_type not in ["template", "creative"]:
                logger.warning(f"Неизвестный тип задачи: {task_type}, используем template")
                task_type = "template"
                
            # Проверка правильности уровня сложности
            if difficulty not in ["standard", "advanced"]:
                logger.warning(f"Неизвестный уровень сложности: {difficulty}, используем standard")
                difficulty = "standard"
            
            # Формируем контекст для модели
            concept_info = f"Понятие: {concept['name']}\nОпределение: {concept['definition']}"
            if concept.get('example'):
                concept_info += f"\nПример: {concept['example']}"
            
            related_info = ""
            if related_concepts:
                # Проверяем, что related_concepts - список
                if not isinstance(related_concepts, list):
                    logger.warning(f"related_concepts должен быть списком, получено: {type(related_concepts)}")
                    related_concepts = []
                else:
                    related_info = "Связанные понятия:\n"
                    for i, rc in enumerate(related_concepts):
                        # Проверяем, что каждый элемент - словарь
                        if not isinstance(rc, dict):
                            logger.warning(f"Элемент {i} в related_concepts должен быть словарем")
                            continue
                        
                        # Проверяем наличие необходимых полей
                        if not rc.get('name') or not rc.get('definition'):
                            logger.warning(f"Связанное понятие {i} не содержит имя или определение")
                            continue
                            
                        related_info += f"- {rc['name']} ({rc.get('relation_type', 'связано с')}): {rc['definition']}\n"
            
            task_description = ""
            if task_type == "template":
                task_description = (
                    "Создай шаблонную задачу с множественным выбором (4 варианта ответа, только один правильный) "
                    "на основе данного понятия. Задача должна проверять именно понимание понятия, "
                    "а не просто факты. Важно, чтобы студент действительно осознал суть понятия и его место "
                    "в системе знаний."
                )
                if difficulty == "advanced":
                    task_description += (
                        "Задача должна быть продвинутого уровня, требующей глубокого понимания понятия и его связей. "
                        "Включи связанные понятия в формулировку задачи, чтобы проверить, как студент понимает "
                        "взаимосвязи между разными элементами системы."
                    )
                else:
                    task_description += (
                        "Задача должна быть стандартного уровня, доступная для понимания. "
                        "Сфокусируйся на основных аспектах понятия и его ключевых характеристиках."
                    )
            else:  # creative
                task_description = (
                    "Создай творческую задачу, требующую развёрнутого ответа, "
                    "на основе данного понятия. Задача должна вести к более глубокому усвоению понятия "
                    "через размышление и творческое применение. Важно, чтобы студент не просто запомнил определение, "
                    "а осмыслил понятие и научился использовать его в разных контекстах."
                )
                if difficulty == "advanced":
                    task_description += (
                        "Задача должна быть продвинутого уровня, требовать анализа и синтеза информации. "
                        "Включи связанные понятия в формулировку задачи, чтобы студент мог построить "
                        "целостную картину и увидеть, как понятия образуют систему."
                    )
                else:
                    task_description += (
                        "Задача должна быть стандартного уровня, доступная для понимания. "
                        "Сфокусируйся на практическом применении понятия и его связи с реальными ситуациями."
                    )
            
            # Примеры вопросов из базы (если есть)
            question_examples = ""
            if concept.get('questions'):
                question_examples = "Примеры вопросов по данному понятию:\n"
                for q in concept['questions']:
                    question_examples += f"- {q}\n"
            
            format_instructions = ""
            if task_type == "template":
                format_instructions = """
                Формат вывода задачи должен быть следующим:
                ```json
                {
                    "question": "Текст вопроса",
                    "options": [
                        {"label": "A", "text": "Вариант A", "is_correct": true/false, "explanation": "Объяснение"},
                        {"label": "B", "text": "Вариант B", "is_correct": true/false, "explanation": "Объяснение"},
                        {"label": "C", "text": "Вариант C", "is_correct": true/false, "explanation": "Объяснение"},
                        {"label": "D", "text": "Вариант D", "is_correct": true/false, "explanation": "Объяснение"}
                    ],
                    "hints": ["Подсказка 1", "Подсказка 2"]
                }
                ```
                Среди вариантов должен быть только один правильный. Для каждого варианта укажи краткое объяснение, почему он правильный или неправильный.
                Добавь 2-3 полезные подсказки, которые могут помочь студенту выбрать правильный ответ.
                """
            else:  # creative
                format_instructions = """
                Формат вывода задачи должен быть следующим:
                ```json
                {
                    "question": "Текст вопроса или задания",
                    "criteria": ["Критерий 1 для оценки ответа", "Критерий 2", "Критерий 3"],
                    "example_answer": "Пример хорошего ответа",
                    "hints": ["Подсказка 1", "Подсказка 2"]
                }
                ```
                """
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Ты - ИИ-репетитор для студентов, изучающих курс 'Системное саморазвитие'. "
                        "Твоя задача - создавать учебные задачи для проверки знаний студентов. "
                        "ВАЖНО: В школе системного менеджмента изучение и усвоение понятий - это самое важное. "
                        "Все задачи, которые ты создаешь, должны быть НАЦЕЛЕНЫ НА ПРОВЕРКУ ЗНАНИЙ ПОНЯТИЙ "
                        "и их лучшего усвоения. Сфокусируйся на точном определении и понимании понятий, "
                        "их взаимосвязях и практическом применении. "
                        "Задачи должны быть связаны с понятиями из графа знаний и адаптированы "
                        "под уровень сложности."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"{concept_info}\n\n"
                        f"{related_info}\n\n"
                        f"{question_examples}\n\n"
                        f"{task_description}\n\n"
                        f"{format_instructions}"
                    )
                }
            ]
            
            try:
                logger.info("Отправляем запрос к OpenRouter API")
                response = await self.generate_completion(messages)
                
                if not response or not response.get('choices'):
                    logger.error(f"Неожиданный формат ответа от API: {response}")
                    raise ValueError("Неожиданный формат ответа от API")
                    
                content = response['choices'][0]['message']['content']
                logger.info("Получен ответ от OpenRouter API")
                
                # Извлекаем JSON данные между метками ```json и ```
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
                
                if match:
                    json_str = match.group(1)
                    logger.info("Извлечены JSON данные между метками ```json```")
                    task = json.loads(json_str)
                    logger.info("JSON данные успешно преобразованы в словарь")
                else:
                    # Если данные не обернуты в тройные обратные кавычки
                    # Пытаемся извлечь JSON напрямую
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    
                    if json_start != -1 and json_end != -1:
                        json_str = content[json_start:json_end+1]
                        task = json.loads(json_str)
                        logger.info("JSON данные извлечены непосредственно из текста")
                    else:
                        # Если не удалось извлечь JSON, используем структуру ответа по умолчанию
                        raise ValueError("Не удалось найти JSON данные в ответе")
                
                # Проверяем и структурируем данные
                if task_type == "template":
                    # Проверяем наличие вопроса
                    if not task.get("question"):
                        task["question"] = f"Что такое {concept['name']}?"
                        logger.warning("Отсутствует вопрос в шаблонной задаче, добавлен вопрос по умолчанию")
                    
                    # Проверяем наличие вариантов ответов
                    if not task.get("options") or len(task.get("options", [])) < 2:
                        correct_option = {"label": "A", "text": concept['definition'], "is_correct": True}
                        task["options"] = [correct_option]
                        
                        # Добавляем неверные варианты
                        for i, related in enumerate(related_concepts[:3], 1):
                            option = {
                                "label": chr(65 + i),  # B, C, D
                                "text": related.get('definition', f"Неверное определение {i}"),
                                "is_correct": False
                            }
                            task["options"].append(option)
                        
                        logger.warning("Отсутствуют или недостаточно вариантов ответов, добавлены варианты по умолчанию")
                    
                    # Проверяем наличие всех необходимых полей в опциях        
                    for i, option in enumerate(task.get("options", [])):
                        if "label" not in option:
                            option["label"] = chr(65 + i)  # A, B, C, D
                            logger.warning(f"Добавлена метка {option['label']} для варианта {i}")
                        if "explanation" not in option:
                            option["explanation"] = "Пояснение отсутствует."
                            logger.warning(f"Добавлено пояснение по умолчанию для варианта {option['label']}")
                    
                    # Добавляем подсказки, если их нет
                    if not task.get("hints"):
                        definition = concept.get('definition', '')
                        # Определяем ключевые слова из определения
                        key_words = [word for word in definition.split() if len(word) > 5][:3]
                        task["hints"] = [
                            f"Обратите внимание на ключевые элементы определения: {', '.join(key_words)}",
                            f"Подумайте о том, как {concept['name']} соотносится с другими понятиями в данной главе."
                        ]
                        logger.warning("Добавлены подсказки по умолчанию для шаблонной задачи")
                else:  # creative
                    # Проверяем наличие необходимых полей для творческой задачи
                    if not task.get("question"):
                        task["question"] = f"Опишите своими словами, что такое {concept['name']} и как это понятие применяется на практике."
                        logger.warning("Отсутствует вопрос в творческой задаче, добавлен вопрос по умолчанию")
                    
                    if not task.get("criteria"):
                        task["criteria"] = [
                            "Точность определения",
                            "Глубина понимания",
                            "Примеры применения"
                        ]
                        logger.warning("Отсутствуют критерии в творческой задаче, добавлены критерии по умолчанию")
                    
                    if not task.get("example_answer"):
                        task["example_answer"] = f"{concept['definition']} Это понятие можно применить в следующих ситуациях..."
                        logger.warning("Отсутствует пример ответа в творческой задаче, добавлен пример по умолчанию")
            
                # Добавляем метаданные
                task['concept_name'] = concept['name']
                task['task_type'] = task_type
                task['difficulty'] = difficulty
                
                logger.info("Задача успешно создана и структурирована")
                return task
            except Exception as api_error:
                logger.error(f"Ошибка при вызове API: {str(api_error)}")
                raise
        except Exception as e:
            logger.error(f"Ошибка при генерации задачи: {str(e)}")
            # Возвращаем запасной вариант задачи
            if task_type == "template":
                return {
                    "question": f"Что такое {concept.get('name', 'понятие')}?",
                    "options": [
                        {"label": "A", "text": concept.get('definition', 'Правильное определение'), "is_correct": True, 
                         "explanation": "Это правильное определение понятия."},
                        {"label": "B", "text": "Неверное определение 1", "is_correct": False, 
                         "explanation": "Это неверное определение."},
                        {"label": "C", "text": "Неверное определение 2", "is_correct": False, 
                         "explanation": "Это неверное определение."},
                        {"label": "D", "text": "Неверное определение 3", "is_correct": False, 
                         "explanation": "Это неверное определение."}
                    ],
                    "hints": [
                        "Обратите внимание на ключевые элементы определения.",
                        "Подумайте о главных характеристиках этого понятия."
                    ],
                    "concept_name": concept.get('name', 'понятие'),
                    "task_type": task_type,
                    "difficulty": difficulty
                }
            else:
                return {
                    "question": f"Опишите своими словами, что такое {concept.get('name', 'понятие')} и как это понятие применяется на практике.",
                    "criteria": [
                        "Точность определения",
                        "Глубина понимания",
                        "Примеры применения"
                    ],
                    "example_answer": f"{concept.get('definition', 'Определение понятия')} Это понятие можно применить в следующих ситуациях...",
                    "concept_name": concept.get('name', 'понятие'),
                    "task_type": task_type,
                    "difficulty": difficulty
                }
    
    async def check_answer(
        self, 
        task: Dict[str, Any], 
        student_answer: str, 
        concept: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Проверка ответа студента
        
        Args:
            task: Задача
            student_answer: Ответ студента
            concept: Понятие, к которому относится задача
            
        Returns:
            Результат проверки
        """
        task_type = task.get("task_type", "")
        
        if task_type == "template":
            # Проверяем ответ на шаблонную задачу с множественным выбором
            answer_label = student_answer.strip().upper()
            
            # Ищем опцию с указанной меткой
            correct_option = None
            selected_option = None
            
            for option in task.get("options", []):
                if option.get("is_correct", False):
                    correct_option = option
                
                if option.get("label", "").upper() == answer_label:
                    selected_option = option
            
            if not selected_option:
                return {
                    "is_correct": False,
                    "feedback": f"Вы выбрали несуществующий вариант. Доступные варианты: A, B, C, D."
                }
            
            is_correct = selected_option.get("is_correct", False)
            
            if is_correct:
                feedback = selected_option.get("explanation", "Верно!")
            else:
                # Находим цифровую метку для правильного ответа (для отображения)
                correct_display_label = ""
                for i, opt in enumerate(task.get("options", []), 1):
                    if opt.get("is_correct", False):
                        correct_display_label = str(i)
                        break
                
                feedback = (
                    f"{selected_option.get('explanation', 'Неверно.')} "
                    f"Правильный ответ: {correct_display_label}. "
                    f"{correct_option.get('explanation', '')}"
                )
            
            return {
                "is_correct": is_correct,
                "feedback": feedback
            }
        else:
            # Проверяем ответ на творческую задачу с использованием мотивационного интервьюирования
            criteria = task.get("criteria", [])
            criteria_text = "\n".join([f"- {c}" for c in criteria])
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Ты - ИИ-репетитор, использующий принципы мотивационного интервьюирования для обратной связи. "
                        "ВАЖНО: Твоя цель - создать поддерживающую, эмпатичную среду, где студент чувствует, что его слышат и ценят. "
                        "Используй эти основные принципы мотивационного интервьюирования:\n"
                        "1. Выражай эмпатию: признавай усилия студента, даже если ответ неверный\n"
                        "2. Развивай несоответствие: мягко указывай на расхождение между тем, что студент знает и что могло бы быть улучшено\n"
                        "3. Избегай споров: не критикуй напрямую, а предлагай альтернативные точки зрения\n"
                        "4. Поддерживай самоэффективность: подчеркивай способность студента улучшить свое понимание\n\n"
                        "Оценивая ответ студента, в первую очередь проверь, насколько глубоко и точно усвоено основное понятие, "
                        "о котором идет речь в задаче. Особое внимание обрати на то, понимает ли студент "
                        "суть понятия, его место в системе знаний и может ли применять его на практике. "
                        "Оцени ответ по 10-балльной шкале, где: "
                        "8-10 баллов - глубокое понимание понятия и его применения; "
                        "5-7 баллов - основное понимание понятия присутствует, но есть неточности; "
                        "1-4 балла - слабое понимание понятия или его неверное применение. "
                        "Твоя цель - дать объективную оценку и конструктивную обратную связь по ответу студента."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Задача: {task.get('question', '')}\n\n"
                        f"Понятие: {concept.get('name', '')}\n"
                        f"Определение понятия: {concept.get('definition', '')}\n\n"
                        f"Критерии оценки:\n{criteria_text}\n\n"
                        f"Пример хорошего ответа: {task.get('example_answer', '')}\n\n"
                        f"Ответ студента: {student_answer}\n\n"
                        "Оцени ответ студента по указанным критериям. В формате JSON укажи следующие поля:\n"
                        "- is_correct (true/false): верно ли студент понял основную суть понятия\n"
                        "- score (число от 1 до 10): общая оценка ответа\n" 
                        "- feedback (текст): эмпатичная обратная связь, которая начинается с признания усилий студента\n"
                        "- strengths (массив строк): 2-3 сильные стороны ответа\n"
                        "- improvements (массив строк): 2-3 области для улучшения, сформулированные как открытые вопросы\n"
                        "- reflection_questions (массив строк): 1-2 вопроса для размышления, которые помогут студенту глубже понять понятие\n\n"
                        "Используй принципы мотивационного интервьюирования в своей обратной связи."
                    )
                }
            ]
            
            try:
                response = await self.generate_completion(messages)
                content = response['choices'][0]['message']['content']
                
                # Пытаемся извлечь JSON с результатом оценки
                json_start = content.find('{')
                json_end = content.rfind('}')
                
                if json_start != -1 and json_end != -1:
                    result_json = content[json_start:json_end+1]
                    try:
                        result = json.loads(result_json)
                        
                        # Проверяем, что в результате есть нужные поля
                        if 'is_correct' in result and 'feedback' in result:
                            # Формируем улучшенную обратную связь с элементами мотивационного интервьюирования
                            feedback = result.get('feedback', '')
                            
                            # Добавляем сильные стороны
                            strengths = result.get('strengths', [])
                            if strengths:
                                feedback += "\n\n*Сильные стороны вашего ответа:*\n"
                                for strength in strengths:
                                    feedback += f"• {strength}\n"
                            
                            # Добавляем области для улучшения
                            improvements = result.get('improvements', [])
                            if improvements:
                                feedback += "\n*Для размышления:*\n"
                                for improvement in improvements:
                                    feedback += f"• {improvement}\n"
                            
                            # Добавляем вопросы для рефлексии
                            reflection_questions = result.get('reflection_questions', [])
                            if reflection_questions:
                                feedback += "\n*Вопросы для углубления понимания:*\n"
                                for question in reflection_questions:
                                    feedback += f"• {question}\n"
                            
                            result['feedback'] = feedback
                            return result
                    except json.JSONDecodeError:
                        # Если не удалось декодировать JSON, используем весь текст как обратную связь
                        logger.warning(f"Не удалось декодировать JSON из ответа: {content}")
                
                # Если не удалось извлечь и обработать JSON, используем эвристики для определения оценки
                is_correct = 'правильный' in content.lower() or 'верный' in content.lower() or 'глубокое понимание' in content.lower()
                
                # Создаем базовую структуру ответа
                feedback = "Я ценю твое стремление разобраться в этом понятии. "
                if is_correct:
                    feedback += "Твой ответ показывает понимание ключевых аспектов. " + content
                else:
                    feedback += "Давай вместе подумаем, как можно углубить понимание этого понятия. " + content
                
                return {
                    "is_correct": is_correct,
                    "feedback": feedback,
                    "strengths": [],
                    "improvements": [
                        "Как бы ты связал(а) это понятие с другими концепциями, которые мы изучали?",
                        "Какие примеры из реальной жизни могли бы проиллюстрировать это понятие?"
                    ],
                    "reflection_questions": [
                        "Что было бы для тебя самым полезным, чтобы лучше понять это понятие?"
                    ]
                }
            except Exception as e:
                logger.error(f"Ошибка при проверке ответа: {str(e)}")
                # Возвращаем базовый ответ в случае ошибки с элементами мотивационного интервьюирования
                return {
                    "is_correct": False,
                    "feedback": "Спасибо за твой ответ. К сожалению, у нас возникли технические трудности при проверке. Не волнуйся, это не связано с качеством твоего ответа. Попробуем еще раз?",
                    "strengths": [],
                    "improvements": [],
                    "reflection_questions": []
                }
