"""
Модели данных для работы с Neo4j
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Concept:
    """Модель понятия в графе знаний"""
    name: str
    definition: str
    chapter: str
    example: Optional[str] = None
    difficulty: str = "basic"
    id: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для Neo4j"""
        return {
            "name": self.name,
            "definition": self.definition,
            "chapter": self.chapter,
            "example": self.example,
            "difficulty": self.difficulty,
            **self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Concept':
        """Создание объекта из словаря Neo4j"""
        concept_id = data.pop("id", None)
        name = data.pop("name", "")
        definition = data.pop("definition", "")
        chapter = data.pop("chapter", "")
        example = data.pop("example", None)
        difficulty = data.pop("difficulty", "basic")
        
        return cls(
            id=concept_id,
            name=name,
            definition=definition,
            chapter=chapter,
            example=example,
            difficulty=difficulty,
            properties=data
        )


@dataclass
class Task:
    """Модель задачи"""
    question: str
    task_type: str  # "multiple_choice" или "creative"
    difficulty: str  # "basic" или "advanced"
    concept_name: str
    options: List[Dict[str, Any]] = field(default_factory=list)  # для multiple_choice
    criteria: List[str] = field(default_factory=list)  # для creative
    hints: List[str] = field(default_factory=list)
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для Neo4j"""
        return {
            "question": self.question,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "concept_name": self.concept_name,
            "options": self.options,
            "criteria": self.criteria,
            "hints": self.hints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Создание объекта из словаря Neo4j"""
        task_id = data.pop("id", None)
        question = data.pop("question", "")
        task_type = data.pop("task_type", "")
        difficulty = data.pop("difficulty", "")
        concept_name = data.pop("concept_name", "")
        options = data.pop("options", [])
        criteria = data.pop("criteria", [])
        hints = data.pop("hints", [])
        
        return cls(
            id=task_id,
            question=question,
            task_type=task_type,
            difficulty=difficulty,
            concept_name=concept_name,
            options=options,
            criteria=criteria,
            hints=hints
        )


@dataclass
class Student:
    """Модель студента"""
    telegram_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    current_chapter: str = ""
    tasks_completed: int = 0
    correct_answers: int = 0
    last_active: datetime = field(default_factory=datetime.now)
    mastered_concepts: List[str] = field(default_factory=list)
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для Neo4j"""
        return {
            "telegram_id": self.telegram_id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "current_chapter": self.current_chapter,
            "tasks_completed": self.tasks_completed,
            "correct_answers": self.correct_answers,
            "last_active": self.last_active.isoformat(),
            "mastered_concepts": self.mastered_concepts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Student':
        """Создание объекта из словаря Neo4j"""
        student_id = data.pop("id", None)
        telegram_id = data.pop("telegram_id", 0)
        username = data.pop("username", None)
        first_name = data.pop("first_name", None)
        last_name = data.pop("last_name", None)
        current_chapter = data.pop("current_chapter", "")
        tasks_completed = data.pop("tasks_completed", 0)
        correct_answers = data.pop("correct_answers", 0)
        
        last_active_str = data.pop("last_active", None)
        last_active = datetime.now()
        if last_active_str:
            try:
                last_active = datetime.fromisoformat(last_active_str)
            except ValueError:
                pass
        
        mastered_concepts = data.pop("mastered_concepts", [])
        
        return cls(
            id=student_id,
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            current_chapter=current_chapter,
            tasks_completed=tasks_completed,
            correct_answers=correct_answers,
            last_active=last_active,
            mastered_concepts=mastered_concepts
        )


@dataclass
class StudentAnswer:
    """Модель ответа студента на задачу"""
    student_id: int
    task_id: int
    answer_text: str
    is_correct: bool
    feedback: str
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для Neo4j"""
        return {
            "student_id": self.student_id,
            "task_id": self.task_id,
            "answer_text": self.answer_text,
            "is_correct": self.is_correct,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudentAnswer':
        """Создание объекта из словаря Neo4j"""
        answer_id = data.pop("id", None)
        student_id = data.pop("student_id", 0)
        task_id = data.pop("task_id", 0)
        answer_text = data.pop("answer_text", "")
        is_correct = data.pop("is_correct", False)
        feedback = data.pop("feedback", "")
        
        timestamp_str = data.pop("timestamp", None)
        timestamp = datetime.now()
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                pass
        
        return cls(
            id=answer_id,
            student_id=student_id,
            task_id=task_id,
            answer_text=answer_text,
            is_correct=is_correct,
            feedback=feedback,
            timestamp=timestamp
        )
