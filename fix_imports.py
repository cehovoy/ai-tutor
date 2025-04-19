#!/usr/bin/env python3
"""
Скрипт для исправления импортов с префиксом ai_tutor
"""
import os
import re
import sys
from pathlib import Path

def find_files_with_ai_tutor_imports(directory):
    """
    Находит все файлы с импортами ai_tutor
    """
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "from ai_tutor." in content or "import ai_tutor." in content:
                        result.append((file_path, content))
    return result

def print_files_with_ai_tutor_imports(files_with_imports):
    """
    Выводит найденные файлы с импортами
    """
    print(f"Найдено {len(files_with_imports)} файлов с импортами ai_tutor:")
    for file_path, _ in files_with_imports:
        print(f"  {file_path}")

def fix_imports(files_with_imports):
    """
    Исправляет импорты в файлах
    """
    for file_path, content in files_with_imports:
        print(f"Исправление импортов в файле {file_path}...")
        
        # Заменяем импорты
        new_content = re.sub(r'from ai_tutor\.', 'from ', content)
        new_content = re.sub(r'import ai_tutor\.', 'import ', new_content)
        
        # Записываем изменения
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()
    
    print(f"Поиск файлов с импортами ai_tutor в директории {directory}...")
    files_with_imports = find_files_with_ai_tutor_imports(directory)
    print_files_with_ai_tutor_imports(files_with_imports)
    
    if files_with_imports:
        answer = input("Исправить импорты? (y/n): ")
        if answer.lower() == 'y':
            fix_imports(files_with_imports)
            print("Импорты исправлены.")
        else:
            print("Операция отменена.")
    else:
        print("Файлы с импортами ai_tutor не найдены.") 