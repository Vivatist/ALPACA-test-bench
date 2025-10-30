"""
Процессоры для очистки и нормализации извлеченного текста.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.base import BaseCleaner
from ..utils.logger import get_logger, log_processing_stage

logger = get_logger(__name__)


class BasicTextCleaner(BaseCleaner):
    """Базовый очиститель текста от общих артефактов."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Basic Cleaner", config)
        
    @log_processing_stage("Basic Text Cleaning")
    def clean_text(self, text: str, metadata: Dict[str, Any] = None, **kwargs) -> str:
        """Выполняет базовую очистку текста."""
        if not text:
            return ""
        
        cleaned = text
        
        # Удаляем лишние пробелы
        if self.config.get("remove_extra_whitespace", True):
            cleaned = self._remove_extra_whitespace(cleaned)
        
        # Нормализуем Unicode
        if self.config.get("normalize_unicode", True):
            cleaned = self._normalize_unicode(cleaned)
        
        # Исправляем кодировку
        if self.config.get("fix_encoding", True):
            cleaned = self._fix_encoding_issues(cleaned)
        
        # Удаляем управляющие символы
        if self.config.get("remove_control_chars", True):
            cleaned = self._remove_control_characters(cleaned)
        
        return cleaned.strip()
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Удаляет избыточные пробелы и переносы."""
        # Заменяем множественные пробелы на один
        text = re.sub(r' +', ' ', text)
        
        # Заменяем множественные переносы строк
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Убираем пробелы в начале и конце строк
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_unicode(self, text: str) -> str:
        """Нормализует Unicode символы."""
        # Приводим к стандартной форме NFC
        text = unicodedata.normalize('NFC', text)
        
        # Заменяем различные виды кавычек и тире
        replacements = {
            '"': '"',  # Левые кавычки
            '"': '"',  # Правые кавычки
            ''': "'",  # Левая одинарная кавычка
            ''': "'",  # Правая одинарная кавычка
            '«': '"',  # Левые французские кавычки
            '»': '"',  # Правые французские кавычки
            '—': '-',  # Длинное тире
            '–': '-',  # Среднее тире
            '…': '...',  # Многоточие
            '№': 'No.',  # Номер
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Исправляет проблемы с кодировкой."""
        # Удаляем символы замещения
        text = text.replace('�', '')
        
        # Исправляем типичные ошибки кодировки
        encoding_fixes = {
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '-',
        }
        
        for wrong, correct in encoding_fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _remove_control_characters(self, text: str) -> str:
        """Удаляет управляющие символы."""
        # Удаляем управляющие символы кроме переносов строк и табуляции
        cleaned = ''.join(char for char in text 
                         if ord(char) >= 32 or char in '\n\t')
        
        return cleaned


class AdvancedTextCleaner(BaseCleaner):
    """Продвинутый очиститель для исправления типичных проблем извлечения."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Advanced Cleaner", config)
        
    @log_processing_stage("Advanced Text Cleaning")
    def clean_text(self, text: str, metadata: Dict[str, Any] = None, **kwargs) -> str:
        """Выполняет продвинутую очистку текста."""
        if not text:
            return ""
        
        cleaned = text
        
        # Исправляем переносы слов
        if self.config.get("fix_hyphenation", True):
            cleaned = self._fix_hyphenation(cleaned)
        
        # Объединяем разорванные слова
        if self.config.get("merge_broken_words", True):
            cleaned = self._merge_broken_words(cleaned)
        
        # Удаляем колонтитулы
        if self.config.get("remove_headers_footers", True):
            cleaned = self._remove_headers_footers(cleaned)
        
        # Нормализуем кавычки
        if self.config.get("normalize_quotes", True):
            cleaned = self._normalize_quotes(cleaned)
        
        # Исправляем списки
        if self.config.get("fix_bullet_points", True):
            cleaned = self._fix_bullet_points(cleaned)
        
        return cleaned.strip()
    
    def _fix_hyphenation(self, text: str) -> str:
        """Исправляет разорванные переносом строки слова."""
        # Ищем слова, разорванные дефисом на конце строки
        pattern = r'(\w+)-\s*\n\s*(\w+)'
        
        def replace_hyphen(match):
            word1 = match.group(1)
            word2 = match.group(2)
            
            # Проверяем, является ли это действительно одним словом
            combined = word1 + word2
            
            # Простая проверка: если первая часть не является полным словом
            if len(word1) < 3 or word2[0].islower():
                return combined
            else:
                return f"{word1}-{word2}"
        
        return re.sub(pattern, replace_hyphen, text)
    
    def _merge_broken_words(self, text: str) -> str:
        """Объединяет слова, разорванные на отдельные строки."""
        lines = text.split('\n')
        merged_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            # Если строка заканчивается неполным словом
            if (current_line and 
                not current_line.endswith('.') and 
                not current_line.endswith('!') and 
                not current_line.endswith('?') and 
                not current_line.endswith(':') and
                i + 1 < len(lines)):
                
                next_line = lines[i + 1].strip()
                
                # Если следующая строка начинается с маленькой буквы
                if (next_line and 
                    next_line[0].islower() and 
                    len(next_line.split()[0]) > 1):
                    
                    # Объединяем строки
                    merged_lines.append(current_line + next_line)
                    i += 2
                    continue
            
            merged_lines.append(current_line)
            i += 1
        
        return '\n'.join(merged_lines)
    
    def _remove_headers_footers(self, text: str) -> str:
        """Удаляет повторяющиеся колонтитулы."""
        lines = text.split('\n')
        
        if len(lines) < 5:
            return text
        
        # Ищем короткие повторяющиеся строки
        line_counts = {}
        for line in lines:
            clean_line = line.strip()
            if clean_line and len(clean_line) < 100:
                line_counts[clean_line] = line_counts.get(clean_line, 0) + 1
        
        # Удаляем строки, которые повторяются более 2 раз
        filtered_lines = []
        for line in lines:
            clean_line = line.strip()
            if not clean_line or line_counts.get(clean_line, 0) <= 2:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _normalize_quotes(self, text: str) -> str:
        """Нормализует кавычки и исправляет их парность."""
        # Заменяем все виды кавычек на стандартные
        text = re.sub(r'[""„‚«»]', '"', text)
        text = re.sub(r"[''‚']", "'", text)
        
        # Исправляем непарные кавычки
        # Простой алгоритм для двойных кавычек
        quote_count = text.count('"')
        if quote_count % 2 != 0:
            # Если нечетное количество, удаляем последнюю
            last_quote = text.rfind('"')
            if last_quote != -1:
                text = text[:last_quote] + text[last_quote + 1:]
        
        return text
    
    def _fix_bullet_points(self, text: str) -> str:
        """Исправляет и нормализует маркированные списки."""
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Ищем различные виды маркеров списков
            bullet_patterns = [
                r'^[•·▪▫◦‣⁃]\s*',  # Различные символы маркеров
                r'^\*\s*',  # Звездочка
                r'^-\s*',   # Дефис
                r'^\+\s*',  # Плюс
                r'^\d+\.\s*',  # Нумерованный список
                r'^\d+\)\s*',  # Нумерованный список со скобкой
            ]
            
            for pattern in bullet_patterns:
                match = re.match(pattern, stripped)
                if match:
                    # Нормализуем к стандартному Markdown формату
                    if re.match(r'^\d+[.)]\s*', stripped):
                        # Нумерованный список — сохраняем исходный номер
                        number_match = re.match(r'^(\d+)[.)]\s*', stripped)
                        content = re.sub(r'^\d+[.)]\s*', '', stripped)
                        number_token = number_match.group(1) if number_match else "1"
                        fixed_lines.append(f"{number_token}. {content}")
                    else:
                        # Маркированный список
                        content = re.sub(pattern, '', stripped)
                        fixed_lines.append(f"- {content}")
                    break
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)


class HTMLCleaner(BaseCleaner):
    """Очиститель для удаления HTML разметки и артефактов."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("HTML Cleaner", config)
        
    @log_processing_stage("HTML Cleaning")
    def clean_text(self, text: str, metadata: Dict[str, Any] = None, **kwargs) -> str:
        """Очищает текст от HTML разметки."""
        if not text:
            return ""
        
        try:
            from bs4 import BeautifulSoup
            
            cleaned = text
            
            # Удаляем HTML теги
            if self.config.get("remove_tags", True):
                cleaned = self._remove_html_tags(cleaned)
            
            # Конвертируем HTML entities
            if self.config.get("convert_entities", True):
                cleaned = self._convert_html_entities(cleaned)
            
            # Сохраняем структуру
            if self.config.get("preserve_structure", True):
                cleaned = self._preserve_html_structure(text)
            
            return cleaned.strip()
            
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic HTML cleaning")
            return self._basic_html_clean(text)
    
    def _remove_html_tags(self, text: str) -> str:
        """Удаляет HTML теги с помощью BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def _convert_html_entities(self, text: str) -> str:
        """Конвертирует HTML entities в обычные символы."""
        import html
        return html.unescape(text)
    
    def _preserve_html_structure(self, text: str) -> str:
        """Сохраняет структуру, конвертируя HTML в Markdown."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(text, 'html.parser')
        
        # Заменяем заголовки
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                heading.string = f"{'#' * i} {heading.get_text()}"
                heading.name = 'p'
        
        # Заменяем списки
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                li.string = f"- {li.get_text()}"
                li.name = 'p'
        
        for ol in soup.find_all('ol'):
            for i, li in enumerate(ol.find_all('li'), 1):
                li.string = f"{i}. {li.get_text()}"
                li.name = 'p'
        
        # Заменяем выделение
        for strong in soup.find_all(['strong', 'b']):
            strong.string = f"**{strong.get_text()}**"
            strong.name = 'span'
        
        for em in soup.find_all(['em', 'i']):
            em.string = f"*{em.get_text()}*"
            em.name = 'span'
        
        return soup.get_text()
    
    def _basic_html_clean(self, text: str) -> str:
        """Базовая очистка HTML без BeautifulSoup."""
        # Удаляем HTML теги
        text = re.sub(r'<[^>]+>', '', text)
        
        # Конвертируем основные HTML entities
        html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'",
            '&nbsp;': ' ',
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '...',
        }
        
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        
        return text


class OCRArtifactsCleaner(BaseCleaner):
    """Очиститель артефактов OCR распознавания."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("OCR Artifacts Cleaner", config)
        
    @log_processing_stage("OCR Artifacts Cleaning")
    def clean_text(self, text: str, metadata: Dict[str, Any] = None, **kwargs) -> str:
        """Очищает типичные артефакты OCR."""
        if not text:
            return ""
        
        cleaned = text
        
        # Исправляем типичные ошибки OCR
        cleaned = self._fix_ocr_character_errors(cleaned)
        
        # Исправляем пробелы в словах
        cleaned = self._fix_word_spacing(cleaned)
        
        # Удаляем артефакты сканирования
        cleaned = self._remove_scan_artifacts(cleaned)
        
        # Исправляем числа
        cleaned = self._fix_numbers(cleaned)
        
        return cleaned.strip()
    
    def _fix_ocr_character_errors(self, text: str) -> str:
        """Исправляет типичные ошибки распознавания символов."""
        # Словарь типичных замен OCR ошибок
        ocr_corrections = {
            # Путаница с I, l, 1
            r'\bl\b': 'I',  # одинокая l -> I
            r'\b1\b(?=[a-zA-Z])': 'I',  # 1 перед буквами -> I
            
            # Путаница с O, 0
            r'\b0(?=[a-zA-Z])': 'O',  # 0 перед буквами -> O
            
            # Типичные замены букв
            'rn': 'm',  # rn часто распознается как m
            'cl': 'd',  # cl может быть d
            'ii': 'll',  # двойное i как ll
            
            # Исправления для русского текста
            'а': 'a',   # русская а может быть английской a
            'р': 'p',   # русская р может быть английской p
            'с': 'c',   # русская с может быть английской c
        }
        
        for wrong, correct in ocr_corrections.items():
            text = re.sub(wrong, correct, text)
        
        return text
    
    def _fix_word_spacing(self, text: str) -> str:
        """Исправляет пробелы внутри слов."""
        # Ищем случаи, где внутри слов есть пробелы
        # Например: "п р и м е р" -> "пример"
        
        # Паттерн для слов с лишними пробелами
        pattern = r'\b([а-яёa-z])\s+([а-яёa-z])\s+([а-яёa-z])\b'
        
        def fix_spaced_word(match):
            # Если это действительно разорванное слово, объединяем
            chars = [match.group(i) for i in range(1, match.lastindex + 1)]
            return ''.join(chars)
        
        # Применяем исправление несколько раз
        for _ in range(3):
            text = re.sub(pattern, fix_spaced_word, text, flags=re.IGNORECASE)
        
        return text
    
    def _remove_scan_artifacts(self, text: str) -> str:
        """Удаляет артефакты сканирования."""
        # Удаляем странные символы и последовательности
        artifacts = [
            r'[|}{\\]+',  # Вертикальные линии и скобки
            r'[_=\-]{5,}',  # Длинные линии подчеркивания
            r'\.{4,}',  # Множественные точки
            r'\s+\.\s+',  # Одинокие точки
            r'[^\w\s\.,!?;:()\-\'"#*\[\]/]+',  # Странные символы
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, ' ', text)
        
        # Удаляем повторяющиеся пробелы
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _fix_numbers(self, text: str) -> str:
        """Исправляет распознавание чисел."""
        # Исправляем буквы в числах
        number_fixes = {
            'O': '0',  # O -> 0 в числах
            'l': '1',  # l -> 1 в числах  
            'S': '5',  # S -> 5 в числах
            'G': '6',  # G -> 6 в числах
        }
        
        # Ищем последовательности, которые выглядят как числа
        def fix_number(match):
            number_str = match.group(0)
            for wrong, correct in number_fixes.items():
                number_str = number_str.replace(wrong, correct)
            return number_str
        
        # Применяем к последовательностям цифр и букв
        text = re.sub(r'\b[\dOlSG]{2,}\b', fix_number, text)
        
        return text