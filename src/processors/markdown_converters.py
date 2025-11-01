"""
Конвертеры текста в Markdown формат.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from ..core.base import BaseConverter
from ..utils.logger import get_logger, log_processing_stage

logger = get_logger(__name__)


class MarkdownifyConverter(BaseConverter):
    """Конвертер HTML в Markdown с помощью markdownify."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Markdownify", config)
        
    @log_processing_stage("Markdownify Conversion")
    def convert_to_markdown(self, text: str, **kwargs) -> str:
        """Конвертирует HTML в Markdown."""
        try:
            from markdownify import markdownify

            # Настройки конвертации
            heading_style = self.config.get("heading_style", "ATX")
            bullets = self.config.get("bullets", "-")
            convert_tags = self.config.get("convert", [
                "b", "strong", "i", "em", "h1", "h2", "h3", "h4", "h5", "h6",
                "p", "br", "ul", "ol", "li", "a", "img", "table", "tr", "td", "th"
            ])
            
            # Если текст не содержит HTML тегов, применяем базовое форматирование
            if not re.search(r'<[^>]+>', text):
                return self._format_plain_text(text)
            
            markdown = markdownify(
                text,
                heading_style=heading_style,
                bullets=bullets,
                convert=convert_tags,
                strip=['script', 'style']
            )
            
            return self._clean_markdown(markdown)
            
        except ImportError:
            logger.warning("markdownify not available, using basic conversion")
            return self._format_plain_text(text)
        except Exception as e:
            logger.error(f"Markdownify conversion failed: {e}")
            return self._format_plain_text(text)
    
    def _format_plain_text(self, text: str) -> str:
        """Форматирует обычный текст как Markdown."""
        if not text.strip():
            return ""
        
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Простая эвристика для заголовков
            if self._looks_like_header(line):
                level = self._determine_header_level(line)
                formatted_lines.append(f"{'#' * level} {line}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _looks_like_header(self, line: str) -> bool:
        """Определяет, похожа ли строка на заголовок."""
        return (
            len(line) < 100 and  # Короткая строка
            not line.endswith('.') and  # Не заканчивается точкой
            not line.endswith(',') and  # Не заканчивается запятой
            (line.isupper() or  # Все заглавные
             line.istitle() or  # Заглавные первые буквы
             re.match(r'^\d+\.?\s*[A-ZА-Я]', line))  # Начинается с номера
        )
    
    def _determine_header_level(self, line: str) -> int:
        """Определяет уровень заголовка."""
        if line.isupper():
            return 1  # Заголовок верхнего уровня
        elif re.match(r'^\d+\.?\s*', line):
            # По номеру раздела
            match = re.match(r'^(\d+)\.?', line)
            if match:
                num = int(match.group(1))
                if num <= 10:
                    return 2
                else:
                    return 3
        
        return 2  # По умолчанию второй уровень
    
    def _clean_markdown(self, markdown: str) -> str:
        """Очищает сгенерированный Markdown."""
        # Удаляем избыточные пустые строки
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Исправляем пробелы вокруг заголовков
        markdown = re.sub(r'\n(#{1,6})\s*([^\n]+)', r'\n\1 \2', markdown)
        
        # Исправляем списки
        markdown = re.sub(r'\n\s*([*+-])\s*([^\n]+)', r'\n\1 \2', markdown)
        
        return markdown.strip()


class Html2TextConverter(BaseConverter):
    """Конвертер HTML в Markdown с помощью html2text."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("html2text", config)
        
    @log_processing_stage("Html2Text Conversion")
    def convert_to_markdown(self, text: str, **kwargs) -> str:
        """Конвертирует HTML в Markdown с помощью html2text."""
        try:
            import html2text
            
            h = html2text.HTML2Text()
            
            # Настройки конвертера
            h.ignore_links = self.config.get("ignore_links", False)
            h.ignore_images = self.config.get("ignore_images", False) 
            h.body_width = self.config.get("body_width", 0)
            h.unicode_snob = self.config.get("unicode_snob", True)
            
            # Если текст не содержит HTML, форматируем как обычный текст
            if not re.search(r'<[^>]+>', text):
                return self._format_as_markdown(text)
            
            markdown = h.handle(text)
            return self._clean_html2text_output(markdown)
            
        except ImportError:
            logger.warning("html2text not available, using basic conversion")
            return self._format_as_markdown(text)
        except Exception as e:
            logger.error(f"html2text conversion failed: {e}")
            return self._format_as_markdown(text)
    
    def _format_as_markdown(self, text: str) -> str:
        """Форматирует обычный текст в Markdown."""
        if not text.strip():
            return ""
        
        # Разбиваем на параграфы
        paragraphs = re.split(r'\n\s*\n', text)
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            lines = paragraph.split('\n')
            if len(lines) == 1 and self._is_likely_header(paragraph):
                # Это заголовок
                level = self._get_header_level(paragraph)
                formatted_paragraphs.append(f"{'#' * level} {paragraph}")
            else:
                # Обычный параграф
                cleaned_paragraph = ' '.join(line.strip() for line in lines)
                formatted_paragraphs.append(cleaned_paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _is_likely_header(self, text: str) -> bool:
        """Определяет, является ли текст заголовком."""
        return (
            len(text) < 120 and
            not text.endswith('.') and
            not text.endswith(',') and
            (text.isupper() or 
             text.istitle() or
             re.match(r'^[0-9IVX]+\.?\s*[A-ZА-Я]', text))
        )
    
    def _get_header_level(self, text: str) -> int:
        """Определяет уровень заголовка."""
        if text.isupper() or re.match(r'^[IVX]+\.?\s*', text):
            return 1
        elif re.match(r'^[0-9]+\.?\s*', text):
            return 2
        else:
            return 3
    
    def _clean_html2text_output(self, markdown: str) -> str:
        """Очищает вывод html2text."""
        # Удаляем избыточные пустые строки
        markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)
        
        # Исправляем заголовки
        markdown = re.sub(r'\n(#{1,6})\s*([^\n]+)', r'\n\1 \2', markdown)
        
        # Убираем лишние пробелы в конце строк
        lines = markdown.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        return '\n'.join(cleaned_lines).strip()


class PandocConverter(BaseConverter):
    """Конвертер с использованием Pandoc (если установлен)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Pandoc", config)
        
    @log_processing_stage("Pandoc Conversion")
    def convert_to_markdown(self, text: str, **kwargs) -> str:
        """Конвертирует текст в Markdown через Pandoc."""
        try:
            import shutil
            import subprocess
            import tempfile

            # Проверяем наличие pandoc
            if not shutil.which('pandoc'):
                raise EnvironmentError("Pandoc not found in PATH")
            
            from_format = self.config.get("from_format", "html")
            to_format = self.config.get("to_format", "markdown")
            extra_args = self.config.get("extra_args", ["--wrap=none"])
            
            # Если это не HTML, используем базовый форматтер
            if from_format == "html" and not re.search(r'<[^>]+>', text):
                return self._basic_markdown_format(text)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', 
                                           delete=False, encoding='utf-8') as tmp_input:
                tmp_input.write(text)
                tmp_input.flush()
                
                # Команда pandoc
                cmd = [
                    'pandoc',
                    '-f', from_format,
                    '-t', to_format,
                    tmp_input.name
                ] + extra_args
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8',
                    timeout=30
                )
                
                # Удаляем временный файл
                Path(tmp_input.name).unlink(missing_ok=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Pandoc failed: {result.stderr}")
                
                return self._clean_pandoc_output(result.stdout)
                
        except (ImportError, EnvironmentError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Pandoc conversion failed: {e}, using basic conversion")
            return self._basic_markdown_format(text)
        except Exception as e:
            logger.error(f"Pandoc conversion error: {e}")
            return self._basic_markdown_format(text)
    
    def _basic_markdown_format(self, text: str) -> str:
        """Базовое форматирование в Markdown."""
        if not text.strip():
            return ""
        
        # Простое форматирование
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                if in_list:
                    in_list = False
                formatted_lines.append("")
                continue
            
            # Заголовки
            if len(stripped) < 80 and not stripped.endswith('.'):
                if stripped.isupper():
                    formatted_lines.append(f"# {stripped}")
                elif stripped.istitle():
                    formatted_lines.append(f"## {stripped}")
                else:
                    formatted_lines.append(stripped)
            
            # Списки
            elif stripped.startswith(('- ', '* ', '+ ')):
                formatted_lines.append(stripped)
                in_list = True
            elif re.match(r'^\d+\.\s', stripped):
                formatted_lines.append(stripped)
                in_list = True
            
            else:
                formatted_lines.append(stripped)
                in_list = False
        
        return '\n'.join(formatted_lines)
    
    def _clean_pandoc_output(self, markdown: str) -> str:
        """Очищает вывод Pandoc."""
        # Удаляем избыточные пустые строки
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Исправляем заголовки без пробела
        markdown = re.sub(r'\n(#{1,6})([^\s])', r'\n\1 \2', markdown)
        
        # Исправляем списки
        markdown = re.sub(r'\n([*+-])\s*([^\n]+)', r'\n\1 \2', markdown)
        
        return markdown.strip()


class CustomMarkdownFormatter(BaseConverter):
    """Кастомный форматтер для преобразования в качественный Markdown."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Custom Markdown Formatter", config)
        
    @log_processing_stage("Custom Markdown Formatting")
    def convert_to_markdown(self, text: str, **kwargs) -> str:
        """Форматирует текст в качественный Markdown."""
        if not text.strip():
            return ""
        
        # Этапы форматирования
        formatted = self._preprocess_text(text)
        formatted = self._format_headers(formatted)
        formatted = self._format_lists(formatted)
        formatted = self._format_paragraphs(formatted)
        formatted = self._format_emphasis(formatted)
        formatted = self._format_tables(formatted)
        formatted = self._postprocess_text(formatted)
        
        return formatted
    
    def _preprocess_text(self, text: str) -> str:
        """Предварительная обработка текста."""
        # Нормализуем переносы строк
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Удаляем избыточные пробелы
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Нормализуем пустые строки
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    def _format_headers(self, text: str) -> str:
        """Форматирует заголовки."""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                formatted_lines.append("")
                continue
            
            # Уже Markdown заголовок
            if re.match(r'^#{1,6}\s', stripped):
                formatted_lines.append(stripped)
                continue
            
            # Определяем заголовки
            if self._is_header(stripped):
                level = self._get_header_level_advanced(stripped)
                formatted_lines.append(f"{'#' * level} {stripped}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _is_header(self, line: str) -> bool:
        """Расширенная проверка на заголовок."""
        if len(line) > 150:  # Слишком длинная строка
            return False
        
        if line.endswith(('.', ',', ';', '!', '?')):  # Заканчивается пунктуацией
            return False
        
        # Различные признаки заголовка
        patterns = [
            r'^[A-ZА-Я][^.]{5,80}$',  # Начинается с заглавной, нет точек
            r'^[0-9IVX]+\.?\s*[A-ZА-Я]',  # Нумерованный заголовок
            r'^[A-ZА-Я\s]{5,50}$',  # Все заглавные
            r'^[A-ZА-Я][a-zа-я]*(\s[A-ZА-Я][a-zа-я]*){1,8}$',  # Title Case
        ]
        
        return any(re.match(pattern, line) for pattern in patterns)
    
    def _get_header_level_advanced(self, line: str) -> int:
        """Определяет уровень заголовка с учетом контекста."""
        # По длине и содержанию
        if len(line) < 30 and line.isupper():
            return 1
        elif re.match(r'^[0-9]+\.?\s*', line):
            # По номеру раздела
            match = re.match(r'^([0-9]+)', line)
            if match:
                num = int(match.group(1))
                if num <= 5:
                    return 1
                elif num <= 20:
                    return 2
                else:
                    return 3
        elif re.match(r'^[IVX]+\.?\s*', line):
            return 1  # Римские цифры - верхний уровень
        elif line.istitle():
            return 2
        
        return 3  # По умолчанию
    
    def _format_lists(self, text: str) -> str:
        """Форматирует списки."""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Различные виды маркеров
            list_patterns = [
                (r'^[•·▪▫◦‣⁃]\s*(.+)', lambda m: f"- {m.group(1)}"),
                (r'^[*+]\s*(.+)', lambda m: f"- {m.group(1)}"),
                (r'^(\d+)[.)]\s*(.+)', lambda m: f"{m.group(1)}. {m.group(2)}"),
            ]
            
            formatted = False
            for pattern, replacement in list_patterns:
                match = re.match(pattern, stripped)
                if match:
                    if callable(replacement):
                        formatted_lines.append(replacement(match))
                    else:
                        formatted_lines.append(re.sub(pattern, replacement, stripped))
                    formatted = True
                    break
            
            if not formatted:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _format_paragraphs(self, text: str) -> str:
        """Форматирует параграфы."""
        # Разделяем на блоки
        blocks = re.split(r'\n\s*\n', text)
        formatted_blocks = []
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            lines = block.split('\n')
            
            # Если это заголовок или список, не трогаем
            if (re.match(r'^#{1,6}\s', lines[0]) or 
                re.match(r'^[-*+]\s', lines[0]) or
                re.match(r'^\d+\.\s', lines[0])):
                formatted_blocks.append(block)
            else:
                # Объединяем строки параграфа
                paragraph = ' '.join(line.strip() for line in lines)
                formatted_blocks.append(paragraph)
        
        return '\n\n'.join(formatted_blocks)
    
    def _format_emphasis(self, text: str) -> str:
        """Форматирует выделение текста."""
        # Ищем слова в CAPS и делаем их жирными
        text = re.sub(r'\b([A-ZА-Я]{3,})\b', r'**\1**', text)
        
        # Слова в кавычках делаем курсивом
        text = re.sub(r'"([^"]{3,30})"', r'*\1*', text)
        
        return text
    
    def _format_tables(self, text: str) -> str:
        """Ищет и форматирует таблицы."""
        lines = text.split('\n')
        formatted_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Ищем строки, похожие на таблицы
            if '|' in line and len(line.split('|')) >= 3:
                # Собираем таблицу
                table_lines = []
                j = i
                while j < len(lines) and '|' in lines[j]:
                    table_lines.append(lines[j].strip())
                    j += 1
                
                if len(table_lines) >= 2:  # Минимум заголовок + данные
                    formatted_table = self._format_table_block(table_lines)
                    formatted_lines.extend(formatted_table.split('\n'))
                    i = j
                    continue
            
            formatted_lines.append(lines[i])
            i += 1
        
        return '\n'.join(formatted_lines)
    
    def _format_table_block(self, table_lines: List[str]) -> str:
        """Форматирует блок таблицы."""
        if not table_lines:
            return ""
        
        # Разбираем строки таблицы
        rows = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.split('|')]
            # Убираем пустые ячейки в начале и конце
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            if cells:
                rows.append(cells)
        
        if not rows:
            return ""
        
        # Определяем максимальное количество колонок
        max_cols = max(len(row) for row in rows)
        
        # Выравниваем все строки
        for row in rows:
            while len(row) < max_cols:
                row.append("")
        
        # Создаем Markdown таблицу
        markdown_lines = []
        
        # Заголовок
        header = "| " + " | ".join(rows[0]) + " |"
        markdown_lines.append(header)
        
        # Разделитель
        separator = "|" + "|".join([" --- " for _ in range(max_cols)]) + "|"
        markdown_lines.append(separator)
        
        # Данные
        for row in rows[1:]:
            row_line = "| " + " | ".join(row) + " |"
            markdown_lines.append(row_line)
        
        return '\n'.join(markdown_lines)
    
    def _postprocess_text(self, text: str) -> str:
        """Финальная обработка."""
        # Удаляем избыточные пустые строки
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Исправляем пробелы вокруг заголовков
        text = re.sub(r'\n(#{1,6})\s*([^\n]+)', r'\n\1 \2', text)
        
        # Исправляем списки
        text = re.sub(r'\n\s*([-*+])\s*([^\n]+)', r'\n\1 \2', text)
        text = re.sub(r'\n\s*(\d+\.)\s*([^\n]+)', r'\n\1 \2', text)
        
        return text.strip()


class PassThroughConverter(BaseConverter):
    """
    Pass-through конвертер для текстов, которые уже в формате Markdown.
    Используется для экстракторов типа MarkItDown, которые сами выдают Markdown.
    Просто возвращает текст без изменений.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PassThrough", config)
    
    @log_processing_stage("PassThrough Conversion")
    def convert_to_markdown(self, text: str, **kwargs) -> str:
        """Возвращает текст без изменений (уже Markdown)."""
        logger.debug(f"PassThrough: text already in Markdown format ({len(text)} chars)")
        return text
    
    def get_metadata(self, text: str, **kwargs) -> Dict[str, Any]:
        """Возвращает метаданные о конвертации."""
        return {
            "converter": "PassThrough",
            "conversion_type": "none",
            "input_length": len(text),
            "output_length": len(text),
            "note": "Text already in Markdown format, no conversion needed"
        }