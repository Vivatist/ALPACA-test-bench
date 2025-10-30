"""
Процессоры для извлечения текста из Word документов (.doc, .docx).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import subprocess

from ..core.base import BaseExtractor
from ..utils.logger import get_logger, log_processing_stage

logger = get_logger(__name__)


class PythonDocxExtractor(BaseExtractor):
    """Процессор на базе python-docx для извлечения текста из .docx файлов."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Python-docx", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() in ['.docx']
    
    @log_processing_stage("Python-docx Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает текст из .docx файла."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            content_parts = []
            
            # Настройки извлечения
            extract_tables = self.config.get("extract_tables", True)
            extract_images = self.config.get("extract_images", False)
            preserve_formatting = self.config.get("preserve_formatting", True)
            
            # Извлекаем основной текст
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    formatted_text = self._format_paragraph(
                        paragraph, preserve_formatting
                    )
                    if formatted_text:
                        content_parts.append(formatted_text)
            
            # Извлекаем таблицы
            if extract_tables and doc.tables:
                content_parts.append("\n# Таблицы\n")
                for table_num, table in enumerate(doc.tables):
                    content_parts.append(f"## Таблица {table_num + 1}")
                    markdown_table = self._extract_table(table)
                    content_parts.append(markdown_table)
            
            # Извлекаем изображения (заглушки)
            if extract_images:
                content_parts.append("\n# Изображения\n")
                # TODO: Реализовать извлечение изображений
                content_parts.append("*Извлечение изображений пока не реализовано*")
            
            return "\n\n".join(content_parts)
            
        except ImportError:
            raise ImportError("python-docx is not installed. Run: pip install python-docx")
        except Exception as e:
            logger.error(f"Python-docx extraction failed: {e}")
            raise
    
    def _format_paragraph(self, paragraph, preserve_formatting: bool) -> str:
        """Форматирует параграф с учетом стилей."""
        if not preserve_formatting:
            return paragraph.text.strip()
        
        text = paragraph.text.strip()
        if not text:
            return ""
        
        # Определяем уровень заголовка по стилю
        style_name = paragraph.style.name.lower()
        
        if 'heading 1' in style_name or 'заголовок 1' in style_name:
            return f"# {text}"
        elif 'heading 2' in style_name or 'заголовок 2' in style_name:
            return f"## {text}"
        elif 'heading 3' in style_name or 'заголовок 3' in style_name:
            return f"### {text}"
        elif 'heading 4' in style_name or 'заголовок 4' in style_name:
            return f"#### {text}"
        elif 'heading 5' in style_name or 'заголовок 5' in style_name:
            return f"##### {text}"
        elif 'heading 6' in style_name or 'заголовок 6' in style_name:
            return f"###### {text}"
        elif 'title' in style_name or 'название' in style_name:
            return f"# {text}"
        else:
            # Проверяем форматирование текста внутри параграфа
            return self._format_runs(paragraph.runs) if paragraph.runs else text
    
    def _format_runs(self, runs) -> str:
        """Форматирует runs (части текста с разным форматированием)."""
        formatted_parts = []
        
        for run in runs:
            text = run.text
            if not text:
                continue
            
            # Применяем форматирование
            if run.bold and run.italic:
                text = f"***{text}***"
            elif run.bold:
                text = f"**{text}**"
            elif run.italic:
                text = f"*{text}*"
            
            # Можно добавить другие виды форматирования
            if hasattr(run, 'underline') and run.underline:
                text = f"<u>{text}</u>"
            
            formatted_parts.append(text)
        
        return "".join(formatted_parts)
    
    def _extract_table(self, table) -> str:
        """Извлекает таблицу в Markdown формате."""
        if not table.rows:
            return "*Пустая таблица*"
        
        markdown_lines = []
        
        # Получаем данные таблицы
        table_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace('\n', ' ')
                row_data.append(cell_text)
            table_data.append(row_data)
        
        if not table_data:
            return "*Пустая таблица*"
        
        # Определяем максимальное количество колонок
        max_cols = max(len(row) for row in table_data)
        
        # Выравниваем все строки до максимального количества колонок
        for row in table_data:
            while len(row) < max_cols:
                row.append("")
        
        # Создаем заголовок (первая строка)
        header = "| " + " | ".join(table_data[0]) + " |"
        markdown_lines.append(header)
        
        # Разделитель
        separator = "|" + "|".join([" --- " for _ in range(max_cols)]) + "|"
        markdown_lines.append(separator)
        
        # Остальные строки
        for row in table_data[1:]:
            row_line = "| " + " | ".join(row) + " |"
            markdown_lines.append(row_line)
        
        return "\n".join(markdown_lines)


class Docx2txtExtractor(BaseExtractor):
    """Простой процессор на базе docx2txt для извлечения текста."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Docx2txt", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() in ['.docx']
    
    @log_processing_stage("Docx2txt Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает текст простым способом."""
        try:
            import docx2txt
            
            text = docx2txt.process(str(file_path))
            
            if not text or not text.strip():
                return "Текст не извлечен или документ пуст"
            
            # Базовая очистка и форматирование
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    # Простая эвристика для заголовков
                    if len(line) < 100 and line.isupper():
                        cleaned_lines.append(f"# {line}")
                    elif len(line) < 80 and not line.endswith('.'):
                        cleaned_lines.append(f"## {line}")
                    else:
                        cleaned_lines.append(line)
                elif cleaned_lines and cleaned_lines[-1]:
                    # Добавляем пустую строку только если предыдущая не пустая
                    cleaned_lines.append("")
            
            return "\n".join(cleaned_lines)
            
        except ImportError:
            raise ImportError("docx2txt is not installed. Run: pip install docx2txt")
        except Exception as e:
            logger.error(f"Docx2txt extraction failed: {e}")
            raise


class LibreOfficeExtractor(BaseExtractor):
    """
    Процессор для .doc файлов через LibreOffice/OpenOffice.
    Требует установленный LibreOffice.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LibreOffice", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() in ['.doc', '.docx']
    
    @log_processing_stage("LibreOffice Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Конвертирует документ в текст через LibreOffice."""
        try:
            # Проверяем наличие LibreOffice
            libreoffice_path = self._find_libreoffice()
            if not libreoffice_path:
                raise EnvironmentError(
                    "LibreOffice not found. Please install LibreOffice."
                )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Конвертируем в текстовый файл
                cmd = [
                    libreoffice_path,
                    '--headless',
                    '--convert-to', 'txt',
                    '--outdir', str(temp_path),
                    str(file_path)
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=60
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
                
                # Читаем результат
                output_file = temp_path / f"{file_path.stem}.txt"
                if not output_file.exists():
                    raise RuntimeError("Converted file not found")
                
                text = output_file.read_text(encoding='utf-8')
                
                # Базовая обработка
                if not text.strip():
                    return "Документ пуст или не удалось извлечь текст"
                
                return self._format_libreoffice_output(text)
                
        except subprocess.TimeoutExpired:
            raise TimeoutError("LibreOffice conversion timeout")
        except Exception as e:
            logger.error(f"LibreOffice extraction failed: {e}")
            raise
    
    def _find_libreoffice(self) -> Optional[str]:
        """Ищет исполняемый файл LibreOffice."""
        import shutil
        
        # Возможные пути к LibreOffice
        possible_paths = [
            'libreoffice',
            'soffice',
            '/usr/bin/libreoffice',
            '/usr/local/bin/libreoffice',
            'C:\\Program Files\\LibreOffice\\program\\soffice.exe',
            'C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe'
        ]
        
        for path in possible_paths:
            if shutil.which(path):
                return path
        
        return None
    
    def _format_libreoffice_output(self, text: str) -> str:
        """Форматирует вывод LibreOffice."""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if formatted_lines and formatted_lines[-1]:
                    formatted_lines.append("")
                continue
            
            # Простая эвристика для заголовков
            if len(line) < 100:
                # Если строка короткая и не заканчивается точкой
                if not line.endswith('.') and not line.endswith(','):
                    # Проверяем, начинается ли со цифры (номер раздела)
                    if line[0].isdigit() or line.isupper():
                        formatted_lines.append(f"# {line}")
                        continue
            
            formatted_lines.append(line)
        
        return "\n".join(formatted_lines)