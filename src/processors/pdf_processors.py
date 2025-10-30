"""
Процессоры для извлечения текста из PDF файлов.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from ..core.base import BaseExtractor, ProcessingResult, ProcessingStage, ProcessingStatus
from ..utils.logger import get_logger, log_processing_stage

logger = get_logger(__name__)


class PyPDFExtractor(BaseExtractor):
    """Процессор на базе PyPDF для извлечения текста из PDF."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PyPDF", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() == '.pdf'
    
    @log_processing_stage("PyPDF Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает текст с помощью PyPDF."""
        try:
            import PyPDF2
            
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Проверяем, защищен ли PDF паролем
                if pdf_reader.is_encrypted:
                    password = self.config.get("password")
                    if password:
                        pdf_reader.decrypt(password)
                    else:
                        raise ValueError("PDF is encrypted and no password provided")
                
                # Извлекаем текст со всех страниц
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"# Страница {page_num + 1}\n\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
            
            return "\n\n".join(text_content)
            
        except ImportError:
            raise ImportError("PyPDF2 is not installed. Run: pip install PyPDF2")
        except Exception as e:
            logger.error(f"PyPDF extraction failed: {e}")
            raise


class PDFPlumberExtractor(BaseExtractor):
    """Процессор на базе pdfplumber для извлечения текста и таблиц."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PDFPlumber", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() == '.pdf'
    
    @log_processing_stage("PDFPlumber Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает текст и таблицы с помощью pdfplumber."""
        try:
            import pdfplumber
            
            content_parts = []
            extract_tables = self.config.get("extract_tables", True)
            table_settings = self.config.get("table_settings", {})
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_content = []
                    page_content.append(f"# Страница {page_num + 1}")
                    
                    # Извлекаем основной текст
                    text = page.extract_text()
                    if text and text.strip():
                        page_content.append(text.strip())
                    
                    # Извлекаем таблицы
                    if extract_tables:
                        tables = page.extract_tables(table_settings)
                        for table_num, table in enumerate(tables):
                            if table:
                                page_content.append(f"\n## Таблица {table_num + 1}")
                                page_content.append(self._format_table_as_markdown(table))
                    
                    if len(page_content) > 1:  # Есть контент кроме заголовка
                        content_parts.append("\n\n".join(page_content))
            
            return "\n\n---\n\n".join(content_parts)
            
        except ImportError:
            raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
            raise
    
    def _format_table_as_markdown(self, table: List[List[str]]) -> str:
        """Конвертирует таблицу в Markdown формат."""
        if not table or not table[0]:
            return ""
        
        # Очищаем None значения
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        # Создаем Markdown таблицу
        markdown_lines = []
        
        # Заголовок
        header = "| " + " | ".join(cleaned_table[0]) + " |"
        markdown_lines.append(header)
        
        # Разделитель
        separator = "|" + "|".join([" --- " for _ in cleaned_table[0]]) + "|"
        markdown_lines.append(separator)
        
        # Данные
        for row in cleaned_table[1:]:
            row_line = "| " + " | ".join(row) + " |"
            markdown_lines.append(row_line)
        
        return "\n".join(markdown_lines)


class PyMuPDFExtractor(BaseExtractor):
    """Процессор на базе PyMuPDF (fitz) для извлечения текста."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PyMuPDF", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() == '.pdf'
    
    @log_processing_stage("PyMuPDF Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает текст с помощью PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            content_parts = []
            get_text_dict = self.config.get("get_text_dict", True)
            extract_fonts = self.config.get("extract_fonts", True)
            
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_content = [f"# Страница {page_num + 1}"]
                
                if get_text_dict:
                    # Извлекаем структурированный текст с информацией о шрифтах
                    text_dict = page.get_text("dict")
                    formatted_text = self._format_text_dict(text_dict, extract_fonts)
                    if formatted_text.strip():
                        page_content.append(formatted_text)
                else:
                    # Простое извлечение текста
                    text = page.get_text()
                    if text.strip():
                        page_content.append(text.strip())
                
                if len(page_content) > 1:
                    content_parts.append("\n\n".join(page_content))
            
            doc.close()
            return "\n\n---\n\n".join(content_parts)
            
        except ImportError:
            raise ImportError("PyMuPDF is not installed. Run: pip install pymupdf")
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise
    
    def _format_text_dict(self, text_dict: Dict, extract_fonts: bool = True) -> str:
        """Форматирует структурированный текст из PyMuPDF."""
        content_lines = []
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            block_lines = []
            current_font_size = None
            
            for line in block["lines"]:
                line_text = ""
                
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    
                    if extract_fonts and text.strip():
                        font_size = span.get("size", 0)
                        flags = span.get("flags", 0)
                        
                        # Определяем заголовки по размеру шрифта
                        if font_size > 16:
                            text = f"# {text}"
                        elif font_size > 14:
                            text = f"## {text}"
                        elif font_size > 12:
                            text = f"### {text}"
                        
                        # Выделение жирным (flag 16 = bold)
                        if flags & 2**4:
                            text = f"**{text}**"
                        
                        # Курсив (flag 2 = italic)
                        if flags & 2**1:
                            text = f"*{text}*"
                    
                    line_text += text
                
                if line_text.strip():
                    block_lines.append(line_text.strip())
            
            if block_lines:
                content_lines.append(" ".join(block_lines))
        
        return "\n\n".join(content_lines)


class PDFMinerExtractor(BaseExtractor):
    """Процессор на базе pdfminer для детального извлечения текста."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("PDFMiner", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() == '.pdf'
    
    @log_processing_stage("PDFMiner Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает текст с помощью pdfminer."""
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams
            
            # Настройки layout анализа
            laparams_config = self.config.get("laparams", {})
            laparams = LAParams(
                boxes_flow=laparams_config.get("boxes_flow", 0.5),
                word_margin=laparams_config.get("word_margin", 0.1),
                char_margin=laparams_config.get("char_margin", 2.0),
                line_margin=laparams_config.get("line_margin", 0.5)
            )
            
            text = extract_text(str(file_path), laparams=laparams)
            
            # Разбиваем на страницы (приблизительно)
            pages = text.split('\f')  # Form feed символ обычно разделяет страницы
            
            if len(pages) <= 1:
                # Если нет разделителей страниц, делим по количеству символов
                page_size = len(text) // max(1, len(text) // 3000)  # ~3000 символов на страницу
                pages = [text[i:i+page_size] for i in range(0, len(text), page_size)]
            
            formatted_pages = []
            for page_num, page_text in enumerate(pages):
                if page_text.strip():
                    formatted_pages.append(f"# Страница {page_num + 1}\n\n{page_text.strip()}")
            
            return "\n\n---\n\n".join(formatted_pages)
            
        except ImportError:
            raise ImportError("pdfminer.six is not installed. Run: pip install pdfminer.six")
        except Exception as e:
            logger.error(f"PDFMiner extraction failed: {e}")
            raise


class CamelotExtractor(BaseExtractor):
    """Процессор на базе Camelot для извлечения таблиц из PDF."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Camelot", config)
        
    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() == '.pdf'
    
    @log_processing_stage("Camelot Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает таблицы с помощью Camelot."""
        try:
            import camelot
            import pandas as pd
            
            flavor = self.config.get("flavor", "lattice")
            table_areas = self.config.get("table_areas")
            columns = self.config.get("columns")
            
            # Параметры для Camelot
            camelot_kwargs = {}
            if table_areas:
                camelot_kwargs["table_areas"] = table_areas
            if columns:
                camelot_kwargs["columns"] = columns
            
            # Извлекаем таблицы
            tables = camelot.read_pdf(str(file_path), flavor=flavor, **camelot_kwargs)
            
            content_parts = []
            content_parts.append("# Извлеченные таблицы из PDF")
            content_parts.append(f"Найдено таблиц: {len(tables)}")
            
            for table_num, table in enumerate(tables):
                content_parts.append(f"\n## Таблица {table_num + 1}")
                content_parts.append(f"Точность: {table.accuracy:.2f}")
                content_parts.append(f"Whitespace: {table.whitespace:.2f}")
                
                # Конвертируем таблицу в Markdown
                df = table.df
                if not df.empty:
                    markdown_table = self._dataframe_to_markdown(df)
                    content_parts.append(markdown_table)
                else:
                    content_parts.append("*Таблица пуста*")
            
            if len(tables) == 0:
                content_parts.append("*Таблицы не найдены*")
            
            return "\n\n".join(content_parts)
            
        except ImportError:
            raise ImportError(
                "Camelot is not installed. Run: pip install camelot-py[cv] "
                "Note: Requires OpenCV and Ghostscript"
            )
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            raise
    
    def _dataframe_to_markdown(self, df) -> str:
        """Конвертирует pandas DataFrame в Markdown таблицу."""
        if df.empty:
            return "*Таблица пуста*"
        
        # Заменяем NaN на пустые строки
        df_clean = df.fillna("")
        
        # Создаем Markdown таблицу
        lines = []
        
        # Заголовки (используем индексы столбцов если нет явных заголовков)
        headers = [f"Колонка {i+1}" for i in range(len(df_clean.columns))]
        header_line = "| " + " | ".join(headers) + " |"
        lines.append(header_line)
        
        # Разделитель
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"
        lines.append(separator)
        
        # Данные
        for _, row in df_clean.iterrows():
            row_values = [str(val).strip() for val in row.values]
            row_line = "| " + " | ".join(row_values) + " |"
            lines.append(row_line)
        
        return "\n".join(lines)