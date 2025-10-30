"""
Инициализация модуля процессоров документов.
"""

# Экспортируем основные классы процессоров
from .pdf_processors import (
    PyPDFExtractor,
    PDFPlumberExtractor, 
    PyMuPDFExtractor,
    PDFMinerExtractor,
    CamelotExtractor
)

from .docx_processors import (
    PythonDocxExtractor,
    Docx2txtExtractor,
    LibreOfficeExtractor
)

from .text_cleaners import (
    BasicTextCleaner,
    AdvancedTextCleaner,
    HTMLCleaner,
    OCRArtifactsCleaner
)

from .unstructured_processors import (
    UnstructuredPartitionExtractor,
    UnstructuredLLMCleaner
)

from .markdown_converters import (
    MarkdownifyConverter,
    Html2TextConverter,
    PandocConverter,
    CustomMarkdownFormatter
)

__all__ = [
    # PDF процессоры
    'PyPDFExtractor',
    'PDFPlumberExtractor',
    'PyMuPDFExtractor',
    'PDFMinerExtractor',
    'CamelotExtractor',
    
    # Word процессоры
    'PythonDocxExtractor',
    'Docx2txtExtractor',
    'LibreOfficeExtractor',
    
    # Очистители текста
    'BasicTextCleaner',
    'AdvancedTextCleaner',
    'HTMLCleaner',
    'OCRArtifactsCleaner',
    'UnstructuredLLMCleaner',
    
    # Конвертеры Markdown
    'MarkdownifyConverter',
    'Html2TextConverter',
    'PandocConverter',
    'CustomMarkdownFormatter',
    'UnstructuredPartitionExtractor'
]