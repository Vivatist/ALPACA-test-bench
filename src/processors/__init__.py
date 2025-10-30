"""
Инициализация модуля процессоров документов.
"""

from .docx_processors import (Docx2txtExtractor, DocExtractor,
                              Win32WordExtractor)
from .markdown_converters import (CustomMarkdownFormatter, Html2TextConverter,
                                  MarkdownifyConverter, PandocConverter)
# Экспортируем основные классы процессоров
from .pdf_processors import (CamelotExtractor, PDFMinerExtractor,
                             PDFPlumberExtractor, PyMuPDFExtractor,
                             PyPDFExtractor)
from .text_cleaners import (AdvancedTextCleaner, BasicTextCleaner, HTMLCleaner,
                            OCRArtifactsCleaner)
from .unstructured_processors import (UnstructuredLLMCleaner,
                                      UnstructuredPartitionExtractor)

__all__ = [
    # PDF процессоры
    'PyPDFExtractor',
    'PDFPlumberExtractor',
    'PyMuPDFExtractor',
    'PDFMinerExtractor',
    'CamelotExtractor',
    
    # Word процессоры
    'Docx2txtExtractor',
    'DocExtractor',
    'Win32WordExtractor',
    
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