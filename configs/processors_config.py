"""
Конфигурация процессоров документов для ALPACA Test Bench.

Этот файл содержит настройки всех доступных процессоров
для различных типов документов.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ProcessorConfig:
    """Базовая конфигурация процессора."""
    name: str
    enabled: bool = True
    priority: int = 0
    timeout: int = 300
    parameters: Dict[str, Any] = field(default_factory=dict)


UNSTRUCTURED_BASE_PARAMETERS: Dict[str, Any] = {
    "strategy": "hi_res",
    "chunking_strategy": "by_title",
    "include_metadata": True,
    "infer_table_structure": True,
    "drop_types": ["Header", "Footer", "PageBreak"],
}

# PDF Процессоры
PDF_PROCESSORS = {
    "unstructured_pdf": ProcessorConfig(
        name="Unstructured Partition (PDF)",
        enabled=True,
        priority=0,
        parameters={
            **UNSTRUCTURED_BASE_PARAMETERS,
            "supported_types": [".pdf"],
        }
    ),
    "pypdf": ProcessorConfig(
        name="PyPDF",
        enabled=True,
        priority=1,
        parameters={
            "extract_images": False,
            "password": None,
            "strict": False
        }
    ),
    "pdfplumber": ProcessorConfig(
        name="PDF Plumber",
        enabled=True,
        priority=2,
        parameters={
            "extract_tables": True,
            "table_settings": {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines"
            }
        }
    ),
    "pymupdf": ProcessorConfig(
        name="PyMuPDF (fitz)",
        enabled=True,
        priority=3,
        parameters={
            "get_text_dict": True,
            "extract_images": False,
            "extract_fonts": True
        }
    ),
    "pdfminer": ProcessorConfig(
        name="PDFMiner",
        enabled=True,
        priority=4,
        parameters={
            "laparams": {
                "boxes_flow": 0.5,
                "word_margin": 0.1,
                "char_margin": 2.0,
                "line_margin": 0.5
            }
        }
    ),
    "camelot": ProcessorConfig(
        name="Camelot (Tables)",
        enabled=True,
        priority=5,
        parameters={
            "flavor": "lattice",
            "table_areas": None,
            "columns": None
        }
    )
}

# Word документы
DOCX_PROCESSORS = {
    "unstructured_docx": ProcessorConfig(
        name="Unstructured Partition (DOC/DOCX)",
        enabled=True,
        priority=0,
        parameters={
            **UNSTRUCTURED_BASE_PARAMETERS,
            "supported_types": [".docx", ".doc"],
        }
    ),
    "python_docx": ProcessorConfig(
        name="Python-docx",
        enabled=True,
        priority=1,
        parameters={
            "extract_tables": True,
            "extract_images": False,
            "preserve_formatting": True
        }
    ),
    "docx2txt": ProcessorConfig(
        name="Docx2txt",
        enabled=True,
        priority=2,
        parameters={}
    )
}

# PowerPoint
PPTX_PROCESSORS = {
    "unstructured_pptx": ProcessorConfig(
        name="Unstructured Partition (PPT/PPTX)",
        enabled=True,
        priority=0,
        parameters={
            **UNSTRUCTURED_BASE_PARAMETERS,
            "supported_types": [".ppt", ".pptx"],
        }
    ),
    "python_pptx": ProcessorConfig(
        name="Python-pptx",
        enabled=True,
        priority=1,
        parameters={
            "extract_notes": True,
            "extract_slide_layouts": True,
            "extract_images": False
        }
    )
}

# Excel
XLSX_PROCESSORS = {
    "unstructured_spreadsheet": ProcessorConfig(
        name="Unstructured Partition (XLS/XLSX)",
        enabled=False,
        priority=0,
        parameters={
            **UNSTRUCTURED_BASE_PARAMETERS,
            "supported_types": [".xlsx", ".xls"],
            "infer_table_structure": True,
        }
    ),
    "openpyxl": ProcessorConfig(
        name="OpenPyXL",
        enabled=True,
        priority=1,
        parameters={
            "data_only": True,
            "read_only": True,
            "max_rows": 1000000,
            "max_cols": 16384
        }
    ),
    "pandas_excel": ProcessorConfig(
        name="Pandas Excel",
        enabled=True,
        priority=2,
        parameters={
            "sheet_name": None,  # Все листы
            "header": 0,
            "na_values": ["", " ", "NULL", "null", "NaN"]
        }
    ),
    "xlrd": ProcessorConfig(
        name="XLRD (Legacy)",
        enabled=False,  # Только для старых .xls файлов
        priority=3,
        parameters={}
    )
}

# OCR для изображений
IMAGE_OCR_PROCESSORS = {
    "tesseract": ProcessorConfig(
        name="Tesseract OCR",
        enabled=True,
        priority=1,
        parameters={
            "lang": "rus+eng",
            "config": "--oem 3 --psm 6",
            "timeout": 60
        }
    ),
    "easyocr": ProcessorConfig(
        name="EasyOCR",
        enabled=True,
        priority=2,
        parameters={
            "languages": ["ru", "en"],
            "gpu": False,
            "width_ths": 0.7,
            "height_ths": 0.7
        }
    )
}

# Постобработка и очистка
TEXT_CLEANERS = {
    "basic": ProcessorConfig(
        name="Basic Cleaner",
        enabled=True,
        priority=1,
        parameters={
            "remove_extra_whitespace": True,
            "normalize_unicode": True,
            "fix_encoding": True,
            "remove_control_chars": True
        }
    ),
    "advanced": ProcessorConfig(
        name="Advanced Cleaner",
        enabled=True,
        priority=2,
        parameters={
            "fix_hyphenation": True,
            "merge_broken_words": True,
            "remove_headers_footers": True,
            "normalize_quotes": True,
            "fix_bullet_points": True
        }
    ),
    "html": ProcessorConfig(
        name="HTML Cleaner",
        enabled=True,
        priority=3,
        parameters={
            "remove_tags": True,
            "convert_entities": True,
            "preserve_structure": True
        }
    ),
    "unstructured_llm": ProcessorConfig(
        name="Unstructured LLM Cleaner",
        enabled=False,
        priority=4,
        parameters={
            "cleaner_functions": UNSTRUCTURED_BASE_PARAMETERS.get(
                "cleaner_functions",
                [
                    "clean_extra_whitespace",
                    "clean_multiple_newlines",
                    "clean_non_ascii_chars",
                    "clean_bullets",
                ],
            ),
            "drop_types": UNSTRUCTURED_BASE_PARAMETERS["drop_types"],
            "use_llm_cleaning": True,
            "repartition_if_missing": True,
            "fallback_partition_kwargs": {
                "strategy": "auto",
                "include_page_breaks": False,
            }
        }
    )
}

# Конвертеры в Markdown
MARKDOWN_CONVERTERS = {
    "pandoc": ProcessorConfig(
        name="Pandoc",
        enabled=False,  # Требует установки pandoc
        priority=1,
        parameters={
            "from_format": "html",
            "to_format": "markdown",
            "extra_args": ["--wrap=none"]
        }
    ),
    "markdownify": ProcessorConfig(
        name="Markdownify",
        enabled=True,
        priority=2,
        parameters={
            "heading_style": "ATX",
            "bullets": "-",
            "convert": ["b", "strong", "i", "em", "h1", "h2", "h3", "h4", "h5", "h6"]
        }
    ),
    "html2text": ProcessorConfig(
        name="html2text",
        enabled=True,
        priority=3,
        parameters={
            "ignore_links": False,
            "ignore_images": False,
            "body_width": 0,
            "unicode_snob": True
        }
    )
}

# Собираем все конфигурации
ALL_PROCESSORS = {
    "pdf": PDF_PROCESSORS,
    "docx": DOCX_PROCESSORS,
    "doc": DOCX_PROCESSORS,  # Те же процессоры
    "pptx": PPTX_PROCESSORS,
    "ppt": PPTX_PROCESSORS,  # Те же процессоры
    "xlsx": XLSX_PROCESSORS,
    "xls": XLSX_PROCESSORS,  # Те же процессоры
    "images": IMAGE_OCR_PROCESSORS,
    "cleaners": TEXT_CLEANERS,
    "markdown": MARKDOWN_CONVERTERS
}

# Настройки качества и метрик
QUALITY_METRICS = {
    "text_length": {
        "enabled": True,
        "weight": 0.1
    },
    "readability": {
        "enabled": True,
        "weight": 0.2
    },
    "structure_preservation": {
        "enabled": True,
        "weight": 0.3
    },
    "formatting_quality": {
        "enabled": True,
        "weight": 0.2
    },
    "error_rate": {
        "enabled": True,
        "weight": 0.2
    }
}

# Путь к файлам и директориям
PATHS = {
    "test_files": Path("test_files"),
    "output": Path("outputs"),
    "temp": Path("temp"),
    "configs": Path("configs"),
    "logs": Path("logs")
}