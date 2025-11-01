"""
Процессоры для извлечения текста с использованием MarkItDown.
MarkItDown - библиотека от Microsoft, которая извлекает содержимое 
из различных форматов и конвертирует его в Markdown.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ..core.base import BaseExtractor
from ..utils.logger import get_logger, log_processing_stage

logger = get_logger(__name__)


class MarkItDownExtractor(BaseExtractor):
    """
    Экстрактор на базе MarkItDown от Microsoft.
    Извлекает текст из PDF, Word, Excel, PowerPoint, изображений и других форматов,
    сразу конвертируя его в Markdown.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MarkItDown", config)
        self._markitdown = None
        
    def _get_markitdown(self):
        """Ленивая инициализация MarkItDown."""
        if self._markitdown is None:
            try:
                from markitdown import MarkItDown
                self._markitdown = MarkItDown()
            except ImportError as e:
                raise ImportError(
                    "MarkItDown is not installed. Install with: pip install markitdown"
                ) from e
        return self._markitdown
    
    def supports_file_type(self, file_type: str) -> bool:
        """
        MarkItDown поддерживает множество форматов:
        PDF, DOCX, XLSX, PPTX, изображения (с OCR), HTML, текст и др.
        """
        supported = [
            '.pdf', '.docx', '.doc', '.xlsx', '.xls',
            '.pptx', '.ppt', '.html', '.htm', '.txt',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp',
            '.csv', '.json', '.xml', '.zip'
        ]
        return file_type.lower() in supported
    
    @log_processing_stage("MarkItDown Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """
        Извлекает текст из файла и конвертирует в Markdown.
        
        Args:
            file_path: Путь к файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            str: Текст в формате Markdown
        """
        try:
            markitdown = self._get_markitdown()
            
            # MarkItDown возвращает объект с атрибутом text_content
            result = markitdown.convert(str(file_path))
            
            if hasattr(result, 'text_content'):
                markdown_text = result.text_content
            elif isinstance(result, str):
                markdown_text = result
            else:
                markdown_text = str(result)
            
            if not markdown_text or not markdown_text.strip():
                logger.warning(f"MarkItDown returned empty content for {file_path.name}")
                return ""
                
            logger.info(
                f"MarkItDown extracted {len(markdown_text)} characters from {file_path.name}"
            )
            
            return markdown_text
            
        except Exception as e:
            logger.error(f"MarkItDown extraction failed for {file_path.name}: {e}")
            raise
    
    def get_metadata(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Возвращает метаданные о файле."""
        metadata = super().get_metadata(file_path, **kwargs)
        metadata.update({
            "extractor": "MarkItDown",
            "output_format": "markdown",
            "direct_markdown": True,  # Флаг что результат уже в Markdown
        })
        return metadata
