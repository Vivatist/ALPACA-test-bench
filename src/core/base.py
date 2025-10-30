"""
Базовые классы и интерфейсы для системы обработки документов.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
from enum import Enum


class ProcessingStage(Enum):
    """Этапы обработки документа."""
    VALIDATION = "validation"
    EXTRACTION = "extraction"
    CLEANING = "cleaning"
    CONVERSION = "conversion"
    EVALUATION = "evaluation"


class ProcessingStatus(Enum):
    """Статусы обработки."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ProcessingResult:
    """Результат обработки документа."""
    content: str
    processor_name: str
    stage: ProcessingStage
    status: ProcessingStatus
    execution_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    intermediate_files: List[Path] = None

    def __post_init__(self):
        if self.intermediate_files is None:
            self.intermediate_files = []


@dataclass 
class DocumentMetadata:
    """Метаданные документа."""
    file_path: Path
    file_size: int
    file_type: str
    mime_type: str
    created_at: Optional[float] = None
    modified_at: Optional[float] = None
    pages: Optional[int] = None
    language: Optional[str] = None
    encoding: Optional[str] = None
    
    @classmethod
    def from_file(cls, file_path: Path) -> "DocumentMetadata":
        """Создает метаданные из файла."""
        import mimetypes
        from pathlib import Path
        
        file_path = Path(file_path)
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return cls(
            file_path=file_path,
            file_size=stat.st_size,
            file_type=file_path.suffix.lower(),
            mime_type=mime_type or "application/octet-stream",
            created_at=stat.st_ctime,
            modified_at=stat.st_mtime
        )


class BaseProcessor(ABC):
    """Базовый класс для всех процессоров документов."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.timeout = self.config.get("timeout", 300)
        
    @abstractmethod
    def process(self, file_path: Path, **kwargs) -> ProcessingResult:
        """
        Основной метод обработки документа.
        
        Args:
            file_path: Путь к файлу для обработки
            **kwargs: Дополнительные параметры
            
        Returns:
            ProcessingResult: Результат обработки
        """
        pass
    
    @abstractmethod
    def supports_file_type(self, file_type: str) -> bool:
        """
        Проверяет, поддерживает ли процессор данный тип файла.
        
        Args:
            file_type: Тип файла (расширение)
            
        Returns:
            bool: True если поддерживает
        """
        pass
    
    def validate_input(self, file_path: Path) -> bool:
        """
        Валидирует входной файл.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            bool: True если файл валиден
        """
        return (
            file_path.exists() and 
            file_path.is_file() and
            self.supports_file_type(file_path.suffix.lower())
        )
    
    def _create_result(
        self, 
        content: str, 
        stage: ProcessingStage,
        status: ProcessingStatus,
        execution_time: float,
        metadata: Dict[str, Any] = None,
        error_message: str = None
    ) -> ProcessingResult:
        """Создает объект результата обработки."""
        return ProcessingResult(
            content=content,
            processor_name=self.name,
            stage=stage,
            status=status,
            execution_time=execution_time,
            metadata=metadata or {},
            error_message=error_message
        )


class BaseExtractor(BaseProcessor):
    """Базовый класс для извлекателей текста."""
    
    def process(self, file_path: Path, **kwargs) -> ProcessingResult:
        start_time = time.time()
        
        try:
            if not self.validate_input(file_path):
                return self._create_result(
                    content="",
                    stage=ProcessingStage.EXTRACTION,
                    status=ProcessingStatus.FAILED,
                    execution_time=time.time() - start_time,
                    error_message=f"Invalid input file: {file_path}"
                )
            
            content = self.extract_text(file_path, **kwargs)
            execution_time = time.time() - start_time
            
            return self._create_result(
                content=content,
                stage=ProcessingStage.EXTRACTION,
                status=ProcessingStatus.COMPLETED,
                execution_time=execution_time,
                metadata={
                    "text_length": len(content),
                    "file_size": file_path.stat().st_size
                }
            )
            
        except Exception as e:
            return self._create_result(
                content="",
                stage=ProcessingStage.EXTRACTION,
                status=ProcessingStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    @abstractmethod
    def extract_text(self, file_path: Path, **kwargs) -> str:
        """Извлекает текст из файла."""
        pass


class BaseCleaner(BaseProcessor):
    """Базовый класс для очистителей текста."""
    
    def process(
        self,
        file_path: Path,
        text: str = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> ProcessingResult:
        start_time = time.time()
        
        try:
            if text is None:
                return self._create_result(
                    content="",
                    stage=ProcessingStage.CLEANING,
                    status=ProcessingStatus.FAILED,
                    execution_time=time.time() - start_time,
                    error_message="No text provided for cleaning"
                )
            
            cleaned_output = self.clean_text(
                text,
                metadata=metadata,
                file_path=file_path,
                **kwargs
            )
            
            if isinstance(cleaned_output, tuple):
                cleaned_text, extra_metadata = cleaned_output
            else:
                cleaned_text = cleaned_output
                extra_metadata = {}
            
            if cleaned_text is None:
                cleaned_text = ""
            
            execution_time = time.time() - start_time
            base_metadata = {
                "original_length": len(text) if text else 0,
                "cleaned_length": len(cleaned_text),
                "reduction_ratio": 1 - (len(cleaned_text) / len(text)) if text else 0
            }
            
            if metadata:
                base_metadata["source_metadata_keys"] = list(metadata.keys())
            
            combined_metadata = {**base_metadata, **(extra_metadata or {})}
            
            return self._create_result(
                content=cleaned_text,
                stage=ProcessingStage.CLEANING,
                status=ProcessingStatus.COMPLETED,
                execution_time=execution_time,
                metadata=combined_metadata
            )
            
        except Exception as e:
            return self._create_result(
                content=text or "",
                stage=ProcessingStage.CLEANING,
                status=ProcessingStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def supports_file_type(self, file_type: str) -> bool:
        """Cleaners поддерживают любой текст."""
        return True
    
    @abstractmethod
    def clean_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """Очищает текст."""
        pass


class BaseConverter(BaseProcessor):
    """Базовый класс для конвертеров в Markdown."""
    
    def process(self, file_path: Path, text: str = None, **kwargs) -> ProcessingResult:
        start_time = time.time()
        
        try:
            if text is None:
                return self._create_result(
                    content="",
                    stage=ProcessingStage.CONVERSION,
                    status=ProcessingStatus.FAILED,
                    execution_time=time.time() - start_time,
                    error_message="No text provided for conversion"
                )
            
            markdown = self.convert_to_markdown(text, **kwargs)
            execution_time = time.time() - start_time
            
            return self._create_result(
                content=markdown,
                stage=ProcessingStage.CONVERSION,
                status=ProcessingStatus.COMPLETED,
                execution_time=execution_time,
                metadata={
                    "input_length": len(text),
                    "output_length": len(markdown),
                    "markdown_elements": self._count_markdown_elements(markdown)
                }
            )
            
        except Exception as e:
            return self._create_result(
                content=text or "",
                stage=ProcessingStage.CONVERSION,
                status=ProcessingStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def supports_file_type(self, file_type: str) -> bool:
        """Converters поддерживают любой текст."""
        return True
    
    @abstractmethod
    def convert_to_markdown(self, text: str, **kwargs) -> str:
        """Конвертирует текст в Markdown."""
        pass
    
    def _count_markdown_elements(self, markdown: str) -> Dict[str, int]:
        """Подсчитывает элементы Markdown для метрик."""
        import re
        
        return {
            "headers": len(re.findall(r'^#{1,6}\s+', markdown, re.MULTILINE)),
            "lists": len(re.findall(r'^\s*[-*+]\s+', markdown, re.MULTILINE)),
            "links": len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown)),
            "bold": len(re.findall(r'\*\*([^*]+)\*\*', markdown)),
            "italic": len(re.findall(r'\*([^*]+)\*', markdown)),
            "code_blocks": len(re.findall(r'```[\s\S]*?```', markdown)),
            "tables": len(re.findall(r'\|.*\|', markdown))
        }


class BaseMetric(ABC):
    """Базовый класс для метрик качества."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def calculate(
        self, 
        processed_text: str, 
        original_file: Path = None, 
        metadata: Dict[str, Any] = None
    ) -> float:
        """
        Вычисляет метрику качества.
        
        Args:
            processed_text: Обработанный текст
            original_file: Путь к исходному файлу
            metadata: Дополнительные метаданные
            
        Returns:
            float: Значение метрики (0.0 - 1.0)
        """
        pass
    
    def get_description(self) -> str:
        """Возвращает описание метрики."""
        return f"{self.name} (вес: {self.weight})"


@dataclass
class QualityScore:
    """Оценка качества обработки."""
    overall_score: float
    metric_scores: Dict[str, float]
    processor_name: str
    execution_time: float
    metadata: Dict[str, Any]
    
    def get_grade(self) -> str:
        """Возвращает буквенную оценку."""
        if self.overall_score >= 0.9:
            return "A+ (Отлично)"
        elif self.overall_score >= 0.8:
            return "A (Очень хорошо)"
        elif self.overall_score >= 0.7:
            return "B (Хорошо)"
        elif self.overall_score >= 0.6:
            return "C (Удовлетворительно)"
        elif self.overall_score >= 0.5:
            return "D (Плохо)"
        else:
            return "F (Неудовлетворительно)"