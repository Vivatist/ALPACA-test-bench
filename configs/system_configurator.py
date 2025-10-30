"""
Конфигуратор системы ALPACA Test Bench.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import setup_logging

from .processors_config import ALL_PROCESSORS, PATHS, QUALITY_METRICS


class SystemConfigurator:
    """Управляет конфигурацией системы."""
    
    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path("configs/system_config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла."""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию."""
        return {
            "system": {
                "version": "1.0.0",
                "debug_mode": False,
                "auto_cleanup": True,
                "cleanup_days": 7
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_logging": True,
                "console_logging": True
            },
            "performance": {
                "max_workers": 4,
                "timeout_seconds": 300,
                "max_file_size_mb": 100,
                "enable_caching": True
            },
            "processors": {
                "pdf_default": "PDFPlumber",
                "docx_default": "Unstructured Partition (DOC/DOCX)",
                "cleaner_default": ["Basic Cleaner", "Advanced Cleaner"],
                "markdown_default": "Custom Markdown Formatter"
            },
            "quality_metrics": {
                "text_length_weight": 0.1,
                "readability_weight": 0.2,
                "structure_weight": 0.3,
                "formatting_weight": 0.2,
                "error_rate_weight": 0.2,
                "enable_detailed_analysis": True
            },
            "ui": {
                "theme": "light",
                "show_debug_info": False,
                "auto_refresh": True,
                "results_per_page": 10
            },
            "storage": {
                "compress_results": False,
                "backup_enabled": False,
                "max_storage_mb": 1000
            }
        }
    
    def save_config(self):
        """Сохраняет конфигурацию в файл."""
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get_processor_config(self, file_type: str) -> List[str]:
        """Получает список рекомендуемых процессоров для типа файла."""
        processors = []
        
        if file_type.lower() == 'pdf':
            processors = ["PDFPlumber", "PyMuPDF", "PyPDF"]
        elif file_type.lower() in ['docx', 'doc']:
            processors = [
                "Unstructured Partition (DOC/DOCX)",
                "Docx2txt"
            ]
        elif file_type.lower() in ['pptx', 'ppt']:
            processors = ["Python-pptx"]
        elif file_type.lower() in ['xlsx', 'xls']:
            processors = ["OpenPyXL", "Pandas"]
        elif file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff']:
            processors = ["Tesseract", "EasyOCR"]
        
        return processors
    
    def get_quality_settings(self) -> Dict[str, float]:
        """Получает настройки метрик качества."""
        return {
            "text_length": self.config["quality_metrics"]["text_length_weight"],
            "readability": self.config["quality_metrics"]["readability_weight"],
            "structure": self.config["quality_metrics"]["structure_weight"],
            "formatting": self.config["quality_metrics"]["formatting_weight"],
            "error_rate": self.config["quality_metrics"]["error_rate_weight"]
        }
    
    def update_processor_settings(self, processor_type: str, settings: Dict[str, Any]):
        """Обновляет настройки процессора."""
        if "processor_settings" not in self.config:
            self.config["processor_settings"] = {}
        
        self.config["processor_settings"][processor_type] = settings
        self.save_config()
    
    def get_performance_settings(self) -> Dict[str, Any]:
        """Получает настройки производительности."""
        return self.config["performance"]
    
    def update_performance_settings(self, settings: Dict[str, Any]):
        """Обновляет настройки производительности."""
        self.config["performance"].update(settings)
        self.save_config()
    
    def get_logging_settings(self) -> Dict[str, Any]:
        """Получает настройки логирования."""
        return self.config["logging"]
    
    def is_processor_enabled(self, processor_name: str) -> bool:
        """Проверяет, включен ли процессор."""
        for file_type, processors in ALL_PROCESSORS.items():
            for proc_id, proc_config in processors.items():
                if proc_config.name == processor_name:
                    return proc_config.enabled
        return False
    
    def enable_processor(self, processor_name: str):
        """Включает процессор."""
        for file_type, processors in ALL_PROCESSORS.items():
            for proc_id, proc_config in processors.items():
                if proc_config.name == processor_name:
                    proc_config.enabled = True
                    return True
        return False
    
    def disable_processor(self, processor_name: str):
        """Выключает процессор."""
        for file_type, processors in ALL_PROCESSORS.items():
            for proc_id, proc_config in processors.items():
                if proc_config.name == processor_name:
                    proc_config.enabled = False
                    return True
        return False
    
    def get_recommended_pipeline(self, file_type: str) -> Dict[str, str]:
        """Получает рекомендуемый pipeline для типа файла."""
        recommendations = {
            'pdf': {
                'extractor': 'PDFPlumber',
                'cleaner': 'Advanced Cleaner',
                'converter': 'Custom Markdown Formatter'
            },
            'docx': {
                'extractor': 'Unstructured Partition (DOC/DOCX)',
                'cleaner': 'Basic Cleaner',
                'converter': 'Custom Markdown Formatter'
            },
            'pptx': {
                'extractor': 'Python-pptx',
                'cleaner': 'Advanced Cleaner',
                'converter': 'Custom Markdown Formatter'
            },
            'xlsx': {
                'extractor': 'OpenPyXL',
                'cleaner': 'Basic Cleaner',
                'converter': 'Custom Markdown Formatter'
            },
            'image': {
                'extractor': 'Tesseract',
                'cleaner': 'OCR Artifacts Cleaner',
                'converter': 'Custom Markdown Formatter'
            }
        }
        
        return recommendations.get(file_type.lower(), {})
    
    def validate_config(self) -> List[str]:
        """Проверяет корректность конфигурации."""
        errors = []
        
        # Проверяем обязательные секции
        required_sections = ["system", "logging", "performance", "processors"]
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Отсутствует секция: {section}")
        
        # Проверяем настройки производительности
        perf = self.config.get("performance", {})
        if perf.get("max_workers", 0) <= 0:
            errors.append("max_workers должно быть больше 0")
        
        if perf.get("timeout_seconds", 0) <= 0:
            errors.append("timeout_seconds должно быть больше 0")
        
        if perf.get("max_file_size_mb", 0) <= 0:
            errors.append("max_file_size_mb должно быть больше 0")
        
        # Проверяем веса метрик качества
        quality = self.config.get("quality_metrics", {})
        weights = [
            quality.get("text_length_weight", 0),
            quality.get("readability_weight", 0),
            quality.get("structure_weight", 0),
            quality.get("formatting_weight", 0),
            quality.get("error_rate_weight", 0)
        ]
        
        if sum(weights) == 0:
            errors.append("Сумма весов метрик не может быть 0")
        
        return errors
    
    def reset_to_defaults(self):
        """Сбрасывает конфигурацию к значениям по умолчанию."""
        self.config = self._get_default_config()
        self.save_config()
    
    def export_config(self, export_file: Path):
        """Экспортирует конфигурацию в файл."""
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def import_config(self, import_file: Path):
        """Импортирует конфигурацию из файла."""
        with open(import_file, 'r', encoding='utf-8') as f:
            imported_config = json.load(f)
        
        # Валидируем импортированную конфигурацию
        temp_config = self.config
        self.config = imported_config
        
        errors = self.validate_config()
        if errors:
            self.config = temp_config
            raise ValueError(f"Некорректная конфигурация: {', '.join(errors)}")
        
        self.save_config()


def setup_system(config_file: Path = None) -> SystemConfigurator:
    """
    Настраивает систему с конфигурацией.
    
    Args:
        config_file: Путь к файлу конфигурации
        
    Returns:
        SystemConfigurator: Настроенный конфигуратор
    """
    configurator = SystemConfigurator(config_file)
    
    # Проверяем конфигурацию
    errors = configurator.validate_config()
    if errors:
        print(f"Предупреждения конфигурации: {', '.join(errors)}")
        print("Использую значения по умолчанию")
        configurator.reset_to_defaults()
    
    # Применяем настройки логирования
    log_config = configurator.get_logging_settings()
    setup_logging(
        log_level=log_config["level"],
        log_format=log_config.get("format")
    )
    
    return configurator


def create_sample_config():
    """Создает образец файла конфигурации."""
    config_path = Path("configs/system_config.json")
    configurator = SystemConfigurator(config_path)
    
    if not config_path.exists():
        configurator.save_config()
        print(f"Создан образец конфигурации: {config_path}")
    else:
        print(f"Файл конфигурации уже существует: {config_path}")


if __name__ == "__main__":
    # Создание образца конфигурации
    create_sample_config()