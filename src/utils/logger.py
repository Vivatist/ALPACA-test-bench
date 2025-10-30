"""
Настройка логирования для ALPACA Test Bench.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os
from datetime import datetime


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Получает настроенный логгер для модуля.
    
    Args:
        name: Имя логгера (обычно __name__)
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        # Логгер уже настроен
        return logger
    
    # Получаем уровень из переменной окружения или используем INFO по умолчанию
    log_level = level or os.getenv('LOG_LEVEL', 'INFO').upper()
    
    try:
        numeric_level = getattr(logging, log_level)
    except AttributeError:
        numeric_level = logging.INFO
    
    logger.setLevel(numeric_level)
    
    # Создаем форматтер
    formatter = logging.Formatter(
        fmt=os.getenv(
            'LOG_FORMAT',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный хендлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый хендлер
    log_dir = Path(os.getenv('LOG_DIR', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"alpaca_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # В файл записываем все
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Предотвращаем дублирование логов
    logger.propagate = False
    
    return logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_format: Optional[str] = None
):
    """
    Глобальная настройка логирования.
    
    Args:
        log_level: Уровень логирования
        log_dir: Директория для лог файлов
        log_format: Формат сообщений
    """
    os.environ['LOG_LEVEL'] = log_level
    os.environ['LOG_DIR'] = log_dir
    
    if log_format:
        os.environ['LOG_FORMAT'] = log_format
    
    # Создаем директорию для логов
    Path(log_dir).mkdir(exist_ok=True)


class ProcessingLogger:
    """Специализированный логгер для процессов обработки документов."""
    
    def __init__(self, process_name: str):
        self.logger = get_logger(f"processing.{process_name}")
        self.process_name = process_name
        
    def log_start(self, file_path: Path):
        """Логирует начало обработки файла."""
        self.logger.info(f"Starting {self.process_name} for: {file_path}")
        
    def log_success(self, file_path: Path, execution_time: float, **metrics):
        """Логирует успешную обработку."""
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(
            f"Completed {self.process_name} for: {file_path} "
            f"in {execution_time:.2f}s. Metrics: {metrics_str}"
        )
        
    def log_failure(self, file_path: Path, error: Exception):
        """Логирует ошибку обработки."""
        self.logger.error(
            f"Failed {self.process_name} for: {file_path}. "
            f"Error: {str(error)}"
        )
        
    def log_warning(self, file_path: Path, message: str):
        """Логирует предупреждение."""
        self.logger.warning(
            f"{self.process_name} warning for {file_path}: {message}"
        )


class PerformanceLogger:
    """Логгер для отслеживания производительности."""
    
    def __init__(self):
        self.logger = get_logger("performance")
        self.start_times = {}
        
    def start_timer(self, operation_id: str):
        """Начинает отсчет времени для операции."""
        import time
        self.start_times[operation_id] = time.time()
        self.logger.debug(f"Started timer for: {operation_id}")
        
    def end_timer(self, operation_id: str, **metrics):
        """Заканчивает отсчет и логирует результат."""
        import time
        
        if operation_id not in self.start_times:
            self.logger.warning(f"No start time found for: {operation_id}")
            return
            
        execution_time = time.time() - self.start_times[operation_id]
        del self.start_times[operation_id]
        
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(
            f"Operation {operation_id} completed in {execution_time:.2f}s. "
            f"Metrics: {metrics_str}"
        )
        
        return execution_time
        
    def log_memory_usage(self, operation_id: str):
        """Логирует использование памяти."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.logger.debug(
                f"Memory usage for {operation_id}: "
                f"RSS={memory_info.rss / 1024 / 1024:.2f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.2f}MB"
            )
            
        except ImportError:
            self.logger.debug("psutil not available, skipping memory logging")


def log_function_calls(func):
    """Декоратор для логирования вызовов функций."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Логируем вызов функции
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {e}")
            raise
            
    return wrapper


def log_processing_stage(stage_name: str):
    """Декоратор для логирования этапов обработки."""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            perf_logger = PerformanceLogger()
            
            operation_id = f"{stage_name}_{func.__name__}"
            perf_logger.start_timer(operation_id)
            
            try:
                logger.info(f"Starting stage: {stage_name}")
                result = func(*args, **kwargs)
                
                execution_time = perf_logger.end_timer(operation_id)
                logger.info(f"Completed stage: {stage_name} in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                perf_logger.end_timer(operation_id)
                logger.error(f"Stage {stage_name} failed: {e}")
                raise
                
        return wrapper
    return decorator