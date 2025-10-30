"""
Система метрик для оценки качества обработки документов.
"""

import re
import string
from typing import Dict, Any, List, Optional
from pathlib import Path
import math

from ..core.base import BaseMetric, QualityScore
from .logger import get_logger

logger = get_logger(__name__)


class TextLengthMetric(BaseMetric):
    """Метрика длины извлеченного текста."""
    
    def __init__(self, weight: float = 0.1):
        super().__init__("Text Length", weight)
    
    def calculate(
        self, 
        processed_text: str, 
        original_file: Path = None, 
        metadata: Dict[str, Any] = None
    ) -> float:
        """
        Оценивает объем извлеченного текста.
        
        Больше текста не всегда лучше, но слишком мало может указывать
        на проблемы с извлечением.
        """
        text_length = len(processed_text.strip())
        
        if text_length == 0:
            return 0.0
        
        # Базовая оценка на основе длины
        if text_length < 100:
            return 0.2  # Очень мало текста
        elif text_length < 500:
            return 0.5  # Мало текста
        elif text_length < 2000:
            return 0.7  # Нормальное количество
        elif text_length < 10000:
            return 0.9  # Хорошее количество
        else:
            return 1.0  # Много текста


class ReadabilityMetric(BaseMetric):
    """Метрика читаемости текста."""
    
    def __init__(self, weight: float = 0.2):
        super().__init__("Readability", weight)
    
    def calculate(
        self, 
        processed_text: str, 
        original_file: Path = None, 
        metadata: Dict[str, Any] = None
    ) -> float:
        """Оценивает читаемость текста по различным индексам."""
        if not processed_text.strip():
            return 0.0
        
        try:
            # Flesch Reading Ease
            flesch_score = self._flesch_reading_ease(processed_text)
            
            # Automated Readability Index
            ari_score = self._automated_readability_index(processed_text)
            
            # Нормализуем оценки (0.0 - 1.0)
            flesch_normalized = self._normalize_flesch(flesch_score)
            ari_normalized = self._normalize_ari(ari_score)
            
            # Комбинируем оценки
            return (flesch_normalized + ari_normalized) / 2
            
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 0.5  # Средняя оценка при ошибке
    
    def _flesch_reading_ease(self, text: str) -> float:
        """Вычисляет индекс Flesch Reading Ease."""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum([self._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0
        
        return 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
    
    def _automated_readability_index(self, text: str) -> float:
        """Вычисляет Automated Readability Index."""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        characters = len(re.sub(r'\s+', '', text))
        
        if sentences == 0 or words == 0:
            return 0
        
        return 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
    
    def _count_syllables(self, word: str) -> int:
        """Подсчитывает количество слогов в слове (приблизительно)."""
        word = word.lower().strip(string.punctuation)
        if len(word) <= 3:
            return 1
        
        vowels = 'аеёиоуыэюя'  # Для русского текста
        syllables = sum(1 for char in word if char in vowels)
        
        # Английские гласные как fallback
        if syllables == 0:
            vowels_en = 'aeiouy'
            syllables = sum(1 for char in word if char in vowels_en)
        
        return max(1, syllables)
    
    def _normalize_flesch(self, score: float) -> float:
        """Нормализует Flesch Reading Ease (0-100) в диапазон 0-1."""
        if score >= 90:
            return 1.0  # Очень легко читать
        elif score >= 80:
            return 0.9  # Легко читать
        elif score >= 70:
            return 0.8  # Довольно легко
        elif score >= 60:
            return 0.7  # Стандартно
        elif score >= 50:
            return 0.6  # Довольно сложно
        elif score >= 30:
            return 0.4  # Сложно
        else:
            return 0.2  # Очень сложно
    
    def _normalize_ari(self, score: float) -> float:
        """Нормализует ARI в диапазон 0-1."""
        if score <= 6:
            return 1.0  # 6-й класс и ниже
        elif score <= 9:
            return 0.8  # 7-9 класс
        elif score <= 13:
            return 0.6  # 10-13 класс
        elif score <= 16:
            return 0.4  # Студент
        else:
            return 0.2  # Аспирант и выше


class StructurePreservationMetric(BaseMetric):
    """Метрика сохранения структуры документа."""
    
    def __init__(self, weight: float = 0.3):
        super().__init__("Structure Preservation", weight)
    
    def calculate(
        self, 
        processed_text: str, 
        original_file: Path = None, 
        metadata: Dict[str, Any] = None
    ) -> float:
        """Оценивает сохранение структурных элементов."""
        if not processed_text.strip():
            return 0.0
        
        structure_scores = []
        
        # Проверяем заголовки
        headers_score = self._evaluate_headers(processed_text)
        structure_scores.append(headers_score)
        
        # Проверяем списки
        lists_score = self._evaluate_lists(processed_text)
        structure_scores.append(lists_score)
        
        # Проверяем параграфы
        paragraphs_score = self._evaluate_paragraphs(processed_text)
        structure_scores.append(paragraphs_score)
        
        # Проверяем таблицы
        tables_score = self._evaluate_tables(processed_text)
        structure_scores.append(tables_score)
        
        return sum(structure_scores) / len(structure_scores)
    
    def _evaluate_headers(self, text: str) -> float:
        """Оценивает наличие и корректность заголовков."""
        # Ищем Markdown заголовки
        headers = re.findall(r'^#{1,6}\s+.+$', text, re.MULTILINE)
        
        if not headers:
            # Ищем альтернативные форматы заголовков
            alt_headers = re.findall(r'^\s*[А-ЯA-Z][^.!?]*:?\s*$', text, re.MULTILINE)
            return 0.3 if alt_headers else 0.1
        
        # Проверяем иерархию заголовков
        header_levels = [len(h.split()[0]) for h in headers]
        has_hierarchy = len(set(header_levels)) > 1
        
        return 1.0 if has_hierarchy else 0.7
    
    def _evaluate_lists(self, text: str) -> float:
        """Оценивает наличие и корректность списков."""
        # Markdown списки
        unordered_lists = re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE)
        ordered_lists = re.findall(r'^\s*\d+\.\s+', text, re.MULTILINE)
        
        total_lists = len(unordered_lists) + len(ordered_lists)
        
        if total_lists == 0:
            return 0.5  # Не все документы содержат списки
        elif total_lists < 5:
            return 0.7
        else:
            return 1.0
    
    def _evaluate_paragraphs(self, text: str) -> float:
        """Оценивает разбиение на параграфы."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) <= 1:
            return 0.3  # Плохое разбиение на параграфы
        
        # Проверяем средняя длину параграфов
        avg_length = sum(len(p) for p in paragraphs) / len(paragraphs)
        
        if 100 <= avg_length <= 500:
            return 1.0  # Оптимальная длина параграфов
        elif avg_length < 50:
            return 0.6  # Слишком короткие параграфы
        elif avg_length > 1000:
            return 0.7  # Слишком длинные параграфы
        else:
            return 0.8
    
    def _evaluate_tables(self, text: str) -> float:
        """Оценивает наличие и корректность таблиц."""
        # Markdown таблицы
        table_rows = re.findall(r'\|.*\|', text)
        
        if not table_rows:
            return 0.5  # Не все документы содержат таблицы
        
        # Проверяем корректность таблиц
        table_separators = re.findall(r'\|[-\s:|]+\|', text)
        
        return 1.0 if table_separators else 0.7


class FormattingQualityMetric(BaseMetric):
    """Метрика качества форматирования Markdown."""
    
    def __init__(self, weight: float = 0.2):
        super().__init__("Formatting Quality", weight)
    
    def calculate(
        self, 
        processed_text: str, 
        original_file: Path = None, 
        metadata: Dict[str, Any] = None
    ) -> float:
        """Оценивает качество Markdown форматирования."""
        if not processed_text.strip():
            return 0.0
        
        quality_scores = []
        
        # Проверяем корректность синтаксиса
        syntax_score = self._evaluate_markdown_syntax(processed_text)
        quality_scores.append(syntax_score)
        
        # Проверяем консистентность
        consistency_score = self._evaluate_consistency(processed_text)
        quality_scores.append(consistency_score)
        
        # Проверяем отсутствие артефактов
        artifacts_score = self._evaluate_artifacts(processed_text)
        quality_scores.append(artifacts_score)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _evaluate_markdown_syntax(self, text: str) -> float:
        """Оценивает корректность Markdown синтаксиса."""
        errors = 0
        total_checks = 0
        
        # Проверяем заголовки
        headers = re.findall(r'^#{1,6}[^#\s]', text, re.MULTILINE)
        errors += len(headers)  # Заголовки без пробела после #
        total_checks += len(re.findall(r'^#{1,6}', text, re.MULTILINE))
        
        # Проверяем ссылки
        broken_links = re.findall(r'\[[^\]]*\]\([^)]*\s[^)]*\)', text)
        errors += len(broken_links)  # Ссылки с пробелами в URL
        total_checks += len(re.findall(r'\[[^\]]*\]\([^)]*\)', text))
        
        # Проверяем выделение текста
        broken_bold = re.findall(r'\*\*[^*]*\*(?!\*)', text)
        errors += len(broken_bold)  # Незакрытое жирное выделение
        total_checks += len(re.findall(r'\*\*[^*]*\*\*', text))
        
        if total_checks == 0:
            return 0.8  # Нет элементов для проверки
        
        return max(0.0, 1.0 - (errors / total_checks))
    
    def _evaluate_consistency(self, text: str) -> float:
        """Оценивает консистентность форматирования."""
        consistency_scores = []
        
        # Консистентность заголовков
        headers = re.findall(r'^(#{1,6})\s+', text, re.MULTILINE)
        if headers:
            header_styles = set(headers)
            consistency_scores.append(0.8 if len(header_styles) <= 3 else 0.6)
        
        # Консистентность списков
        list_markers = re.findall(r'^\s*([-*+]|\d+\.)\s+', text, re.MULTILINE)
        if list_markers:
            unique_markers = len(set([m.strip('0123456789') for m in list_markers]))
            consistency_scores.append(1.0 if unique_markers <= 2 else 0.7)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.8
    
    def _evaluate_artifacts(self, text: str) -> float:
        """Оценивает отсутствие артефактов извлечения."""
        artifact_patterns = [
            r'�+',  # Символы замещения
            r'\x00-\x1f',  # Управляющие символы
            r'\.{4,}',  # Множественные точки
            r'\s{5,}',  # Избыточные пробелы
            r'[^\w\s.,!?;:()\-\'"#*\[\]|`~/@&%$№+=<>{}\\]{3,}',  # Странные символы
        ]
        
        artifacts = 0
        for pattern in artifact_patterns:
            artifacts += len(re.findall(pattern, text, re.MULTILINE))
        
        # Нормализуем по длине текста
        text_length = len(text)
        if text_length == 0:
            return 0.0
        
        artifact_ratio = artifacts / text_length
        
        if artifact_ratio == 0:
            return 1.0
        elif artifact_ratio < 0.01:
            return 0.9
        elif artifact_ratio < 0.05:
            return 0.7
        elif artifact_ratio < 0.1:
            return 0.5
        else:
            return 0.2


class ErrorRateMetric(BaseMetric):
    """Метрика уровня ошибок в тексте."""
    
    def __init__(self, weight: float = 0.2):
        super().__init__("Error Rate", weight)
    
    def calculate(
        self, 
        processed_text: str, 
        original_file: Path = None, 
        metadata: Dict[str, Any] = None
    ) -> float:
        """Оценивает уровень ошибок в обработанном тексте."""
        if not processed_text.strip():
            return 0.0
        
        error_indicators = []
        
        # Разорванные слова
        broken_words_score = self._evaluate_broken_words(processed_text)
        error_indicators.append(broken_words_score)
        
        # Некорректные символы
        invalid_chars_score = self._evaluate_invalid_characters(processed_text)
        error_indicators.append(invalid_chars_score)
        
        # OCR ошибки (если применимо)
        ocr_errors_score = self._evaluate_ocr_errors(processed_text)
        error_indicators.append(ocr_errors_score)
        
        return sum(error_indicators) / len(error_indicators)
    
    def _evaluate_broken_words(self, text: str) -> float:
        """Оценивает количество разорванных слов."""
        words = text.split()
        
        if not words:
            return 0.0
        
        broken_count = 0
        
        for word in words:
            # Слова, заканчивающиеся на дефис (возможная разорванность)
            if word.endswith('-') and len(word) > 2:
                broken_count += 1
            
            # Слишком короткие "слова" (возможные фрагменты)
            if len(word.strip(string.punctuation)) == 1 and word.isalpha():
                broken_count += 1
        
        broken_ratio = broken_count / len(words)
        return max(0.0, 1.0 - (broken_ratio * 10))  # Штраф за разорванные слова
    
    def _evaluate_invalid_characters(self, text: str) -> float:
        """Оценивает наличие некорректных символов."""
        # Допустимые символы для русского и английского текста
        valid_chars = set(
            string.ascii_letters + 
            string.digits + 
            string.punctuation + 
            string.whitespace +
            'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        )
        
        invalid_count = sum(1 for char in text if char not in valid_chars)
        
        if len(text) == 0:
            return 0.0
        
        invalid_ratio = invalid_count / len(text)
        return max(0.0, 1.0 - (invalid_ratio * 50))  # Сильный штраф за некорректные символы
    
    def _evaluate_ocr_errors(self, text: str) -> float:
        """Оценивает типичные OCR ошибки."""
        # Типичные OCR замещения
        ocr_patterns = [
            (r'\b[Il1]\b', 'I/l/1 confusion'),
            (r'\b[Oo0]\b', 'O/o/0 confusion'),
            (r'[,.](?=[A-ZА-Я])', 'Missing space after punctuation'),
            (r'[A-ZА-Я]{10,}', 'Long uppercase sequences'),
            (r'\d[A-Za-zА-Яа-я]', 'Number-letter confusion'),
        ]
        
        total_errors = 0
        
        for pattern, _ in ocr_patterns:
            matches = re.findall(pattern, text)
            total_errors += len(matches)
        
        if len(text) == 0:
            return 0.0
        
        error_ratio = total_errors / len(text.split())
        return max(0.0, 1.0 - (error_ratio * 5))


class MetricsCalculator:
    """Основной калькулятор метрик качества."""
    
    def __init__(self):
        self.metrics = [
            TextLengthMetric(weight=0.1),
            ReadabilityMetric(weight=0.2),
            StructurePreservationMetric(weight=0.3),
            FormattingQualityMetric(weight=0.2),
            ErrorRateMetric(weight=0.2)
        ]
        
    def calculate_all_metrics(
        self, 
        text: str, 
        original_file: Path = None, 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Вычисляет все метрики для текста."""
        results = {}
        
        for metric in self.metrics:
            try:
                score = metric.calculate(text, original_file, metadata)
                results[metric.name] = score
            except Exception as e:
                logger.error(f"Failed to calculate {metric.name}: {e}")
                results[metric.name] = 0.0
        
        return results
    
    def calculate_overall_score(
        self,
        text: str,
        original_file: Path = None,
        processor_name: str = "Unknown",
        execution_time: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> QualityScore:
        """Вычисляет общую оценку качества."""
        metric_scores = self.calculate_all_metrics(text, original_file, metadata)
        
        # Вычисляем взвешенную общую оценку
        total_weight = sum(metric.weight for metric in self.metrics)
        weighted_score = sum(
            metric_scores.get(metric.name, 0) * metric.weight
            for metric in self.metrics
        ) / total_weight
        
        return QualityScore(
            overall_score=weighted_score,
            metric_scores=metric_scores,
            processor_name=processor_name,
            execution_time=execution_time,
            metadata=metadata or {}
        )
    
    def add_custom_metric(self, metric: BaseMetric):
        """Добавляет пользовательскую метрику."""
        self.metrics.append(metric)
        logger.info(f"Added custom metric: {metric.name}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по всем метрикам."""
        return {
            "total_metrics": len(self.metrics),
            "metrics": [
                {
                    "name": metric.name,
                    "weight": metric.weight,
                    "description": metric.get_description()
                }
                for metric in self.metrics
            ],
            "total_weight": sum(metric.weight for metric in self.metrics)
        }