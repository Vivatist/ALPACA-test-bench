"""
Основной pipeline для обработки документов.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.metrics import MetricsCalculator
from .base import (BaseCleaner, BaseConverter, BaseExtractor, BaseProcessor,
                   DocumentMetadata, ProcessingResult, ProcessingStage,
                   ProcessingStatus, QualityScore)

logger = get_logger(__name__)


class DocumentPipeline:
    """Основной pipeline для обработки документов."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.extractors: Dict[str, Dict[str, BaseExtractor]] = {}
        self.cleaners: List[BaseCleaner] = []
        self.converters: Dict[str, BaseConverter] = {}
        self.metrics_calculator = MetricsCalculator()
        
        # Настройки выполнения
        self.max_workers = self.config.get("max_workers", 4)
        self.timeout = self.config.get("timeout", 300)
        self.save_intermediate = self.config.get("save_intermediate", True)
        
    def register_extractor(self, file_types: List[str], extractor: BaseExtractor):
        """Регистрирует экстрактор для определенных типов файлов."""
        for file_type in file_types:
            normalized = file_type.lower()
            self.extractors.setdefault(normalized, {})[extractor.name] = extractor
        logger.info(
            "Registered extractor %s for types: %s",
            extractor.name,
            ", ".join(file_types)
        )
    
    def register_cleaner(self, cleaner: BaseCleaner):
        """Регистрирует очиститель текста."""
        self.cleaners.append(cleaner)
        logger.info(f"Registered cleaner: {cleaner.name}")
    
    def register_converter(self, name: str, converter: BaseConverter):
        """Регистрирует конвертер в Markdown."""
        self.converters[name] = converter
        logger.info(f"Registered converter: {name}")
    
    def process_document(
        self,
        file_path: Path,
        extractor_name: str = None,
        cleaner_names: List[str] = None,
        converter_name: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Обрабатывает документ через полный pipeline.
        
        Args:
            file_path: Путь к файлу
            extractor_name: Имя конкретного экстрактора (если не указан, выбирается автоматически)
            cleaner_names: Список имен очистителей (если не указан, используются все)
            converter_name: Имя конвертера (если не указан, используется первый доступный)
            
        Returns:
            Dict с результатами всех этапов обработки
        """
        file_path = Path(file_path)
        
        # Получаем метаданные файла
        metadata = DocumentMetadata.from_file(file_path)
        
        # Результаты обработки
        results = {
            "file_metadata": metadata,
            "extraction_results": {},
            "cleaning_results": {},
            "conversion_results": {},
            "quality_scores": {},
            "pipeline_metadata": {
                "start_time": time.time(),
                "stages_completed": [],
                "errors": []
            }
        }
        
        try:
            # Этап 1: Извлечение текста
            extraction_results = self._extract_text(file_path, extractor_name, **kwargs)
            results["extraction_results"] = extraction_results
            results["pipeline_metadata"]["stages_completed"].append("extraction")
            
            if not extraction_results:
                raise ValueError("No successful text extraction")
            
            # Этап 2: Очистка текста
            cleaning_results = self._clean_texts(
                extraction_results,
                source_file=file_path,
                cleaner_names=cleaner_names,
                **kwargs
            )
            results["cleaning_results"] = cleaning_results
            results["pipeline_metadata"]["stages_completed"].append("cleaning")
            
            # Этап 3: Конвертация в Markdown
            conversion_results = self._convert_to_markdown(
                cleaning_results, converter_name, **kwargs
            )
            results["conversion_results"] = conversion_results
            results["pipeline_metadata"]["stages_completed"].append("conversion")
            
            # Этап 4: Оценка качества
            quality_scores = self._evaluate_quality(
                conversion_results, file_path, **kwargs
            )
            results["quality_scores"] = quality_scores
            results["pipeline_metadata"]["stages_completed"].append("evaluation")
            
            # Сохранение промежуточных результатов
            if self.save_intermediate:
                self._save_intermediate_results(results, file_path)
            
        except Exception as e:
            logger.error(f"Pipeline failed for {file_path}: {e}")
            results["pipeline_metadata"]["errors"].append(str(e))
        
        finally:
            results["pipeline_metadata"]["end_time"] = time.time()
            results["pipeline_metadata"]["total_time"] = (
                results["pipeline_metadata"]["end_time"] - 
                results["pipeline_metadata"]["start_time"]
            )
        
        return results
    
    def _extract_text(
        self, 
        file_path: Path, 
        extractor_name: str = None,
        **kwargs
    ) -> Dict[str, ProcessingResult]:
        """Извлекает текст с помощью доступных экстракторов."""
        file_type = file_path.suffix.lower()
        results: Dict[str, ProcessingResult] = {}
        processors = self.extractors.get(file_type, {})
        
        if not processors:
            logger.warning("No extractors registered for file type %s", file_type)
            return results
        
        if extractor_name:
            extractor = processors.get(extractor_name)
            if extractor is None:
                raise ValueError(
                    f"Extractor '{extractor_name}' is not registered for type {file_type}"
                )
            try:
                result = extractor.process(file_path, **kwargs)
                results[extractor_name] = result
            except Exception as exc:
                logger.error("Extractor %s failed: %s", extractor_name, exc)
        else:
            for name, extractor in processors.items():
                try:
                    result = extractor.process(file_path, **kwargs)
                    results[name] = result
                except Exception as exc:
                    logger.error("Extractor %s failed: %s", name, exc)
        
        return results
    
    def _clean_texts(
        self,
        extraction_results: Dict[str, ProcessingResult],
        source_file: Path,
        cleaner_names: List[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, ProcessingResult]]:
        """Очищает извлеченные тексты."""
        cleaning_results = {}
        
        cleaners_to_use = (
            [c for c in self.cleaners if c.name in cleaner_names] 
            if cleaner_names 
            else self.cleaners
        )
        
        for extractor_name, extraction_result in extraction_results.items():
            if extraction_result.status != ProcessingStatus.COMPLETED:
                continue
                
            cleaning_results[extractor_name] = {}
            
            for cleaner in cleaners_to_use:
                try:
                    result = cleaner.process(
                        file_path=source_file,
                        text=extraction_result.content,
                        metadata=extraction_result.metadata,
                        **kwargs
                    )
                    cleaning_results[extractor_name][cleaner.name] = result
                except Exception as e:
                    logger.error(f"Cleaner {cleaner.name} failed: {e}")
        
        return cleaning_results
    
    def _convert_to_markdown(
        self,
        cleaning_results: Dict[str, Dict[str, ProcessingResult]],
        converter_name: str = None,
        **kwargs
    ) -> Dict[str, Dict[str, Dict[str, ProcessingResult]]]:
        """Конвертирует очищенные тексты в Markdown."""
        conversion_results = {}
        
        converters_to_use = (
            {converter_name: self.converters[converter_name]} 
            if converter_name and converter_name in self.converters
            else self.converters
        )
        
        for extractor_name, cleaner_results in cleaning_results.items():
            conversion_results[extractor_name] = {}
            
            for cleaner_name, cleaning_result in cleaner_results.items():
                if cleaning_result.status != ProcessingStatus.COMPLETED:
                    continue
                
                conversion_results[extractor_name][cleaner_name] = {}
                
                for conv_name, converter in converters_to_use.items():
                    try:
                        result = converter.process(
                            file_path=None,
                            text=cleaning_result.content,
                            **kwargs
                        )
                        conversion_results[extractor_name][cleaner_name][conv_name] = result
                    except Exception as e:
                        logger.error(f"Converter {conv_name} failed: {e}")
        
        return conversion_results
    
    def _evaluate_quality(
        self,
        conversion_results: Dict[str, Dict[str, Dict[str, ProcessingResult]]],
        original_file: Path,
        **kwargs
    ) -> Dict[str, Dict[str, Dict[str, QualityScore]]]:
        """Оценивает качество результатов."""
        quality_scores = {}
        
        for extractor_name, cleaner_results in conversion_results.items():
            quality_scores[extractor_name] = {}
            
            for cleaner_name, converter_results in cleaner_results.items():
                quality_scores[extractor_name][cleaner_name] = {}
                
                for converter_name, result in converter_results.items():
                    if result.status != ProcessingStatus.COMPLETED:
                        continue
                    
                    try:
                        score = self.metrics_calculator.calculate_overall_score(
                            text=result.content,
                            original_file=original_file,
                            processor_name=f"{extractor_name}→{cleaner_name}→{converter_name}",
                            execution_time=result.execution_time,
                            metadata=result.metadata
                        )
                        quality_scores[extractor_name][cleaner_name][converter_name] = score
                    except Exception as e:
                        logger.error(f"Quality evaluation failed: {e}")
        
        return quality_scores
    
    def _save_intermediate_results(self, results: Dict[str, Any], file_path: Path):
        """Сохраняет промежуточные результаты."""
        try:
            from ..utils.file_manager import FileManager
            
            file_manager = FileManager()
            experiment_dir = file_manager.get_output_dir(file_path)

            file_manager.save_extraction_results(
                results["extraction_results"],
                experiment_dir,
                original_path=file_path,
            )
            file_manager.save_cleaning_results(
                results["cleaning_results"],
                experiment_dir,
                extraction_results=results["extraction_results"],
                original_path=file_path,
            )
            file_manager.save_conversion_results(
                results["conversion_results"],
                experiment_dir,
                extraction_results=results["extraction_results"],
                cleaning_results=results["cleaning_results"],
                original_path=file_path,
            )
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    def compare_processors(
        self,
        file_path: Path,
        extractor_names: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Сравнивает различные процессоры на одном файле.
        
        Args:
            file_path: Путь к файлу
            extractor_names: Список экстракторов для сравнения
            
        Returns:
            Dict с результатами сравнения
        """
        results = {}
        
        available = self.extractors.get(file_path.suffix.lower(), {})
        extractors_to_test = (
            extractor_names if extractor_names 
            else list(available.keys())
        )
        
        for extractor_name in extractors_to_test:
            logger.info(f"Testing extractor: {extractor_name}")
            result = self.process_document(
                file_path=file_path,
                extractor_name=extractor_name,
                **kwargs
            )
            results[extractor_name] = result
        
        # Создаем сравнительный отчет
        comparison_report = self._create_comparison_report(results)
        
        return {
            "individual_results": results,
            "comparison_report": comparison_report
        }
    
    def _create_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Создает отчет сравнения процессоров."""
        report = {
            "best_performers": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Анализируем результаты и находим лучших исполнителей
        best_overall = None
        best_score = 0.0
        
        for processor_name, result in results.items():
            quality_scores = result.get("quality_scores", {})
            
            # Находим лучшую комбинацию для этого процессора
            best_combo_score = 0.0
            best_combo = None
            
            for extractor in quality_scores:
                for cleaner in quality_scores[extractor]:
                    for converter in quality_scores[extractor][cleaner]:
                        score = quality_scores[extractor][cleaner][converter]
                        if hasattr(score, 'overall_score') and score.overall_score > best_combo_score:
                            best_combo_score = score.overall_score
                            best_combo = f"{extractor}→{cleaner}→{converter}"
            
            if best_combo_score > best_score:
                best_score = best_combo_score
                best_overall = (processor_name, best_combo, best_combo_score)
            
            report["performance_summary"][processor_name] = {
                "best_combination": best_combo,
                "best_score": best_combo_score,
                "total_execution_time": result.get("pipeline_metadata", {}).get("total_time", 0)
            }
        
        if best_overall:
            report["best_performers"]["overall"] = {
                "processor": best_overall[0],
                "combination": best_overall[1], 
                "score": best_overall[2]
            }
        
        # Генерируем рекомендации
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Генерирует рекомендации на основе анализа результатов."""
        recommendations = []
        
        performance_summary = report.get("performance_summary", {})
        
        if performance_summary:
            # Найти самый быстрый процессор
            fastest = min(
                performance_summary.items(),
                key=lambda x: x[1].get("total_execution_time", float('inf'))
            )
            recommendations.append(
                f"Самый быстрый: {fastest[0]} "
                f"({fastest[1].get('total_execution_time', 0):.2f}s)"
            )
            
            # Найти процессор с лучшим качеством
            best_quality = max(
                performance_summary.items(),
                key=lambda x: x[1].get("best_score", 0)
            )
            recommendations.append(
                f"Лучшее качество: {best_quality[0]} "
                f"(оценка: {best_quality[1].get('best_score', 0):.2f})"
            )
            
            # Рекомендации по балансу скорость/качество
            for name, perf in performance_summary.items():
                score = perf.get("best_score", 0)
                time_taken = perf.get("total_execution_time", 0)
                
                if score > 0.8 and time_taken < 10:
                    recommendations.append(f"Рекомендуется {name}: отличный баланс качества и скорости")
                elif score > 0.9:
                    recommendations.append(f"Для максимального качества используйте {name}")
                elif time_taken < 5:
                    recommendations.append(f"Для быстрой обработки используйте {name}")
        
        return recommendations

    def batch_process(
        self,
        input_directory: Path,
        file_patterns: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Пакетная обработка файлов в директории.
        
        Args:
            input_directory: Путь к директории с файлами
            file_patterns: Паттерны файлов для обработки
            
        Returns:
            Dict с результатами пакетной обработки
        """
        input_directory = Path(input_directory)
        
        if not input_directory.exists():
            raise ValueError(f"Directory does not exist: {input_directory}")
        
        # Находим файлы для обработки
        files_to_process = []
        patterns = file_patterns or ["*.pdf", "*.docx", "*.doc", "*.pptx", "*.ppt", "*.xlsx", "*.xls", "*.jpg", "*.jpeg", "*.png", "*.tiff"]
        
        for pattern in patterns:
            files_to_process.extend(input_directory.glob(pattern))
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Обрабатываем файлы
        results = {}
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for file_path in files_to_process:
                future = executor.submit(self.process_document, file_path, **kwargs)
                futures[future] = file_path
            
            for future in futures:
                file_path = futures[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results[str(file_path)] = result
                    logger.info(f"Successfully processed: {file_path}")
                except TimeoutError:
                    failed_files.append((str(file_path), "Timeout"))
                    logger.error(f"Timeout processing: {file_path}")
                except Exception as e:
                    failed_files.append((str(file_path), str(e)))
                    logger.error(f"Failed to process {file_path}: {e}")
        
        return {
            "successful_results": results,
            "failed_files": failed_files,
            "summary": {
                "total_files": len(files_to_process),
                "successful": len(results),
                "failed": len(failed_files)
            }
        }

    def get_extractors_for_type(self, file_type: str) -> List[str]:
        """Возвращает список доступных экстракторов для указанного типа файла."""
        return list(self.extractors.get(file_type.lower(), {}).keys())
