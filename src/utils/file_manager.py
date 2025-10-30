"""
Менеджер файлов для сохранения и управления результатами обработки.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import shutil
import os

from ..core.base import ProcessingResult, QualityScore
from .logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """Управляет файлами и директориями для результатов обработки."""
    
    def __init__(self, base_output_dir: Path = None):
        self.base_output_dir = Path(base_output_dir or "outputs")
        self.base_output_dir.mkdir(exist_ok=True)
        
    def get_output_dir(self, input_file: Path) -> Path:
        """
        Создает структуру директорий для результатов обработки файла.
        
        Args:
            input_file: Путь к исходному файлу
            
        Returns:
            Path: Путь к директории для результатов
        """
        # Создаем уникальную папку на основе имени файла и времени
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = self._make_safe_filename(input_file.stem)
        
        output_dir = self.base_output_dir / f"{safe_filename}_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Создаем подпапки
        (output_dir / "extraction").mkdir(exist_ok=True)
        (output_dir / "cleaning").mkdir(exist_ok=True)
        (output_dir / "markdown").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)
        
        return output_dir
    
    def save_extraction_results(
        self, 
        results: Dict[str, ProcessingResult], 
        output_dir: Path
    ):
        """Сохраняет результаты извлечения текста."""
        extraction_dir = output_dir / "extraction"
        
        for processor_name, result in results.items():
            if result.status.value == "completed":
                # Сохраняем текст
                text_file = extraction_dir / f"{processor_name}.txt"
                text_file.write_text(result.content, encoding='utf-8')
                
                # Сохраняем метаданные
                metadata_file = extraction_dir / f"{processor_name}_metadata.json"
                metadata = {
                    "processor": processor_name,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata,
                    "error_message": result.error_message
                }
                metadata_file.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False), 
                    encoding='utf-8'
                )
                
        logger.info(f"Saved extraction results to {extraction_dir}")
    
    def save_cleaning_results(
        self, 
        results: Dict[str, Dict[str, ProcessingResult]], 
        output_dir: Path
    ):
        """Сохраняет результаты очистки текста."""
        cleaning_dir = output_dir / "cleaning"
        
        for extractor_name, cleaner_results in results.items():
            extractor_dir = cleaning_dir / extractor_name
            extractor_dir.mkdir(exist_ok=True)
            
            for cleaner_name, result in cleaner_results.items():
                if result.status.value == "completed":
                    # Сохраняем очищенный текст
                    text_file = extractor_dir / f"{cleaner_name}.txt"
                    text_file.write_text(result.content, encoding='utf-8')
                    
                    # Сохраняем метаданные
                    metadata_file = extractor_dir / f"{cleaner_name}_metadata.json"
                    metadata = {
                        "extractor": extractor_name,
                        "cleaner": cleaner_name,
                        "status": result.status.value,
                        "execution_time": result.execution_time,
                        "metadata": result.metadata,
                        "error_message": result.error_message
                    }
                    metadata_file.write_text(
                        json.dumps(metadata, indent=2, ensure_ascii=False), 
                        encoding='utf-8'
                    )
        
        logger.info(f"Saved cleaning results to {cleaning_dir}")
    
    def save_conversion_results(
        self, 
        results: Dict[str, Dict[str, Dict[str, ProcessingResult]]], 
        output_dir: Path
    ):
        """Сохраняет результаты конвертации в Markdown."""
        markdown_dir = output_dir / "markdown"
        
        for extractor_name, cleaner_results in results.items():
            for cleaner_name, converter_results in cleaner_results.items():
                pipeline_name = f"{extractor_name}_{cleaner_name}"
                pipeline_dir = markdown_dir / pipeline_name
                pipeline_dir.mkdir(exist_ok=True)
                
                for converter_name, result in converter_results.items():
                    if result.status.value == "completed":
                        # Сохраняем Markdown
                        md_file = pipeline_dir / f"{converter_name}.md"
                        md_file.write_text(result.content, encoding='utf-8')
                        
                        # Сохраняем метаданные
                        metadata_file = pipeline_dir / f"{converter_name}_metadata.json"
                        metadata = {
                            "extractor": extractor_name,
                            "cleaner": cleaner_name,
                            "converter": converter_name,
                            "status": result.status.value,
                            "execution_time": result.execution_time,
                            "metadata": result.metadata,
                            "error_message": result.error_message
                        }
                        metadata_file.write_text(
                            json.dumps(metadata, indent=2, ensure_ascii=False), 
                            encoding='utf-8'
                        )
        
        logger.info(f"Saved markdown results to {markdown_dir}")
    
    def save_quality_report(
        self, 
        quality_scores: Dict[str, Dict[str, Dict[str, QualityScore]]], 
        report_file: Path
    ):
        """Сохраняет отчет по качеству обработки."""
        report_data = {
            "generation_time": datetime.now().isoformat(),
            "results": {}
        }
        
        best_overall = None
        best_score = 0.0
        
        for extractor_name, cleaner_results in quality_scores.items():
            report_data["results"][extractor_name] = {}
            
            for cleaner_name, converter_results in cleaner_results.items():
                report_data["results"][extractor_name][cleaner_name] = {}
                
                for converter_name, score in converter_results.items():
                    pipeline_name = f"{extractor_name}→{cleaner_name}→{converter_name}"
                    
                    score_data = {
                        "pipeline": pipeline_name,
                        "overall_score": score.overall_score,
                        "grade": score.get_grade(),
                        "execution_time": score.execution_time,
                        "metric_scores": score.metric_scores,
                        "metadata": score.metadata
                    }
                    
                    report_data["results"][extractor_name][cleaner_name][converter_name] = score_data
                    
                    # Отслеживаем лучший результат
                    if score.overall_score > best_score:
                        best_score = score.overall_score
                        best_overall = {
                            "pipeline": pipeline_name,
                            "score": score.overall_score,
                            "grade": score.get_grade()
                        }
        
        # Добавляем сводку
        report_data["summary"] = {
            "total_pipelines": sum(
                len(converter_results) 
                for cleaner_results in quality_scores.values()
                for converter_results in cleaner_results.values()
            ),
            "best_performer": best_overall
        }
        
        report_file.write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False), 
            encoding='utf-8'
        )
        
        logger.info(f"Saved quality report to {report_file}")
    
    def create_comparison_report(
        self,
        comparison_results: Dict[str, Any],
        output_file: Path
    ):
        """Создает HTML отчет сравнения процессоров."""
        html_content = self._generate_comparison_html(comparison_results)
        
        output_file.write_text(html_content, encoding='utf-8')
        logger.info(f"Created comparison report: {output_file}")
    
    def _generate_comparison_html(self, results: Dict[str, Any]) -> str:
        """Генерирует HTML для отчета сравнения."""
        html = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ALPACA Test Bench - Отчет сравнения</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #f5f5f5;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .header {{ 
                    text-align: center; 
                    margin-bottom: 30px; 
                    color: #333;
                }}
                .summary {{ 
                    background: #e8f4f8; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-bottom: 20px;
                }}
                .processor-results {{ 
                    margin-bottom: 30px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px;
                }}
                .processor-header {{ 
                    background: #f0f0f0; 
                    padding: 10px; 
                    font-weight: bold;
                    border-bottom: 1px solid #ddd;
                }}
                .metrics-table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 10px 0;
                }}
                .metrics-table th, .metrics-table td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left;
                }}
                .metrics-table th {{ 
                    background: #f9f9f9; 
                    font-weight: bold;
                }}
                .score-excellent {{ color: #28a745; font-weight: bold; }}
                .score-good {{ color: #17a2b8; font-weight: bold; }}
                .score-fair {{ color: #ffc107; font-weight: bold; }}
                .score-poor {{ color: #dc3545; font-weight: bold; }}
                .recommendations {{ 
                    background: #fff3cd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    border-left: 4px solid #ffc107;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧪 ALPACA Test Bench</h1>
                    <h2>Отчет сравнения процессоров</h2>
                    <p>Сгенерирован: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
                </div>
        """
        
        # Добавляем сводку
        comparison_report = results.get("comparison_report", {})
        best_performer = comparison_report.get("best_performers", {}).get("overall")
        
        if best_performer:
            html += f"""
                <div class="summary">
                    <h3>🏆 Лучший результат</h3>
                    <p><strong>Процессор:</strong> {best_performer['processor']}</p>
                    <p><strong>Конфигурация:</strong> {best_performer['combination']}</p>
                    <p><strong>Оценка:</strong> <span class="score-excellent">{best_performer['score']:.2f}</span> ({best_performer['grade']})</p>
                </div>
            """
        
        # Добавляем результаты по процессорам
        individual_results = results.get("individual_results", {})
        
        for processor_name, result in individual_results.items():
            html += f"""
                <div class="processor-results">
                    <div class="processor-header">
                        {processor_name}
                    </div>
                    <div style="padding: 15px;">
            """
            
            # Извлекаем лучший результат для этого процессора
            quality_scores = result.get("quality_scores", {})
            best_config = None
            best_score_val = 0.0
            
            for extractor in quality_scores:
                for cleaner in quality_scores[extractor]:
                    for converter in quality_scores[extractor][cleaner]:
                        score = quality_scores[extractor][cleaner][converter]
                        if hasattr(score, 'overall_score') and score.overall_score > best_score_val:
                            best_score_val = score.overall_score
                            best_config = {
                                'pipeline': f"{extractor}→{cleaner}→{converter}",
                                'score': score,
                                'metrics': score.metric_scores
                            }
            
            if best_config:
                score_class = self._get_score_class(best_config['score'].overall_score)
                
                html += f"""
                        <p><strong>Лучшая конфигурация:</strong> {best_config['pipeline']}</p>
                        <p><strong>Общая оценка:</strong> <span class="{score_class}">{best_config['score'].overall_score:.2f}</span> ({best_config['score'].get_grade()})</p>
                        <p><strong>Время выполнения:</strong> {best_config['score'].execution_time:.2f} сек</p>
                        
                        <table class="metrics-table">
                            <thead>
                                <tr>
                                    <th>Метрика</th>
                                    <th>Значение</th>
                                    <th>Оценка</th>
                                </tr>
                            </thead>
                            <tbody>
                """
                
                for metric_name, metric_value in best_config['metrics'].items():
                    metric_class = self._get_score_class(metric_value)
                    html += f"""
                                <tr>
                                    <td>{metric_name}</td>
                                    <td><span class="{metric_class}">{metric_value:.3f}</span></td>
                                    <td>{self._get_metric_description(metric_value)}</td>
                                </tr>
                    """
                
                html += """
                            </tbody>
                        </table>
                """
            
            html += """
                    </div>
                </div>
            """
        
        # Добавляем рекомендации
        recommendations = comparison_report.get("recommendations", [])
        if recommendations:
            html += """
                <div class="recommendations">
                    <h3>💡 Рекомендации</h3>
                    <ul>
            """
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            
            html += """
                    </ul>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_score_class(self, score: float) -> str:
        """Возвращает CSS класс для оценки."""
        if score >= 0.8:
            return "score-excellent"
        elif score >= 0.6:
            return "score-good"
        elif score >= 0.4:
            return "score-fair"
        else:
            return "score-poor"
    
    def _get_metric_description(self, score: float) -> str:
        """Возвращает описание оценки метрики."""
        if score >= 0.9:
            return "Отлично"
        elif score >= 0.7:
            return "Хорошо"
        elif score >= 0.5:
            return "Удовлетворительно"
        elif score >= 0.3:
            return "Плохо"
        else:
            return "Очень плохо"
    
    def _make_safe_filename(self, filename: str) -> str:
        """Создает безопасное имя файла."""
        # Удаляем или заменяем небезопасные символы
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        safe_name = ''.join(c if c in safe_chars else '_' for c in filename)
        
        # Ограничиваем длину
        return safe_name[:50] if len(safe_name) > 50 else safe_name
    
    def cleanup_old_results(self, days_old: int = 7):
        """Удаляет старые результаты."""
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        deleted_count = 0
        for item in self.base_output_dir.iterdir():
            if item.is_dir() and item.stat().st_mtime < cutoff_time:
                try:
                    shutil.rmtree(item)
                    deleted_count += 1
                    logger.info(f"Deleted old result directory: {item}")
                except Exception as e:
                    logger.error(f"Failed to delete {item}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old result directories")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Возвращает информацию об использовании дискового пространства."""
        total_size = 0
        file_count = 0
        dir_count = 0
        
        for root, dirs, files in os.walk(self.base_output_dir):
            dir_count += len(dirs)
            for file in files:
                file_path = Path(root) / file
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except OSError:
                    pass
        
        return {
            "total_size_mb": total_size / (1024 * 1024),
            "total_files": file_count,
            "total_directories": dir_count,
            "base_directory": str(self.base_output_dir)
        }