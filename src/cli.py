"""
CLI интерфейс для ALPACA Test Bench.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional

import click

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent))

from configs.processors_config import ALL_PROCESSORS
from core.pipeline import DocumentPipeline
from processors import *
from utils import FileManager, get_logger, setup_logging


def _is_unstructured_available() -> bool:
    return importlib.util.find_spec("unstructured") is not None


def _register_components(pipeline: DocumentPipeline, file_ext: str) -> List[str]:
    file_ext = file_ext.lower()

    if file_ext == '.pdf':
        pipeline.register_extractor(['.pdf'], PyPDFExtractor())
        pipeline.register_extractor(['.pdf'], PDFPlumberExtractor())
        pipeline.register_extractor(['.pdf'], PyMuPDFExtractor())
    elif file_ext == '.docx':
        pipeline.register_extractor(['.docx'], Docx2txtExtractor())
    elif file_ext == '.doc':
        pipeline.register_extractor(['.doc'], LibreOfficeExtractor())
    else:
        raise ValueError(f"Неподдерживаемый тип файла: {file_ext}")

    if _is_unstructured_available():
        already_registered = any(
            'Unstructured Partition' in extractors
            for extractors in pipeline.extractors.values()
        )
        if not already_registered:
            unstructured_types = [
                '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.md', '.html'
            ]
            language_hints = ['rus', 'eng']
            unstructured_params = {
                "supported_types": unstructured_types,
                "strategy": "hi_res",
                "chunking_strategy": "by_title",
                "include_metadata": True,
                "infer_table_structure": True,
                "languages": language_hints,
                "partition_kwargs": {
                    "languages": language_hints,
                    "ocr_languages": "+".join(language_hints),
                },
                "fallback_partition_kwargs": {
                    "languages": language_hints,
                    "ocr_languages": "+".join(language_hints),
                },
            }
            pipeline.register_extractor(
                unstructured_types,
                UnstructuredPartitionExtractor(unstructured_params)
            )

    existing_cleaners = {cleaner.name for cleaner in pipeline.cleaners}

    if "Basic Cleaner" not in existing_cleaners:
        pipeline.register_cleaner(BasicTextCleaner())
    if "Advanced Cleaner" not in existing_cleaners:
        pipeline.register_cleaner(AdvancedTextCleaner())
    if "HTML Cleaner" not in existing_cleaners:
        pipeline.register_cleaner(HTMLCleaner())

    if _is_unstructured_available():
        if "Unstructured LLM Cleaner" not in existing_cleaners:
            pipeline.register_cleaner(
                UnstructuredLLMCleaner({
                    "use_llm_cleaning": False,
                    "repartition_if_missing": True,
                    "languages": ['rus', 'eng'],
                    "fallback_partition_kwargs": {
                        "languages": ['rus', 'eng'],
                        "ocr_languages": 'rus+eng',
                    },
                })
            )

    if "custom" not in pipeline.converters:
        pipeline.register_converter("custom", CustomMarkdownFormatter())

    return pipeline.get_extractors_for_type(file_ext)


@click.group()
@click.option('--log-level', default='INFO', help='Уровень логирования')
@click.option('--log-dir', default='logs', help='Директория для логов')
def cli(log_level: str, log_dir: str):
    """ALPACA Test Bench - CLI интерфейс для тестирования процессоров документов."""
    setup_logging(log_level, log_dir)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--processor', '-p', multiple=True, help='Процессоры для использования')
@click.option('--cleaner', '-c', multiple=True, help='Очистители текста')
@click.option('--converter', '-conv', help='Конвертер в Markdown')
@click.option('--output-dir', '-o', default='outputs', help='Директория для результатов')
@click.option(
    '--save-intermediate',
    is_flag=True,
    help='Сохранять промежуточные результаты'
)
def process(
    file_path: str,
    processor: List[str],
    cleaner: List[str],
    converter: Optional[str],
    output_dir: str,
    save_intermediate: bool
):
    """Обработать один файл."""
    file_path = Path(file_path)
    logger = get_logger(__name__)
    
    click.echo(f"Обработка файла: {file_path}")
    
    # Настройка pipeline
    pipeline = DocumentPipeline({
        "save_intermediate": save_intermediate
    })
    
    # Регистрируем процессоры в зависимости от типа файла
    file_ext = file_path.suffix.lower()
    
    try:
        _register_components(pipeline, file_ext)
        
        # Обработка
        result = pipeline.process_document(
            file_path=file_path,
            extractor_name=processor[0] if processor else None,
            cleaner_names=list(cleaner) if cleaner else None,
            converter_name=converter
        )
        
        # Вывод результатов
        click.echo("\n=== РЕЗУЛЬТАТЫ ОБРАБОТКИ ===")
        
        # Извлечение
        extraction_results = result.get("extraction_results", {})
        for name, res in extraction_results.items():
            status = "✅" if res.status.value == "completed" else "❌"
            click.echo(f"{status} Извлечение ({name}): {res.execution_time:.2f}s")
        
        # Качество
        quality_scores = result.get("quality_scores", {})
        best_score = 0.0
        best_combo = "N/A"
        
        for extractor in quality_scores:
            for cleaner_name in quality_scores[extractor]:
                for conv_name in quality_scores[extractor][cleaner_name]:
                    score = quality_scores[extractor][cleaner_name][conv_name]
                    if (
                        hasattr(score, 'overall_score')
                        and score.overall_score > best_score
                    ):
                        best_score = score.overall_score
                        best_combo = f"{extractor}→{cleaner_name}→{conv_name}"
        
        click.echo(f"\n📊 Лучший результат: {best_score:.3f} ({best_combo})")
        
        # Общее время
        total_time = result.get("pipeline_metadata", {}).get("total_time", 0)
        click.echo(f"⏱️  Общее время: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        click.echo(f"❌ Ошибка: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--processors', '-p', multiple=True, help='Процессоры для сравнения')
@click.option('--output-format', type=click.Choice(['json', 'html']), default='json')
@click.option('--output-file', '-o', help='Файл для сохранения результатов')
def compare(
    file_path: str,
    processors: List[str],
    output_format: str,
    output_file: Optional[str]
):
    """Сравнить различные процессоры на одном файле."""
    file_path = Path(file_path)
    logger = get_logger(__name__)
    
    click.echo(f"Сравнение процессоров для файла: {file_path}")
    
    # Настройка pipeline
    pipeline = DocumentPipeline()
    file_ext = file_path.suffix.lower()
    
    try:
        available_processors = _register_components(pipeline, file_ext)
    except ValueError as exc:
        click.echo(f"❌ {exc}")
        sys.exit(1)
    
    # Определяем процессоры для тестирования
    processors_to_test = list(processors) if processors else available_processors
    
    click.echo(f"Тестируем процессоры: {', '.join(processors_to_test)}")
    
    try:
        # Сравнение
        with click.progressbar(processors_to_test, label="Обработка") as bar:
            results = {}
            for processor_name in bar:
                result = pipeline.process_document(
                    file_path=file_path,
                    extractor_name=processor_name
                )
                results[processor_name] = result
        
        # Создание отчета сравнения
        comparison_report = pipeline._create_comparison_report(results)
        
        # Вывод результатов
        click.echo("\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
        
        summary = comparison_report.get("performance_summary", {})
        best = comparison_report.get("best_performers", {}).get("overall")
        
        if best:
            click.echo(f"🏆 Лучший результат: {best['processor']} (оценка: {best['score']:.3f})")
        
        click.echo(f"\n📊 Сводка по процессорам:")
        for proc_name, perf in summary.items():
            score = perf.get('best_score', 0)
            time_taken = perf.get('total_execution_time', 0)
            click.echo(f"  • {proc_name}: {score:.3f} ({time_taken:.2f}s)")
        
        # Рекомендации
        recommendations = comparison_report.get("recommendations", [])
        if recommendations:
            click.echo(f"\n💡 Рекомендации:")
            for rec in recommendations:
                click.echo(f"  • {rec}")
        
        # Сохранение результатов
        if output_file:
            file_manager = FileManager()
            output_path = Path(output_file)
            
            if output_format == 'json':
                output_path.write_text(
                    json.dumps({
                        "comparison_report": comparison_report,
                        "individual_results": results
                    }, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )
            elif output_format == 'html':
                file_manager.create_comparison_report(
                    {"comparison_report": comparison_report, "individual_results": results},
                    output_path
                )
            
            click.echo(f"📄 Результаты сохранены в: {output_path}")
    
    except Exception as e:
        logger.error(f"Ошибка сравнения: {e}")
        click.echo(f"❌ Ошибка: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.option('--pattern', '-p', multiple=True, default=['*.pdf'], help='Паттерны файлов')
@click.option('--output-dir', '-o', default='batch_results', help='Директория результатов')
@click.option('--max-workers', default=4, help='Количество параллельных процессов')
@click.option('--timeout', default=300, help='Таймаут на файл (секунды)')
def batch(
    input_directory: str,
    pattern: List[str],
    output_dir: str,
    max_workers: int,
    timeout: int
):
    """Пакетная обработка файлов в директории."""
    input_dir = Path(input_directory)
    logger = get_logger(__name__)
    
    click.echo(f"Пакетная обработка директории: {input_dir}")
    click.echo(f"Паттерны файлов: {', '.join(pattern)}")
    
    # Настройка pipeline
    pipeline = DocumentPipeline({
        "max_workers": max_workers,
        "timeout": timeout
    })
    
    # Регистрация всех доступных процессоров
    try:
        # PDF
        pipeline.register_extractor(['.pdf'], PyPDFExtractor())
        pipeline.register_extractor(['.pdf'], PDFPlumberExtractor())
        
        # Word
        pipeline.register_extractor(['.docx'], Docx2txtExtractor())
        
        # Очистители
        pipeline.register_cleaner(BasicTextCleaner())
        pipeline.register_cleaner(AdvancedTextCleaner())
        
        # Конвертеры
        pipeline.register_converter("custom", CustomMarkdownFormatter())
        
        # Запуск пакетной обработки
        results = pipeline.batch_process(
            input_directory=input_dir,
            file_patterns=list(pattern)
        )
        
        # Вывод результатов
        summary = results.get("summary", {})
        click.echo(f"\n=== РЕЗУЛЬТАТЫ ПАКЕТНОЙ ОБРАБОТКИ ===")
        click.echo(f"📄 Всего файлов: {summary.get('total_files', 0)}")
        click.echo(f"✅ Успешно обработано: {summary.get('successful', 0)}")
        click.echo(f"❌ Ошибок: {summary.get('failed', 0)}")
        
        # Неудачные файлы
        failed_files = results.get("failed_files", [])
        if failed_files:
            click.echo(f"\n❌ Файлы с ошибками:")
            for file_path, error in failed_files:
                click.echo(f"  • {file_path}: {error}")
        
        click.echo(f"\n📁 Результаты сохранены в: {Path(output_dir).absolute()}")
        
    except Exception as e:
        logger.error(f"Ошибка пакетной обработки: {e}")
        click.echo(f"❌ Ошибка: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_processors():
    """Показать список доступных процессоров."""
    click.echo("📋 Доступные процессоры:\n")
    
    for file_type, processors in ALL_PROCESSORS.items():
        if file_type in ['cleaners', 'markdown']:
            continue
            
        click.echo(f"📄 {file_type.upper()}:")
        for proc_name, config in processors.items():
            status = "✅" if config.enabled else "❌"
            click.echo(f"  {status} {config.name} (приоритет: {config.priority})")
        click.echo()
    
    click.echo("🧹 Очистители:")
    for proc_name, config in ALL_PROCESSORS.get('cleaners', {}).items():
        status = "✅" if config.enabled else "❌"
        click.echo(f"  {status} {config.name}")
    
    click.echo("\n📝 Конвертеры Markdown:")
    for proc_name, config in ALL_PROCESSORS.get('markdown', {}).items():
        status = "✅" if config.enabled else "❌"
        click.echo(f"  {status} {config.name}")


@cli.command()
@click.option('--days', default=7, help='Удалить результаты старше N дней')
def cleanup(days: int):
    """Очистить старые результаты."""
    file_manager = FileManager()
    
    click.echo(f"Очистка результатов старше {days} дней...")
    
    try:
        file_manager.cleanup_old_results(days_old=days)
        
        # Показываем информацию о хранилище
        storage_info = file_manager.get_storage_info()
        click.echo(f"\n📊 Информация о хранилище:")
        click.echo(f"  💾 Размер: {storage_info['total_size_mb']:.1f} МБ")
        click.echo(f"  📄 Файлов: {storage_info['total_files']}")
        click.echo(f"  📁 Директорий: {storage_info['total_directories']}")
        
    except Exception as e:
        click.echo(f"❌ Ошибка очистки: {e}", err=True)


if __name__ == '__main__':
    cli()