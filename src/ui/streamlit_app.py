"""
Streamlit веб-интерфейс для ALPACA Test Bench.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Optional
import importlib.util

# Настройка страницы
st.set_page_config(
    page_title="ALPACA Test Bench",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Импорты основных компонентов
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from core.pipeline import DocumentPipeline
    from processors import *
    from utils import get_logger, setup_logging, FileManager
    from configs.processors_config import ALL_PROCESSORS, QUALITY_METRICS
    
    setup_logging("INFO")
    logger = get_logger(__name__)
    
except ImportError as e:
    st.error(f"Ошибка импорта: {e}")
    st.stop()


class StreamlitApp:
    """Основное приложение Streamlit."""
    
    def __init__(self):
        self.pipeline = DocumentPipeline()
        self.file_manager = FileManager()
        self.unstructured_available = self._is_unstructured_available()
        self.setup_pipeline()
        
    def setup_pipeline(self):
        """Настраивает pipeline с доступными процессорами."""
        # Регистрируем PDF процессоры
        try:
            self.pipeline.register_extractor(['.pdf'], PyPDFExtractor())
            self.pipeline.register_extractor(['.pdf'], PDFPlumberExtractor())
            self.pipeline.register_extractor(['.pdf'], PyMuPDFExtractor())
        except Exception as e:
            st.warning(f"Некоторые PDF процессоры недоступны: {e}")
        
        # Регистрируем Word процессоры
        try:
            self.pipeline.register_extractor(['.docx'], PythonDocxExtractor())
            self.pipeline.register_extractor(['.docx'], Docx2txtExtractor())
        except Exception as e:
            st.warning(f"Некоторые Word процессоры недоступны: {e}")

        # Unstructured интеграция
        if self.unstructured_available:
            try:
                unstructured_types = [
                    '.pdf',
                    '.docx',
                    '.doc',
                    '.pptx',
                    '.ppt',
                    '.md',
                    '.html',
                ]
                unstructured_params = {
                    "supported_types": unstructured_types,
                    "strategy": "hi_res",
                    "chunking_strategy": "by_title",
                    "include_metadata": True,
                    "infer_table_structure": True,
                }
                self.pipeline.register_extractor(
                    unstructured_types,
                    UnstructuredPartitionExtractor(unstructured_params)
                )
                self.pipeline.register_cleaner(
                    UnstructuredLLMCleaner(
                        {
                            "use_llm_cleaning": False,
                            "repartition_if_missing": True,
                        }
                    )
                )
            except Exception as e:
                st.warning(f"Unstructured интеграция недоступна: {e}")
        
        # Регистрируем очистители
        self.pipeline.register_cleaner(BasicTextCleaner())
        self.pipeline.register_cleaner(AdvancedTextCleaner())
        self.pipeline.register_cleaner(HTMLCleaner())
        
        # Регистрируем конвертеры
        self.pipeline.register_converter("custom", CustomMarkdownFormatter())
        try:
            self.pipeline.register_converter("markdownify", MarkdownifyConverter())
        except:
            pass
        try:
            self.pipeline.register_converter("html2text", Html2TextConverter())
        except:
            pass

    @staticmethod
    def _is_unstructured_available() -> bool:
        return importlib.util.find_spec("unstructured") is not None
    
    def run(self):
        """Запуск приложения."""
        if not self.unstructured_available:
            st.sidebar.info(
                "Установите пакет `unstructured[all-docs]`, чтобы включить"
                " структурированное извлечение и LLM-нормализацию."
            )
        st.title("🧪 ALPACA Test Bench")
        st.subheader("Испытательный стенд для тестирования библиотек оцифровки документов")
        
        # Sidebar для навигации
        page = st.sidebar.selectbox(
            "Выберите режим работы",
            [
                "📄 Тестирование файла",
                "🔄 Сравнение процессоров", 
                "📊 Пакетная обработка",
                "⚙️ Настройки",
                "📈 Аналитика"
            ]
        )
        
        if page == "📄 Тестирование файла":
            self.single_file_testing()
        elif page == "🔄 Сравнение процессоров":
            self.processor_comparison()
        elif page == "📊 Пакетная обработка":
            self.batch_processing()
        elif page == "⚙️ Настройки":
            self.settings_page()
        elif page == "📈 Аналитика":
            self.analytics_page()
    
    def single_file_testing(self):
        """Страница тестирования одного файла."""
        st.header("Тестирование одного файла")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите файл для обработки",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'jpg', 'jpeg', 'png'],
            help="Поддерживаемые форматы: PDF, Word, PowerPoint, Excel, изображения"
        )
        
        if not uploaded_file:
            st.info("Пожалуйста, загрузите файл для начала тестирования")
            return
        
        # Сохраняем загруженный файл
        temp_file = Path(f"temp_{uploaded_file.name}")
        temp_file.write_bytes(uploaded_file.getvalue())
        
        try:
            # Настройки обработки
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Настройки обработки")
                
                # Выбор процессоров
                available_processors = self._get_available_processors(temp_file.suffix)
                selected_processors = st.multiselect(
                    "Выберите процессоры для тестирования",
                    available_processors,
                    default=available_processors[:2] if available_processors else []
                )
                
                # Настройки очистки
                enable_cleaning = st.checkbox("Включить очистку текста", value=True)
                if enable_cleaning:
                    cleaning_choices = [
                        "Basic Cleaner",
                        "Advanced Cleaner",
                        "HTML Cleaner",
                    ]
                    if self.unstructured_available:
                        cleaning_choices.append("Unstructured LLM Cleaner")
                    cleaning_options = st.multiselect(
                        "Методы очистки",
                        cleaning_choices,
                        default=["Basic Cleaner", "Advanced Cleaner"],
                    )
                else:
                    cleaning_options = []
                
                # Настройки конвертации
                converter_options = st.multiselect(
                    "Конвертеры в Markdown",
                    ["custom", "markdownify", "html2text"],
                    default=["custom"]
                )
            
            with col2:
                st.subheader("Информация о файле")
                
                file_info = {
                    "Имя файла": uploaded_file.name,
                    "Размер": f"{len(uploaded_file.getvalue()) / 1024:.1f} КБ",
                    "Тип": uploaded_file.type or "Неизвестно"
                }
                
                for key, value in file_info.items():
                    st.text(f"{key}: {value}")
            
            # Кнопка запуска обработки
            if st.button("🚀 Запустить обработку", type="primary"):
                if not selected_processors:
                    st.error("Выберите хотя бы один процессор")
                    return
                
                self._process_single_file(
                    temp_file, 
                    selected_processors,
                    cleaning_options,
                    converter_options
                )
        
        finally:
            # Удаляем временный файл
            if temp_file.exists():
                temp_file.unlink()
    
    def _get_available_processors(self, file_extension: str) -> List[str]:
        """Получает список доступных процессоров для типа файла."""
        if not file_extension:
            return []
        processors = self.pipeline.get_extractors_for_type(file_extension.lower())
        return processors
    
    def _process_single_file(
        self, 
        file_path: Path, 
        processors: List[str],
        cleaners: List[str],
        converters: List[str]
    ):
        """Обрабатывает файл и отображает результаты."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        
        for i, processor in enumerate(processors):
            status_text.text(f"Обработка с помощью {processor}...")
            progress_bar.progress((i + 1) / len(processors))
            
            try:
                # Обработка файла
                result = self.pipeline.process_document(
                    file_path=file_path,
                    extractor_name=processor,
                    cleaner_names=cleaners,
                    converter_name=converters[0] if converters else None
                )
                
                results[processor] = result
                
            except Exception as e:
                st.error(f"Ошибка при обработке с {processor}: {e}")
                logger.error(f"Processing failed with {processor}: {e}")
        
        status_text.text("Обработка завершена!")
        progress_bar.progress(100)
        
        if results:
            self._display_results(results, file_path.name)
    
    def _display_results(self, results: Dict[str, Any], filename: str):
        """Отображает результаты обработки."""
        st.header("Результаты обработки")
        
        # Создаем табы для разных видов результатов
        tabs = st.tabs(["📊 Сравнение", "📝 Тексты", "🎯 Качество", "📈 Метрики"])
        
        with tabs[0]:
            self._display_comparison_results(results)
        
        with tabs[1]:
            self._display_extracted_texts(results)
        
        with tabs[2]:
            self._display_quality_scores(results)
        
        with tabs[3]:
            self._display_detailed_metrics(results)
    
    def _display_comparison_results(self, results: Dict[str, Any]):
        """Отображает сравнительные результаты."""
        st.subheader("Сравнение процессоров")
        
        # Собираем данные для сравнения
        comparison_data = []
        
        for processor_name, result in results.items():
            quality_scores = result.get("quality_scores", {})
            
            # Находим лучший результат для этого процессора
            best_score = 0.0
            best_combination = "N/A"
            execution_time = result.get("pipeline_metadata", {}).get("total_time", 0)
            
            for extractor in quality_scores:
                for cleaner in quality_scores[extractor]:
                    for converter in quality_scores[extractor][cleaner]:
                        score = quality_scores[extractor][cleaner][converter]
                        if hasattr(score, 'overall_score') and score.overall_score > best_score:
                            best_score = score.overall_score
                            best_combination = f"{extractor}→{cleaner}→{converter}"
            
            comparison_data.append({
                "Процессор": processor_name,
                "Лучшая оценка": best_score,
                "Лучшая комбинация": best_combination,
                "Время выполнения": execution_time,
                "Оценка": self._get_grade(best_score)
            })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Таблица результатов
            st.dataframe(df, use_container_width=True)
            
            # График сравнения
            col1, col2 = st.columns(2)
            
            with col1:
                # График качества
                fig_quality = px.bar(
                    df, 
                    x="Процессор", 
                    y="Лучшая оценка",
                    title="Сравнение качества обработки",
                    color="Лучшая оценка",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_quality, use_container_width=True)
            
            with col2:
                # График времени выполнения
                fig_time = px.bar(
                    df,
                    x="Процессор",
                    y="Время выполнения",
                    title="Время выполнения (секунды)",
                    color="Время выполнения",
                    color_continuous_scale="RdYlBu_r"
                )
                st.plotly_chart(fig_time, use_container_width=True)
    
    def _display_extracted_texts(self, results: Dict[str, Any]):
        """Отображает извлеченные тексты."""
        st.subheader("Извлеченные тексты")
        
        for processor_name, result in results.items():
            with st.expander(f"Результат: {processor_name}"):
                
                extraction_results = result.get("extraction_results", {})
                conversion_results = result.get("conversion_results", {})
                
                # Показываем исходный извлеченный текст
                st.subheader("Исходный текст")
                for extractor_name, extraction_result in extraction_results.items():
                    if extraction_result.status.value == "completed":
                        text = extraction_result.content
                        st.text_area(
                            f"Извлечено с помощью {extractor_name}",
                            value=text[:1000] + "..." if len(text) > 1000 else text,
                            height=150,
                            key=f"extracted_{processor_name}_{extractor_name}"
                        )
                
                # Показываем финальный Markdown
                st.subheader("Финальный Markdown")
                for extractor in conversion_results:
                    for cleaner in conversion_results[extractor]:
                        for converter in conversion_results[extractor][cleaner]:
                            conv_result = conversion_results[extractor][cleaner][converter]
                            if conv_result.status.value == "completed":
                                markdown = conv_result.content
                                st.code(markdown[:1500] + "..." if len(markdown) > 1500 else markdown, language="markdown")
    
    def _display_quality_scores(self, results: Dict[str, Any]):
        """Отображает оценки качества."""
        st.subheader("Детальная оценка качества")
        
        for processor_name, result in results.items():
            quality_scores = result.get("quality_scores", {})
            
            if quality_scores:
                st.write(f"**{processor_name}**")
                
                # Собираем все оценки
                all_scores = []
                for extractor in quality_scores:
                    for cleaner in quality_scores[extractor]:
                        for converter in quality_scores[extractor][cleaner]:
                            score = quality_scores[extractor][cleaner][converter]
                            if hasattr(score, 'overall_score'):
                                all_scores.append({
                                    "Комбинация": f"{extractor}→{cleaner}→{converter}",
                                    "Общая оценка": score.overall_score,
                                    "Оценка": score.get_grade(),
                                    "Время": score.execution_time,
                                    **score.metric_scores
                                })
                
                if all_scores:
                    scores_df = pd.DataFrame(all_scores)
                    st.dataframe(scores_df, use_container_width=True)
    
    def _display_detailed_metrics(self, results: Dict[str, Any]):
        """Отображает детальные метрики."""
        st.subheader("Детальный анализ метрик")
        
        # Собираем все метрики для анализа
        all_metrics = []
        
        for processor_name, result in results.items():
            quality_scores = result.get("quality_scores", {})
            
            for extractor in quality_scores:
                for cleaner in quality_scores[extractor]:
                    for converter in quality_scores[extractor][cleaner]:
                        score = quality_scores[extractor][cleaner][converter]
                        if hasattr(score, 'metric_scores'):
                            metrics_row = {
                                "Процессор": processor_name,
                                "Комбинация": f"{extractor}→{cleaner}→{converter}",
                                **score.metric_scores,
                                "Общая оценка": score.overall_score
                            }
                            all_metrics.append(metrics_row)
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            
            # Radar chart для сравнения метрик
            if len(all_metrics) > 1:
                st.subheader("Радар-диаграмма метрик")
                
                # Выбираем лучшие результаты каждого процессора
                best_results = metrics_df.loc[metrics_df.groupby('Процессор')['Общая оценка'].idxmax()]
                
                metric_columns = [col for col in best_results.columns 
                                if col not in ['Процессор', 'Комбинация', 'Общая оценка']]
                
                fig = go.Figure()
                
                for _, row in best_results.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row[col] for col in metric_columns],
                        theta=metric_columns,
                        fill='toself',
                        name=row['Процессор']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Сравнение метрик качества"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _get_grade(self, score: float) -> str:
        """Возвращает буквенную оценку."""
        if score >= 0.9:
            return "A+ (Отлично)"
        elif score >= 0.8:
            return "A (Очень хорошо)" 
        elif score >= 0.7:
            return "B (Хорошо)"
        elif score >= 0.6:
            return "C (Удовлетворительно)"
        elif score >= 0.5:
            return "D (Плохо)"
        else:
            return "F (Неудовлетворительно)"
    
    def processor_comparison(self):
        """Страница сравнения процессоров."""
        st.header("Сравнение процессоров")
        st.info("Эта функция будет реализована в следующей версии")
    
    def batch_processing(self):
        """Страница пакетной обработки."""
        st.header("Пакетная обработка")
        st.info("Эта функция будет реализована в следующей версии")
    
    def settings_page(self):
        """Страница настроек."""
        st.header("Настройки системы")
        
        # Настройки логирования
        st.subheader("Логирование")
        log_level = st.selectbox(
            "Уровень логирования",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1
        )
        
        # Настройки производительности
        st.subheader("Производительность")
        max_workers = st.slider("Количество параллельных процессов", 1, 8, 4)
        timeout = st.slider("Таймаут обработки (секунды)", 30, 600, 300)
        
        # Настройки качества
        st.subheader("Метрики качества")
        for metric_name, config in QUALITY_METRICS.items():
            enabled = st.checkbox(f"Включить {metric_name}", value=config["enabled"])
            if enabled:
                weight = st.slider(f"Вес {metric_name}", 0.0, 1.0, config["weight"])
        
        if st.button("Применить настройки"):
            st.success("Настройки применены!")
    
    def analytics_page(self):
        """Страница аналитики."""
        st.header("Аналитика и статистика")
        
        # Информация о хранилище
        st.subheader("Информация о хранилище")
        storage_info = self.file_manager.get_storage_info()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Размер данных", f"{storage_info['total_size_mb']:.1f} МБ")
        
        with col2:
            st.metric("Файлов", storage_info['total_files'])
        
        with col3:
            st.metric("Директорий", storage_info['total_directories'])
        
        with col4:
            if st.button("Очистить старые результаты"):
                self.file_manager.cleanup_old_results(days_old=7)
                st.success("Старые результаты удалены")


def main():
    """Главная функция приложения."""
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
        logger.error(f"Streamlit app error: {e}")


if __name__ == "__main__":
    main()