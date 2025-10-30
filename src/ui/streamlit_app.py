"""
Streamlit веб-интерфейс для ALPACA Test Bench.
"""

import importlib.util
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    from pathlib import Path

    # Добавляем корневую директорию проекта в sys.path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from configs.processors_config import ALL_PROCESSORS, QUALITY_METRICS
    from src.core.pipeline import DocumentPipeline
    from src.processors.docx_processors import (Docx2txtExtractor,
                                                DocExtractor,
                                                PythonDocxExtractor,
                                                Win32WordExtractor)
    from src.processors.markdown_converters import (CustomMarkdownFormatter,
                                                    Html2TextConverter,
                                                    MarkdownifyConverter,
                                                    PandocConverter)
    from src.processors.pdf_processors import (PDFPlumberExtractor,
                                               PyMuPDFExtractor,
                                               PyPDFExtractor)
    from src.processors.text_cleaners import (AdvancedTextCleaner,
                                              BasicTextCleaner, HTMLCleaner)
    from src.utils.file_manager import FileManager
    from src.utils.llm_client import LLMConfig, build_llm_callable
    from src.utils.logger import get_logger, setup_logging

    # Опциональные импорты
    try:
        from src.processors.unstructured_processors import (
            UnstructuredLLMCleaner, UnstructuredPartitionExtractor)
        UNSTRUCTURED_AVAILABLE = True
    except ImportError:
        UNSTRUCTURED_AVAILABLE = False
    
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
        self.unstructured_available = UNSTRUCTURED_AVAILABLE
        self.setup_pipeline()
        self._ensure_llm_state()

    @staticmethod
    def _default_llm_state() -> Dict[str, Any]:
        return {
            "enabled": False,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "",
            "base_url": "",
            "temperature": 0.0,
            "system_prompt": "",
            "timeout": 60,
            "max_output_tokens": 512,
            "chunk_size": 2048,
        }

    def _ensure_llm_state(self) -> Dict[str, Any]:
        if "llm_settings" not in st.session_state:
            st.session_state["llm_settings"] = self._default_llm_state()
        return st.session_state["llm_settings"]
        
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
            st.warning(f"Docx процессоры недоступны: {e}")

        # Универсальный экстрактор для .doc (пробует antiword, catdoc, LibreOffice, MS Word)
        try:
            self.pipeline.register_extractor(['.doc'], DocExtractor())
        except Exception as e:
            st.warning(f"DOC экстрактор недоступен: {e}")

        try:
            win32_available = (
                platform.system().lower().startswith("win") and
                importlib.util.find_spec("win32com") is not None
            )
            if win32_available:
                self.pipeline.register_extractor(['.doc', '.docx'], Win32WordExtractor())
            else:
                logger.info("pywin32/MS Word COM извлечение недоступно: модуль win32com не найден")
        except Exception as e:
            st.warning(f"MS Word COM экстрактор недоступен: {e}")

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
                        "ocr_languages": 'rus+eng',
                    },
                    "fallback_partition_kwargs": {
                        "languages": language_hints,
                        "ocr_languages": 'rus+eng',
                    },
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
                            "languages": language_hints,
                            "fallback_partition_kwargs": {
                                "languages": language_hints,
                                "ocr_languages": 'rus+eng',
                            },
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
        self.pipeline.register_converter("markdownify", MarkdownifyConverter())
        self.pipeline.register_converter("html2text", Html2TextConverter())
        self.pipeline.register_converter("pandoc", PandocConverter())

    def _get_unstructured_cleaner(self) -> Optional[Any]:
        for cleaner in self.pipeline.cleaners:
            if getattr(cleaner, 'name', '') == "Unstructured LLM Cleaner":
                return cleaner
        return None

    def _configure_unstructured_llm_cleaner(
        self,
        llm_required: bool,
        llm_callable: Optional[Callable[[str], str]],
    ) -> None:
        from src.utils.logger import logger
        cleaner = self._get_unstructured_cleaner()
        if cleaner is None:
            logger.warning("UnstructuredLLMCleaner не найден в pipeline")
            return

        llm_state = self._ensure_llm_state()
        should_enable = bool(llm_required and llm_callable is not None)
        cleaner.config["use_llm_cleaning"] = should_enable
        cleaner.config["chunk_size"] = llm_state.get("chunk_size", cleaner.config.get("chunk_size", 2048))
        cleaner.llm_callable = llm_callable
        logger.info(f"Настроен UnstructuredLLMCleaner: use_llm_cleaning={should_enable}, llm_callable={'установлен' if llm_callable else 'отсутствует'}")

    def _resolve_llm_callable(self) -> Optional[Callable[[str], str]]:
        from src.utils.logger import logger
        llm_state = self._ensure_llm_state()
        if not llm_state.get("enabled"):
            logger.info("LLM выключен в настройках")
            return None

        api_key = (llm_state.get("api_key") or "").strip()
        model = (llm_state.get("model") or "").strip()
        if not api_key or not model:
            logger.error("Не указан API ключ или модель")
            raise ValueError("Укажите API ключ и модель для LLM.")

        provider = (llm_state.get("provider") or "openai").strip().lower()
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=(llm_state.get("base_url") or "").strip() or None,
            temperature=float(llm_state.get("temperature", 0.0)),
            system_prompt=llm_state.get("system_prompt") or None,
            timeout=int(llm_state.get("timeout", 60)),
            max_output_tokens=(llm_state.get("max_output_tokens") or None),
        )
        logger.info(f"Создан LLM callable для провайдера {provider}, модель {model}")
        callable_fn = build_llm_callable(config)
        return callable_fn

    def _render_llm_settings(self) -> None:
        if not self.unstructured_available:
            return

        llm_state = self._ensure_llm_state()
        st.subheader("LLM очистка (unstructured)")
        with st.form("llm_settings_form"):
            enabled = st.checkbox(
                "Включить LLM очистку",
                value=llm_state.get("enabled", False),
            )
            provider = st.selectbox(
                "Провайдер",
                ["OpenAI"],
                index=0,
            )
            model = st.text_input(
                "Модель",
                value=llm_state.get("model", "gpt-4o-mini"),
                help="Название модели, доступной в выбранном провайдере",
            )
            api_key = st.text_input(
                "API ключ",
                value=llm_state.get("api_key", ""),
                type="password",
                help="Ключ не сохраняется на диск и живет только в сессии Streamlit",
            )
            base_url = st.text_input(
                "Базовый URL",
                value=llm_state.get("base_url", ""),
                placeholder="https://api.openai.com/v1/chat/completions",
            )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(llm_state.get("temperature", 0.0)),
                step=0.05,
            )
            max_tokens = st.number_input(
                "Max output tokens (0 = по умолчанию)",
                min_value=0,
                max_value=8192,
                value=int(llm_state.get("max_output_tokens", 512) or 0),
            )
            chunk_size = st.number_input(
                "Размер чанка при очистке",
                min_value=256,
                max_value=4096,
                step=256,
                value=int(llm_state.get("chunk_size", 2048)),
            )
            timeout = st.number_input(
                "Таймаут запроса (сек)",
                min_value=10,
                max_value=180,
                value=int(llm_state.get("timeout", 60)),
            )
            system_prompt = st.text_area(
                "Системный промпт",
                value=llm_state.get("system_prompt", ""),
                height=150,
            )

            submitted = st.form_submit_button("Сохранить LLM настройки")

        if submitted:
            st.session_state["llm_settings"] = {
                "enabled": enabled,
                "provider": provider.lower(),
                "model": model.strip(),
                "api_key": api_key.strip(),
                "base_url": base_url.strip(),
                "temperature": float(temperature),
                "system_prompt": system_prompt.strip(),
                "timeout": int(timeout),
                "max_output_tokens": int(max_tokens) if max_tokens else None,
                "chunk_size": int(chunk_size),
            }
            st.success("Настройки LLM сохранены")

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
                    ["custom", "markdownify", "html2text", "pandoc"],
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

        llm_required = (
            self.unstructured_available and "Unstructured LLM Cleaner" in cleaners
        )
        llm_callable: Optional[Callable[[str], str]] = None

        if llm_required:
            try:
                llm_callable = self._resolve_llm_callable()
            except Exception as exc:
                st.warning(f"LLM очистка не активирована: {exc}")
                llm_callable = None

        self._configure_unstructured_llm_cleaner(llm_required, llm_callable)

        if llm_required and llm_callable is None:
            st.info("Unstructured LLM Cleaner будет работать без LLM — выполняется только базовая очистка.")
        
        for i, processor in enumerate(processors):
            status_text.text(f"Обработка с помощью {processor}...")
            progress_bar.progress((i + 1) / len(processors))
            
            try:
                # Обработка файла
                result = self.pipeline.process_document(
                    file_path=file_path,
                    extractor_name=processor,
                    cleaner_names=cleaners,
                    converter_name=converters[0] if converters else None,
                    llm_callable=llm_callable,
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
        
        self._render_llm_settings()

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