"""
Процессоры, интегрированные с библиотекой unstructured.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..core.base import (
    BaseCleaner,
    BaseExtractor,
    ProcessingResult,
    ProcessingStatus,
)
from ..utils.logger import get_logger, log_processing_stage

logger = get_logger(__name__)


class UnstructuredPartitionExtractor(BaseExtractor):
    """Экстрактор, использующий unstructured.partition_* функции."""

    DEFAULT_SUPPORTED_TYPES = (
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        ".md",
        ".html",
        ".txt",
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__("Unstructured Partition", config)
        supported = config.get("supported_types", self.DEFAULT_SUPPORTED_TYPES)
        self.supported_types = tuple(sorted(set(ft.lower() for ft in supported)))
        self.drop_types = set(config.get("drop_types", ["Header", "Footer", "PageBreak"]))
        self.force_metadata = config.get("include_metadata", True)
        self.table_as_html = config.get("table_as_html", True)

    def supports_file_type(self, file_type: str) -> bool:
        return file_type.lower() in self.supported_types

    @log_processing_stage("Unstructured Partition Extraction")
    def extract_text(self, file_path: Path, **kwargs) -> str:
        try:
            elements, raw_elements = self._partition_file(file_path, **kwargs)
        except ImportError as exc:
            raise ImportError(
                "unstructured is not installed. Install with: pip install "
                "\"unstructured[all-docs]\""
            ) from exc

        serialized = [self._serialize_element(element) for element in raw_elements]
        relevant = [
            element
            for element in serialized
            if element["type"] not in self.drop_types
        ]

        markdown = self._elements_to_markdown(relevant)

        self.last_elements = relevant

        # Метаданные нужны, чтобы передавать элементы на этап очистки и оценки
        self.last_metadata = {
            "element_types": [element["type"] for element in relevant],
            "element_count": len(relevant),
            "elements": relevant,
        }

        return markdown

    def process(self, file_path: Path, **kwargs) -> ProcessingResult:
        result = super().process(file_path, **kwargs)
        if result.status == ProcessingStatus.COMPLETED:
            metadata = dict(result.metadata)
            metadata.update(getattr(self, "last_metadata", {}))
            result.metadata = metadata
        return result

    def _partition_file(self, file_path: Path, **kwargs):
        file_type = file_path.suffix.lower()
        partition_kwargs = {
            "include_page_breaks": False,
            "hi_res_model_name": self.config.get("hi_res_model_name"),
            "strategy": self.config.get("strategy"),
            "chunking_strategy": self.config.get("chunking_strategy", "by_title"),
            "infer_table_structure": self.config.get("infer_table_structure", True),
        }
        partition_kwargs.update(self.config.get("partition_kwargs", {}))
        partition_kwargs.update(kwargs.get("partition_kwargs", {}))

        try:
            if file_type == ".pdf":
                module = "unstructured.partition.pdf"
                partition_pdf = self._load_callable(module, "partition_pdf")
                elements = partition_pdf(filename=str(file_path), **partition_kwargs)
            elif file_type in (".doc", ".docx"):
                try:
                    module = "unstructured.partition.doc"
                    partition_doc = self._load_callable(module, "partition_doc")
                    elements = partition_doc(filename=str(file_path), **partition_kwargs)
                except ModuleNotFoundError:
                    partition_docx = self._load_callable(
                        "unstructured.partition.docx",
                        "partition_docx",
                    )
                    elements = partition_docx(filename=str(file_path), **partition_kwargs)
            elif file_type in (".ppt", ".pptx"):
                try:
                    module = "unstructured.partition.ppt"
                    partition_ppt = self._load_callable(module, "partition_ppt")
                    elements = partition_ppt(filename=str(file_path), **partition_kwargs)
                except ModuleNotFoundError:
                    partition_pptx = self._load_callable(
                        "unstructured.partition.pptx",
                        "partition_pptx",
                    )
                    elements = partition_pptx(filename=str(file_path), **partition_kwargs)
            elif file_type in (".xlsx", ".xls"):
                module = "unstructured.partition.xlsx"
                partition_xlsx = self._load_callable(module, "partition_xlsx")
                elements = partition_xlsx(filename=str(file_path), **partition_kwargs)
            elif file_type == ".md":
                module = "unstructured.partition.md"
                partition_md = self._load_callable(module, "partition_md")
                elements = partition_md(filename=str(file_path), **partition_kwargs)
            elif file_type == ".html":
                module = "unstructured.partition.html"
                partition_html = self._load_callable(module, "partition_html")
                elements = partition_html(filename=str(file_path), **partition_kwargs)
            else:
                module = "unstructured.partition.auto"
                partition_auto = self._load_callable(module, "partition")
                elements = partition_auto(filename=str(file_path), **partition_kwargs)
        except ImportError:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("unstructured partition failed for %s: %s", file_path, exc)
            raise

        return self._filter_elements(elements), elements

    def _filter_elements(self, elements: Iterable[Any]) -> List[Any]:
        filtered = []
        for element in elements:
            element_type = getattr(element, "category", getattr(element, "type", "Text"))
            if element_type in self.drop_types:
                continue
            filtered.append(element)
        return filtered

    @staticmethod
    def _load_callable(module: str, attribute: str) -> Callable[..., Any]:
        return getattr(import_module(module), attribute)

    def _serialize_element(self, element: Any) -> Dict[str, Any]:
        element_type = getattr(element, "category", getattr(element, "type", "Text"))
        metadata = getattr(element, "metadata", None)
        if metadata is not None and hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        elif isinstance(metadata, dict):
            metadata = metadata
        else:
            metadata = {}

        base = {
            "type": element_type,
            "text": getattr(element, "text", "") or "",
            "metadata": metadata,
        }

        if self.force_metadata and hasattr(element, "to_dict"):
            try:
                serialized = element.to_dict()
                base["metadata"].update(serialized.get("metadata", {}))
                base.setdefault("id", serialized.get("id"))
            except Exception:  # pylint: disable=broad-except
                pass

        return base

    def _elements_to_markdown(self, elements: Iterable[Dict[str, Any]]) -> str:
        output: List[str] = []
        for element in elements:
            text = (element.get("text") or "").strip()
            if not text:
                continue

            element_type = element.get("type", "NarrativeText")

            if element_type == "Title":
                level = element.get("metadata", {}).get("category_depth") or 1
                level = max(1, min(6, int(level)))
                output.append(f"{'#' * level} {text}")
            elif element_type == "ListItem":
                output.append(f"- {text}")
            elif element_type == "Table":
                table_md = self._table_to_markdown(element)
                output.append(table_md)
            else:
                output.append(text)

        return "\n\n".join(output)

    def _table_to_markdown(self, element: Dict[str, Any]) -> str:
        if not self.table_as_html:
            return element.get("text", "")

        try:
            table_html = element.get("metadata", {}).get("text_as_html")
            if not table_html:
                return element.get("text", "")
            from markdownify import markdownify as html_to_md  # type: ignore

            return html_to_md(table_html, heading_style="ATX")
        except Exception:  # pylint: disable=broad-except
            return element.get("text", "")


class UnstructuredLLMCleaner(BaseCleaner):
    """Очистка текста с помощью unstructured.cleaners и опциональной LLM нормализации."""

    DEFAULT_CLEANERS = [
        "clean_extra_whitespace",
        "clean_multiple_newlines",
        "clean_non_ascii_chars",
        "clean_bullets",
    ]

    MARKDOWN_TYPE_MAP = {
        "Title": "#",
        "ListItem": "-",
        "NarrativeText": "",
        "Address": "",
        "Table": None,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        defaults = {
            "cleaner_functions": self.DEFAULT_CLEANERS,
            "drop_types": ["Footer", "Header", "PageBreak"],
            "use_llm_cleaning": False,
            "llm_callable": None,
            "chunk_size": 2048,
            "repartition_if_missing": True,
            "fallback_partition_kwargs": {},
        }
        cfg = defaults.copy()
        if config:
            cfg.update(config)
        super().__init__("Unstructured LLM Cleaner", cfg)
        self.llm_callable: Optional[Callable[..., str]] = cfg.get("llm_callable")
        self.drop_types = set(cfg.get("drop_types", []))

    def clean_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_path: Optional[Path] = None,
        llm_callable: Optional[Callable[..., str]] = None,
        **kwargs,
    ) -> Any:
        try:
            cleaners_core = import_module("unstructured.cleaners.core")
        except ImportError as exc:
            raise ImportError(
                "unstructured is not installed. Install with: pip install "
                "\"unstructured[all-docs]\""
            ) from exc

        elements = self._resolve_elements(text, metadata, file_path, kwargs)
        cleaner_functions = self._resolve_cleaner_functions(cleaners_core)

        cleaned_elements: List[Dict[str, Any]] = []
        applied_functions = [func.__name__ for func in cleaner_functions]
        llm = llm_callable or self.llm_callable or kwargs.get("llm_callable")
        llm_used = False

        for element in elements:
            element_type = element.get("type", "NarrativeText")
            if element_type in self.drop_types:
                continue

            cleaned_text = element.get("text", "") or ""
            for func in cleaner_functions:
                cleaned_text = func(cleaned_text)

            if self.config.get("use_llm_cleaning") and llm:
                llm_response = self._call_llm(
                    llm,
                    cleaned_text,
                    element_type=element_type,
                    metadata=element.get("metadata", {}),
                    context=self._build_context(cleaned_elements),
                )
                if llm_response:
                    cleaned_text = llm_response.strip()
                    llm_used = True

            cleaned_element = {
                "type": element_type,
                "text": cleaned_text.strip(),
                "metadata": element.get("metadata", {}),
            }
            cleaned_elements.append(cleaned_element)

        markdown = self._elements_to_markdown(cleaned_elements)
        extra_metadata = {
            "cleaned_elements": cleaned_elements,
            "applied_cleaners": applied_functions,
            "llm_cleaning_applied": llm_used,
        }
        return markdown, extra_metadata

    def _resolve_elements(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
        file_path: Optional[Path],
        kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if metadata and metadata.get("elements"):
            return metadata["elements"]

        if self.config.get("repartition_if_missing") and file_path:
            try:
                module = "unstructured.partition.auto"
                partition_auto = self._load_callable(module, "partition")

                partition_kwargs = self.config.get("fallback_partition_kwargs", {})
                partition_kwargs.update(kwargs.get("partition_kwargs", {}))
                elements = partition_auto(filename=str(file_path), **partition_kwargs)
                return [self._serialize_element(el) for el in elements]
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Fallback partition failed for %s: %s", file_path, exc)

        return [
            {
                "type": "NarrativeText",
                "text": text,
                "metadata": metadata or {},
            }
        ]

    def _resolve_cleaner_functions(self, cleaners_core) -> List[Callable[[str], str]]:
        functions: List[Callable[[str], str]] = []
        for name in self.config.get("cleaner_functions", []):
            func = getattr(cleaners_core, name, None)
            if callable(func):
                functions.append(func)
            else:
                logger.warning(
                    "Cleaner function %s not found in unstructured.cleaners.core",
                    name,
                )
        if not functions:
            functions.append(cleaners_core.clean_extra_whitespace)
        return functions

    def _call_llm(
        self,
        llm_callable: Callable[..., Any],
        text: str,
        **kwargs,
    ) -> Optional[str]:
        try:
            response = llm_callable(text, **kwargs)
        except TypeError:
            prompt = self._build_prompt(text, **kwargs)
            response = llm_callable(prompt)
        if isinstance(response, dict):
            return response.get("text") or response.get("content")
        if isinstance(response, str):
            return response
        return None

    def _build_prompt(
        self,
        text: str,
        element_type: str,
        metadata: Dict[str, Any],
        context: str,
    ) -> str:
        return (
            "Выступи в роли редактора. Очисти и нормализуй элемент документа.\n"
            f"Тип элемента: {element_type}.\n"
            f"Контекст: {context}.\n"
            "Сохраняй смысл и структуру, не удаляй данные. Верни результат в Markdown.\n"
            f"Текст: {text}\n"
            f"Метаданные: {metadata}\n"
        )

    def _build_context(
        self,
        elements: List[Dict[str, Any]],
        max_tokens: int = 600,
    ) -> str:
        if not elements:
            return ""
        context_parts: List[str] = []
        total_length = 0
        for element in reversed(elements):
            snippet = element.get("text", "")
            if not snippet:
                continue
            total_length += len(snippet)
            if total_length > max_tokens:
                break
            context_parts.append(snippet)
        return "\n\n".join(reversed(context_parts))

    def _elements_to_markdown(self, elements: Iterable[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for element in elements:
            text = (element.get("text") or "").strip()
            if not text:
                continue
            element_type = element.get("type", "NarrativeText")
            prefix = self.MARKDOWN_TYPE_MAP.get(element_type, "")
            if prefix is None:
                table_md = element.get("metadata", {}).get("text_as_html")
                if table_md:
                    try:
                        from markdownify import markdownify as html_to_md  # type: ignore

                        lines.append(html_to_md(table_md, heading_style="ATX"))
                        continue
                    except Exception:  # pylint: disable=broad-except
                        lines.append(text)
                        continue
                lines.append(text)
                continue
            if prefix:
                lines.append(f"{prefix} {text}")
            else:
                lines.append(text)
        return "\n\n".join(lines)

    def _serialize_element(self, element: Any) -> Dict[str, Any]:
        element_type = getattr(
            element,
            "category",
            getattr(element, "type", "NarrativeText"),
        )
        metadata = getattr(element, "metadata", {})
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        elif not isinstance(metadata, dict):
            metadata = {}
        return {
            "type": element_type,
            "text": getattr(element, "text", ""),
            "metadata": metadata,
        }
