"""Вспомогательные утилиты для подключения LLM в unstructured."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "Ты выступаешь в роли редактора. Очисти и нормализуй полученный фрагмент документа, "
    "сохрани деловую лексику и структуру. Используй Markdown и не удаляй фактические данные."
)


@dataclass
class LLMConfig:
    """Конфигурация LLM подключения."""

    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.0
    system_prompt: Optional[str] = None
    timeout: int = 60
    max_output_tokens: Optional[int] = None


def build_llm_callable(config: LLMConfig) -> Callable[[str], str]:
    """Создает функцию вызова LLM согласно конфигурации."""

    provider = (config.provider or "").strip().lower()
    if provider == "openai":
        return _build_openai_callable(config)

    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _build_openai_callable(config: LLMConfig) -> Callable[[str], str]:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - маловероятно
        raise ImportError(
            "requests is required for OpenAI LLM integration. Install via pip install requests"
        ) from exc

    # Обработка base_url: добавляем /chat/completions если отсутствует
    base_url = (config.base_url or "https://api.openai.com/v1").strip()
    if base_url.endswith("/v1") or base_url.endswith("/v1/"):
        base_url = base_url.rstrip("/") + "/chat/completions"
    elif not base_url.endswith("/chat/completions"):
        base_url = base_url.rstrip("/") + "/chat/completions"
    
    url = base_url
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    system_prompt = (config.system_prompt or _DEFAULT_SYSTEM_PROMPT).strip()

    def _invoke(prompt: str, **kwargs) -> str:
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(max(0.0, min(config.temperature, 1.0))),
        }

        max_tokens = kwargs.get("max_tokens") or config.max_output_tokens
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=config.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - сетевые ошибки
            logger.error("OpenAI request failed: %s", exc)
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        data = response.json()
        try:
            choice = data["choices"][0]["message"]["content"]
            return choice.strip()
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected OpenAI response format: %s", data)
            raise RuntimeError("Unexpected OpenAI response format") from exc

    return _invoke
