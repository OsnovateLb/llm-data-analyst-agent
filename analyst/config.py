"""Настройки приложения. Всё, что можно менять без правки кода, читается из окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # OpenAI-совместимый API (по умолчанию OpenRouter) и модель. Пустая модель — автовыбор бесплатной.
    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", ""))
    max_tokens: int = field(default_factory=lambda: _int_env("AGENT_MAX_TOKENS", 8000))
    # Сколько раз агент может обратиться к модели за один анализ.
    max_steps: int = field(default_factory=lambda: _int_env("AGENT_MAX_STEPS", 14))

    # Ограничения песочницы.
    exec_timeout_s: int = field(default_factory=lambda: _int_env("SANDBOX_TIMEOUT_S", 60))
    exec_memory_mb: int = field(default_factory=lambda: _int_env("SANDBOX_MEMORY_MB", 2048))
    max_stdout_chars: int = 6000
    max_figures_per_call: int = 3

    # Ограничения на входные данные.
    max_upload_mb: int = field(default_factory=lambda: _int_env("MAX_UPLOAD_MB", 20))
    max_rows: int = field(default_factory=lambda: _int_env("MAX_ROWS", 200_000))
    max_instruction_chars: int = 2000


SETTINGS = Settings()

# Модули, которые разрешено импортировать коду агента.
ALLOWED_IMPORTS = frozenset(
    {
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scipy",
        "math",
        "statistics",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "re",
        "json",
        "textwrap",
        "string",
        "warnings",
        "decimal",
        "fractions",
        "calendar",
    }
)
