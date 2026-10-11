"""Клиент OpenAI-совместимого Chat Completions API с поддержкой tool calling.

По умолчанию — OpenRouter (https://openrouter.ai) с бесплатными моделями: ключ бесплатный,
банковская карта не нужна. Работает на стандартной библиотеке, без SDK.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
USER_AGENT = "llm-data-analyst-agent/1.0"
NETWORK_RETRIES = 5

# Бесплатные модели OpenRouter периодически меняются. Если модель не выбрана явно, агент берёт
# доступные бесплатные модели с поддержкой tools в этом порядке и при сбое переключается на следующую.
PREFERRED_FREE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "thinkingmachines/inkling:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "nvidia/nemotron-3.5-lightning:free",
    "thinkingmachines/inkling-small:free",
    "google/gemma-4-26b-a4b-it:free",
    "openrouter/free",  # роутер OpenRouter: сам выбирает доступную бесплатную модель
]
NON_CHAT_HINTS = ("lyria", "content-safety", "guard", "embed", "rerank", "tts", "whisper")
DEPRIORITIZED_HINTS = ("code", "coder", "laguna", "nano", "2.6b", "mini", "mercury", "dots")


class LLMError(RuntimeError):
    """Ошибка API. status_code — HTTP-код, если есть."""

    status_code: int | None = None


class FatalLLMError(LLMError):
    """Ошибка, при которой смена модели не поможет: неверный ключ, дневной лимит, нет сети."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON-строка, как её вернула модель


@dataclass
class ChatReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)

    def as_message(self) -> dict[str, Any]:
        """Сообщение ассистента для истории диалога в формате Chat Completions."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


def urlopen_json(req, timeout: int) -> Any:
    """urlopen с повтором при сетевых сбоях (обрыв TLS, таймаут, сброс соединения)."""
    for attempt in range(NETWORK_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as exc:
            if attempt == NETWORK_RETRIES - 1:
                raise FatalLLMError(f"Сетевая ошибка после {NETWORK_RETRIES} попыток: {exc}") from exc
            wait = 2 ** (attempt + 1)
            log.warning("сетевой сбой (%s), повтор через %s c", getattr(exc, "reason", exc), wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = "", timeout: int = 180):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._models_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
                "HTTP-Referer": "https://github.com/OsnovateLb/llm-data-analyst-agent",
                "X-Title": "llm-data-analyst-agent",
            },
        )
        return urlopen_json(req, timeout=self.timeout)

    # ---------------------------------------------------------------- модели
    def list_free_models(self) -> list[dict]:
        models = self._request("GET", "/models").get("data", [])

        def is_free(m: dict) -> bool:
            p = m.get("pricing") or {}
            return m.get("id", "").endswith(":free") or (
                str(p.get("prompt")) in ("0", "0.0") and str(p.get("completion")) in ("0", "0.0")
            )

        def is_text_chat(m: dict) -> bool:
            outputs = (m.get("architecture") or {}).get("output_modalities") or ["text"]
            return "text" in outputs and not any(h in m.get("id", "").lower() for h in NON_CHAT_HINTS)

        free = [m for m in models if is_free(m) and is_text_chat(m)]
        self._models_meta = {m["id"]: m for m in free}
        return free

    def free_model_candidates(self) -> list[str]:
        """Бесплатные модели с поддержкой tools: сначала из списка предпочтений, затем остальные."""
        try:
            free = self.list_free_models()
        except LLMError as exc:
            log.warning("не удалось получить список моделей (%s); использую список по умолчанию", exc)
            return list(PREFERRED_FREE_MODELS)

        def has_tools(m: dict) -> bool:
            params = m.get("supported_parameters")
            return params is None or "tools" in params

        usable = [m for m in free if has_tools(m)]
        ids = {m["id"] for m in usable}
        preferred = [mid for mid in PREFERRED_FREE_MODELS if mid in ids]
        rest = sorted(
            (m for m in usable if m["id"] not in preferred),
            key=lambda m: (any(h in m["id"].lower() for h in DEPRIORITIZED_HINTS), -(m.get("context_length") or 0)),
        )
        candidates = preferred + [m["id"] for m in rest]
        return candidates or list(PREFERRED_FREE_MODELS)

    def supports_images(self, model: str | None = None) -> bool:
        meta = self._models_meta.get(model or self.model) or {}
        return "image" in ((meta.get("architecture") or {}).get("input_modalities") or [])

    # ------------------------------------------------------------------ чат
    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 8000) -> ChatReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        payload: dict = {}
        for attempt in range(5):
            try:
                payload = self._request("POST", "/chat/completions", body)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                if exc.code in (401, 403):
                    raise FatalLLMError(f"Ошибка авторизации ({exc.code}): проверьте API-ключ. {detail}") from exc
                if exc.code == 429 and ("per-day" in detail or "per day" in detail.lower()):
                    raise FatalLLMError(f"Исчерпан дневной лимит бесплатных запросов OpenRouter. {detail}") from exc
                if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                    wait = int(exc.headers.get("Retry-After") or 0) or 5 * 2**attempt
                    log.warning("API ответил %s, повтор через %s c", exc.code, wait)
                    time.sleep(wait)
                    continue
                err = LLMError(f"HTTP {exc.code}: {detail}")
                err.status_code = exc.code
                raise err from exc
        if payload.get("error"):
            err = LLMError(f"Ошибка провайдера: {payload['error']}")
            err.status_code = (payload["error"] or {}).get("code") if isinstance(payload["error"], dict) else None
            raise err

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if not content.strip() and not message.get("tool_calls"):
            content = message.get("reasoning") or ""  # «думающие» модели иногда отвечают в reasoning
        calls = []
        for i, c in enumerate(message.get("tool_calls") or []):
            fn = c.get("function") or {}
            args = fn.get("arguments")
            calls.append(
                ToolCall(
                    id=c.get("id") or f"call_{i}",
                    name=fn.get("name") or "",
                    arguments=args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False),
                )
            )
        return ChatReply(content=content, tool_calls=calls, finish_reason=choice.get("finish_reason"), usage=payload.get("usage") or {})
