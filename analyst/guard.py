"""Защита от prompt-injection и опасного кода.

Слои защиты (подробнее в README):
1. Проверка инструкции пользователя на типовые injection-фразы.
2. Сканирование содержимого датасета: подозрительные ячейки показываются пользователю
   и явно помечаются для модели как данные, а не команды.
3. Всё недоверенное (вывод кода, данные) оборачивается в теги, которые нельзя «закрыть»
   изнутри, а системный промпт запрещает выполнять инструкции из этих блоков.
4. Статическая проверка кода агента (AST) до запуска + изоляция в песочнице (sandbox.py).
5. Проверка итогового отчёта: канареечный токен (утечка системного промпта),
   удаление внешних ссылок и картинок (канал утечки данных через URL).
"""

from __future__ import annotations

import ast
import re
import secrets
from dataclasses import dataclass

import pandas as pd

from .config import ALLOWED_IMPORTS

# ---------------------------------------------------------------------------
# 1–2. Поиск injection-фраз в тексте
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    # английский
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?|messages?)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)?\s*(instructions?|rules?|prompts?)",
    r"forget\s+(all\s+|everything|your)\s*(instructions?|rules?|above)?",
    r"(reveal|print|show|output|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
    r"system\s*prompt",
    r"you\s+are\s+now\b",
    r"\bact\s+as\s+(an?\s+)?(different|new|unrestricted)",
    r"developer\s+mode",
    r"\bjailbreak",
    r"new\s+instructions?\s*:",
    r"</?\s*(system|assistant|instructions?|tool_output|user_instructions)\s*>",
    r"\[/?INST\]",
    r"<\|im_(start|end)\|>",
    # русский
    r"игнорир\w*\s+(все\s+|всех\s+)?(предыдущ|прошл|вышеуказ|систем|исходн)\w*\s+(инструкц|указан|правил|промпт)\w*",
    r"не\s+(обращай|следуй)\w*\s+(внимани\w*\s+)?(на\s+)?(предыдущ|систем|прошл)\w*\s+(инструкц|указан|правил)\w*",
    r"забудь\s+(все\s+|всё\s+)?(предыдущ\w*\s+)?(инструкц|правил|указан)\w*",
    r"систем\w*\s+промпт",
    r"(покажи|выведи|раскрой|повтори)\s+(свой\s+|свои\s+|системн\w*\s+)*(промпт|инструкц)\w*",
    r"теперь\s+ты\s+",
    r"ты\s+теперь\s+",
    r"новые\s+инструкции\s*:",
    r"режим\s+разработчика",
    # попытки добраться до секретов
    r"api[\s_-]*key",
    r"anthropic_api_key",
    r"os\.environ",
    r"api[\s_-]*ключ",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


@dataclass
class Finding:
    where: str
    snippet: str

    def as_text(self) -> str:
        return f"{self.where}: «{self.snippet}»"


def find_injections(text: str) -> list[str]:
    """Возвращает найденные подозрительные фрагменты (пустой список — всё чисто)."""
    if not text:
        return []
    return [m.group(0) for m in _INJECTION_RE.finditer(text)]


def check_instruction(text: str, max_chars: int) -> tuple[str, list[str]]:
    """Нормализует инструкцию пользователя и возвращает (очищенный текст, найденные проблемы)."""
    cleaned = _strip_control_chars(text or "").strip()
    problems: list[str] = []
    if len(cleaned) > max_chars:
        problems.append(f"инструкция длиннее {max_chars} символов")
        cleaned = cleaned[:max_chars]
    problems += [f"подозрительная фраза: «{s}»" for s in find_injections(cleaned)]
    return cleaned, problems


def scan_dataframe(df: pd.DataFrame, max_values_per_col: int = 2000, max_findings: int = 20) -> list[Finding]:
    """Ищет injection-фразы в названиях столбцов и текстовых ячейках."""
    findings: list[Finding] = []
    for col in df.columns:
        for hit in find_injections(str(col)):
            findings.append(Finding(f"название столбца {str(col)[:60]!r}", hit))
    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        values = series.dropna().astype(str)
        values = values[values.str.len() > 15].drop_duplicates().head(max_values_per_col)
        for idx, value in values.items():
            hits = find_injections(value)
            if hits:
                findings.append(Finding(f"столбец {str(col)[:60]!r}, строка {idx}", value[:120]))
            if len(findings) >= max_findings:
                return findings
    return findings


# ---------------------------------------------------------------------------
# 3. Обёртка недоверенного текста
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_RE = re.compile(r"<\s*/?\s*(untrusted_data|tool_output|user_instructions|system)[^>]*>", re.IGNORECASE)


def _strip_control_chars(text: str) -> str:
    return _CONTROL_RE.sub("", text)


def neutralize(text: str) -> str:
    """Не даёт содержимому закрыть наш тег и выдать себя за системный текст."""
    text = _strip_control_chars(text)
    return _TAG_RE.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), text)


def wrap_untrusted(text: str, tag: str = "untrusted_data") -> str:
    return f"<{tag}>\n{neutralize(text)}\n</{tag}>"


# ---------------------------------------------------------------------------
# 4. Статическая проверка кода агента
# ---------------------------------------------------------------------------

FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "help",
        "memoryview",
        "__builtins__",
        "__loader__",
        "__spec__",
    }
)

# Методы, которые читают/пишут произвольные источники или исполняют код.
FORBIDDEN_ATTRS = frozenset(
    {
        "system",
        "popen",
        "read_pickle",
        "to_pickle",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "to_sql",
        "read_html",
        "read_xml",
        "read_clipboard",
        "to_clipboard",
        "environ",
        "getenv",
        "putenv",
        "load_module",
        "exec_module",
    }
)


class CodeRejected(ValueError):
    pass


def check_code(code: str) -> None:
    """Бросает CodeRejected, если код использует запрещённые конструкции."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeRejected(f"Синтаксическая ошибка: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise CodeRejected("Относительные импорты запрещены")
            _check_module(node.module or "")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise CodeRejected(f"Запрещённое имя: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                raise CodeRejected(f"Запрещённый атрибут: .{node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            # Строки с dunder-именами нужны только для обхода проверок через reflection.
            if re.search(r"__\w+__", node.value):
                raise CodeRejected("Строки с dunder-именами запрещены")


def _check_module(name: str) -> None:
    top = name.split(".")[0]
    if top not in ALLOWED_IMPORTS:
        raise CodeRejected(
            f"Импорт модуля '{name}' запрещён. Разрешены: {', '.join(sorted(ALLOWED_IMPORTS))}"
        )


# ---------------------------------------------------------------------------
# 5. Проверка итогового отчёта
# ---------------------------------------------------------------------------


def new_canary() -> str:
    return "CANARY-" + secrets.token_hex(8)


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?:|ftp:|data:|javascript:)[^)]*\)", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_BARE_URL_RE = re.compile(r"(?:https?|ftp)://\S+", re.IGNORECASE)


def sanitize_report_text(text: str, canary: str) -> str:
    """Удаляет из текста отчёта картинки/ссылки на внешние ресурсы и HTML; ловит утечку промпта."""
    if not isinstance(text, str):
        text = str(text)
    if canary and canary in text:
        return "[фрагмент удалён: попытка вывести служебные инструкции]"
    text = _MD_IMAGE_RE.sub("[изображение удалено]", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub("[ссылка удалена]", text)
    text = _HTML_TAG_RE.sub("", text)
    return text


def sanitize_report(obj, canary: str):
    """Рекурсивно чистит все строки в структуре отчёта."""
    if isinstance(obj, str):
        return sanitize_report_text(obj, canary)
    if isinstance(obj, list):
        return [sanitize_report(x, canary) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_report(v, canary) for k, v in obj.items()}
    return obj
