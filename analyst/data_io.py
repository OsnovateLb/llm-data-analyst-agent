"""Чтение загруженного файла (CSV/Excel) с понятными ошибками."""

from __future__ import annotations

import io

import pandas as pd

from .config import SETTINGS


class DataLoadError(ValueError):
    pass


def load_table(raw: bytes, file_name: str) -> pd.DataFrame:
    if len(raw) > SETTINGS.max_upload_mb * 1024 * 1024:
        raise DataLoadError(f"Файл больше {SETTINGS.max_upload_mb} МБ")
    name = file_name.lower()
    try:
        if name.endswith((".xlsx", ".xlsm", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        elif name.endswith((".csv", ".tsv", ".txt")):
            df = _read_csv(raw)
        else:
            raise DataLoadError("Поддерживаются файлы CSV, TSV и Excel (.xlsx)")
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Не удалось прочитать файл: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise DataLoadError("Файл пустой или не содержит столбцов")
    if len(df) > SETTINGS.max_rows:
        df = df.head(SETTINGS.max_rows)
    # Названия столбцов приводим к строкам — так с ними проще работать в коде агента.
    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns)]
    return df


def _read_csv(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        # sep=None + engine=python: pandas сам определяет разделитель (, ; \t |)
        lines = text.splitlines()
        while lines and lines[0].startswith("#"):  # комментарии-шапка (например, «# Data source: ...»)
            lines.pop(0)
        return pd.read_csv(io.StringIO("\n".join(lines)), sep=None, engine="python")
    raise DataLoadError(f"Не удалось определить кодировку файла: {last_error}")
