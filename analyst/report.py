"""Экспорт отчёта агента в Markdown."""

from __future__ import annotations

from typing import Any


def report_to_markdown(report: dict[str, Any], file_name: str) -> str:
    out = [f"# {report.get('title', 'Отчёт')}", "", f"_Датасет: {file_name}_", "", report.get("summary", ""), ""]

    metrics = report.get("key_metrics") or []
    if metrics:
        out += ["## Ключевые метрики", "", "| Метрика | Значение | Комментарий |", "|---|---|---|"]
        for m in metrics:
            row = [str(m.get(k, "")).replace("|", "/").replace("\n", " ") for k in ("name", "value", "comment")]
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    insights = report.get("insights") or []
    if insights:
        out += ["## Инсайты", ""]
        for i, ins in enumerate(insights, 1):
            out += [f"**{i}. {ins.get('title', '')}**", "", ins.get("detail", ""), "", f"_Подтверждение: {ins.get('evidence', '')}_", ""]

    figures = report.get("figures") or []
    if figures:
        out += ["## Графики", ""]
        out += [f"- {f.get('id')}: {f.get('caption', '')}" for f in figures]
        out.append("")

    for key, title in (("data_quality", "Качество данных"), ("recommendations", "Рекомендации"), ("limitations", "Ограничения")):
        items = report.get(key) or []
        if items:
            out += [f"## {title}", ""] + [f"- {x}" for x in items] + [""]
    return "\n".join(out)
