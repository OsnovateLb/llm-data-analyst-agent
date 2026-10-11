"""Streamlit-интерфейс: загрузка датасета → LLM-агент анализирует его кодом → отчёт.

Запуск:  streamlit run app.py
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

from analyst.agent import AgentResult, DataAnalystAgent, Step
from analyst.config import SETTINGS
from analyst.data_io import DataLoadError, load_table
from analyst.guard import check_instruction, scan_dataframe
from analyst.llm import FatalLLMError, LLMError, OpenAICompatibleClient
from analyst.report import report_to_markdown
from analyst.sandbox import Sandbox

MAX_RUNS_PER_SESSION = 5
AUTO = "Авто: лучшая доступная бесплатная модель"

st.set_page_config(page_title="LLM-аналитик данных", page_icon="📊", layout="wide")


def secret(name: str) -> str | None:
    """Секрет из .streamlit/secrets.toml (Streamlit Cloud) или переменной окружения."""
    try:
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 — файла секретов может не быть
        value = None
    return value or os.environ.get(name)


# ---------------------------------------------------------------- доступ к приложению
app_password = secret("APP_PASSWORD")
if app_password and not st.session_state.get("authed"):
    st.title("📊 LLM-аналитик данных")
    pwd = st.text_input("Пароль доступа", type="password")
    if pwd:
        if hmac.compare_digest(pwd, app_password):
            st.session_state.authed = True
            st.rerun()
        st.error("Неверный пароль")
    st.stop()

@st.cache_data(ttl=3600, show_spinner=False)
def candidate_models(key: str) -> list[str]:
    """Бесплатные модели с поддержкой tools (список OpenRouter кэшируется на час)."""
    return OpenAICompatibleClient(key, SETTINGS.base_url).free_model_candidates()


# ---------------------------------------------------------------- боковая панель
with st.sidebar:
    st.header("Настройки")
    server_key = secret("LLM_API_KEY") or secret("OPENROUTER_API_KEY")
    api_key = server_key or st.text_input(
        "OpenRouter API key", type="password", help="Бесплатный ключ: https://openrouter.ai/keys. Ключ не сохраняется."
    )
    model_choice = st.selectbox(
        "Модель", [AUTO] + (candidate_models(api_key) if api_key else []), help="Бесплатные модели OpenRouter с поддержкой инструментов"
    )
    max_steps = st.slider("Максимум шагов агента", 4, 20, SETTINGS.max_steps)
    st.caption(
        "Агент работает через OpenRouter на бесплатных моделях и сам пишет и выполняет Python-код "
        "(pandas, matplotlib) в изолированной песочнице: "
        "без сети, без доступа к файлам сервера и к секретам."
    )
    runs_left = MAX_RUNS_PER_SESSION - st.session_state.get("runs", 0)
    st.caption(f"Осталось запусков в этой сессии: {runs_left}")

# ---------------------------------------------------------------- основная часть
st.title("📊 LLM-аналитик данных")
st.write(
    "Загрузите CSV или Excel — ИИ-агент сам исследует данные, выполняя код, "
    "и подготовит отчёт с метриками, графиками и выводами."
)

uploaded = st.file_uploader("Датасет", type=["csv", "tsv", "txt", "xlsx", "xls"])
instruction = st.text_area(
    "На что обратить внимание (необязательно)",
    placeholder="Например: интересует динамика выручки по месяцам и какие категории товаров растут быстрее",
    max_chars=SETTINGS.max_instruction_chars,
)

df = None
if uploaded is not None:
    try:
        df = load_table(uploaded.getvalue(), uploaded.name)
    except DataLoadError as exc:
        st.error(str(exc))
    else:
        st.caption(f"{uploaded.name}: {df.shape[0]} строк × {df.shape[1]} столбцов")
        st.dataframe(df.head(20))

start = st.button("Проанализировать", type="primary", disabled=df is None or runs_left <= 0)


def render_step(container, step: Step, compact: bool = False) -> None:
    """compact=True — внутри st.status (вложенные expander-ы Streamlit запрещает)."""
    target = container if compact else container.expander(f"🐍 {step.purpose or 'Выполнение кода'}")
    with target:
        if compact:
            st.markdown(f"🐍 **{step.purpose or 'Выполнение кода'}**")
        st.code(step.code, language="python")
        if step.stdout:
            st.text(step.stdout[:3000])
        if step.error:
            st.error(step.error[-1500:])
        for fig in step.figures:
            st.image(str(fig), width=420)


def render_report(result: AgentResult, file_name: str) -> None:
    report = result.report
    st.header(report.get("title", "Отчёт"))
    st.write(report.get("summary", ""))

    metrics = report.get("key_metrics") or []
    if metrics:
        st.subheader("Ключевые метрики")
        cols = st.columns(min(4, len(metrics)))
        for i, m in enumerate(metrics):
            with cols[i % len(cols)]:
                st.metric(m.get("name", ""), m.get("value", ""))
                if m.get("comment"):
                    st.caption(m["comment"])

    insights = report.get("insights") or []
    if insights:
        st.subheader("Инсайты")
        for i, ins in enumerate(insights, 1):
            st.markdown(f"**{i}. {ins.get('title', '')}**")
            st.write(ins.get("detail", ""))
            st.caption(f"Подтверждение: {ins.get('evidence', '')}")

    figures = [f for f in report.get("figures") or [] if f.get("id") in result.figures]
    if figures:
        st.subheader("Графики")
        cols = st.columns(2)
        for i, f in enumerate(figures):
            with cols[i % 2]:
                st.image(str(result.figures[f["id"]]), caption=f.get("caption", ""))

    for key, title in (("data_quality", "Качество данных"), ("recommendations", "Рекомендации"), ("limitations", "Ограничения")):
        items = report.get(key) or []
        if items:
            st.subheader(title)
            for x in items:
                st.markdown(f"- {x}")

    st.download_button(
        "Скачать отчёт (Markdown)",
        report_to_markdown(report, file_name),
        file_name="report.md",
        mime="text/markdown",
    )


if start and df is not None:
    if not api_key:
        st.error("Укажите OpenRouter API key в боковой панели (бесплатно: https://openrouter.ai/keys).")
        st.stop()

    clean_instruction, problems = check_instruction(instruction, SETTINGS.max_instruction_chars)
    if problems:
        st.error(
            "Инструкция похожа на попытку изменить поведение модели и не будет отправлена:\n\n- "
            + "\n- ".join(problems)
            + "\n\nОпишите, на что обратить внимание при анализе данных."
        )
        st.stop()

    findings = scan_dataframe(df)
    if findings:
        st.warning(
            "В данных найден текст, похожий на инструкции для ИИ. Агент будет считать его данными и не выполнит:\n\n- "
            + "\n- ".join(f.as_text() for f in findings[:5])
        )

    # Предыдущая песочница больше не нужна.
    old = st.session_state.pop("sandbox", None)
    if old is not None:
        old.cleanup()
    sandbox = Sandbox()
    sandbox.load_dataframe(df)
    st.session_state.sandbox = sandbox
    st.session_state.runs = st.session_state.get("runs", 0) + 1

    client = OpenAICompatibleClient(api_key, SETTINGS.base_url)
    if model_choice != AUTO:
        candidates = [model_choice]
    elif SETTINGS.model:
        candidates = [SETTINGS.model]
    else:
        candidates = candidate_models(api_key)
    agent = DataAnalystAgent(client, sandbox, candidates=candidates, max_steps=max_steps)

    with st.status("Агент анализирует данные…", expanded=True) as status:

        def on_event(kind, payload):
            if kind == "thinking":
                status.update(label=f"Агент анализирует данные… шаг {payload} (модель {client.model})")
            elif kind == "model":
                status.markdown(f"🔁 Переключаюсь на модель `{payload['model']}`: {payload['reason'][:200]}")
            elif kind == "text":
                status.markdown(f"💬 {payload[:500]}")
            elif kind == "result":
                render_step(status, payload, compact=True)

        try:
            result = agent.run(
                uploaded.name,
                n_rows=df.shape[0],
                n_cols=df.shape[1],
                user_instruction=clean_instruction,
                data_findings=[f.as_text() for f in findings],
                on_event=on_event,
            )
        except FatalLLMError as exc:
            status.update(label="Ошибка", state="error")
            st.error(f"Анализ остановлен: {exc}")
            st.stop()
        except LLMError as exc:
            status.update(label="Ошибка API", state="error")
            st.error(f"Ни одна из бесплатных моделей не ответила: {exc}. Попробуйте позже или выберите модель вручную.")
            st.stop()

        state = "complete" if result.report else "error"
        status.update(label=f"Анализ завершён ({result.stopped_reason}), запусков кода: {len(result.steps)}", state=state, expanded=False)

    st.session_state.result = result
    st.session_state.file_name = uploaded.name

result: AgentResult | None = st.session_state.get("result")
if result is not None:
    if result.report:
        render_report(result, st.session_state.get("file_name", "dataset"))
    else:
        st.error(f"Агент не подготовил отчёт ({result.stopped_reason}). Попробуйте ещё раз или увеличьте лимит шагов.")
    if result.steps:
        with st.expander(f"Журнал работы агента: {len(result.steps)} запусков кода"):
            for step in result.steps:
                st.markdown(f"🐍 **{step.purpose or 'Выполнение кода'}**")
                st.code(step.code, language="python")
                if step.stdout:
                    st.text(step.stdout[:3000])
                if step.error:
                    st.error(step.error[-1500:])
    fmt = lambda n: f"{n:,}".replace(",", " ")  # noqa: E731
    tokens = f"вход {fmt(result.input_tokens)}, выход {fmt(result.output_tokens)}"
    st.caption(f"Модель: {', '.join(result.models_used)} · токены: {tokens}")
