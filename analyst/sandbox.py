"""Изолированный запуск Python-кода, который пишет агент (инструмент `run_python`)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import ALLOWED_IMPORTS, SETTINGS
from .guard import CodeRejected, check_code

RUNNER = Path(__file__).with_name("_runner.py")
MARKER = "@@SANDBOX_RESULT@@"


@dataclass
class ExecResult:
    stdout: str = ""
    error: str | None = None
    figures: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def library_paths() -> list[str]:
    """Каталоги с установленными библиотеками из sys.path текущего процесса (в т.ч. пользовательский
    site-packages). Папка приложения сюда не попадает — песочница не должна читать код и секреты приложения."""
    return [
        p
        for p in sys.path
        if p and os.path.isdir(p) and os.path.basename(os.path.normpath(p)).lower() in ("site-packages", "dist-packages")
    ]


class Sandbox:
    """Одна песочница на один анализ: своя временная папка, туда кладётся датасет."""

    def __init__(self, timeout_s: int | None = None, memory_mb: int | None = None):
        self.timeout_s = timeout_s or SETTINGS.exec_timeout_s
        self.memory_mb = memory_mb or SETTINGS.exec_memory_mb
        self.workdir = Path(tempfile.mkdtemp(prefix="agent_sbx_")).resolve()

    def load_dataframe(self, df: pd.DataFrame) -> None:
        df.to_pickle(self.workdir / "data.pkl")

    def _env(self) -> dict[str, str]:
        # Минимальное окружение: никакие секреты родительского процесса не передаются.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.workdir),
            "TMPDIR": str(self.workdir),
            "TEMP": str(self.workdir),
            "TMP": str(self.workdir),
            "MPLCONFIGDIR": str(self.workdir / ".mpl"),
            "MPLBACKEND": "Agg",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONIOENCODING": "utf-8",
            "SANDBOX_ALLOWED_IMPORTS": ",".join(sorted(ALLOWED_IMPORTS)),
            "SANDBOX_CPU_S": str(self.timeout_s),
            "SANDBOX_MEM_MB": str(self.memory_mb),
            "SANDBOX_SYS_PATH": os.pathsep.join(library_paths()),
        }
        if os.name == "nt":  # без SYSTEMROOT Python на Windows не стартует
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return env

    def run(self, code: str) -> ExecResult:
        try:
            check_code(code)
        except CodeRejected as exc:
            return ExecResult(error=f"Код отклонён проверкой безопасности: {exc}")

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(RUNNER), str(self.workdir)],
                input=code,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_s + 15,  # +время на импорт pandas/matplotlib
                cwd=self.workdir,
                env=self._env(),
            )
        except subprocess.TimeoutExpired:
            return ExecResult(error=f"Превышено время выполнения ({self.timeout_s} c). Упростите вычисления.")

        payload = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith(MARKER):
                try:
                    payload = json.loads(line[len(MARKER):])
                except json.JSONDecodeError:
                    payload = None
                break
        if payload is None:
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            hint = " (превышен лимит времени CPU или памяти)" if proc.returncode and proc.returncode < 0 else ""
            return ExecResult(error=f"Процесс песочницы завершился с кодом {proc.returncode}{hint}.\n{tail}")

        stdout = payload.get("stdout", "")
        if len(stdout) > SETTINGS.max_stdout_chars:
            half = SETTINGS.max_stdout_chars // 2
            stdout = stdout[:half] + "\n... [вывод сокращён] ...\n" + stdout[-half:]
        figures = [
            self.workdir / name
            for name in payload.get("figures", [])
            if isinstance(name, str) and re.fullmatch(r"fig_\d+\.png", name) and (self.workdir / name).is_file()
        ]
        return ExecResult(stdout=stdout, error=payload.get("error"), figures=figures)

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)
