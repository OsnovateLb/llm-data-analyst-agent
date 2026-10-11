"""Процесс-исполнитель кода агента. Запускается из sandbox.py как отдельный процесс:

    python -I _runner.py <workdir>      (код агента приходит через stdin)

Внутри:
  * лимиты CPU/памяти/размера файлов (на Linux/macOS);
  * датафрейм `df` загружается заранее;
  * встроенные функции урезаны, импорт — только из белого списка;
  * audit hook (PEP 578) блокирует сеть, запуск процессов, ctypes,
    запись файлов вне рабочей папки и чтение файлов вне разрешённых каталогов;
  * переменные окружения (в т.ч. API-ключ) сюда не передаются вообще.
Результат печатается одной JSON-строкой после маркера.
"""

import builtins
import contextlib
import io
import json
import os
import sys
import traceback

MARKER = "@@SANDBOX_RESULT@@"
ALLOWED_IMPORTS = set(os.environ.get("SANDBOX_ALLOWED_IMPORTS", "").split(","))


def _set_limits(cpu_s: int, mem_mb: int) -> None:
    try:
        import resource  # нет на Windows
    except ImportError:
        return
    for limit, value in (
        (resource.RLIMIT_CPU, cpu_s),
        (resource.RLIMIT_AS, mem_mb * 1024 * 1024),
        (resource.RLIMIT_FSIZE, 50 * 1024 * 1024),
    ):
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(limit, (value, value))


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _extend_sys_path() -> list[str]:
    """Флаг -I отключает пользовательский site-packages (на Windows pip часто ставит пакеты именно туда:
    %APPDATA%\\Python\\Python3XX\\site-packages). Родительский процесс передаёт пути к библиотекам явно."""
    extra = [p for p in os.environ.get("SANDBOX_SYS_PATH", "").split(os.pathsep) if p and os.path.isdir(p)]
    for p in extra:
        if p not in sys.path:
            sys.path.append(p)
    return extra


def main() -> None:
    extra_paths = _extend_sys_path()
    workdir = _norm(sys.argv[1])
    os.chdir(workdir)
    _set_limits(int(os.environ.get("SANDBOX_CPU_S", "60")), int(os.environ.get("SANDBOX_MEM_MB", "2048")))

    # Флаг -I отключает PYTHONIOENCODING, поэтому явно работаем в UTF-8 (важно на Windows с cp1251).
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")
    real_stdout = sys.stdout
    code = sys.stdin.buffer.read().decode("utf-8", errors="replace")

    # --- Тяжёлые импорты и прогрев ДО включения ограничений ---------------
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    with contextlib.suppress(Exception):
        import seaborn  # noqa: F401
    with contextlib.suppress(Exception):
        import scipy.stats  # noqa: F401

    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.title("warm-up")
    fig.savefig(io.BytesIO(), format="png")
    plt.close("all")
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = pd.read_pickle(os.path.join(workdir, "data.pkl"))

    # --- Audit hook --------------------------------------------------------
    import site
    import sysconfig

    read_roots = {workdir}
    for p in [sys.prefix, sys.base_prefix, sys.exec_prefix, *site.getsitepackages(), *extra_paths]:
        read_roots.add(_norm(p))
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        with contextlib.suppress(Exception):
            read_roots.add(_norm(sysconfig.get_path(key)))
    read_roots.add(_norm(matplotlib.get_data_path()))
    for extra in ("/usr/share/fonts", "/usr/share/zoneinfo", "/usr/lib", "/usr/local/lib"):
        read_roots.add(_norm(extra))
    read_files = {_norm(p) for p in ("/dev/null", "/dev/urandom", "/etc/localtime")}

    def inside(path: str, roots) -> bool:
        return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)

    blocked_prefixes = (
        "socket.",
        "subprocess.",
        "os.system",
        "os.exec",
        "os.spawn",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
        "os.kill",
        "os.killpg",
        "os.startfile",
        "os.putenv",
        "os.unsetenv",
        "os.chmod",
        "os.chown",
        "os.symlink",
        "os.link",
        "os.truncate",
        "ctypes.",
        "winreg.",
        "_winapi.",
        "webbrowser.",
        "urllib.",
        "http.",
        "ftplib.",
        "smtplib.",
        "telnetlib.",
        "pty.",
        "shutil.",
        "sys.addaudithook",
        "sys.settrace",
        "sys.setprofile",
        "pickle.find_class",
    )
    blocked_imports = {
        "socket",
        "_socket",
        "ssl",
        "subprocess",
        "_posixsubprocess",
        "ctypes",
        "_ctypes",
        "multiprocessing",
        "urllib.request",
        "http.client",
        "requests",
        "httpx",
        "ftplib",
        "smtplib",
        "telnetlib",
        "pty",
        "webbrowser",
    }
    state = {"busy": False}

    def hook(event: str, args) -> None:
        if state["busy"]:
            return
        if event.startswith(blocked_prefixes):
            raise PermissionError(f"Песочница: операция '{event}' запрещена")
        if event == "import" and args and args[0] in blocked_imports:
            raise PermissionError(f"Песочница: импорт '{args[0]}' запрещён")
        if event in ("open", "os.listdir", "os.scandir", "os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.replace"):
            target = args[0] if args else None
            if target is None or isinstance(target, int):
                return
            state["busy"] = True
            try:
                path = _norm(os.fsdecode(target))
            finally:
                state["busy"] = False
            writing = event != "open" or (
                len(args) > 1 and isinstance(args[1], str) and any(c in args[1] for c in "wax+")
            )
            if event == "open" and len(args) > 2 and isinstance(args[2], int):
                writing = writing or bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
            if event in ("os.listdir", "os.scandir"):
                writing = False
            if writing:
                if not inside(path, {workdir}):
                    raise PermissionError("Песочница: запись разрешена только в рабочую папку")
            elif not (inside(path, read_roots) or path in read_files):
                raise PermissionError("Песочница: чтение файлов вне рабочей папки запрещено")

    # --- Урезанные builtins и импорт по белому списку ------------------------
    real_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"Импорт модуля '{name}' запрещён в песочнице")
        return real_import(name, globals, locals, fromlist, level)

    safe_builtins = {
        k: v
        for k, v in vars(builtins).items()
        if k
        not in {
            "eval",
            "exec",
            "compile",
            "open",
            "input",
            "breakpoint",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
            "exit",
            "quit",
            "help",
            "memoryview",
            "__import__",
        }
    }
    safe_builtins["__import__"] = safe_import

    namespace = {"__builtins__": safe_builtins, "__name__": "__agent__", "df": df, "pd": pd, "np": np, "plt": plt}

    sys.addaudithook(hook)

    buffer = io.StringIO()
    error = None
    try:
        compiled = compile(code, "<agent_code>", "exec")
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compiled, namespace)
    except BaseException as exc:  # noqa: BLE001 — отдаём агенту любую ошибку
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        # Оставляем только кадры кода агента, без путей сервера.
        lines = [ln for ln in tb if "<agent_code>" in ln or not ln.startswith("  File")]
        error = "".join(lines)[-3000:]

    # --- Сохранение графиков --------------------------------------------------
    figures = []
    counter_path = os.path.join(workdir, "fig_counter.txt")
    try:
        with builtins.open(counter_path, encoding="utf-8") as fh:
            counter = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        counter = 0
    for num in plt.get_fignums():
        try:
            counter += 1
            name = f"fig_{counter}.png"
            plt.figure(num).savefig(os.path.join(workdir, name), dpi=90, bbox_inches="tight")
            figures.append(name)
        except Exception as exc:  # noqa: BLE001
            error = (error or "") + f"\nНе удалось сохранить график: {exc}"
    plt.close("all")
    with builtins.open(counter_path, "w", encoding="utf-8") as fh:
        fh.write(str(counter))

    real_stdout.write("\n" + MARKER + json.dumps({"stdout": buffer.getvalue(), "error": error, "figures": figures}) + "\n")
    real_stdout.flush()


if __name__ == "__main__":
    main()
