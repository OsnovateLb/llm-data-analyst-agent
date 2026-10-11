import os
import sys

import numpy as np
import pandas as pd
import pytest

from analyst.sandbox import Sandbox


@pytest.fixture(scope="module")
def sandbox():
    sb = Sandbox(timeout_s=20)
    sb.load_dataframe(pd.DataFrame({"a": np.arange(50), "b": np.linspace(0, 1, 50), "c": ["x", "y"] * 25}))
    yield sb
    sb.cleanup()


def test_runs_pandas_and_saves_figure(sandbox):
    res = sandbox.run("print(df.groupby('c')['a'].sum().to_dict())\nplt.figure()\ndf['b'].hist()\nplt.title('Гистограмма')")
    assert res.ok, res.error
    assert "{'x': 600, 'y': 625}" in res.stdout
    assert len(res.figures) == 1 and res.figures[0].is_file()


def test_files_in_workdir_persist_between_calls(sandbox):
    assert sandbox.run("df[df.a > 10].to_csv('part.csv', index=False)").ok
    res = sandbox.run("print(len(pd.read_csv('part.csv')))")
    assert res.ok and res.stdout.strip() == "39"


def test_variables_do_not_persist(sandbox):
    sandbox.run("x = 5")
    res = sandbox.run("print(x)")
    assert not res.ok and "NameError" in res.error


@pytest.mark.parametrize(
    "code, expected",
    [
        ("pd.read_csv('https://example.com/data.csv')", "запрещена"),
        ("df.to_csv('/tmp/leak.csv')", "запись разрешена только"),
        ("import os", "отклонён"),
        ("open('data.pkl')", "отклонён"),
    ],
)
def test_attacks_blocked(sandbox, code, expected):
    res = sandbox.run(code)
    assert not res.ok
    assert expected in res.error


@pytest.mark.skipif(sys.platform == "win32", reason="на Windows другой путь")
def test_reading_system_files_blocked(sandbox):
    res = sandbox.run("print(pd.read_csv('/etc/passwd', sep=':', header=None))")
    assert not res.ok and "чтение файлов" in res.error


def test_secrets_not_visible(sandbox, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    # Процесс песочницы получает минимальное окружение без секретов родителя.
    env = sandbox._env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "sk-test-secret" not in " ".join(env.values())


def test_timeout():
    sb = Sandbox(timeout_s=3)
    sb.load_dataframe(pd.DataFrame({"a": [1]}))
    try:
        res = sb.run("while True:\n    pass")
        assert not res.ok
    finally:
        sb.cleanup()


def test_library_paths_passed_without_app_dir(sandbox):
    from pathlib import Path

    from analyst.sandbox import library_paths

    paths = library_paths()
    assert paths and all(Path(p).name in ("site-packages", "dist-packages") for p in paths)
    app_dir = str(Path(__file__).resolve().parents[1])
    assert app_dir not in sandbox._env()["SANDBOX_SYS_PATH"].split(os.pathsep)


def test_cyrillic_code_and_output(sandbox):
    res = sandbox.run("print('Выручка по регионам')\nplt.figure()\nplt.title('Динамика')")
    assert res.ok, res.error
    assert "Выручка по регионам" in res.stdout


def test_app_secrets_not_readable(sandbox):
    from pathlib import Path

    secrets = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml.example"
    res = sandbox.run(f"print(pd.read_csv(r'{secrets}'))")
    assert not res.ok and "чтение файлов" in res.error
