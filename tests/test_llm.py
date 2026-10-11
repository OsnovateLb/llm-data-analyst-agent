"""Тесты HTTP-клиента на локальном сервере, имитирующем OpenRouter."""

import json
import ssl
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from analyst import llm


class FakeOpenRouter(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, code, payload, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [
            {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "google/lyria-3-pro-preview", "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"output_modalities": ["audio"]}},
            {"id": "no-tools/model:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["temperature"]},
            {"id": "vision/model:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["tools"],
             "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}},
            {"id": "nvidia/nemotron-3-super-120b-a12b:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["tools"]},
        ]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state.setdefault("bodies", []).append(body)
        if len(self.state["bodies"]) == 1:
            return self._send(503, {"error": {"message": "overloaded"}}, {"Retry-After": "0"})
        self._send(200, {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "run_python", "arguments": "{\"purpose\": \"x\", \"code\": \"print(1)\"}"}}
                ]},
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        })


@pytest.fixture
def server():
    FakeOpenRouter.state = {}
    httpd = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/api/v1"
    httpd.shutdown()


def test_candidates_need_tools_and_skip_non_chat(server):
    client = llm.OpenAICompatibleClient("k", server)
    cands = client.free_model_candidates()
    assert cands == ["nvidia/nemotron-3-super-120b-a12b:free", "vision/model:free"]
    assert client.supports_images("vision/model:free") and not client.supports_images("nvidia/nemotron-3-super-120b-a12b:free")


def test_chat_parses_tool_calls_and_retries_503(server, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = llm.OpenAICompatibleClient("sk-or-test", server, "m:free")
    r = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "run_python"}}])
    assert r.tool_calls[0].name == "run_python" and json.loads(r.tool_calls[0].arguments)["code"] == "print(1)"
    assert r.usage["prompt_tokens"] == 12
    assert r.as_message()["tool_calls"][0]["function"]["name"] == "run_python"
    body = FakeOpenRouter.state["bodies"][-1]
    assert body["tool_choice"] == "auto" and body["model"] == "m:free"


def test_daily_limit_is_fatal(server, monkeypatch):
    def handler(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send(429, {"error": {"message": "Rate limit exceeded: free-models-per-day"}})

    monkeypatch.setattr(FakeOpenRouter, "do_POST", handler)
    with pytest.raises(llm.FatalLLMError, match="дневной лимит"):
        llm.OpenAICompatibleClient("k", server, "m").chat([{"role": "user", "content": "x"}])


def test_network_errors_retried(monkeypatch):
    calls = {"n": 0}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return b'{"ok": 1}'

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError(ssl.SSLEOFError(8, "EOF occurred in violation of protocol"))
        return Resp()

    monkeypatch.setattr(llm.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm.urlopen_json("https://x", 5) == {"ok": 1}


def test_model_list_failure_falls_back(monkeypatch):
    client = llm.OpenAICompatibleClient("k")
    monkeypatch.setattr(client, "list_free_models", lambda: (_ for _ in ()).throw(llm.FatalLLMError("net")))
    assert client.free_model_candidates() == llm.PREFERRED_FREE_MODELS
