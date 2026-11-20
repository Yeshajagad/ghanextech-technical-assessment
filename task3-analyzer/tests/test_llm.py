from types import SimpleNamespace

import httpx
import ollama
import pytest

from analyzer import config
from analyzer.llm import LLMError, complete

SCHEMA = {"type": "object"}


class FakeOllama:
    def __init__(self, content="{}", error=None):
        self.content = content
        self.error = error
        self.kwargs = None

    def chat(self, **kwargs):
        if self.error:
            raise self.error
        self.kwargs = kwargs
        return SimpleNamespace(message=SimpleNamespace(content=self.content))


def test_reply_text_is_returned():
    assert complete("sys", "user", SCHEMA, client=FakeOllama('{"a": 1}')) == '{"a": 1}'


def test_messages_schema_and_settings_are_sent():
    client = FakeOllama()
    complete("be brief", "analyse", SCHEMA, client=client)
    assert client.kwargs["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "analyse"},
    ]
    assert client.kwargs["format"] == SCHEMA
    assert client.kwargs["options"] == {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX}


def test_qwen3_is_told_not_to_think(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_CHAT_MODEL", "qwen3")
    client = FakeOllama()
    complete("s", "u", SCHEMA, client=client)
    assert client.kwargs["think"] is False


def test_other_models_are_not_sent_the_think_option(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_CHAT_MODEL", "llama3.2")
    client = FakeOllama()
    complete("s", "u", SCHEMA, client=client)
    assert "think" not in client.kwargs


def test_empty_reply_becomes_empty_text():
    assert complete("s", "u", SCHEMA, client=FakeOllama(content=None)) == ""


def test_ollama_not_running():
    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        complete("s", "u", SCHEMA, client=FakeOllama(error=ConnectionError("refused")))


def test_slow_model():
    with pytest.raises(LLMError, match="too long"):
        complete("s", "u", SCHEMA, client=FakeOllama(error=httpx.ReadTimeout("slow")))


def test_model_not_installed():
    with pytest.raises(LLMError, match="ollama pull"):
        complete("s", "u", SCHEMA, client=FakeOllama(error=ollama.ResponseError("missing", 404)))


def test_other_ollama_errors():
    with pytest.raises(LLMError, match="Ollama returned an error"):
        complete("s", "u", SCHEMA, client=FakeOllama(error=ollama.ResponseError("boom", 500)))


def test_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "banana")
    with pytest.raises(LLMError, match="LLM_PROVIDER"):
        complete("s", "u", SCHEMA)


# ---------- Groq (skipped when the groq package is not installed) ----------

class FakeGroq:
    def __init__(self, content=None, error=None):
        def create(**kwargs):
            self.kwargs = kwargs
            if error:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


@pytest.fixture
def groq_provider(monkeypatch):
    pytest.importorskip("groq")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")


def test_groq_asks_for_json(groq_provider):
    client = FakeGroq(content='{"a": 1}')
    assert complete("s", "u", SCHEMA, client=client) == '{"a": 1}'
    assert client.kwargs["response_format"] == {"type": "json_object"}
    assert client.kwargs["temperature"] == 0


def test_groq_without_key(groq_provider, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        complete("s", "u", SCHEMA)


def test_groq_errors_are_translated(groq_provider):
    import groq

    request = httpx.Request("POST", "https://api.groq.com")
    cases = [
        (groq.RateLimitError("x", response=httpx.Response(429, request=request), body=None), "rate limit"),
        (groq.AuthenticationError("x", response=httpx.Response(401, request=request), body=None), "key was rejected"),
        (groq.APIStatusError("x", response=httpx.Response(400, request=request), body=None), "valid answer"),
        (groq.APIConnectionError(request=request), "Could not reach"),
    ]
    for error, expected in cases:
        with pytest.raises(LLMError, match=expected):
            complete("s", "u", SCHEMA, client=FakeGroq(error=error))