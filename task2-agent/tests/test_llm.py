import json
from types import SimpleNamespace

import ollama
import pytest

from agent import config
from agent.llm import LLMError, Reply, ToolCall, _to_groq, _to_ollama, chat

TOOLS = [{"type": "function", "function": {"name": "calculator"}}]

CONVERSATION = [
    {"role": "system", "content": "Be nice."},
    {"role": "user", "content": "What is 2+3?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [ToolCall(id="call_0", name="calculator", arguments={"expression": "2+3"})],
    },
    {"role": "tool", "tool_call_id": "call_0", "name": "calculator", "content": "2+3 = 5"},
]


# ---------- message conversion ----------

def test_conversion_for_ollama():
    converted = _to_ollama(CONVERSATION)
    assert converted[1] == {"role": "user", "content": "What is 2+3?"}
    assert converted[2]["tool_calls"] == [
        {"function": {"name": "calculator", "arguments": {"expression": "2+3"}}}
    ]
    assert converted[3] == {"role": "tool", "tool_name": "calculator", "content": "2+3 = 5"}


def test_conversion_for_groq():
    converted = _to_groq(CONVERSATION)
    call = converted[2]["tool_calls"][0]
    assert call["id"] == "call_0" and call["type"] == "function"
    assert json.loads(call["function"]["arguments"]) == {"expression": "2+3"}
    assert converted[3] == {"role": "tool", "tool_call_id": "call_0", "content": "2+3 = 5"}


# ---------- Ollama ----------

def ollama_response(content="", calls=()):
    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=args)) for name, args in calls
    ]
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls or None))


class FakeOllama:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def chat(self, **kwargs):
        if self.error:
            raise self.error
        self.kwargs = kwargs
        return self.response


def test_plain_answer():
    client = FakeOllama(ollama_response("Hello!"))
    assert chat(CONVERSATION, TOOLS, client=client) == Reply("Hello!", [])


def test_tool_call_is_parsed():
    client = FakeOllama(ollama_response(calls=[("calculator", {"expression": "2+3"})]))
    reply = chat(CONVERSATION, TOOLS, client=client)
    assert reply.tool_calls == [ToolCall("call_0", "calculator", {"expression": "2+3"})]


def test_tools_and_settings_are_sent():
    client = FakeOllama(ollama_response("ok"))
    chat(CONVERSATION, TOOLS, client=client)
    assert client.kwargs["tools"] == TOOLS
    assert client.kwargs["options"] == {"temperature": 0}


def test_qwen3_is_told_not_to_think(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_CHAT_MODEL", "qwen3")
    client = FakeOllama(ollama_response("ok"))
    chat(CONVERSATION, TOOLS, client=client)
    assert client.kwargs["think"] is False


def test_other_models_are_not_sent_the_think_option(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_CHAT_MODEL", "llama3.2")
    client = FakeOllama(ollama_response("ok"))
    chat(CONVERSATION, TOOLS, client=client)
    assert "think" not in client.kwargs


def test_ollama_not_running():
    client = FakeOllama(error=ConnectionError("refused"))
    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        chat(CONVERSATION, TOOLS, client=client)


def test_model_not_installed():
    client = FakeOllama(error=ollama.ResponseError("not found", 404))
    with pytest.raises(LLMError, match="ollama pull"):
        chat(CONVERSATION, TOOLS, client=client)


def test_model_without_tool_support():
    client = FakeOllama(error=ollama.ResponseError("registry.ollama.ai/library/gemma does not support tools", 400))
    with pytest.raises(LLMError, match="cannot use tools"):
        chat(CONVERSATION, TOOLS, client=client)


def test_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "banana")
    with pytest.raises(LLMError, match="LLM_PROVIDER"):
        chat(CONVERSATION, TOOLS)


# ---------- Groq ----------

def groq_response(content="", calls=()):
    tool_calls = [
        SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=args))
        for cid, name, args in calls
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))


@pytest.fixture
def groq_provider(monkeypatch):
    pytest.importorskip("groq")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")


def test_groq_tool_call_is_parsed(groq_provider):
    client = FakeGroq(groq_response(calls=[("abc", "calculator", '{"expression": "2+3"}')]))
    reply = chat(CONVERSATION, TOOLS, client=client)
    assert reply.tool_calls == [ToolCall("abc", "calculator", {"expression": "2+3"})]


def test_groq_broken_arguments_become_empty(groq_provider):
    client = FakeGroq(groq_response(calls=[("abc", "calculator", "{not json")]))
    assert chat(CONVERSATION, TOOLS, client=client).tool_calls[0].arguments == {}


def test_groq_plain_answer(groq_provider):
    assert chat(CONVERSATION, TOOLS, client=FakeGroq(groq_response("Hi"))).content == "Hi"


def test_groq_without_key(groq_provider, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        chat(CONVERSATION, TOOLS)