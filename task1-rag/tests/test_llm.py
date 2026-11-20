from types import SimpleNamespace

import ollama
import pytest

from rag import config
from rag.llm import LLMError, answer_stream, build_prompt

CHUNKS = [{"text": "The sky is blue.", "source": "a.pdf", "page": 3}]


def part(text):
    return SimpleNamespace(message=SimpleNamespace(content=text))


class FakeOllama:
    def __init__(self, pieces=("The sky ", "is blue [1]."), error=None):
        self.pieces = pieces
        self.error = error
        self.kwargs = None

    def chat(self, **kwargs):
        if self.error:
            raise self.error
        self.kwargs = kwargs
        return iter([part(p) for p in self.pieces])


def test_prompt_numbers_the_passages():
    prompt = build_prompt("Why?", CHUNKS)
    assert "[1] a.pdf, page 3" in prompt
    assert prompt.endswith("Question: Why?")


def test_answer_is_streamed_in_pieces():
    client = FakeOllama()
    text = list(answer_stream("What colour is the sky?", CHUNKS, client=client))
    assert text == ["The sky ", "is blue [1]."]
    assert client.kwargs["options"] == {"temperature": 0}
    assert client.kwargs["messages"][0]["role"] == "system"


def test_empty_pieces_are_skipped():
    client = FakeOllama(pieces=("", "hi", ""))
    assert list(answer_stream("q", CHUNKS, client=client)) == ["hi"]


def test_qwen3_is_told_not_to_think(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_CHAT_MODEL", "qwen3")
    client = FakeOllama()
    list(answer_stream("q", CHUNKS, client=client))
    assert client.kwargs["think"] is False


def test_llama_is_not_sent_the_think_option():
    client = FakeOllama()
    list(answer_stream("q", CHUNKS, client=client))
    assert "think" not in client.kwargs


def test_ollama_not_running_gives_friendly_error():
    client = FakeOllama(error=ConnectionError("refused"))
    with pytest.raises(LLMError, match="Cannot reach Ollama"):
        list(answer_stream("q", CHUNKS, client=client))


def test_missing_chat_model_tells_the_user_what_to_pull():
    client = FakeOllama(error=ollama.ResponseError("not found", 404))
    with pytest.raises(LLMError, match="ollama pull"):
        list(answer_stream("q", CHUNKS, client=client))


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "banana")
    with pytest.raises(LLMError, match="LLM_PROVIDER"):
        list(answer_stream("q", CHUNKS))


def test_groq_without_key_is_rejected(monkeypatch):
    pytest.importorskip("groq")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        list(answer_stream("q", CHUNKS))


def test_groq_answer_is_streamed(monkeypatch):
    pytest.importorskip("groq")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")

    def delta(text):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    class FakeGroq:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: iter([delta("Hi "), delta(None), delta("there")]))
        )

    assert list(answer_stream("q", CHUNKS, client=FakeGroq())) == ["Hi ", "there"]