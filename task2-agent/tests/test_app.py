"""Run the real Streamlit app in a test harness, with the model replaced by a fake."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from agent import config, llm
from agent.llm import LLMError, Reply, ToolCall

APP = Path(__file__).resolve().parent.parent / "app.py"


def start_app(monkeypatch, chat):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")  # skips the Ollama health check
    monkeypatch.setattr(llm, "chat", chat)
    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_app_starts_with_tools_and_examples_listed(monkeypatch):
    at = start_app(monkeypatch, lambda messages, tools: Reply("unused"))
    assert not at.exception
    assert len(at.sidebar.button) == 7  # six examples plus "Clear chat"
    assert "calculator" in " ".join(m.value for m in at.sidebar.markdown)


def test_plain_question_shows_only_the_answer(monkeypatch):
    at = start_app(monkeypatch, lambda messages, tools: Reply("Hello there!"))
    at.chat_input[0].set_value("Hi").run()

    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    assert "Hello there!" in at.chat_message[1].markdown[0].value
    assert len(at.expander) == 0  # no tools used, so no "Tools used" box


def test_tool_use_is_shown_to_the_user(monkeypatch):
    replies = iter([
        Reply("", [ToolCall("c0", "calculator", {"expression": "2+3"})]),
        Reply("It is 5."),
    ])
    at = start_app(monkeypatch, lambda messages, tools: next(replies))
    at.chat_input[0].set_value("2+3?").run()

    assert at.expander[0].label == "Tools used (1)"
    assert "2+3 = 5" in at.expander[0].code[0].value


def test_conversation_is_remembered_between_turns(monkeypatch):
    seen = []

    def fake_chat(messages, tools):
        seen.append([m["content"] for m in messages if m["role"] in ("user", "assistant")])
        return Reply("Noted.")

    at = start_app(monkeypatch, fake_chat)
    at.chat_input[0].set_value("My name is Yesha").run()
    at.chat_input[0].set_value("What is my name?").run()

    assert seen[1] == ["My name is Yesha", "Noted.", "What is my name?"]


def test_model_errors_are_shown_without_crashing(monkeypatch):
    def broken(messages, tools):
        raise LLMError("Cannot reach Ollama.")

    at = start_app(monkeypatch, broken)
    at.chat_input[0].set_value("Hi").run()

    assert not at.exception
    assert at.error[0].value == "Cannot reach Ollama."
    assert at.session_state["messages"] == []  # the failed question is not kept