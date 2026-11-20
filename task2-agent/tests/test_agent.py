from agent import config
from agent.agent import SYSTEM_PROMPT, run_agent
from agent.llm import Reply, ToolCall
from agent.tools import TOOLS


class ScriptedChat:
    """Stands in for the model: returns prepared replies one by one and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []  # a copy of the messages at each call

    def __call__(self, messages, tools):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name, **arguments):
    return ToolCall(id=f"id_{name}", name=name, arguments=arguments)


def test_greeting_is_answered_without_tools():
    chat = ScriptedChat(Reply("Hello! How can I help?"))
    result = run_agent([], "Hi", chat=chat)
    assert result.answer == "Hello! How can I help?"
    assert result.steps == []
    assert len(chat.seen) == 1


def test_one_tool_call_then_an_answer():
    chat = ScriptedChat(
        Reply("", [call("calculator", expression="2 + 3")]),
        Reply("The answer is 5."),
    )
    result = run_agent([], "What is 2 + 3?", chat=chat)

    assert result.answer == "The answer is 5."
    assert [(s.name, s.result, s.is_error) for s in result.steps] == [("calculator", "2 + 3 = 5", False)]
    # The second request to the model contains the tool's output.
    last = chat.seen[1][-1]
    assert last["role"] == "tool" and last["content"] == "2 + 3 = 5"


def test_tools_can_be_chained():
    chat = ScriptedChat(
        Reply("", [call("calculator", expression="20 * 9 / 5")]),
        Reply("", [call("calculator", expression="36 + 32")]),
        Reply("20°C is 68°F."),
    )
    result = run_agent([], "Convert 20C to F", chat=chat)
    assert [s.arguments["expression"] for s in result.steps] == ["20 * 9 / 5", "36 + 32"]
    assert result.answer == "20°C is 68°F."


def test_several_calls_in_one_reply_are_all_run():
    chat = ScriptedChat(
        Reply("", [call("calculator", expression="1+1"), call("calculator", expression="2+2")]),
        Reply("2 and 4."),
    )
    result = run_agent([], "two sums", chat=chat)
    assert [s.result for s in result.steps] == ["1+1 = 2", "2+2 = 4"]


def test_tool_error_reaches_the_model_and_nothing_crashes():
    chat = ScriptedChat(
        Reply("", [call("calculator", expression="1/0")]),
        Reply("That cannot be calculated because it divides by zero."),
    )
    result = run_agent([], "1/0?", chat=chat)
    assert result.steps[0].is_error
    assert "Division by zero" in chat.seen[1][-1]["content"]
    assert "divides by zero" in result.answer


def test_unknown_tool_is_reported_to_the_model():
    chat = ScriptedChat(Reply("", [call("teleport", where="moon")]), Reply("I cannot do that."))
    result = run_agent([], "go", chat=chat)
    assert result.steps[0].is_error
    assert "no tool called 'teleport'" in result.steps[0].result


def test_bad_arguments_are_reported_to_the_model():
    chat = ScriptedChat(Reply("", [call("get_weather")]), Reply("Which city?"))
    result = run_agent([], "weather?", chat=chat)
    assert "Missing required parameter 'location'" in result.steps[0].result


def test_agent_stops_after_the_step_limit():
    forever = [Reply("", [call("calculator", expression="1+1")]) for _ in range(config.MAX_STEPS + 3)]
    chat = ScriptedChat(*forever)
    result = run_agent([], "loop", chat=chat)
    assert len(chat.seen) == config.MAX_STEPS
    assert "could not finish" in result.answer


def test_history_is_sent_before_the_new_question():
    history = [
        {"role": "user", "content": "My name is Yesha."},
        {"role": "assistant", "content": "Nice to meet you, Yesha."},
    ]
    chat = ScriptedChat(Reply("Your name is Yesha."))
    run_agent(history, "What is my name?", chat=chat)
    sent = chat.seen[0]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "My name is Yesha."
    assert sent[-1]["content"] == "What is my name?"


def test_only_recent_history_is_sent(monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY", 2)
    history = [{"role": "user", "content": f"message {i}"} for i in range(6)]
    chat = ScriptedChat(Reply("ok"))
    run_agent(history, "now", chat=chat)
    assert [m["content"] for m in chat.seen[0][1:-1]] == ["message 4", "message 5"]


def test_thinking_text_is_removed_from_the_answer():
    chat = ScriptedChat(Reply("<think>let me see</think>\nThe answer is 5."))
    assert run_agent([], "q", chat=chat).answer == "The answer is 5."


def test_empty_answer_gets_a_fallback():
    assert "could not come up" in run_agent([], "q", chat=ScriptedChat(Reply("  "))).answer


def test_the_system_prompt_names_every_tool():
    for name in TOOLS:
        assert name in SYSTEM_PROMPT