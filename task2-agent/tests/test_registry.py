import pytest

from agent.tools import TOOLS, run_tool, tool_schemas


def test_three_tools_are_registered():
    assert set(TOOLS) == {"calculator", "get_weather", "search_wikipedia"}


@pytest.mark.parametrize("schema", tool_schemas(), ids=lambda s: s["function"]["name"])
def test_every_schema_is_complete(schema):
    function = schema["function"]
    assert schema["type"] == "function"
    assert len(function["description"]) > 30  # the model needs a real description
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    for name in parameters["required"]:
        assert name in parameters["properties"]
    for spec in parameters["properties"].values():
        assert spec["type"] in ("string", "number")
        assert spec["description"]


def test_running_a_tool():
    text, is_error = run_tool("calculator", {"expression": "2 + 3"})
    assert (text, is_error) == ("2 + 3 = 5", False)


def test_unknown_tool_is_reported_not_raised():
    text, is_error = run_tool("teleport", {})
    assert is_error
    assert "no tool called 'teleport'" in text
    assert "calculator" in text  # lists what exists, so the model can recover


def test_bad_arguments_are_reported():
    text, is_error = run_tool("calculator", {})
    assert is_error
    assert "Missing required parameter 'expression'" in text


def test_tool_errors_are_returned_as_text():
    text, is_error = run_tool("calculator", {"expression": "1/0"})
    assert is_error
    assert text == "Error: Division by zero is not possible."


def test_unexpected_crash_is_contained(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(TOOLS["calculator"], "func", boom)
    text, is_error = run_tool("calculator", {"expression": "1+1"})
    assert is_error
    assert "failed unexpectedly" in text