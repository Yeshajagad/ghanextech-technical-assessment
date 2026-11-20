import httpx
import pytest

from agent.tools.base import Tool, ToolError, check_arguments, get_json

TOOL = Tool(
    name="demo",
    description="A demo tool.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "count": {"type": "number"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
    func=lambda **kw: kw,
)


def client_returning(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- schema ----------

def test_schema_has_the_shape_models_expect():
    schema = TOOL.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "demo"
    assert schema["function"]["parameters"]["required"] == ["city"]


# ---------- check_arguments ----------

def test_valid_arguments_pass():
    assert check_arguments(TOOL, {"city": "Paris", "count": 2}) == {"city": "Paris", "count": 2}


def test_missing_required_parameter():
    with pytest.raises(ToolError, match="Missing required parameter 'city'"):
        check_arguments(TOOL, {})


def test_wrong_type_is_rejected():
    with pytest.raises(ToolError, match="'city' must be a string"):
        check_arguments(TOOL, {"city": 5})


def test_boolean_is_not_a_number():
    with pytest.raises(ToolError, match="'count' must be a number"):
        check_arguments(TOOL, {"city": "Paris", "count": True})


def test_enum_value_is_normalised():
    assert check_arguments(TOOL, {"city": "Paris", "units": " Fahrenheit "})["units"] == "fahrenheit"


def test_enum_rejects_other_values():
    with pytest.raises(ToolError, match="one of: celsius, fahrenheit"):
        check_arguments(TOOL, {"city": "Paris", "units": "kelvin"})


def test_unknown_parameters_are_ignored():
    assert check_arguments(TOOL, {"city": "Paris", "colour": "red"}) == {"city": "Paris"}


def test_arguments_must_be_an_object():
    with pytest.raises(ToolError, match="JSON object"):
        check_arguments(TOOL, ["Paris"])


# ---------- get_json ----------

def test_get_json_returns_the_body():
    client = client_returning(lambda request: httpx.Response(200, json={"ok": True}))
    assert get_json("https://example.com", {}, client) == {"ok": True}


def test_timeout_becomes_a_tool_error():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ToolError, match="too long"):
        get_json("https://example.com", {}, client_returning(handler))


def test_rate_limit_becomes_a_tool_error():
    client = client_returning(lambda request: httpx.Response(429))
    with pytest.raises(ToolError, match="limiting requests"):
        get_json("https://example.com", {}, client)


def test_server_error_includes_the_status():
    client = client_returning(lambda request: httpx.Response(503))
    with pytest.raises(ToolError, match="503"):
        get_json("https://example.com", {}, client)


def test_connection_problem_becomes_a_tool_error():
    def handler(request):
        raise httpx.ConnectError("no network")

    with pytest.raises(ToolError, match="Could not reach"):
        get_json("https://example.com", {}, client_returning(handler))


def test_unreadable_reply_becomes_a_tool_error():
    client = client_returning(lambda request: httpx.Response(200, text="not json"))
    with pytest.raises(ToolError, match="could not be read"):
        get_json("https://example.com", {}, client)