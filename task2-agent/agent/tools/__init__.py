"""The list of tools the agent can use, and one safe way to run them."""

import logging

from agent.tools import calculator, weather, wikipedia
from agent.tools.base import ToolError, check_arguments

log = logging.getLogger(__name__)

TOOLS = {tool.name: tool for tool in (calculator.TOOL, weather.TOOL, wikipedia.TOOL)}


def tool_schemas():
    """The tool definitions to send to the model."""
    return [tool.schema() for tool in TOOLS.values()]


def run_tool(name, arguments):
    """Run one tool call and return (text, is_error). It never raises.

    Errors come back as text so the model can read them and explain the problem.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: there is no tool called '{name}'. Available tools: {', '.join(TOOLS)}.", True

    try:
        clean = check_arguments(tool, arguments)
        return str(tool.func(**clean)), False
    except ToolError as error:
        return f"Error: {error}", True
    except Exception:
        log.exception("Tool %s failed unexpectedly", name)
        return "Error: the tool failed unexpectedly.", True