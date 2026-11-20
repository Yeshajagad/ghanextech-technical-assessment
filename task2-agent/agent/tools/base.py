"""Shared pieces for every tool: the Tool container, argument checks and the HTTP helper."""

from dataclasses import dataclass
from typing import Callable

import httpx

from agent import config


class ToolError(Exception):
    """A problem the model should hear about, worded so a user can understand it."""


@dataclass
class Tool:
    name: str
    description: str  # the model reads this to decide when to use the tool
    parameters: dict  # JSON schema describing the arguments
    func: Callable

    def schema(self):
        """The definition sent to the model (the format Ollama and Groq both accept)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


PYTHON_TYPES = {"string": str, "number": (int, float)}


def check_arguments(tool, arguments):
    """Return the arguments cleaned up, or raise ToolError saying what is wrong."""
    if not isinstance(arguments, dict):
        raise ToolError("Arguments must be a JSON object.")

    properties = tool.parameters.get("properties", {})
    for name in tool.parameters.get("required", []):
        if name not in arguments:
            raise ToolError(f"Missing required parameter '{name}'.")

    clean = {}
    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            continue  # ignore parameters the tool does not define
        expected = PYTHON_TYPES[spec["type"]]
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ToolError(f"Parameter '{name}' must be a {spec['type']}.")
        if "enum" in spec:
            value = value.strip().lower()
            if value not in spec["enum"]:
                raise ToolError(f"Parameter '{name}' must be one of: {', '.join(spec['enum'])}.")
        clean[name] = value
    return clean


def get_json(url, params, client=None):
    """GET a URL and return its JSON. Network problems become a ToolError."""
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=config.HTTP_TIMEOUT, headers={"User-Agent": config.USER_AGENT})
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        raise ToolError("The service took too long to respond. Try again in a moment.")
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 429:
            raise ToolError("The service is limiting requests right now. Try again in a minute.")
        raise ToolError(f"The service returned an error ({error.response.status_code}).")
    except httpx.RequestError:
        raise ToolError("Could not reach the service. Check the internet connection.")
    except ValueError:
        raise ToolError("The service sent a reply that could not be read.")
    finally:
        if own_client:
            client.close()