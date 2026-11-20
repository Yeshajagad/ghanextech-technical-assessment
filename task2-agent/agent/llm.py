"""Talk to the chat model, with tool calling.

Local: Ollama (qwen3 by default). Deployed demo: Groq (LLM_PROVIDER=groq).

The agent keeps its messages in one neutral format and this file translates
them for whichever provider is in use:
  {"role": "system" | "user", "content": text}
  {"role": "assistant", "content": text, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "tool_call_id": id, "name": tool name, "content": text}
"""

import json
from dataclasses import dataclass, field

import ollama

from agent import config


class LLMError(Exception):
    """Raised with a message that is safe to show to the user."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str
    tool_calls: list = field(default_factory=list)


def chat(messages, tools, client=None):
    """Send the conversation and the tool definitions; return a Reply."""
    if config.LLM_PROVIDER == "ollama":
        return _chat_ollama(messages, tools, client)
    if config.LLM_PROVIDER == "groq":
        return _chat_groq(messages, tools, client)
    raise LLMError(f"LLM_PROVIDER must be 'ollama' or 'groq' (got '{config.LLM_PROVIDER}').")


# ---------- Ollama ----------

def _to_ollama(messages):
    converted = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{"function": {"name": c.name, "arguments": c.arguments}} for c in m["tool_calls"]]
            converted.append({"role": "assistant", "content": m["content"], "tool_calls": calls})
        elif m["role"] == "tool":
            converted.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
        else:
            converted.append({"role": m["role"], "content": m["content"]})
    return converted


def _chat_ollama(messages, tools, client):
    client = client or ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT)
    extra = {}
    if "qwen3" in config.OLLAMA_CHAT_MODEL:
        extra["think"] = False  # qwen3 would otherwise print its reasoning first

    try:
        response = client.chat(
            model=config.OLLAMA_CHAT_MODEL,
            messages=_to_ollama(messages),
            tools=tools,
            options={"temperature": 0},
            **extra,
        )
    except ConnectionError:
        raise LLMError(
            f"Cannot reach Ollama at {config.OLLAMA_HOST}. "
            "Start it with `ollama serve` (or open the Ollama app)."
        )
    except ollama.ResponseError as error:
        if error.status_code == 404:
            raise LLMError(
                f"The model '{config.OLLAMA_CHAT_MODEL}' is not installed. "
                f"Run: ollama pull {config.OLLAMA_CHAT_MODEL}"
            )
        if "does not support tools" in str(error.error):
            raise LLMError(
                f"The model '{config.OLLAMA_CHAT_MODEL}' cannot use tools. Try qwen3 or llama3.2."
            )
        raise LLMError(f"Ollama returned an error: {error.error}")

    message = response.message
    calls = [
        # Ollama does not give calls an id, so make one up from the position.
        ToolCall(id=f"call_{i}", name=c.function.name, arguments=dict(c.function.arguments))
        for i, c in enumerate(message.tool_calls or [])
    ]
    return Reply(content=message.content or "", tool_calls=calls)


# ---------- Groq ----------

def _to_groq(messages):
    converted = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in m["tool_calls"]
            ]
            converted.append({"role": "assistant", "content": m["content"], "tool_calls": calls})
        elif m["role"] == "tool":
            converted.append(
                {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
            )
        else:
            converted.append({"role": m["role"], "content": m["content"]})
    return converted


def _chat_groq(messages, tools, client):
    import groq  # only the deployed demo needs this package

    if client is None:
        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set. Add it to the app's secrets.")
        client = groq.Groq(api_key=config.GROQ_API_KEY)

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=_to_groq(messages),
            tools=tools,
            temperature=0,
        )
    except groq.AuthenticationError:
        raise LLMError("The Groq API key was rejected. Check that it is correct.")
    except groq.RateLimitError:
        raise LLMError("Groq free-tier rate limit reached. Wait a minute and try again.")
    except groq.APIConnectionError:
        raise LLMError("Could not reach the Groq API. Check your internet connection.")
    except groq.APIStatusError as error:
        if error.status_code == 400:
            raise LLMError("The model produced an invalid tool call. Please try rephrasing.")
        raise LLMError(f"The Groq API returned an error ({error.status_code}).")

    message = response.choices[0].message
    calls = []
    for c in message.tool_calls or []:
        try:
            arguments = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}  # the tool check will report the missing parameters
        calls.append(ToolCall(id=c.id, name=c.function.name, arguments=arguments))
    return Reply(content=message.content or "", tool_calls=calls)